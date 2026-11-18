"""FastAPI application.

Endpoints
----------
GET  /healthz                 liveness
GET  /schemas/{schema}/tables list ordinary base tables (catalog read only)
GET  /schemas/{schema}/tables/{table}/structure
                              full supported-structure view + rejections
POST /jobs                    create a generation/verification job
GET  /jobs/{job_id}           status, conflict report, logs
GET  /jobs/{job_id}/download  zip package, available only after SUCCEEDED
GET  /jobs/{job_id}/row-load-plan
                              single-transaction row-by-row INSERT replay of
                              the published sample for FK-enabled targets
GET  /docs                    OpenAPI Swagger UI
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response

from .config import settings
from .db import source_connection
from .introspect import (
    IntrospectionError,
    introspect_tables,
    list_base_tables,
)
from .jobs import STORE, package_zip, run_job
from .models import GenerateRequest, JobStatus


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="数仓关联样本生成服务",
    version="1.0.0",
    description=(
        "只读源库表结构（不读行数据），基于 z3 生成同时满足指定行数、整数闭区间与"
        "主键/唯一键/可空/MATCH SIMPLE/FULL 复合及循环外键约束的合成样本，"
        "并在隔离空库实际装载验证通过后才开放 SQL 包下载。"
    ),
    lifespan=lifespan,
)


@app.get("/healthz", tags=["ops"])
def healthz():
    return {"status": "ok"}


@app.get("/schemas/{schema}/tables", tags=["catalog"])
def list_tables(schema: str):
    """List ordinary base tables; non-base relations are flagged, not read."""
    try:
        with source_connection(settings.source_dsn) as conn:
            base, others = list_base_tables(conn, schema)
    except Exception as exc:
        raise HTTPException(502, f"无法连接源库: {exc.__class__.__name__}: {exc}")
    return {
        "schema": schema,
        "base_tables": base,
        "excluded_non_base_relations": others,
        "max_selectable_tables": settings.max_tables,
        "max_rows_per_table": settings.max_rows_per_table,
    }


@app.get("/schemas/{schema}/tables/{table}/structure", tags=["catalog"])
def table_structure(schema: str, table: str):
    """Return the supported structural view, or explicit rejections."""
    try:
        with source_connection(settings.source_dsn) as conn:
            found = introspect_tables(conn, schema, [table])
    except IntrospectionError as exc:
        return JSONResponse(
            status_code=422,
            content={
                "table": f"{schema}.{table}",
                "supported": False,
                "rejections": [
                    {"object": o, "reason": r} for o, r in exc.rejections
                ],
            },
        )
    except Exception as exc:
        raise HTTPException(502, f"无法连接源库: {exc.__class__.__name__}: {exc}")
    ts = found[table]
    return {"table": f"{schema}.{table}", "supported": True, "structure": ts.model_dump()}


@app.post("/jobs", tags=["jobs"], status_code=202)
def create_job(req: GenerateRequest, background: BackgroundTasks):
    if not req.tables:
        raise HTTPException(422, "tables 不能为空")
    if len({t.table for t in req.tables}) != len(req.tables):
        raise HTTPException(422, "请求中存在重复表")
    if len(req.tables) > settings.max_tables:
        raise HTTPException(
            422, f"至多选择 {settings.max_tables} 张表（不含外键闭包扩展）"
        )
    job_id = STORE.create()
    background.add_task(run_job, job_id, req)
    return {"job_id": job_id, "status": JobStatus.PENDING.value}


@app.get("/jobs/{job_id}", tags=["jobs"])
def get_job(job_id: str):
    job = STORE.get(job_id)
    if job is None:
        raise HTTPException(404, "job 不存在或已过期")
    return job.model_dump()


@app.get("/jobs/{job_id}/download", tags=["jobs"])
def download_job(job_id: str):
    job = STORE.get(job_id)
    if job is None:
        raise HTTPException(404, "job 不存在或已过期")
    if job.status != JobStatus.SUCCEEDED:
        raise HTTPException(
            409,
            f"作业尚未验证成功（当前状态 {job.status.value}）；失败作业不提供部分包下载",
        )
    package = STORE.package(job_id)
    if package is None:
        raise HTTPException(409, "SQL 包不可用")
    data = package_zip(package)
    return Response(
        content=data,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="sample_{job_id}.zip"',
            "X-Contains-Real-Patient-Data": "false",
        },
    )


@app.get(
    "/jobs/{job_id}/row-load-plan",
    tags=["jobs"],
    responses={
        200: {"description": "可执行的逐行装载计划（JSON；加 ?format=sql 返回纯 SQL 脚本）"},
        404: {"description": "job 不存在或已过期"},
        409: {"description": "作业尚未验证成功，拒绝提供计划"},
        422: {"description": "即时外键引用环：返回涉及行与外键，不提供部分脚本"},
    },
)
def row_load_plan(job_id: str, format: str | None = None):
    """Row-by-row replay of the published sample into an FK-enabled target.

    The plan is computed once at publish time from the verified structures
    and rows; this endpoint only serves the stored artefact, so source
    catalog changes and concurrent queries cannot alter it. Immediate
    foreign keys are topologically ordered (never deferred, constraints stay
    enabled); DEFERRABLE INITIALLY DEFERRED keys hold at COMMIT. If the
    immediate references form a cycle, the involved rows and foreign keys
    are returned (422) and no partial script is produced.
    """
    job = STORE.get(job_id)
    if job is None:
        raise HTTPException(404, "job 不存在或已过期")
    if job.status != JobStatus.SUCCEEDED:
        raise HTTPException(
            409,
            f"作业尚未验证成功（当前状态 {job.status.value}）；"
            f"未成功作业不提供行装载计划",
        )
    outcome = STORE.row_plan(job_id)
    if outcome is None:
        raise HTTPException(409, "行装载计划不可用")
    state = outcome.get("state")
    if state == "cycle":
        return JSONResponse(status_code=422, content=outcome["detail"])
    if state != "ready":
        raise HTTPException(
            500, f"行装载计划生成失败: {outcome.get('message', '未知错误')}"
        )
    body = outcome["body"]
    if format == "sql":
        return Response(
            content=body["sql"], media_type="text/plain; charset=utf-8"
        )
    return body
