"""Row-load-plan tests: ordering, NULL/match semantics, self references,
immediate vs deferred cycles, API behaviour, snapshot immutability."""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.jobs import STORE, run_job
from app.main import app
from app.models import (
    Column,
    ForeignKey,
    GenerateRequest,
    JobStatus,
    MatchType,
    TableSchema,
    UniqueConstraint,
)
from app.rowplan import ImmediateCycleError, compute_row_load_plan
from app.sqlgen import render_constraints, render_schema

client = TestClient(app)


# ------------------------------------------------------------------ helpers
def _col(name, nullable=False, pg_type="integer"):
    return Column(
        name=name,
        pg_type=pg_type,
        ordinal=1,
        nullable=nullable,
        has_default=False,
        is_identity=False,
        is_generated=False,
    )


def _fk(name, columns, ref_table, ref_columns, match=MatchType.SIMPLE,
        not_deferrable=True, initially_deferred=False):
    return ForeignKey(
        name=name,
        columns=columns,
        ref_table=ref_table,
        ref_columns=ref_columns,
        match=match,
        on_update="NO ACTION",
        on_delete="NO ACTION",
        not_deferrable=not_deferrable,
        initially_deferred=initially_deferred,
    )


def _table(name, columns, pk=None, fks=(), uniques=()):
    ucs = []
    if pk:
        ucs.append(
            UniqueConstraint(name=f"{name}_pk", columns=list(pk), is_primary=True)
        )
    ucs += [
        UniqueConstraint(name=n, columns=list(cols), is_primary=False)
        for n, cols in uniques
    ]
    return TableSchema(
        schema_name="public",
        name=name,
        columns=columns,
        unique_constraints=ucs,
        foreign_keys=list(fks),
    )


def _parent_child(**fk_kw):
    p = _table("p", [_col("id")], pk=["id"])
    c = _table(
        "c",
        [_col("id"), _col("pid")],
        pk=["id"],
        fks=[_fk("c_p_fk", ["pid"], "p", ["id"], **fk_kw)],
    )
    return {"p": p, "c": c}


def _cycle_tables(**fk_kw):
    a = _table(
        "a", [_col("id"), _col("b_id")], pk=["id"],
        fks=[_fk("a_b_fk", ["b_id"], "b", ["id"], **fk_kw)],
    )
    b = _table(
        "b", [_col("id"), _col("a_id")], pk=["id"],
        fks=[_fk("b_a_fk", ["a_id"], "a", ["id"], **fk_kw)],
    )
    return {"a": a, "b": b}


def _run(req_dict):
    req = GenerateRequest(**req_dict)
    jid = STORE.create()
    run_job(jid, req)
    return jid, STORE.get(jid)


def _load(name):
    raw = json.load(open(f"/workspace/examples/{name}"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _order_pairs(body):
    return [(e["table"], e["row_index"]) for e in body["order"]]


def _issues_no_set_constraints(sql: str) -> bool:
    """True when no executable (non-comment) line defers constraints."""
    return all(
        "SET CONSTRAINTS" not in ln
        for ln in sql.splitlines()
        if not ln.lstrip().startswith("--")
    )


# ------------------------------------------------------- unit: ordering rules
def test_linear_plan_orders_referenced_rows_first():
    tables = _parent_child()
    data = {
        "p": [{"id": 1}, {"id": 2}],
        "c": [{"id": 10, "pid": 2}, {"id": 11, "pid": 1}],
    }
    body = compute_row_load_plan("job1", "public", tables, ["p", "c"], data)
    pos = {(t, i): seq for seq, (t, i) in enumerate(_order_pairs(body), 1)}
    assert pos[("p", 1)] < pos[("c", 0)]  # p row id=2 before the c row pid=2
    assert pos[("p", 0)] < pos[("c", 1)]
    assert body["insert_count"] == 4
    assert body["deferred_foreign_keys"] == []
    assert body["sql"].count("INSERT INTO") == 4
    assert _issues_no_set_constraints(body["sql"])
    assert "BEGIN;" in body["sql"] and "COMMIT;" in body["sql"]


def test_match_simple_mixed_null_needs_no_referent():
    p = _table(
        "p", [_col("k1"), _col("k2", nullable=True)],
        uniques=[("p_uk", ["k1", "k2"])],
    )
    c = _table(
        "c",
        [_col("id"), _col("f1", nullable=True), _col("f2", nullable=True)],
        pk=["id"],
        fks=[_fk("c_p_fk", ["f1", "f2"], "p", ["k1", "k2"])],
    )
    tables = {"p": p, "c": c}
    # Parent table is EMPTY: only NULL-exempt child rows may exist.
    data = {
        "p": [],
        "c": [{"id": 1, "f1": None, "f2": 10}, {"id": 2, "f1": None, "f2": None}],
    }
    body = compute_row_load_plan("job2", "public", tables, ["p", "c"], data)
    assert body["insert_count"] == 2
    assert body["immediate_fk_edges"] == 0


def test_match_full_all_null_exempt_and_full_key_ordered():
    p = _table("p", [_col("k1"), _col("k2")], uniques=[("p_uk", ["k1", "k2"])])
    c = _table(
        "c",
        [_col("id"), _col("f1", nullable=True), _col("f2", nullable=True)],
        pk=["id"],
        fks=[_fk("c_p_fk", ["f1", "f2"], "p", ["k1", "k2"], match=MatchType.FULL)],
    )
    tables = {"p": p, "c": c}
    data = {
        "p": [{"k1": 1, "k2": 10}],
        "c": [
            {"id": 1, "f1": None, "f2": None},  # all-NULL key: exempt
            {"id": 2, "f1": 1, "f2": 10},       # full key: needs p row 0
        ],
    }
    body = compute_row_load_plan("job3", "public", tables, ["p", "c"], data)
    # Deterministic: p row first, exempt child row next, dependent row last.
    assert _order_pairs(body) == [("p", 0), ("c", 0), ("c", 1)]


def test_match_full_mixed_null_rejected_defensively():
    p = _table("p", [_col("k1"), _col("k2")], uniques=[("p_uk", ["k1", "k2"])])
    c = _table(
        "c",
        [_col("id"), _col("f1", nullable=True), _col("f2", nullable=True)],
        pk=["id"],
        fks=[_fk("c_p_fk", ["f1", "f2"], "p", ["k1", "k2"], match=MatchType.FULL)],
    )
    tables = {"p": p, "c": c}
    # Verified published data can never contain this; the planner must not
    # silently treat it as exempt.
    data = {"p": [{"k1": 1, "k2": 10}], "c": [{"id": 1, "f1": None, "f2": 10}]}
    with pytest.raises(ValueError):
        compute_row_load_plan("job3b", "public", tables, ["p", "c"], data)


def test_self_referencing_row_written_directly():
    node = _table(
        "node", [_col("id"), _col("parent", nullable=True)], pk=["id"],
        fks=[_fk("node_parent_fk", ["parent"], "node", ["id"])],
    )
    tables = {"node": node}
    data = {
        "node": [
            {"id": 1, "parent": 1},  # references itself: no ordering edge
            {"id": 2, "parent": 1},  # references row 0
        ]
    }
    body = compute_row_load_plan("job4", "public", tables, ["node"], data)
    assert _order_pairs(body) == [("node", 0), ("node", 1)]
    assert body["immediate_fk_edges"] == 1  # the self-loop added no edge


def test_two_row_self_cycle_reported():
    node = _table(
        "node", [_col("id"), _col("parent", nullable=True)], pk=["id"],
        fks=[_fk("node_parent_fk", ["parent"], "node", ["id"])],
    )
    tables = {"node": node}
    data = {"node": [{"id": 1, "parent": 2}, {"id": 2, "parent": 1}]}
    with pytest.raises(ImmediateCycleError) as exc:
        compute_row_load_plan("job5", "public", tables, ["node"], data)
    detail = exc.value.detail
    assert detail["kind"] == "immediate_fk_cycle"
    assert "sql" not in detail  # never a partial script
    rows = detail["cycles"][0]["rows"]
    assert {r["row_index"] for r in rows} == {0, 1}
    assert all(r["key"] is not None and r["values"] for r in rows)
    fks = detail["cycles"][0]["foreign_keys"]
    assert {f["constraint"] for f in fks} == {"node_parent_fk"}
    assert {(f["from_row"]["row_index"], f["to_row"]["row_index"]) for f in fks} == {
        (0, 1),
        (1, 0),
    }


def test_immediate_cross_table_cycle_reported():
    tables = _cycle_tables()
    data = {"a": [{"id": 1, "b_id": 10}], "b": [{"id": 10, "a_id": 1}]}
    with pytest.raises(ImmediateCycleError) as exc:
        compute_row_load_plan("job6", "public", tables, ["a", "b"], data)
    fks = {f["constraint"] for f in exc.value.detail["cycles"][0]["foreign_keys"]}
    assert fks == {"a_b_fk", "b_a_fk"}


def test_initially_immediate_deferrable_cycle_still_reported():
    # DEFERRABLE INITIALLY IMMEDIATE is immediate unless SET CONSTRAINTS is
    # issued — and the plan never defers it.
    tables = _cycle_tables(not_deferrable=False, initially_deferred=False)
    data = {"a": [{"id": 1, "b_id": 10}], "b": [{"id": 10, "a_id": 1}]}
    with pytest.raises(ImmediateCycleError):
        compute_row_load_plan("job7", "public", tables, ["a", "b"], data)


def test_initially_deferred_cycle_is_planned_without_edges():
    tables = _cycle_tables(not_deferrable=False, initially_deferred=True)
    data = {"a": [{"id": 1, "b_id": 10}], "b": [{"id": 10, "a_id": 1}]}
    body = compute_row_load_plan("job8", "public", tables, ["a", "b"], data)
    assert body["insert_count"] == 2
    assert body["immediate_fk_edges"] == 0
    assert sorted(body["deferred_foreign_keys"]) == ["a.a_b_fk", "b.b_a_fk"]


def test_zero_row_sample_yields_empty_transaction():
    t = _table("t", [_col("id")], pk=["id"])
    body = compute_row_load_plan("job9", "public", {"t": t}, ["t"], {"t": []})
    assert body["insert_count"] == 0
    assert "no rows" in body["sql"]
    assert "BEGIN;" in body["sql"] and "COMMIT;" in body["sql"]


# ------------------------------------------- real database: scripts actually load
def test_linear_plan_script_loads_with_fks_enabled(fresh_verify_database):
    tables = _parent_child()
    data = {
        "p": [{"id": 1}, {"id": 2}],
        "c": [{"id": 10, "pid": 2}, {"id": 11, "pid": 1}],
    }
    body = compute_row_load_plan("j", "public", tables, ["p", "c"], data)
    dsn, _ = fresh_verify_database
    with psycopg.connect(dsn, autocommit=False) as conn:
        cur = conn.cursor()
        cur.execute(render_schema(tables, ["p", "c"]))
        cur.execute(render_constraints(tables, ["p", "c"]))  # FK live BEFORE rows
        cur.execute(body["sql"])  # one transaction, constraints untouched
        conn.commit()
        cur.execute('SELECT count(*) FROM "public"."p"')
        assert cur.fetchone()[0] == 2
        cur.execute('SELECT count(*) FROM "public"."c"')
        assert cur.fetchone()[0] == 2


def test_mixed_null_simple_script_loads(fresh_verify_database):
    p = _table(
        "p", [_col("k1"), _col("k2", nullable=True)],
        uniques=[("p_uk", ["k1", "k2"])],
    )
    c = _table(
        "c",
        [_col("id"), _col("f1", nullable=True), _col("f2", nullable=True)],
        pk=["id"],
        fks=[_fk("c_p_fk", ["f1", "f2"], "p", ["k1", "k2"])],
    )
    tables = {"p": p, "c": c}
    data = {
        "p": [{"k1": 1, "k2": 10}],
        "c": [{"id": 1, "f1": None, "f2": 10}, {"id": 2, "f1": 1, "f2": 10}],
    }
    body = compute_row_load_plan("j", "public", tables, ["p", "c"], data)
    dsn, _ = fresh_verify_database
    with psycopg.connect(dsn, autocommit=False) as conn:
        cur = conn.cursor()
        cur.execute(render_schema(tables, ["p", "c"]))
        cur.execute(render_constraints(tables, ["p", "c"]))
        cur.execute(body["sql"])
        conn.commit()
        cur.execute('SELECT count(*) FROM "public"."c"')
        assert cur.fetchone()[0] == 2


def test_deferred_cycle_script_loads_and_commit_enforces(fresh_verify_database):
    tables = _cycle_tables(not_deferrable=False, initially_deferred=True)
    data = {"a": [{"id": 1, "b_id": 10}], "b": [{"id": 10, "a_id": 1}]}
    body = compute_row_load_plan("j", "public", tables, ["a", "b"], data)
    dsn, _ = fresh_verify_database
    with psycopg.connect(dsn, autocommit=False) as conn:
        cur = conn.cursor()
        cur.execute(render_schema(tables, ["a", "b"]))
        cur.execute(render_constraints(tables, ["a", "b"]))
        # The mutually referencing rows load in ANY order because both FKs
        # are INITIALLY DEFERRED; the script's own COMMIT validates them.
        cur.execute(body["sql"])
        conn.commit()
        cur.execute('SELECT count(*) FROM "public"."a"')
        assert cur.fetchone()[0] == 1
        # The deferred FK is still enforced at COMMIT: a dangling row fails.
        with pytest.raises(psycopg.Error) as excinfo:
            cur.execute(
                'BEGIN; INSERT INTO "public"."b" ("id", "a_id") '
                "VALUES (99, 777); COMMIT;"
            )
        assert excinfo.value.sqlstate == "23503"
        conn.rollback()


# ------------------------------------------------------------------ API level
def test_acyclic_job_plan_replays_into_fk_enabled_schema(
    source_dsn, fresh_verify_database
):
    jid, job = _run(_load("request_row_load_demo.json"))
    assert job.status == JobStatus.SUCCEEDED
    r = client.get(f"/jobs/{jid}/row-load-plan")
    assert r.status_code == 200
    body = r.json()
    assert body["insert_count"] == 9
    assert body["deferred_foreign_keys"] == []
    assert set(body["selected_tables"]) == {"warehouse", "dim_item"}
    assert _issues_no_set_constraints(body["sql"])
    # Every dim_item row is sequenced after the warehouse row it references.
    wh_seq = {
        e["values"]["wh_id"]: e["seq"]
        for e in body["order"]
        if e["table"] == "warehouse"
    }
    for e in body["order"]:
        if e["table"] == "dim_item":
            assert e["seq"] > wh_seq[e["values"]["wh_id"]]
    # Real replay: schema + constraints from the package (FKs enabled),
    # then the single-transaction row-by-row script.
    pkg = STORE.package(jid)
    dsn, _ = fresh_verify_database
    with psycopg.connect(dsn, autocommit=False) as conn:
        cur = conn.cursor()
        cur.execute(pkg["01_schema.sql"])
        cur.execute(pkg["03_constraints.sql"])
        cur.execute(body["sql"])
        conn.commit()
        cur.execute("SELECT count(*) FROM public.warehouse")
        assert cur.fetchone()[0] == 3
        cur.execute("SELECT count(*) FROM public.dim_item")
        assert cur.fetchone()[0] == 6


def test_cyclic_job_plan_reports_immediate_cycle(source_dsn):
    jid, job = _run(_load("request_cyclic_four_tables.json"))
    assert job.status == JobStatus.SUCCEEDED
    r = client.get(f"/jobs/{jid}/row-load-plan")
    assert r.status_code == 422
    body = r.json()
    assert body["kind"] == "immediate_fk_cycle"
    assert "sql" not in body  # no partial script
    fk_names = {
        fk["constraint"] for c in body["cycles"] for fk in c["foreign_keys"]
    }
    assert {"employee_dept_fk", "dept_mgr_fk"} <= fk_names
    rows = [row for c in body["cycles"] for row in c["rows"]]
    assert rows
    assert all({"table", "row_index", "key", "values"} <= set(row) for row in rows)
    # The cycle report does not affect package availability.
    assert client.get(f"/jobs/{jid}/download").status_code == 200


def test_plan_rejected_unless_job_succeeded(source_dsn):
    assert client.get("/jobs/no-such-job/row-load-plan").status_code == 404
    pending = STORE.create()
    assert client.get(f"/jobs/{pending}/row-load-plan").status_code == 409
    jid, job = _run(_load("request_unsat_conflict.json"))
    assert job.status == JobStatus.FAILED
    assert client.get(f"/jobs/{jid}/row-load-plan").status_code == 409


def test_plan_sql_format_and_cycle_has_no_script(source_dsn):
    jid, job = _run(_load("request_row_load_demo.json"))
    assert job.status == JobStatus.SUCCEEDED
    r = client.get(f"/jobs/{jid}/row-load-plan", params={"format": "sql"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert "BEGIN;" in r.text and "COMMIT;" in r.text
    assert r.text.count("INSERT INTO") == 9
    jid2, job2 = _run(_load("request_cyclic_four_tables.json"))
    assert job2.status == JobStatus.SUCCEEDED
    r2 = client.get(f"/jobs/{jid2}/row-load-plan", params={"format": "sql"})
    assert r2.status_code == 422


def test_plan_ignores_source_db_and_concurrent_reads(source_dsn, monkeypatch):
    jid, job = _run(_load("request_row_load_demo.json"))
    assert job.status == JobStatus.SUCCEEDED
    first = client.get(f"/jobs/{jid}/row-load-plan")
    assert first.status_code == 200

    # The endpoint must not consult the source database at all.
    import app.jobs as jobsmod
    import app.main as mainmod

    def boom(*_a, **_k):
        raise AssertionError("row-load-plan touched the source database")

    monkeypatch.setattr(mainmod, "source_connection", boom)
    monkeypatch.setattr(jobsmod, "source_connection", boom)
    again = client.get(f"/jobs/{jid}/row-load-plan")
    assert again.status_code == 200
    assert again.content == first.content

    # Concurrent readers all observe the same stored plan.
    def fetch(_):
        return client.get(f"/jobs/{jid}/row-load-plan").content

    with ThreadPoolExecutor(max_workers=8) as pool:
        bodies = list(pool.map(fetch, range(8)))
    assert all(b == first.content for b in bodies)


def _admin_source_dsn() -> str:
    info = psycopg.conninfo.conninfo_to_dict(
        os.environ.get(
            "SOURCE_DSN", "postgresql://postgres@/appdb?host=/tmp/pgdata/source"
        )
    )
    info["user"] = "postgres"
    info.pop("password", None)
    return psycopg.conninfo.make_conninfo(**info)


def test_plan_immune_to_later_source_catalog_changes(source_dsn):
    jid, job = _run(_load("request_row_load_demo.json"))
    assert job.status == JobStatus.SUCCEEDED
    before = client.get(f"/jobs/{jid}/row-load-plan").content
    admin = _admin_source_dsn()
    with psycopg.connect(admin, autocommit=True) as c:
        c.cursor().execute("ALTER TABLE public.warehouse ADD COLUMN scratch integer")
    try:
        after = client.get(f"/jobs/{jid}/row-load-plan").content
        assert after == before
    finally:
        with psycopg.connect(admin, autocommit=True) as c:
            c.cursor().execute("ALTER TABLE public.warehouse DROP COLUMN scratch")
