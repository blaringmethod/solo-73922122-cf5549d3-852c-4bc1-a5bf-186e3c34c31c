"""Row-by-row load plans for published samples.

The regular package loads into an empty database structurally: tables are
created without foreign keys, all rows are inserted, and the FKs are added
afterwards. A *row-load plan* is the stricter replay artefact requested by
the warehouse loader: the target schema already exists with EVERY constraint
enabled (primary keys, unique keys and all foreign keys), and the published
rows must be replayed one INSERT at a time inside a single transaction.

Ordering rules (the service's safety guarantees applied row by row):

* constraints are never disabled and immediate foreign keys are never
  deferred — the script issues no ``SET CONSTRAINTS`` at all;
* a ``DEFERRABLE INITIALLY DEFERRED`` foreign key is checked at COMMIT; the
  published rows satisfy it globally, so it imposes no ordering edge;
* NULL-bearing keys follow SQL match semantics: under MATCH SIMPLE any NULL
  component exempts the row, under MATCH FULL an all-NULL key is exempt —
  exempt rows impose no ordering edge;
* a row whose self-referencing key points at itself is satisfied by its own
  INSERT and imposes no ordering edge;
* every other immediate reference adds an edge "referenced row first", and a
  deterministic Kahn topological sort linearises the rows.

If the immediate edges contain a row-level cycle no linear order exists: the
plan is refused wholesale — the rows and foreign keys sitting on the cycle
are reported and no partial script is produced.

The plan is computed exactly once, at publish time, from the structures and
rows that were verified; later source-catalog changes cannot alter it and
concurrent readers all observe the same stored plan. The source database is
never queried for row data (here or anywhere else in the service).
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Callable

from .models import ForeignKey, MatchType, TableSchema
from .solver import Row
from .sqlgen import q


@dataclass(frozen=True)
class RowId:
    """Stable identity of one published row: table + ordinal in the row list."""

    table: str
    index: int  # 0-based position in the published row list of the table


@dataclass(frozen=True)
class FkEdge:
    """One immediate-FK dependency: ``from_row`` needs ``to_row`` present."""

    fk_name: str
    columns: tuple[str, ...]
    ref_table: str
    ref_columns: tuple[str, ...]
    from_row: RowId
    to_row: RowId


class ImmediateCycleError(Exception):
    """The published rows cannot be linearised under immediate foreign keys.

    Carries a JSON-ready ``detail`` describing every row and foreign key that
    sits on an immediate-reference cycle. No partial script is ever produced.
    """

    def __init__(self, detail: dict, n_rows: int):
        self.detail = detail
        self.n_rows = n_rows
        super().__init__(
            f"unsortable immediate FK cycle involving {n_rows} row(s)"
        )


# --------------------------------------------------------------------- edges
def _fk_exempt(fk: ForeignKey, row: Row) -> bool:
    """True when the row's key needs no referent under SQL match semantics."""
    vals = [row[c] for c in fk.columns]
    if len(vals) == 1:
        # Single-column key: NULL is exempt under both match types.
        return vals[0] is None
    null_flags = [v is None for v in vals]
    if fk.match == MatchType.FULL:
        if all(null_flags):
            return True  # all-NULL composite key is exempt
        if any(null_flags):
            # MATCH FULL forbids mixed NULL keys; verified published data can
            # never contain one, so reaching this is a service-side bug.
            raise ValueError(
                f"MATCH FULL 外键 {fk.name} 出现混合 NULL 键，已发布数据不应存在该状态"
            )
        return False
    # MATCH SIMPLE: any NULL component exempts the row.
    return any(null_flags)


def _reference_index(
    ts: TableSchema, rows: list[Row], columns: tuple[str, ...]
) -> dict[tuple, int]:
    """Map referenced-key value tuple -> row ordinal.

    Rows with NULL in any referenced column can never be the target of a
    foreign-key match and are excluded. Referenced columns are unique in the
    source catalog, so a duplicate non-NULL key means corrupted input.
    """
    index: dict[tuple, int] = {}
    for i, row in enumerate(rows):
        key = tuple(row[c] for c in columns)
        if any(v is None for v in key):
            continue
        if key in index:
            raise ValueError(
                f"表 {ts.name} 的被引用键 {columns} 出现重复值 {key}，违反唯一性"
            )
        index[key] = i
    return index


def _build_edges(
    tables: dict[str, TableSchema],
    selected: list[str],
    data: dict[str, list[Row]],
) -> tuple[dict[RowId, set[RowId]], list[FkEdge], list[str]]:
    """Return (dependencies, FK edges, deferred FK labels)."""
    deps: dict[RowId, set[RowId]] = {
        RowId(tname, i): set()
        for tname in selected
        for i in range(len(data[tname]))
    }
    edges: list[FkEdge] = []
    deferred: list[str] = []
    indexes: dict[tuple[str, tuple[str, ...]], dict[tuple, int]] = {}
    for tname in selected:
        ts = tables[tname]
        for fk in ts.foreign_keys:
            if fk.initially_deferred:
                # Checked at COMMIT; the published rows satisfy it globally,
                # so it imposes no ordering edge.
                deferred.append(f"{tname}.{fk.name}")
                continue
            ref_cols = tuple(fk.ref_columns)
            idx_key = (fk.ref_table, ref_cols)
            if idx_key not in indexes:
                indexes[idx_key] = _reference_index(
                    tables[fk.ref_table], data[fk.ref_table], ref_cols
                )
            index = indexes[idx_key]
            for i, row in enumerate(data[tname]):
                if _fk_exempt(fk, row):
                    continue
                key = tuple(row[c] for c in fk.columns)
                j = index.get(key)
                if j is None:
                    raise ValueError(
                        f"行 {tname}[{i}] 的外键 {fk.name} 在被引用表 "
                        f"{fk.ref_table} 中找不到目标，已发布数据不应悬空"
                    )
                src, dst = RowId(tname, i), RowId(fk.ref_table, j)
                if src == dst:
                    # Self-referencing row: satisfied by its own INSERT.
                    continue
                deps[src].add(dst)
                edges.append(
                    FkEdge(
                        fk.name,
                        tuple(fk.columns),
                        fk.ref_table,
                        ref_cols,
                        src,
                        dst,
                    )
                )
    return deps, edges, deferred


# --------------------------------------------------------------------- sort
def _topo_sort(
    deps: dict[RowId, set[RowId]], selected: list[str]
) -> tuple[list[RowId], list[RowId], Callable[[RowId], tuple]]:
    """Deterministic Kahn sort; returns (order, remaining, sort_key)."""
    table_order = {t: pos for pos, t in enumerate(selected)}

    def sort_key(rid: RowId) -> tuple:
        return (table_order[rid.table], rid.index)

    dependents: dict[RowId, set[RowId]] = {rid: set() for rid in deps}
    indeg: dict[RowId, int] = {}
    for rid, targets in deps.items():
        indeg[rid] = len(targets)
        for t in targets:
            dependents[t].add(rid)
    heap = [(sort_key(rid), rid) for rid, d in indeg.items() if d == 0]
    heapq.heapify(heap)
    order: list[RowId] = []
    while heap:
        _, rid = heapq.heappop(heap)
        order.append(rid)
        for dep in dependents[rid]:
            indeg[dep] -= 1
            if indeg[dep] == 0:
                heapq.heappush(heap, (sort_key(dep), dep))
    remaining = sorted((rid for rid, d in indeg.items() if d > 0), key=sort_key)
    return order, remaining, sort_key


def _cycle_sccs(
    remaining: list[RowId],
    deps: dict[RowId, set[RowId]],
    sort_key: Callable[[RowId], tuple],
) -> list[list[RowId]]:
    """Tarjan strongly connected components (size >= 2) within ``remaining``.

    Self-loops cannot occur (self-referencing rows add no edge), so every
    component of size >= 2 is exactly one immediate-reference cycle.
    """
    nodes = set(remaining)
    adj = {
        rid: sorted((d for d in deps[rid] if d in nodes), key=sort_key)
        for rid in nodes
    }
    index_of: dict[RowId, int] = {}
    low: dict[RowId, int] = {}
    on_stack: set[RowId] = set()
    stack: list[RowId] = []
    sccs: list[list[RowId]] = []
    counter = 0

    def visit(node: RowId) -> None:
        nonlocal counter
        index_of[node] = low[node] = counter
        counter += 1
        stack.append(node)
        on_stack.add(node)
        for nxt in adj[node]:
            if nxt not in index_of:
                visit(nxt)
                low[node] = min(low[node], low[nxt])
            elif nxt in on_stack:
                low[node] = min(low[node], index_of[nxt])
        if low[node] == index_of[node]:
            scc = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                scc.append(w)
                if w == node:
                    break
            if len(scc) > 1:
                sccs.append(sorted(scc, key=sort_key))

    for start in remaining:  # arrives sorted: deterministic traversal
        if start not in index_of:
            visit(start)
    return sccs


# ------------------------------------------------------------------ reporting
def _row_descriptor(
    tables: dict[str, TableSchema], data: dict[str, list[Row]], rid: RowId
) -> dict:
    ts = tables[rid.table]
    row = data[rid.table][rid.index]
    pk = ts.pk
    return {
        "table": rid.table,
        "row_index": rid.index,
        "key": ({c: row[c] for c in pk.columns} if pk else None),
        "values": dict(row),
    }


def _cycle_detail(
    tables: dict[str, TableSchema],
    data: dict[str, list[Row]],
    sccs: list[list[RowId]],
    edges: list[FkEdge],
) -> dict:
    cycles = []
    for scc in sccs:
        members = set(scc)
        scc_edges = [
            e for e in edges if e.from_row in members and e.to_row in members
        ]
        cycles.append(
            {
                "rows": [_row_descriptor(tables, data, rid) for rid in scc],
                "foreign_keys": [
                    {
                        "constraint": e.fk_name,
                        "columns": list(e.columns),
                        "references": {
                            "table": e.ref_table,
                            "columns": list(e.ref_columns),
                        },
                        "from_row": {
                            "table": e.from_row.table,
                            "row_index": e.from_row.index,
                        },
                        "to_row": {
                            "table": e.to_row.table,
                            "row_index": e.to_row.index,
                        },
                    }
                    for e in scc_edges
                ],
            }
        )
    return {
        "kind": "immediate_fk_cycle",
        "message": (
            "已发布样本中存在即时外键引用环：这些行互相依赖，不存在满足全部"
            "即时外键的逐行插入顺序；不提供部分脚本。可改用 SQL 包"
            "（02_data.sql 在无外键阶段装载，03_constraints.sql 事后补外键），"
            "或在目标端使用初始延迟外键。"
        ),
        "cycles": cycles,
    }


# ------------------------------------------------------------------ rendering
def _render_sql(
    job_id: str,
    schema_name: str,
    tables: dict[str, TableSchema],
    order: list[RowId],
    data: dict[str, list[Row]],
    deferred: list[str],
) -> str:
    deferred_note = ", ".join(deferred) if deferred else "(none)"
    header = f"""-- ============================================================
-- Row-by-row load plan — job {job_id}
-- Replays the published sample into an EMPTY target schema whose tables
-- already enforce every constraint (primary keys, unique keys and all
-- foreign keys). Everything runs in ONE transaction:
--   * constraints stay ENABLED; immediate foreign keys are NEVER deferred
--     (no SET CONSTRAINTS is issued);
--   * DEFERRABLE INITIALLY DEFERRED foreign keys are checked at COMMIT and
--     hold for this published row set: {deferred_note}
--   * NULL keys follow MATCH SIMPLE / MATCH FULL exemption rules;
--     self-referencing rows are written directly.
-- ============================================================
"""
    lines = [header, "BEGIN;", ""]
    if not order:
        lines.append("-- the published sample contains no rows")
    for seq, rid in enumerate(order, 1):
        ts = tables[rid.table]
        row = data[rid.table][rid.index]
        cols = ", ".join(q(c.name) for c in ts.columns)
        vals = ", ".join(
            "NULL" if row[c.name] is None else str(row[c.name])
            for c in ts.columns
        )
        lines.append(f"-- [{seq}] {rid.table} row {rid.index + 1}")
        lines.append(
            f"INSERT INTO {q(schema_name)}.{q(rid.table)} ({cols}) "
            f"VALUES ({vals});"
        )
        lines.append("")
    lines.append("COMMIT;")
    lines.append("")
    return "\n".join(lines)


# -------------------------------------------------------------------- entry
def compute_row_load_plan(
    job_id: str,
    schema_name: str,
    tables: dict[str, TableSchema],
    selected: list[str],
    data: dict[str, list[Row]],
) -> dict:
    """Compute the JSON-ready row-load plan for a published sample.

    Raises ImmediateCycleError when the immediate foreign keys form a
    row-level cycle; in that case no script (not even a partial one) is
    produced.
    """
    deps, edges, deferred = _build_edges(tables, selected, data)
    order, remaining, sort_key = _topo_sort(deps, selected)
    if remaining:
        sccs = _cycle_sccs(remaining, deps, sort_key)
        if not sccs:  # defensive: remaining nodes always contain a cycle
            sccs = [remaining]
        detail = _cycle_detail(tables, data, sccs, edges)
        raise ImmediateCycleError(detail, sum(len(s) for s in sccs))
    plan_entries = [
        {
            "seq": seq,
            "table": rid.table,
            "row_index": rid.index,
            "values": dict(data[rid.table][rid.index]),
        }
        for seq, rid in enumerate(order, 1)
    ]
    return {
        "job_id": job_id,
        "schema_name": schema_name,
        "source": "published_snapshot",
        "selected_tables": list(selected),
        "insert_count": len(order),
        "immediate_fk_edges": len(edges),
        "deferred_foreign_keys": deferred,
        "order": plan_entries,
        "sql": _render_sql(job_id, schema_name, tables, order, data, deferred),
    }
