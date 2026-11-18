"""Solver unit tests: intervals, uniqueness, nullability, match types, cycles."""
from __future__ import annotations

import pytest

from app.models import (
    Column,
    ForeignKey,
    IntRange,
    MatchType,
    TableSchema,
    TableSpec,
    UniqueConstraint,
)
from app.solver import Generator, Unsatisfiable


def col(name, pg_type="integer", nullable=False):
    return Column(
        name=name,
        pg_type=pg_type,
        ordinal=1,
        nullable=nullable,
        has_default=False,
        is_identity=False,
        is_generated=False,
    )


def spec(table, rows, ranges):
    return TableSpec(
        table=table,
        rows=rows,
        ranges={c: IntRange(low=lo, high=hi) for c, (lo, hi) in ranges.items()},
    )


def _check_unique(data, key_cols):
    seen = set()
    for row in data:
        key = tuple(row[c] for c in key_cols)
        if any(v is None for v in key):
            continue
        assert key not in seen, f"duplicate non-NULL key {key}"
        seen.add(key)


def test_pk_uniqueness_and_ranges():
    ts = TableSchema(
        schema_name="public",
        name="t",
        columns=[col("id"), col("v", nullable=True)],
        unique_constraints=[UniqueConstraint(name="pk", columns=["id"], is_primary=True)],
        foreign_keys=[],
    )
    specs = {"t": spec("t", 5, {"id": (1, 5), "v": (10, 20)})}
    data = Generator({"t": ts}, specs).solve()["t"]
    assert len(data) == 5
    assert sorted(r["id"] for r in data) == [1, 2, 3, 4, 5]
    assert all(10 <= r["v"] <= 20 for r in data)


def test_rows_exceed_range_cardinality_is_unsat():
    ts = TableSchema(
        schema_name="public",
        name="t",
        columns=[col("id")],
        unique_constraints=[UniqueConstraint(name="pk", columns=["id"], is_primary=True)],
        foreign_keys=[],
    )
    specs = {"t": spec("t", 5, {"id": (1, 2)})}
    with pytest.raises(Unsatisfiable) as exc:
        Generator({"t": ts}, specs).solve()
    kinds = {g.kind for g in exc.value.groups}
    assert "unique" in kinds and "range" in kinds


def test_nullable_column_can_be_null():
    ts = TableSchema(
        schema_name="public",
        name="t",
        columns=[col("id"), col("v", nullable=True)],
        unique_constraints=[UniqueConstraint(name="pk", columns=["id"], is_primary=True)],
        foreign_keys=[],
    )
    # NOT NULL id has only 2 values but 3 rows requested; v being nullable does
    # not help ids — this stays unsat, proving nullability is column-scoped.
    specs_bad = {"t": spec("t", 3, {"id": (1, 2), "v": (1, 1)})}
    with pytest.raises(Unsatisfiable):
        Generator({"t": ts}, specs_bad).solve()

    # A nullable unique column may repeat NULLs freely.
    ts2 = TableSchema(
        schema_name="public",
        name="t",
        columns=[col("id"), col("v", nullable=True)],
        unique_constraints=[
            UniqueConstraint(name="pk", columns=["id"], is_primary=True),
            UniqueConstraint(name="uk_v", columns=["v"], is_primary=False),
        ],
        foreign_keys=[],
    )
    specs = {"t": spec("t", 4, {"id": (1, 10), "v": (100, 100)})}
    data = Generator({"t": ts2}, specs).solve()["t"]
    # v has a single-point interval; NULL repetition must be used to satisfy UK.
    nulls = sum(1 for r in data if r["v"] is None)
    assert nulls >= 3
    _check_unique(data, ["v"])


def test_self_referencing_foreign_key():
    ts = TableSchema(
        schema_name="public",
        name="node",
        columns=[col("id"), col("parent", nullable=True)],
        unique_constraints=[
            UniqueConstraint(name="pk", columns=["id"], is_primary=True)
        ],
        foreign_keys=[
            ForeignKey(
                name="fk_parent",
                columns=["parent"],
                ref_table="node",
                ref_columns=["id"],
                match=MatchType.SIMPLE,
                on_update="NO ACTION",
                on_delete="SET NULL",
                not_deferrable=True,
                initially_deferred=False,
            )
        ],
    )
    specs = {"node": spec("node", 4, {"id": (1, 4), "parent": (1, 4)})}
    data = Generator({"node": ts}, specs).solve()["node"]
    ids = {r["id"] for r in data}
    for r in data:
        assert r["parent"] is None or r["parent"] in ids


def test_circular_two_tables_resolved():
    a = TableSchema(
        schema_name="public",
        name="a",
        columns=[col("id"), col("b_id", nullable=True)],
        unique_constraints=[UniqueConstraint(name="pa", columns=["id"], is_primary=True)],
        foreign_keys=[
            ForeignKey(
                name="fk_a_b", columns=["b_id"], ref_table="b", ref_columns=["id"],
                match=MatchType.SIMPLE, on_update="NO ACTION", on_delete="SET NULL",
                not_deferrable=True, initially_deferred=False,
            )
        ],
    )
    b = TableSchema(
        schema_name="public",
        name="b",
        columns=[col("id"), col("a_id")],
        unique_constraints=[UniqueConstraint(name="pb", columns=["id"], is_primary=True)],
        foreign_keys=[
            ForeignKey(
                name="fk_b_a", columns=["a_id"], ref_table="a", ref_columns=["id"],
                match=MatchType.SIMPLE, on_update="NO ACTION", on_delete="NO ACTION",
                not_deferrable=True, initially_deferred=False,
            )
        ],
    )
    specs = {
        "a": spec("a", 3, {"id": (1, 3), "b_id": (10, 12)}),
        "b": spec("b", 3, {"id": (10, 12), "a_id": (1, 3)}),
    }
    out = Generator({"a": a, "b": b}, specs).solve()
    a_ids = {r["id"] for r in out["a"]}
    b_ids = {r["id"] for r in out["b"]}
    for r in out["a"]:
        assert r["b_id"] is None or r["b_id"] in b_ids
    for r in out["b"]:
        assert r["a_id"] in a_ids


def _composite_table(match: MatchType):
    parent = TableSchema(
        schema_name="public",
        name="p",
        columns=[col("k1"), col("k2", nullable=True)],
        unique_constraints=[
            UniqueConstraint(name="pk_p", columns=["k1", "k2"], is_primary=True)
        ],
        foreign_keys=[],
    )
    child = TableSchema(
        schema_name="public",
        name="c",
        columns=[col("id"), col("f1", nullable=True), col("f2", nullable=True)],
        unique_constraints=[
            UniqueConstraint(name="pk_c", columns=["id"], is_primary=True)
        ],
        foreign_keys=[
            ForeignKey(
                name="fk_c_p",
                columns=["f1", "f2"],
                ref_table="p",
                ref_columns=["k1", "k2"],
                match=match,
                on_update="NO ACTION",
                on_delete="NO ACTION",
                not_deferrable=True,
                initially_deferred=False,
            )
        ],
    )
    return parent, child


def test_match_full_forbids_mixed_null_via_solver_ranges():
    # Force the child f1 to NULL via an impossible non-null situation is hard
    # directly; instead assert a feasible solve honours FULL all-or-nothing.
    parent, child = _composite_table(MatchType.FULL)
    specs = {
        "p": spec("p", 2, {"k1": (1, 2), "k2": (10, 11)}),
        "c": spec("c", 4, {"id": (100, 103), "f1": (1, 2), "f2": (10, 11)}),
    }
    out = Generator({"p": parent, "c": child}, specs).solve()
    pkeys = {(r["k1"], r["k2"]) for r in out["p"]}
    for r in out["c"]:
        if r["f1"] is None and r["f2"] is None:
            continue  # all-NULL allowed
        # MATCH FULL: never a mixed key
        assert (r["f1"] is None) == (r["f2"] is None)
        assert (r["f1"], r["f2"]) in pkeys


def test_match_simple_allows_mixed_null():
    parent, child = _composite_table(MatchType.SIMPLE)
    specs = {
        "p": spec("p", 2, {"k1": (1, 2), "k2": (10, 11)}),
        "c": spec("c", 6, {"id": (100, 105), "f1": (1, 2), "f2": (10, 11)}),
    }
    out = Generator({"p": parent, "c": child}, specs).solve()
    # solver prefers non-NULL; feasibility is the key assertion here.
    assert len(out["c"]) == 6


def test_match_simple_vs_full_mixed_null_distinction():
    """A scenario feasible only if mixed NULL composite keys are allowed."""
    # f2 is NOT NULL and single-point, f1 is nullable and single-point, and a
    # unique key spans (f1, f2): the second child row can only differ from the
    # first via f1=NULL, i.e. a *mixed* composite key (NULL, 10). Every model
    # must contain exactly one such row, so the SIMPLE/FULL distinction is
    # deterministic and does not depend on the solver's model choice.
    parent, child_simple = _composite_table(MatchType.SIMPLE)
    _, child_full = _composite_table(MatchType.FULL)
    for child in (child_simple, child_full):
        child.unique_constraints.append(
            UniqueConstraint(name="uk_fk", columns=["f1", "f2"], is_primary=False)
        )
        child.columns = [
            Column(
                name="f2",
                pg_type="integer",
                ordinal=3,
                nullable=False,
                has_default=False,
                is_identity=False,
                is_generated=False,
            )
            if c.name == "f2"
            else c
            for c in child.columns
        ]
    need_mixed = {
        "p": spec("p", 1, {"k1": (1, 1), "k2": (10, 10)}),
        "c": spec("c", 2, {"id": (100, 101), "f1": (1, 1), "f2": (10, 10)}),
    }
    # SIMPLE: the mixed key (f1=NULL, f2=10) is exempt from the FK check and
    # distinct under the unique key, so the scenario is satisfiable — and the
    # mixed row is forced, not merely possible.
    simple_out = Generator({"p": parent, "c": child_simple}, need_mixed).solve()
    mixed = [
        r for r in simple_out["c"] if (r["f1"] is None) != (r["f2"] is None)
    ]
    assert len(mixed) == 1
    assert mixed[0]["f1"] is None and mixed[0]["f2"] == 10

    # FULL: the same scenario is infeasible — the mixed key is forbidden and
    # no all-NULL key is available (f2 is NOT NULL).
    with pytest.raises(Unsatisfiable):
        Generator({"p": parent, "c": child_full}, need_mixed).solve()


def test_zero_rows_table():
    ts = TableSchema(
        schema_name="public",
        name="empty_t",
        columns=[col("id")],
        unique_constraints=[UniqueConstraint(name="pk", columns=["id"], is_primary=True)],
        foreign_keys=[],
    )
    specs = {"empty_t": spec("empty_t", 0, {"id": (1, 10)})}
    out = Generator({"empty_t": ts}, specs).solve()
    assert out["empty_t"] == []


def test_type_bounds_enforced():
    ts = TableSchema(
        schema_name="public",
        name="t",
        columns=[col("id", pg_type="smallint")],
        unique_constraints=[UniqueConstraint(name="pk", columns=["id"], is_primary=True)],
        foreign_keys=[],
    )
    with pytest.raises(ValueError):
        IntRange(low=-40000, high=1).validated(ts.column("id"))
