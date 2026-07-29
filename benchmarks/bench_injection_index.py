#!/usr/bin/env python3
"""
Micro-benchmark: cost/benefit of the schema-v7 secondary injection_stats index
(idx_injection_stats_by_stat) against the original idx_injection_stats_filter.

Synthesizes a production-shaped injection_stats database twice in a scratch directory —
once with only the old (tag, timestamp, ...) index, once with both indexes — replicating
db.py's schema, pragmas (WAL + synchronous=NORMAL) and the #277 bulk lane's 50k-row
executemany transactions. Measures (a) bulk-insert throughput under each index setup (the
write cost of maintaining the second index) and (b) the end-of-run plot query shape
(equality on tag/stat_name/signal_type/injection_stage with run-wide timestamp bounds —
the ~165-query pass that scanned the whole tag partition per query on the old index),
with an EXPLAIN QUERY PLAN line per setup so the chosen index is visible.

The both-indexes query bench runs twice: once without sqlite_stat1 (what production sees —
db.py never runs ANALYZE, and without stats SQLite's default cost model prefers the range
term on idx_injection_stats_filter over the four equality columns of the new index) and
once after ANALYZE, whose stats flip the planner onto idx_injection_stats_by_stat. The gap
between those two rows is the finding, not a methodology artifact.

    python benchmarks/bench_injection_index.py [--rows 20000000] [--queries 12] [--repeats 2]

Prints rows/s and per-query ms plus an extrapolation to release scale (367M rows), and
writes a JSON result to benchmarks/results/ (or --output).
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
import time

import numpy as np
from _common import time_repeats, write_result

TAG = "bench_injection_index"
# Production-shaped categorical axes: ~18 stat_names x 4 signal_types x 3 stages x 10 rounds
STAT_NAMES = (
    ["global_mean", "global_median", "global_std", "global_mad", "global_skew", "global_kurtosis"]
    + ["eti_snr", "eti_drift_rate", "eti_signal_width", "eti_starting_bin", "eti_slope_pixel"]
    + ["eti_y_intercept", "rfi_snr", "rfi_drift_rate", "rfi_signal_width", "rfi_starting_bin"]
    + ["rfi_slope_pixel", "rfi_y_intercept"]
)
SIGNAL_TYPES = ["false_no_signal", "false_with_rfi", "true_only_eti", "true_eti_rfi"]
STAGES = ["A", "B", "C"]
ROUNDS = 10
BASE_TIMESTAMP = 1_700_000_000.0
TIMESTAMP_STEP = 0.001  # Sequential timestamps, as the writer thread commits them
CHUNK_ROWS = 50_000  # config.db.bulk_chunk_rows — the #277 bulk lane's transaction size
RELEASE_ROWS = 367_000_000  # ~10-round release campaign
QUERIES_PER_PLOT_PASS = 165  # plot_injection_stats query count per end-of-run call

# Schema + indexes replicated from db.py's _init_database()
CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS injection_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL NOT NULL,
        stat_name TEXT NOT NULL,
        value REAL NOT NULL,
        round_number INTEGER,
        chunk_number INTEGER,
        sample_index INTEGER,
        background_index INTEGER,
        signal_class TEXT,
        signal_type TEXT,
        injection_stage TEXT,
        is_finite INTEGER DEFAULT 1,
        slope_clamped INTEGER DEFAULT 0,
        tag TEXT,
        metadata TEXT,
        superseded INTEGER DEFAULT 0
    )
"""
OLD_INDEX_SQL = """
    CREATE INDEX IF NOT EXISTS idx_injection_stats_filter
    ON injection_stats(tag, timestamp, stat_name, signal_type, injection_stage)
"""
NEW_INDEX_SQL = """
    CREATE INDEX IF NOT EXISTS idx_injection_stats_by_stat
    ON injection_stats(tag, stat_name, signal_type, injection_stage, timestamp)
"""
INSERT_SQL = """
    INSERT INTO injection_stats
    (timestamp, stat_name, value, round_number, chunk_number, sample_index,
     background_index, signal_class, signal_type, injection_stage, is_finite,
     slope_clamped, tag, metadata)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""
# The end-of-run plot query shape: equality on the categorical columns, run-wide time bounds
QUERY_SQL = """
    SELECT value FROM injection_stats
    WHERE tag = ? AND stat_name = ? AND signal_type = ? AND injection_stage = ?
    AND timestamp BETWEEN ? AND ?
"""

METADATA_JSON = json.dumps(
    {"machine_name": "bench", "user_name": "bench", "ip_address": "0.0.0.0", "process_id": 0},
    sort_keys=True,
)


def make_chunk(start: int, n: int, total_rows: int, seed: int) -> list[tuple]:
    """Rows [start, start+n) of the synthetic dataset — deterministic in (start, seed), so
    both database variants ingest byte-identical data."""
    rng = np.random.default_rng(seed + start)
    values = rng.normal(loc=1.0, scale=0.3, size=n)
    n_combos = len(STAT_NAMES) * len(SIGNAL_TYPES) * len(STAGES)
    rows = []
    for offset in range(n):
        idx = start + offset
        combo = idx % n_combos
        stat_name = STAT_NAMES[combo % len(STAT_NAMES)]
        signal_type = SIGNAL_TYPES[(combo // len(STAT_NAMES)) % len(SIGNAL_TYPES)]
        stage = STAGES[(combo // (len(STAT_NAMES) * len(SIGNAL_TYPES))) % len(STAGES)]
        rows.append(
            (
                BASE_TIMESTAMP + idx * TIMESTAMP_STEP,
                stat_name,
                float(values[offset]),
                1 + (idx * ROUNDS) // total_rows,  # Sequential rounds 1..10
                idx // CHUNK_ROWS,
                idx % 512,
                idx % 1024,
                signal_type.split("_", 1)[0],  # false_* -> "false", true_* -> "true"
                signal_type,
                stage,
                1,
                0,
                TAG,
                METADATA_JSON,
            )
        )
    return rows


def create_db(path: str, with_new_index: bool) -> sqlite3.Connection:
    """Scratch injection_stats db with db.py's schema and pragmas (WAL, synchronous=NORMAL)."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(CREATE_TABLE_SQL)
    conn.execute(OLD_INDEX_SQL)
    if with_new_index:
        conn.execute(NEW_INDEX_SQL)
    conn.commit()
    return conn


def bench_inserts(conn: sqlite3.Connection, total_rows: int, seed: int) -> dict:
    """Bulk-insert total_rows in CHUNK_ROWS-sized executemany transactions (the bulk lane
    pattern), timing only the executemany + commit (row synthesis excluded)."""
    cursor = conn.cursor()
    insert_s = 0.0
    for start in range(0, total_rows, CHUNK_ROWS):
        chunk = make_chunk(start, min(CHUNK_ROWS, total_rows - start), total_rows, seed)
        begin = time.perf_counter()
        cursor.executemany(INSERT_SQL, chunk)
        conn.commit()
        insert_s += time.perf_counter() - begin
    return {"rows": total_rows, "insert_s": insert_s, "rows_per_s": total_rows / insert_s}


def bench_queries(conn: sqlite3.Connection, combos: list[tuple], repeats: int) -> dict:
    """Best-of-`repeats` wall time per sampled (stat_name, signal_type, stage) combo, over
    run-wide timestamp bounds (MIN/MAX +-1 s, as the span-tightened plot pass computes them)."""
    cursor = conn.cursor()
    t_min, t_max = cursor.execute(
        "SELECT MIN(timestamp), MAX(timestamp) FROM injection_stats WHERE tag = ?", (TAG,)
    ).fetchone()
    bounds = (t_min - 1.0, t_max + 1.0)

    per_query_ms = []
    rows_fetched = 0
    for stat_name, signal_type, stage in combos:
        params = (TAG, stat_name, signal_type, stage, *bounds)
        rows_fetched = len(cursor.execute(QUERY_SQL, params).fetchall())  # Warm + row count
        durations = time_repeats(lambda p=params: cursor.execute(QUERY_SQL, p).fetchall(), repeats)
        per_query_ms.append(min(durations) * 1000.0)

    plan = cursor.execute(f"EXPLAIN QUERY PLAN {QUERY_SQL}", params).fetchall()
    return {
        "queries": len(combos),
        "repeats": repeats,
        "rows_per_query": rows_fetched,
        "per_query_ms_mean": sum(per_query_ms) / len(per_query_ms),
        "per_query_ms_max": max(per_query_ms),
        "query_plan": " | ".join(row[3] for row in plan),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20_000_000, help="Synthetic rows per db")
    parser.add_argument("--queries", type=int, default=12, help="Sampled combos to query")
    parser.add_argument("--repeats", type=int, default=2, help="Repeats per query (best kept)")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", default=None, help="Result JSON path")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    all_combos = [(s, t, g) for g in STAGES for t in SIGNAL_TYPES for s in STAT_NAMES]
    combos = [all_combos[i] for i in rng.choice(len(all_combos), args.queries, replace=False)]

    results: dict = {}
    with tempfile.TemporaryDirectory(prefix="bench_injection_index_") as scratch:
        for label, with_new_index in (("old_index_only", False), ("both_indexes", True)):
            db_path = os.path.join(scratch, f"{label}.db")
            conn = create_db(db_path, with_new_index)
            print(f"[{label}] inserting {args.rows} rows in {CHUNK_ROWS}-row transactions...")
            insert = bench_inserts(conn, args.rows, args.seed)
            print(
                f"[{label}] insert: {insert['rows_per_s']:>12,.0f} rows/s "
                f"({insert['insert_s']:.2f}s total)"
            )
            query = bench_queries(conn, combos, args.repeats)
            print(
                f"[{label}] query:  {query['per_query_ms_mean']:>12.2f} ms/query mean "
                f"(max {query['per_query_ms_max']:.2f} ms, {query['rows_per_query']} rows/query)"
            )
            print(f"[{label}] plan:   {query['query_plan']}")
            results[label] = {"insert": insert, "query": query}
            if with_new_index:
                # The production regime has no sqlite_stat1 (db.py never runs ANALYZE), and
                # without stats the planner sticks to the old index. Measure the post-ANALYZE
                # regime too — the realizable benefit, and the gap production leaves unclaimed.
                begin = time.perf_counter()
                conn.execute("ANALYZE")
                conn.commit()
                analyze_s = time.perf_counter() - begin
                query = bench_queries(conn, combos, args.repeats)
                alabel = "both_indexes_analyzed"
                print(
                    f"[{alabel}] query:  {query['per_query_ms_mean']:>6.2f} ms/query mean "
                    f"(ANALYZE itself took {analyze_s:.2f}s)"
                )
                print(f"[{alabel}] plan:   {query['query_plan']}")
                results[alabel] = {"analyze_s": analyze_s, "query": query}
            conn.close()

    old = results["old_index_only"]
    new = results["both_indexes"]
    analyzed = results["both_indexes_analyzed"]
    write_cost_delta_pct = (old["insert"]["rows_per_s"] / new["insert"]["rows_per_s"] - 1) * 100
    # Every setup scales ~linearly in partition size for this query (old index: whole-partition
    # scan; new index: matched rows are a fixed fraction of the partition), so extrapolate
    # linearly to release scale
    scale = RELEASE_ROWS / args.rows

    def per_query_s(setup: dict) -> float:
        return setup["query"]["per_query_ms_mean"] / 1000 * scale

    extrapolation = {
        "release_rows": RELEASE_ROWS,
        "queries_per_plot_pass": QUERIES_PER_PLOT_PASS,
        "old_index_per_query_s": per_query_s(old),
        "both_indexes_per_query_s": per_query_s(new),
        "both_indexes_analyzed_per_query_s": per_query_s(analyzed),
        "old_index_plot_pass_hours": per_query_s(old) * QUERIES_PER_PLOT_PASS / 3600,
        "both_indexes_plot_pass_hours": per_query_s(new) * QUERIES_PER_PLOT_PASS / 3600,
        "both_indexes_analyzed_plot_pass_hours": (
            per_query_s(analyzed) * QUERIES_PER_PLOT_PASS / 3600
        ),
    }
    results["comparison"] = {
        "write_cost_delta_pct": write_cost_delta_pct,
        "query_speedup_no_stats": (
            old["query"]["per_query_ms_mean"] / new["query"]["per_query_ms_mean"]
        ),
        "query_speedup_analyzed": (
            old["query"]["per_query_ms_mean"] / analyzed["query"]["per_query_ms_mean"]
        ),
        "new_index_used_without_stats": (
            "idx_injection_stats_by_stat" in new["query"]["query_plan"]
        ),
        "extrapolation": extrapolation,
    }
    print(
        f"second index write cost: {write_cost_delta_pct:+.1f}% insert throughput; "
        f"query speedup: {results['comparison']['query_speedup_no_stats']:,.1f}x without stats, "
        f"{results['comparison']['query_speedup_analyzed']:,.1f}x after ANALYZE"
    )
    if not results["comparison"]["new_index_used_without_stats"]:
        print(
            "WARNING: without sqlite_stat1 the planner never chose idx_injection_stats_by_stat "
            "— production runs no ANALYZE, so the index buys nothing as-is"
        )
    print(
        f"extrapolated to {RELEASE_ROWS / 1e6:.0f}M rows x {QUERIES_PER_PLOT_PASS} queries: "
        f"{extrapolation['old_index_plot_pass_hours']:.2f} h (old) -> "
        f"{extrapolation['both_indexes_plot_pass_hours']:.2f} h (both, no stats) -> "
        f"{extrapolation['both_indexes_analyzed_plot_pass_hours'] * 3600:.1f} s (both, analyzed) "
        "per plot pass"
    )

    path = write_result(
        "bench_injection_index",
        {
            "rows": args.rows,
            "queries": args.queries,
            "repeats": args.repeats,
            "seed": args.seed,
            "chunk_rows": CHUNK_ROWS,
        },
        results,
        args.output,
    )
    print(f"Result written to {path}")


if __name__ == "__main__":
    main()
