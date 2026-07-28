#!/usr/bin/env python3
"""
Micro-benchmark: the schema-v7 index reshapes on training_stats and latent_snapshots
(companion to bench_injection_index.py, which covers the injection_stats side of v7).

For each table it synthesizes a production-shaped database twice in a scratch directory —
once with the old (tag, timestamp, ...) filter index, once with the v7 equality-first
replacement — replicating db.py's schema, pragmas (WAL + synchronous=NORMAL) and each
table's real write pattern (training_stats: writer-thread flush batches of 5,000 rows;
latent_snapshots: one 3,840-row executemany per snapshot capture). It then times every
production query shape observed in the codebase against both variants, with an EXPLAIN
QUERY PLAN line per shape so the chosen index is visible. No ANALYZE is ever run — db.py
never runs it, so a plan that needs sqlite_stat1 is a plan production never gets.

Shapes covered (and their call sites):
- training_stats: the run-wide loss-curve fetch (train.py plot_beta_vae_loss_curves),
  the stat-scoped snr_range_floor/ceil fetch (train.py _get_snr_by_round), the kl_dim_*
  IN fetch (train.py plot_posterior_collapse), and the dashboard's stat_name IN + ORDER BY
  poll (dashboard.py load_training_stats).
- latent_snapshots: the per-frame capture fetch (train.py plot_latent_space_gif — up to
  latent_viz_gif_max_frames=500 of these per run), the dashboard's latest-capture lookup
  (ORDER BY round/epoch/step DESC LIMIT 1), and the DISTINCT key enumeration
  (query_latent_snapshot_keys) as a regression check — it must stay a bounded tag scan.

Verdict baked into schema v7 (measured on an M-series Mac, sqlite 3.50): the
latent_snapshots reshape ships — 67x per frame fetch at 2M rows (linear in partition size:
~1.5 h -> ~2 s per GIF pass at the 100M-row release scale), the latest-capture lookup
becomes a backward index walk (595 ms -> ~0 ms), keys_distinct unchanged, write cost
within noise (both shapes append at the b-tree right edge — round/epoch/step grow
monotonically like timestamp). The training_stats reshape was REJECTED on this bench's
numbers: it won only the single-stat shape (3.5x on ~9 ms) while regressing the dominant
run-window loss-curve fetch 1.8x (stat-sorted index order scatters table-row lookups that
timestamp order visits sequentially), the kl_dims/dashboard IN shapes ~1.2x, and insert
throughput ~20% — on a table whose tag partitions stay ~58k rows.

    python benchmarks/bench_db_index_shapes.py [--training-tags 18] [--latent-captures 520]

Prints per-shape ms and rows/s per variant plus a release-scale extrapolation of the GIF
frame pass, and writes a JSON result to benchmarks/results/ (or --output).
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

BASE_TIMESTAMP = 1_700_000_000.0
FLUSH_BATCH_ROWS = 5_000  # config.db.write_buffer_max_size — the writer's flush batch
METADATA_JSON = json.dumps(
    {"machine_name": "bench", "user_name": "bench", "ip_address": "0.0.0.0", "process_id": 0},
    sort_keys=True,
)

# --- training_stats: ~29 rows/epoch (21 scalars + latent_dim=8 kl_dim rows), 20 rounds x
# --- 100 epochs per tag; a campaign db accumulates one such partition per run tag ---
TRAINING_STATS = (
    ["total_loss", "reconstruction_loss", "kl_loss", "true_loss", "false_loss"]
    + ["val_total_loss", "val_reconstruction_loss", "val_kl_loss", "val_true_loss"]
    + ["val_false_loss", "gradient_norm_mean", "gradient_norm_max", "gradient_norm_std"]
    + ["clipping_rate", "learning_rate", "epoch_duration", "steps_per_epoch"]
    + ["train_samples", "snr_range_floor", "snr_range_ceil", "beta"]
    + [f"kl_dim_{d:02d}" for d in range(8)]
)
KL_DIM_STATS = [s for s in TRAINING_STATS if s.startswith("kl_dim_")]
DASHBOARD_STATS = TRAINING_STATS[:5] + TRAINING_STATS[5:10]  # losses + val_ variants
ROUNDS, EPOCHS = 20, 100

TRAINING_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS training_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL NOT NULL,
        model_name TEXT NOT NULL,
        stat_name TEXT NOT NULL,
        value REAL NOT NULL,
        round_number INTEGER,
        epoch_number INTEGER,
        tag TEXT,
        metadata TEXT,
        superseded INTEGER DEFAULT 0,
        is_finite INTEGER DEFAULT 1
    )
"""
TRAINING_OLD_INDEX = """
    CREATE INDEX idx_training_stats_filter
    ON training_stats(tag, timestamp, model_name, stat_name)
"""
TRAINING_NEW_INDEX = """
    CREATE INDEX idx_training_stats_by_stat
    ON training_stats(tag, model_name, stat_name, timestamp)
"""
TRAINING_INSERT = """
    INSERT INTO training_stats
    (timestamp, model_name, stat_name, value, round_number, epoch_number, tag, metadata)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""
ALIVE = "superseded = 0 AND is_finite = 1"
TRAINING_QUERIES = {
    "loss_curves": (
        "SELECT timestamp, stat_name, value, round_number, epoch_number FROM training_stats"
        f" WHERE tag = ? AND model_name = 'beta_vae' AND round_number >= 1 AND {ALIVE}"
        " AND timestamp >= ? AND timestamp <= ?"
    ),
    "stat_scoped": (
        "SELECT value, round_number FROM training_stats"
        " WHERE tag = ? AND model_name = 'beta_vae' AND stat_name = 'snr_range_floor'"
        f" AND {ALIVE} AND timestamp >= ? AND timestamp <= ?"
    ),
    "kl_dims": (
        "SELECT timestamp, stat_name, value, round_number, epoch_number FROM training_stats"
        " WHERE tag = ? AND model_name = 'beta_vae' AND round_number >= 1"
        f" AND stat_name IN ({','.join('?' * len(KL_DIM_STATS))})"
        f" AND {ALIVE} AND timestamp >= ? AND timestamp <= ?"
    ),
    "dashboard": (
        "SELECT timestamp, stat_name, value, round_number, epoch_number FROM training_stats"
        " WHERE tag = ? AND model_name = 'beta_vae' AND superseded = 0"
        f" AND stat_name IN ({','.join('?' * len(DASHBOARD_STATS))})"
        " ORDER BY round_number, epoch_number, timestamp"
    ),
}

# --- latent_snapshots: 3,840 rows per capture (960 cadences x 4 signal types), captures
# --- keyed by (round, epoch, step); ~100M rows per tag at release scale ---
SIGNAL_TYPES = ["false_no_signal", "false_with_rfi", "true_only_eti", "true_eti_rfi"]
CADENCES_PER_CAPTURE = 3_840
STEPS_PER_EPOCH = 10  # Captures per epoch at latent_viz_step_interval defaults
RELEASE_LATENT_ROWS = 100_000_000  # docs/DATABASE.md growth expectations, full-scale run
GIF_MAX_FRAMES = 500  # config.training.latent_viz_gif_max_frames

LATENT_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS latent_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL NOT NULL,
        model_name TEXT NOT NULL,
        round_number INTEGER NOT NULL,
        epoch_number INTEGER NOT NULL,
        step_number INTEGER NOT NULL,
        cadence_index INTEGER NOT NULL,
        signal_type TEXT NOT NULL,
        latent_vector TEXT NOT NULL,
        snr_base INTEGER,
        snr_range INTEGER,
        tag TEXT,
        metadata TEXT,
        superseded INTEGER DEFAULT 0
    )
"""
LATENT_OLD_INDEX = """
    CREATE INDEX idx_latent_snapshots_filter
    ON latent_snapshots(tag, timestamp, model_name, round_number, epoch_number, step_number)
"""
LATENT_NEW_INDEX = """
    CREATE INDEX idx_latent_snapshots_by_key
    ON latent_snapshots(tag, round_number, epoch_number, step_number, model_name, timestamp)
"""
LATENT_INSERT = """
    INSERT INTO latent_snapshots
    (timestamp, model_name, round_number, epoch_number, step_number, cadence_index,
     signal_type, latent_vector, snr_base, snr_range, tag, metadata)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""
LATENT_QUERIES = {
    "frame_fetch": (
        "SELECT signal_type, latent_vector FROM latent_snapshots"
        " WHERE tag = ? AND model_name = ? AND round_number = ? AND epoch_number = ?"
        " AND step_number = ? AND timestamp >= ? AND superseded = 0"
    ),
    "latest_key": (
        "SELECT round_number, epoch_number, step_number FROM latent_snapshots"
        " WHERE tag = ? AND superseded = 0"
        " ORDER BY round_number DESC, epoch_number DESC, step_number DESC LIMIT 1"
    ),
    "keys_distinct": (
        "SELECT DISTINCT model_name, round_number, epoch_number, step_number,"
        " snr_base, snr_range FROM latent_snapshots"
        " WHERE tag = ? AND timestamp >= ? AND superseded = 0"
        " ORDER BY model_name, round_number, epoch_number, step_number"
    ),
}


def create_db(path: str, table_sql: str, index_sql: str) -> sqlite3.Connection:
    """Scratch db with db.py's schema for one table and pragmas (WAL, synchronous=NORMAL)."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(table_sql)
    conn.execute(index_sql)
    conn.commit()
    return conn


def insert_batches(conn: sqlite3.Connection, insert_sql: str, batches) -> dict:
    """executemany + commit per batch (the writer-thread pattern), timing inserts only."""
    cursor = conn.cursor()
    insert_s, total = 0.0, 0
    for batch in batches:
        begin = time.perf_counter()
        cursor.executemany(insert_sql, batch)
        conn.commit()
        insert_s += time.perf_counter() - begin
        total += len(batch)
    return {"rows": total, "insert_s": insert_s, "rows_per_s": total / insert_s}


def training_batches(n_tags: int, seed: int):
    """Per-epoch stat rows for n_tags sequential runs, in FLUSH_BATCH_ROWS-sized batches."""
    rng = np.random.default_rng(seed)
    batch, ts = [], BASE_TIMESTAMP
    for t in range(n_tags):
        tag = f"bench_run_{t:02d}"
        for round_number in range(1, ROUNDS + 1):
            for epoch in range(1, EPOCHS + 1):
                ts += 30.0  # ~30 s/epoch keeps per-tag time spans disjoint, like real runs
                for stat in TRAINING_STATS:
                    batch.append(
                        (
                            ts,
                            "beta_vae",
                            stat,
                            float(rng.normal()),
                            round_number,
                            epoch,
                            tag,
                            METADATA_JSON,
                        )
                    )
                    if len(batch) >= FLUSH_BATCH_ROWS:
                        yield batch
                        batch = []
    if batch:
        yield batch


def latent_batches(n_captures: int, tag: str, seed: int):
    """One 3,840-row batch per snapshot capture, capture keys advancing like a real run.
    latent_vector payloads cycle a pre-serialized pool — index cost only depends on the
    key columns, and 2M json.dumps calls would dominate the bench for nothing."""
    rng = np.random.default_rng(seed)
    pool = [json.dumps(np.round(rng.normal(size=(6, 8)), 8).tolist()) for _ in range(256)]
    ts = BASE_TIMESTAMP
    for c in range(n_captures):
        step_idx = c % STEPS_PER_EPOCH
        epoch_idx = (c // STEPS_PER_EPOCH) % EPOCHS
        round_number = 1 + c // (STEPS_PER_EPOCH * EPOCHS)
        ts += 60.0
        yield [
            (
                ts,
                "beta_vae",
                round_number,
                epoch_idx + 1,
                (step_idx + 1) * 10,
                i,
                SIGNAL_TYPES[i % 4],
                pool[(c + i) % 256],
                10,
                2 + round_number,
                tag,
                METADATA_JSON,
            )
            for i in range(CADENCES_PER_CAPTURE)
        ]


def bench_query(cursor, sql: str, params: tuple, repeats: int) -> dict:
    rows = len(cursor.execute(sql, params).fetchall())  # Warm + row count
    durations = time_repeats(lambda: cursor.execute(sql, params).fetchall(), repeats)
    plan = cursor.execute(f"EXPLAIN QUERY PLAN {sql}", params).fetchall()
    return {"rows": rows, "ms": min(durations) * 1000.0, "plan": " | ".join(r[3] for r in plan)}


def run_variant(scratch, label, table_sql, index_sql, insert_sql, batches, queries, repeats):
    conn = create_db(os.path.join(scratch, f"{label}.db"), table_sql, index_sql)
    insert = insert_batches(conn, insert_sql, batches)
    print(f"[{label}] insert: {insert['rows_per_s']:>10,.0f} rows/s ({insert['rows']:,} rows)")
    cursor = conn.cursor()
    shapes = {}
    for name, (sql, params) in queries.items():
        shapes[name] = bench_query(cursor, sql, params, repeats)
        print(
            f"[{label}] {name}: {shapes[name]['ms']:>9.2f} ms ({shapes[name]['rows']:,} rows)"
            f"\n[{label}]   plan: {shapes[name]['plan']}"
        )
    conn.close()
    return {"insert": insert, "queries": shapes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-tags", type=int, default=18, help="Run tags (58k rows each)")
    parser.add_argument("--latent-captures", type=int, default=520, help="3,840-row captures")
    parser.add_argument("--frames", type=int, default=12, help="Sampled GIF frame fetches")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per query (best kept)")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", default=None, help="Result JSON path")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    results: dict = {}

    with tempfile.TemporaryDirectory(prefix="bench_db_index_shapes_") as scratch:
        # training_stats: query the middle tag over its own run window (start_time/end_time
        # as train.py passes them), bounds wide enough to cover the whole partition
        tag = f"bench_run_{args.training_tags // 2:02d}"
        span = (BASE_TIMESTAMP, BASE_TIMESTAMP + args.training_tags * ROUNDS * EPOCHS * 30.0 + 1)
        tqueries = {
            "loss_curves": (TRAINING_QUERIES["loss_curves"], (tag, *span)),
            "stat_scoped": (TRAINING_QUERIES["stat_scoped"], (tag, *span)),
            "kl_dims": (TRAINING_QUERIES["kl_dims"], (tag, *KL_DIM_STATS, *span)),
            "dashboard": (TRAINING_QUERIES["dashboard"], (tag, *DASHBOARD_STATS)),
        }
        for label, index_sql in (
            ("training_old", TRAINING_OLD_INDEX),
            ("training_new", TRAINING_NEW_INDEX),
        ):
            results[label] = run_variant(
                scratch,
                label,
                TRAINING_TABLE_SQL,
                index_sql,
                TRAINING_INSERT,
                training_batches(args.training_tags, args.seed),
                tqueries,
                args.repeats,
            )

        # latent_snapshots: one tag, GIF-style frame fetches on sampled capture keys
        capture_ids = rng.choice(args.latent_captures, size=args.frames, replace=False)
        frame_keys = [
            (
                1 + int(c) // (STEPS_PER_EPOCH * EPOCHS),
                (int(c) // STEPS_PER_EPOCH) % EPOCHS + 1,
                (int(c) % STEPS_PER_EPOCH + 1) * 10,
            )
            for c in capture_ids
        ]
        lqueries = {
            f"frame_fetch_{i}": (
                LATENT_QUERIES["frame_fetch"],
                ("bench_gif", "beta_vae", r, e, s, BASE_TIMESTAMP),
            )
            for i, (r, e, s) in enumerate(frame_keys)
        }
        lqueries["latest_key"] = (LATENT_QUERIES["latest_key"], ("bench_gif",))
        lqueries["keys_distinct"] = (LATENT_QUERIES["keys_distinct"], ("bench_gif", BASE_TIMESTAMP))
        for label, index_sql in (
            ("latent_old", LATENT_OLD_INDEX),
            ("latent_new", LATENT_NEW_INDEX),
        ):
            results[label] = run_variant(
                scratch,
                label,
                LATENT_TABLE_SQL,
                index_sql,
                LATENT_INSERT,
                latent_batches(args.latent_captures, "bench_gif", args.seed),
                lqueries,
                args.repeats,
            )

    # Comparisons + the release-scale GIF pass extrapolation. The old shape's frame fetch
    # scans the tag's whole window (linear in partition rows); the new shape seeks one
    # capture (~flat), so only the old side is scaled.
    def frame_ms(variant: str) -> float:
        shapes = results[variant]["queries"]
        vals = [s["ms"] for n, s in shapes.items() if n.startswith("frame_fetch_")]
        return sum(vals) / len(vals)

    latent_rows = results["latent_old"]["insert"]["rows"]
    scale = RELEASE_LATENT_ROWS / latent_rows
    comparison = {
        "training_write_cost_delta_pct": (
            results["training_old"]["insert"]["rows_per_s"]
            / results["training_new"]["insert"]["rows_per_s"]
            - 1
        )
        * -100,
        "latent_write_cost_delta_pct": (
            results["latent_old"]["insert"]["rows_per_s"]
            / results["latent_new"]["insert"]["rows_per_s"]
            - 1
        )
        * -100,
        "training_speedups": {
            name: results["training_old"]["queries"][name]["ms"]
            / results["training_new"]["queries"][name]["ms"]
            for name in tqueries
        },
        "latent_speedups": {
            name: results["latent_old"]["queries"][name]["ms"]
            / results["latent_new"]["queries"][name]["ms"]
            for name in lqueries
        },
        "frame_fetch_speedup": frame_ms("latent_old") / frame_ms("latent_new"),
        "gif_pass_extrapolation": {
            "release_rows": RELEASE_LATENT_ROWS,
            "frames": GIF_MAX_FRAMES,
            "old_index_hours": frame_ms("latent_old") / 1000 * scale * GIF_MAX_FRAMES / 3600,
            "new_index_hours": frame_ms("latent_new") / 1000 * GIF_MAX_FRAMES / 3600,
        },
    }
    results["comparison"] = comparison
    print(
        f"write cost: training {comparison['training_write_cost_delta_pct']:+.1f}%, "
        f"latent {comparison['latent_write_cost_delta_pct']:+.1f}%; "
        f"frame fetch speedup {comparison['frame_fetch_speedup']:,.1f}x; "
        f"GIF pass at {RELEASE_LATENT_ROWS / 1e6:.0f}M rows x {GIF_MAX_FRAMES} frames: "
        f"{comparison['gif_pass_extrapolation']['old_index_hours']:.2f} h (old) -> "
        f"{comparison['gif_pass_extrapolation']['new_index_hours'] * 3600:.1f} s (new)"
    )

    path = write_result(
        "bench_db_index_shapes",
        {
            "training_tags": args.training_tags,
            "latent_captures": args.latent_captures,
            "frames": args.frames,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        results,
        args.output,
    )
    print(f"Result written to {path}")


if __name__ == "__main__":
    main()
