#!/usr/bin/env python3
"""
Render a per-band inference-performance plot for one streaming-CSV inference run.

Groups per-cadence energy-detection preprocessing wall-clock BY BAND (L/S/C/X) and by
frequency, to reveal whether a band or frequency region is systematically faster/slower.
Fires automatically at the tail of every inference run (alongside benchmark_report.py) and
is also runnable standalone against a database copied off a cluster.

Two data sources, joined per cadence:
1. the SQLite DB `pipeline_stages` table — per-cadence preprocessing wall is the umbrella
   span `inference.preprocess_cadence_<N>` (1-based N in planner order), EXCLUDING its
   `.read_ed` / `.dedup` / `.extract` children (they roll up into the umbrella);
2. the run's inference catalog CSV(s) — the `Band` and `Frequency` columns.

Join (plan-index): the umbrella span carries only N, and the DB has no table mapping N to a
band, so the cadence -> band/frequency map is reconstructed from the catalog by mirroring
preprocessing.group_observations_from_csv — cadences are the CSV rows grouped by
config.inference.cadence_group_by_cols (default
["Target", "Session", "Band", "Cadence ID", "Frequency"]), keeping only groups with exactly
cadence_expected_obs (default 6) observations, in first-appearance order; the i-th such valid
group is cadence N=i (== PendingCadence.index that names the span). ASSUMES the run used the
default group-by columns / expected-obs. A runtime guard compares the mapped cadence count to
the number of umbrella spans and skips the plot (warning, never a crash) if they disagree — so
a resumed run or a non-default grouping degrades to no plot rather than a misleading one.

Reads the SQLite file directly (stdlib sqlite3 + csv + numpy + matplotlib only — no aetherscan
imports), so it also runs against a database fetched from a cluster with
utils/fetch_run_outputs.sh:

    python utils/perband_report.py --save-tag test_20260101_120000 \
        --catalog /path/to/catalog.csv
    python utils/perband_report.py --save-tag test --catalog a.csv b.csv \
        --db-path /path/to/aetherscan.db
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import logging
import os
import re
import socket
import sqlite3
import sys
from collections import OrderedDict
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless environments

import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

logger = logging.getLogger(__name__)

# Per-band colors + the Panel-A x-axis band order (fixed by the spec)
BAND_COLORS = {"L": "#4C72B0", "S": "#55A868", "C": "#C44E52", "X": "#8172B3"}
BAND_ORDER = ["L", "S", "C", "X"]
_UNKNOWN_BAND_COLOR = "#888888"

# Planner defaults (config.inference.*) the plan-index join assumes — see module docstring
DEFAULT_GROUP_BY_COLS = ["Target", "Session", "Band", "Cadence ID", "Frequency"]
DEFAULT_H5_PATH_COL = ".h5 path"
DEFAULT_EXPECTED_OBS = 6

DPI = 130


@dataclass
class CadenceBandRow:
    """One cadence's joined per-band datum: planner index, band, frequency, preprocess wall."""

    n: int  # 1-based planner index (from the umbrella span name)
    band: str
    frequency_mhz: float | None
    preprocess_s: float


# ---------------------------------------------------------------------------
# DB access: per-cadence preprocessing umbrella durations
# ---------------------------------------------------------------------------


# Anchored so a hypothetical future child span ending in `_<digits>` can never be misclassified
# as an umbrella — the pattern is a promise, not a coincidence of the current
# `.read_ed`/`.dedup`/`.extract` child naming.
_UMBRELLA_RE = re.compile(r"^inference\.preprocess_cadence_\d+$")


def _is_umbrella_stage(stage: str) -> bool:
    """True iff `stage` is an `inference.preprocess_cadence_<N>` UMBRELLA span, not one of its
    `.read_ed`/`.dedup`/`.extract` children."""
    return _UMBRELLA_RE.match(stage) is not None


def load_umbrella_preprocess_durations(db_path: str, tag: str) -> dict[int, float]:
    """
    Map planner index N -> per-cadence preprocessing wall (seconds), read from the umbrella
    `inference.preprocess_cadence_<N>` spans of `pipeline_stages` for this tag. Children are
    excluded. On the rare retried cadence (a second umbrella span for the same N), the latest
    attempt wins (rows are read start_time-ordered). Returns {} when the tag has no such spans.
    """
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                "SELECT stage, duration_s FROM pipeline_stages "
                "WHERE tag = ? AND stage LIKE 'inference.preprocess_cadence_%' "
                "ORDER BY start_time",
                (tag,),
            )
            rows = cursor.fetchall()
        except sqlite3.OperationalError as e:
            logger.warning(f"Could not read pipeline_stages from {db_path}: {e}")
            return {}

    durations: dict[int, float] = {}
    for row in rows:
        stage = str(row["stage"])
        if not _is_umbrella_stage(stage):
            continue
        n = int(stage.rsplit("_", 1)[-1])
        durations[n] = float(row["duration_s"])  # later (retried) attempt overwrites earlier
    return durations


# ---------------------------------------------------------------------------
# Catalog access: plan-index -> (band, frequency)
# ---------------------------------------------------------------------------


def _norm_band(value) -> str:
    """Canonicalize a catalog Band cell to an upper-case, stripped label ('?' when empty)."""
    text = str(value).strip().upper() if value is not None else ""
    return text or "?"


def _parse_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _group_catalog_csv(
    csv_path: str, group_by_cols: list[str], h5_path_col: str
) -> OrderedDict[tuple, dict] | None:
    """
    Group one catalog CSV's rows into cadences by `group_by_cols` (first-appearance order),
    mirroring preprocessing.group_observations_from_csv: rows with a missing/empty h5-path cell
    are skipped (so they don't inflate a cadence's obs count). Each group records its member
    count plus the band/frequency of its first row (both are group-by columns, so constant
    within the group). Returns None when the CSV can't be read or is missing ANY required column
    — 'Band'/'Frequency', any group-by column, or the h5-path column — matching
    group_observations_from_csv's all-columns-required contract (it degrades to no plot rather
    than silently regrouping on a subset of columns, which could mis-count cadences).
    """
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            required = ["Band", "Frequency", h5_path_col, *group_by_cols]
            missing = [c for c in dict.fromkeys(required) if c not in header]
            if missing:
                logger.warning(
                    f"Per-band plot: catalog {csv_path} is missing required column(s) {missing} "
                    f"(have {header}); cannot reproduce the planner's cadence grouping — skipping"
                )
                return None

            groups: OrderedDict[tuple, dict] = OrderedDict()
            for row in reader:
                h5 = row.get(h5_path_col)
                if h5 is None or not str(h5).strip():
                    continue
                key = tuple(row.get(c) for c in group_by_cols)
                if key not in groups:
                    groups[key] = {
                        "count": 0,
                        "band": _norm_band(row.get("Band")),
                        "frequency_mhz": _parse_float(row.get("Frequency")),
                    }
                groups[key]["count"] += 1
    except FileNotFoundError:
        logger.warning(f"Per-band plot: catalog CSV not found: {csv_path}")
        return None
    return groups


def map_catalog_cadences(
    catalog_csv_paths,
    group_by_cols: list[str] | None = None,
    h5_path_col: str = DEFAULT_H5_PATH_COL,
    expected_obs: int = DEFAULT_EXPECTED_OBS,
) -> list[tuple[int, str, float | None]] | None:
    """
    Reconstruct the planner's cadence order across one or more catalog CSVs and return
    [(N, band, frequency_mhz), ...] for every VALID cadence (obs count == expected_obs), with N
    the global 1-based planner index (== PendingCadence.index). CSVs are processed in the given
    order and their valid groups share one running index, matching plan_cadences(). Returns
    None when any CSV can't be read or is missing any column the planner grouping requires
    (Band/Frequency, a group-by column, or the h5-path column — see _group_catalog_csv).
    """
    if group_by_cols is None:
        group_by_cols = DEFAULT_GROUP_BY_COLS
    if isinstance(catalog_csv_paths, (str, os.PathLike)):
        catalog_csv_paths = [catalog_csv_paths]

    records: list[tuple[int, str, float | None]] = []
    n = 0
    for csv_path in catalog_csv_paths:
        groups = _group_catalog_csv(str(csv_path), group_by_cols, h5_path_col)
        if groups is None:
            return None
        for info in groups.values():
            if info["count"] != expected_obs:
                continue  # flagged group — planner skips it, so it gets no index / span
            n += 1
            records.append((n, info["band"], info["frequency_mhz"]))
    return records


# ---------------------------------------------------------------------------
# Aggregation + summary
# ---------------------------------------------------------------------------


def _ordered_bands(bands) -> list[str]:
    """Known bands first in BAND_ORDER, then any unexpected extras in first-seen order."""
    present = list(bands)
    return [b for b in BAND_ORDER if b in present] + [b for b in present if b not in BAND_ORDER]


def aggregate_by_band(rows: list[CadenceBandRow]) -> OrderedDict[str, dict]:
    """
    Per-band reduction of the joined cadence rows. Returns an OrderedDict (bands ordered
    [L, S, C, X] then any extras) of {n, median, mean, p90, max, values} over each band's
    preprocessing wall times. The `n` (per-band count) is what the unit test asserts against.
    """
    grouped: OrderedDict[str, list[float]] = OrderedDict()
    for row in rows:
        grouped.setdefault(row.band, []).append(row.preprocess_s)

    agg: OrderedDict[str, dict] = OrderedDict()
    for band in _ordered_bands(grouped.keys()):
        arr = np.asarray(grouped[band], dtype=float)
        agg[band] = {
            "n": int(arr.size),
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "p90": float(np.percentile(arr, 90)),
            "max": float(np.max(arr)),
            "values": grouped[band],
        }
    return agg


def format_summary_table(agg: OrderedDict[str, dict]) -> str:
    """A small fixed-width text table (band, n, median, mean, p90, max) over the per-band agg."""
    header = f"{'band':<6}{'n':>6}{'median':>10}{'mean':>10}{'p90':>10}{'max':>10}"
    lines = [header, "-" * len(header)]
    for band, stats in agg.items():
        lines.append(
            f"{band:<6}{stats['n']:>6}"
            f"{stats['median']:>9.1f}s{stats['mean']:>9.1f}s"
            f"{stats['p90']:>9.1f}s{stats['max']:>9.1f}s"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _render_figure(
    rows: list[CadenceBandRow],
    agg: OrderedDict[str, dict],
    title_tag: str,
    output_path: str,
) -> None:
    """Write the 2-panel figure: per-band distribution (Panel A) + wall-vs-frequency (Panel B).

    ``title_tag`` is the already-composed suptitle identifier (the machine-scoped display tag when
    the caller has one, else ``"{tag}, {hostname}"``).
    """
    fig = Figure(figsize=(15, 6))
    fig.suptitle(f"Inference performance by band ({title_tag})", fontsize=15, fontweight="bold")
    ax_a, ax_b = fig.subplots(1, 2)

    # --- Panel A: per-band boxplot (no fliers) + jittered strip, x ordered [L, S, C, X] ---
    present_bands = list(agg.keys())
    positions = list(range(1, len(present_bands) + 1))
    ax_a.boxplot(
        [agg[b]["values"] for b in present_bands],
        positions=positions,
        widths=0.5,
        showfliers=False,
        medianprops={"color": "black"},
    )
    rng = np.random.default_rng(0)  # deterministic jitter
    trans = blended_transform_factory(ax_a.transData, ax_a.transAxes)
    for pos, band in zip(positions, present_bands, strict=True):
        vals = agg[band]["values"]
        xs = pos + rng.uniform(-0.16, 0.16, size=len(vals))
        ax_a.scatter(
            xs,
            vals,
            s=14,
            color=BAND_COLORS.get(band, _UNKNOWN_BAND_COLOR),
            alpha=0.55,
            edgecolors="none",
            zorder=3,
        )
        ax_a.text(
            pos,
            0.99,
            f"med {agg[band]['median']:.1f}s\nmax {agg[band]['max']:.1f}s\nn={agg[band]['n']}",
            transform=trans,
            ha="center",
            va="top",
            fontsize=8,
        )
    ax_a.set_yscale("log")
    ax_a.set_xticks(positions)
    ax_a.set_xticklabels(present_bands)
    ax_a.set_xlabel("Band", fontsize=11, fontweight="bold")
    ax_a.set_ylabel("Preprocess wall (s)", fontsize=11, fontweight="bold")
    ax_a.set_title("Per-cadence preprocessing by band", fontsize=11)
    ax_a.grid(True, axis="y", alpha=0.3)

    # --- Panel B: preprocess wall vs frequency (continuous MHz x-axis), colored by band ---
    for band in present_bands:
        pts = [
            (r.frequency_mhz, r.preprocess_s)
            for r in rows
            if r.band == band and r.frequency_mhz is not None
        ]
        if not pts:
            continue
        fx, fy = zip(*pts, strict=True)
        ax_b.scatter(
            fx,
            fy,
            s=16,
            color=BAND_COLORS.get(band, _UNKNOWN_BAND_COLOR),
            alpha=0.6,
            edgecolors="none",
            label=f"{band} (n={len(pts)})",
        )
    ax_b.set_yscale("log")
    ax_b.set_xlabel("Frequency (MHz)", fontsize=11, fontweight="bold")
    ax_b.set_ylabel("Preprocess wall (s)", fontsize=11, fontweight="bold")
    ax_b.set_title("Per-cadence preprocessing vs frequency", fontsize=11)
    ax_b.grid(True, alpha=0.3)
    ax_b.legend(title="Band", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    # dirname is "" for a bare filename (cwd) — makedirs("") would raise, so only create a
    # parent directory when output_path actually has one
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    fig.clear()


def render_perband_report(
    db_path: str,
    tag: str,
    catalog_csv_path,
    output_path: str,
    hostname: str,
    display_tag: str | None = None,
) -> str | None:
    """
    Join per-cadence preprocessing walls (pipeline_stages umbrella spans) to catalog band /
    frequency (plan-index join) and write the per-band performance PNG to `output_path`.

    Returns the written path on success, or None (with a logged warning, never an exception)
    when the plot is skipped: no umbrella spans, an unreadable/column-less catalog, or a
    mismatch between the mapped catalog-cadence count and the umbrella-span count (the
    plan-index assumption not holding, e.g. a resumed run or non-default cadence_group_by_cols).
    `catalog_csv_path` may be a single path or a list of paths (processed in planner order).

    `tag` is always the plain DB tag: it keys the `pipeline_stages` query below and must not carry
    the machine name. `display_tag` is the presentation-only machine-scoped tag
    (`{cmd}_{machine}_{datetime}`) the caller derives; when given it labels the on-figure title
    (matching every other plot), else the title falls back to `"{tag}, {hostname}"`.
    """
    durations = load_umbrella_preprocess_durations(db_path, tag)
    if not durations:
        logger.warning(
            f"Per-band inference plot skipped: no preprocess umbrella spans for tag {tag!r}"
        )
        return None

    cadence_map = map_catalog_cadences(catalog_csv_path)
    if cadence_map is None:
        logger.warning("Per-band inference plot skipped: catalog CSV unreadable or missing columns")
        return None

    # Runtime guard on the plan-index join: the number of valid catalog cadences must equal the
    # number of umbrella spans, else the N-th span and the N-th catalog cadence aren't the same
    # cadence and any band mapping would be misleading — skip rather than mislead.
    if len(cadence_map) != len(durations):
        logger.warning(
            f"Per-band inference plot skipped: mapped {len(cadence_map)} catalog cadence(s) but "
            f"found {len(durations)} preprocess umbrella span(s) for tag {tag!r} — the plan-index "
            f"join assumption does not hold (resumed run or non-default cadence_group_by_cols?); "
            f"not rendering to avoid a misleading plot"
        )
        return None

    rows: list[CadenceBandRow] = []
    for n, band, freq in cadence_map:
        if n not in durations:
            logger.warning(
                f"Per-band inference plot skipped: catalog cadence N={n} has no matching "
                f"umbrella span for tag {tag!r}"
            )
            return None
        rows.append(CadenceBandRow(n=n, band=band, frequency_mhz=freq, preprocess_s=durations[n]))

    title_tag = display_tag if display_tag else f"{tag}, {hostname}"
    agg = aggregate_by_band(rows)
    logger.info(f"Per-band preprocessing summary ({title_tag}):\n{format_summary_table(agg)}")
    _render_figure(rows, agg, title_tag, output_path)
    return output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def default_db_path() -> str:
    output_path = os.environ.get(
        "AETHERSCAN_OUTPUT_PATH", "/datax/scratch/zachy/outputs/aetherscan"
    )
    return os.path.join(output_path, "db", "aetherscan.db")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a per-band inference-performance plot (preprocessing wall grouped "
        "by band/frequency) for one Aetherscan inference run."
    )
    parser.add_argument("--save-tag", required=True, help="Run tag to report on")
    parser.add_argument(
        "--catalog",
        required=True,
        nargs="+",
        help="Inference catalog CSV path(s), in planner order (same as config.data.inference_files)",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to aetherscan.db (default: {AETHERSCAN_OUTPUT_PATH}/db/aetherscan.db)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the plot PNG (default: <db parent>/../plots)",
    )
    parser.add_argument(
        "--hostname",
        default=None,
        help="Hostname for the suptitle (default: this machine's hostname)",
    )
    args = parser.parse_args(argv)

    db_path = args.db_path or default_db_path()
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}", file=sys.stderr)
        return 1

    output_dir = args.output_dir or os.path.join(os.path.dirname(os.path.dirname(db_path)), "plots")
    output_path = os.path.join(output_dir, f"perband_inference_perf_{args.save_tag}.png")
    hostname = args.hostname or socket.gethostname()

    result = render_perband_report(db_path, args.save_tag, args.catalog, output_path, hostname)
    if result is None:
        print("Per-band plot was skipped (see warnings above).")
        return 1

    # Re-derive the summary for the standalone stdout emit (render logs it too)
    durations = load_umbrella_preprocess_durations(db_path, args.save_tag)
    cadence_map = map_catalog_cadences(args.catalog) or []
    rows = [
        CadenceBandRow(n=n, band=band, frequency_mhz=freq, preprocess_s=durations[n])
        for n, band, freq in cadence_map
        if n in durations
    ]
    print(f"Per-band inference performance for tag {args.save_tag!r} ({len(rows)} cadence(s))")
    print()
    print(format_summary_table(aggregate_by_band(rows)))
    print()
    print(f"Plot written to {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
