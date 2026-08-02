#!/usr/bin/env python3
"""
Render a benchmarking report for one pipeline run from its pipeline_stages DB rows.

Given a --save-tag (and optionally a --db-path), this tool:
1. prints the run's stage tree — every recorded pipeline_stages span, grouped by its
   hierarchical dot-name, with call counts, total/self durations, and percentages;
2. writes plots/benchmark_report_{tag}.png: a flame-style stage timeline (one lane per
   depth) plus a "top 10 slowest stages" table ranked by self time;
3. prints a rules-driven "suggestions" section that joins pipeline_stages with
   system_resources (e.g. "data generation ≥ 30% of round wall-clock", "GPU util < 40%
   during epochs → input-bound").

Reads the SQLite file directly (stdlib sqlite3 + numpy + matplotlib only — no aetherscan
imports), so it also runs against a database fetched from a cluster with
utils/fetch_run_outputs.sh:

    python utils/benchmark_report.py --save-tag train_20260101_120000
    python utils/benchmark_report.py --save-tag test_20260101_120000 --db-path /path/to/aetherscan.db
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sqlite3
import sys
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless environments

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Thresholds for the rules-driven suggestions section
DATA_GEN_ROUND_FRACTION = 0.30  # data_generation / round wall-clock
GPU_UTIL_INPUT_BOUND = 40.0  # mean GPU % during epochs/encode below this -> input-bound
PREPROCESS_WALL_FRACTION = 0.60  # summed ED preprocessing / inference wall-clock
RAM_PEAK_WARN = 90.0  # peak system RAM % during the run

# Timeline lane colors keyed by top-level stage family (fallback cycles matplotlib tab10)
_FAMILY_COLORS = {"train": "tab:blue", "inference": "tab:green"}


# ---------------------------------------------------------------------------
# Stage tree
# ---------------------------------------------------------------------------


@dataclass
class StageNode:
    """One node of the stage tree: a dot-name component plus its spans and children."""

    name: str  # This component ("round_02"), not the full dotted name
    full_name: str  # Full dotted name ("train.round_02")
    spans: list[dict] = field(default_factory=list)  # Rows recorded exactly at this name
    children: dict[str, StageNode] = field(default_factory=dict)

    @property
    def own_duration(self) -> float:
        """Summed duration of the spans recorded exactly at this node's name."""
        return sum(span["duration_s"] for span in self.spans)

    @property
    def total_duration(self) -> float:
        """Own duration when this node has spans of its own (an umbrella span already
        covers its children's time); otherwise the sum over children (a pure grouping
        node like "train" that never gets its own row)."""
        if self.spans:
            return self.own_duration
        return sum(child.total_duration for child in self.children.values())

    @property
    def self_duration(self) -> float:
        """Time not covered by child stages: total - sum(child totals), floored at 0
        (children running concurrently with the parent — e.g. producer data generation
        overlapping a round — can sum past the parent's wall-clock)."""
        return max(0.0, self.total_duration - sum(c.total_duration for c in self.children.values()))

    def sorted_children(self) -> list[StageNode]:
        """Children in chronological order (min span start), grouping nodes last-resort
        by name so the tree is stable even for nodes without spans."""

        def sort_key(node: StageNode):
            starts = [s["start_time"] for s in node.iter_spans()]
            return (min(starts) if starts else float("inf"), node.name)

        return sorted(self.children.values(), key=sort_key)

    def iter_spans(self):
        yield from self.spans
        for child in self.children.values():
            yield from child.iter_spans()


def build_stage_tree(rows: list[dict]) -> StageNode:
    """
    Fold pipeline_stages rows into a tree keyed by dot-name components. Returns a
    synthetic root (name "", full_name "") whose children are the top-level families
    ("train", "inference"). Rows sharing a name (retries, per-cadence repeats) accumulate
    as multiple spans on one node.
    """
    root = StageNode(name="", full_name="")
    for row in rows:
        parts = str(row["stage"]).split(".")
        node = root
        for depth, part in enumerate(parts):
            if part not in node.children:
                node.children[part] = StageNode(name=part, full_name=".".join(parts[: depth + 1]))
            node = node.children[part]
        node.spans.append(row)
    return root


def format_duration(seconds: float) -> str:
    """Compact human duration: 45.3s / 12m 05s / 2h 13m."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        minutes, secs = divmod(int(round(seconds)), 60)
        return f"{minutes}m {secs:02d}s"
    hours, rem = divmod(int(round(seconds)), 3600)
    return f"{hours}h {rem // 60:02d}m"


def format_tree(root: StageNode) -> list[str]:
    """
    Render the stage tree as aligned text lines. Percentages are relative to the parent
    node's total (top-level families show 100%); n is the number of spans recorded at
    that exact name (0 for pure grouping nodes).
    """
    lines = []

    def _walk(node: StageNode, depth: int, parent_total: float):
        pct = 100.0 * node.total_duration / parent_total if parent_total > 0 else 0.0
        label = f"{'  ' * depth}{node.name}"
        lines.append(
            f"{label:<48}{len(node.spans):>4}  "
            f"{format_duration(node.total_duration):>10}  {pct:6.1f}%"
        )
        for child in node.sorted_children():
            _walk(child, depth + 1, node.total_duration)

    header = f"{'stage':<48}{'n':>4}  {'total':>10}  {'% of parent':>7}"
    lines.append(header)
    lines.append("-" * len(header))
    for top in root.sorted_children():
        _walk(top, 0, top.total_duration)
    return lines


def collect_nodes(root: StageNode) -> list[StageNode]:
    """Flatten the tree (excluding the synthetic root) in depth-first order."""
    nodes: list[StageNode] = []

    def _walk(node: StageNode):
        for child in node.sorted_children():
            nodes.append(child)
            _walk(child)

    _walk(root)
    return nodes


def top_slowest(root: StageNode, k: int = 10) -> list[StageNode]:
    """Top-k nodes ranked by self time (time not attributed to any child stage) — the
    stages where wall-clock actually went, rather than umbrella spans."""
    nodes = [n for n in collect_nodes(root) if n.spans]
    nodes.sort(key=lambda n: n.self_duration, reverse=True)
    return nodes[:k]


# ---------------------------------------------------------------------------
# DB access
# ---------------------------------------------------------------------------


def load_rows(db_path: str, tag: str) -> list[dict]:
    """Load this tag's pipeline_stages rows (chronological) as dicts."""
    # contextlib.closing: sqlite3's context manager only commits/rolls back — it does NOT
    # close the connection (same below)
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                "SELECT stage, start_time, end_time, duration_s, tag, metadata "
                "FROM pipeline_stages WHERE tag = ? ORDER BY start_time",
                (tag,),
            )
        except sqlite3.OperationalError as e:
            raise SystemExit(
                f"Could not read pipeline_stages from {db_path}: {e}. "
                f"(Database predates the benchmarking schema?)"
            ) from e
        return [dict(row) for row in cursor.fetchall()]


def load_resource_series(db_path: str, tag: str, resource_type: str, name_like: str):
    """(timestamps, values) arrays from system_resources for one tag + resource family."""
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        try:
            cursor = conn.execute(
                "SELECT timestamp, value FROM system_resources "
                "WHERE tag = ? AND resource_type = ? AND resource_name LIKE ? "
                "ORDER BY timestamp",
                (tag, resource_type, name_like),
            )
            rows = cursor.fetchall()
        except sqlite3.OperationalError:
            rows = []
    if not rows:
        return np.array([]), np.array([])
    arr = np.asarray(rows, dtype=np.float64)
    return arr[:, 0], arr[:, 1]


def mean_in_window(
    timestamps: np.ndarray, values: np.ndarray, start: float, end: float
) -> float | None:
    """Mean of the samples falling inside [start, end], or None when there are none."""
    if timestamps.size == 0:
        return None
    mask = (timestamps >= start) & (timestamps <= end)
    if not mask.any():
        return None
    return float(values[mask].mean())


# ---------------------------------------------------------------------------
# Suggestions
# ---------------------------------------------------------------------------


def build_suggestions(root: StageNode, db_path: str, tag: str) -> list[str]:
    """Apply the bottleneck heuristics and return human-readable suggestion strings."""
    suggestions: list[str] = []

    gpu_ts, gpu_vals = load_resource_series(db_path, tag, "gpu", "%_utilization")
    ram_ts, ram_vals = load_resource_series(db_path, tag, "ram", "system_total")

    train = root.children.get("train")
    inference = root.children.get("inference")

    # Rule 1: per-round data generation dominating the round's wall-clock
    if train is not None:
        for node in train.sorted_children():
            if not node.name.startswith("round_") or not node.spans:
                continue
            datagen = node.children.get("data_generation")
            if datagen is None or node.total_duration <= 0:
                continue
            # Guard against a data_generation node that is a pure grouping node (children but
            # no own span): the producer-overlap branch below takes max/min over datagen.spans,
            # which would raise on an empty sequence (mirrors Rule 3's empty-spans guard).
            if not datagen.spans:
                continue
            frac = datagen.total_duration / node.total_duration
            if frac >= DATA_GEN_ROUND_FRACTION:
                source = None
                for span in datagen.spans:
                    if span.get("metadata"):
                        source = json.loads(span["metadata"]).get("source")
                if source == "in-process":
                    fix = "enable --overlap-data-generation so it runs while epochs train"
                else:
                    # Producer path: generation runs in the background during the previous
                    # round's epochs, so a big span only matters if this round actually
                    # waited on it — i.e. generation was still running when the round
                    # started. Fully-overlapped generation is free.
                    datagen_end = max(s["end_time"] for s in datagen.spans)
                    round_start = min(s["start_time"] for s in node.spans)
                    if datagen_end <= round_start + 1.0:  # 1 s tolerance
                        continue  # Fully overlapped — nothing to suggest
                    fix = (
                        "the round waited on background generation — raise worker "
                        "count (manager.n_processes) / --data-gen-task-size, or reduce "
                        "--num-samples-beta-vae"
                    )
                suggestions.append(
                    f"{node.full_name}: data_generation took "
                    f"{format_duration(datagen.total_duration)} = {frac:.0%} of the "
                    f"round's wall-clock (>= {DATA_GEN_ROUND_FRACTION:.0%}) — {fix}."
                )

        # Rule 2: GPU utilization during epoch spans
        for node in train.sorted_children():
            epochs = node.children.get("epochs")
            if epochs is None or not epochs.spans:
                continue
            for span in epochs.spans:
                mean_util = mean_in_window(gpu_ts, gpu_vals, span["start_time"], span["end_time"])
                if mean_util is not None and mean_util < GPU_UTIL_INPUT_BOUND:
                    suggestions.append(
                        f"{epochs.full_name}: mean GPU utilization {mean_util:.0f}% "
                        f"(< {GPU_UTIL_INPUT_BOUND:.0f}%) — training is input-bound; "
                        f"check the tf.data feed (batch sizes, host-side preprocessing)."
                    )

    # Rule 3: energy-detection preprocessing dominating inference wall-clock
    if inference is not None:
        preprocess_total = sum(
            node.total_duration
            for node in inference.children.values()
            if node.name.startswith("preprocess_cadence_")
        )
        # Guard against an inference subtree of pure grouping nodes with no measured spans
        # (min/max over an empty sequence would raise); such a run has no wall to compare.
        spans = list(inference.iter_spans())
        if spans:
            wall = max(s["end_time"] for s in spans) - min(s["start_time"] for s in spans)
            if wall > 0 and preprocess_total / wall >= PREPROCESS_WALL_FRACTION:
                suggestions.append(
                    f"inference: energy-detection preprocessing summed to "
                    f"{format_duration(preprocess_total)} = {preprocess_total / wall:.0%} of "
                    f"the run's wall-clock (>= {PREPROCESS_WALL_FRACTION:.0%}) — it is the "
                    f"bottleneck; raise ED worker count (manager.n_processes) before anything "
                    f"GPU-side."
                )

        # Rule 2b: GPU utilization during per-cadence encode spans (averaged over spans)
        encode_means = []
        for node in inference.children.values():
            encode = node.children.get("encode")
            if encode is None:
                continue
            for span in encode.spans:
                mean_util = mean_in_window(gpu_ts, gpu_vals, span["start_time"], span["end_time"])
                if mean_util is not None:
                    encode_means.append(mean_util)
        if encode_means and float(np.mean(encode_means)) < GPU_UTIL_INPUT_BOUND:
            suggestions.append(
                f"inference: mean GPU utilization across encode spans is "
                f"{np.mean(encode_means):.0f}% (< {GPU_UTIL_INPUT_BOUND:.0f}%) — encoding "
                f"is input-bound; consider a larger --per-replica-batch-size."
            )

    # Rule 4: RAM pressure anywhere in the run
    if ram_vals.size > 0:
        peak = float(ram_vals.max())
        if peak >= RAM_PEAK_WARN:
            when = float(ram_ts[int(ram_vals.argmax())])
            holder = None
            for node in collect_nodes(root):
                for span in node.spans:
                    if span["start_time"] <= when <= span["end_time"] and (
                        holder is None or len(node.full_name) > len(holder)
                    ):
                        holder = node.full_name
            suggestions.append(
                f"peak system RAM hit {peak:.1f}% (>= {RAM_PEAK_WARN:.0f}%)"
                + (f" during {holder}" if holder else "")
                + " — OOM risk; reduce chunk/sample sizes or disable keep_round_data."
            )

    return suggestions


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def render_report_png(root: StageNode, rows: list[dict], tag: str, output_path: str) -> None:
    """Write the report figure: flame-style stage timeline + top-10-slowest table."""
    run_start = min(row["start_time"] for row in rows)
    run_end = max(row["end_time"] for row in rows)

    fig, (ax_tl, ax_tb) = plt.subplots(
        2,
        1,
        figsize=(16, 11),
        gridspec_kw={"height_ratios": [3, 1.4]},
    )
    fig.suptitle(f"Aetherscan Pipeline: Benchmark Report ({tag})", fontsize=16, fontweight="bold")

    # --- Timeline: one lane per dot-depth actually present (grouping-only depths like a
    # bare "train" never get spans of their own, so depth 1 is usually empty), bars
    # colored by top-level family ---
    depths_present = sorted({len(str(row["stage"]).split(".")) for row in rows})
    lane_of = {depth: i for i, depth in enumerate(depths_present)}
    n_lanes = len(depths_present)
    fallback_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    families = sorted({str(row["stage"]).split(".", maxsplit=1)[0] for row in rows})

    def family_color(stage: str):
        family = stage.split(".", maxsplit=1)[0]
        if family in _FAMILY_COLORS:
            return _FAMILY_COLORS[family]
        return fallback_colors[families.index(family) % 10]

    run_minutes = max((run_end - run_start) / 60, 1e-9)
    for row in rows:
        depth = len(str(row["stage"]).split("."))
        start_min = (row["start_time"] - run_start) / 60
        width_min = max(row["duration_s"] / 60, run_minutes * 0.001)  # keep slivers visible
        y = n_lanes - 1 - lane_of[depth]  # shallowest lane on top
        ax_tl.barh(
            y,
            width_min,
            left=start_min,
            height=0.8,
            color=family_color(str(row["stage"])),
            alpha=0.75,
            edgecolor="white",
            linewidth=0.4,
        )
        # Leaf-component label inside the bar when there's room (>2.5% of the run width)
        if width_min > 0.025 * run_minutes:
            ax_tl.text(
                start_min + width_min / 2,
                y,
                str(row["stage"]).split(".")[-1],
                ha="center",
                va="center",
                fontsize=7,
                color="black",
                clip_on=True,
            )

    ax_tl.set_yticks(range(n_lanes))
    ax_tl.set_yticklabels([f"depth {depth}" for depth in reversed(depths_present)], fontsize=9)
    ax_tl.set_xlabel("Time since first stage (minutes)", fontsize=11, fontweight="bold")
    ax_tl.set_title(
        "Stage timeline (bars nest top-down by dot-depth; overlaps = concurrent stages)",
        fontsize=11,
    )
    ax_tl.grid(True, axis="x", alpha=0.3)

    # --- Top-10-slowest table (by self time) ---
    ax_tb.axis("off")
    slowest = top_slowest(root, k=10)
    total_wall = max(run_end - run_start, 1e-9)
    cells = [
        [
            node.full_name,
            str(len(node.spans)),
            format_duration(node.total_duration),
            format_duration(node.self_duration),
            f"{100.0 * node.self_duration / total_wall:.1f}%",
        ]
        for node in slowest
    ]
    table = ax_tb.table(
        cellText=cells,
        colLabels=["stage", "n", "total", "self", "self % of wall"],
        colWidths=[0.52, 0.06, 0.14, 0.14, 0.14],
        cellLoc="left",
        # Explicit bbox keeps the table below the axes title instead of overlapping it
        bbox=(0.0, 0.0, 1.0, 0.86),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    ax_tb.set_title("Top 10 slowest stages (by self time)", fontsize=11, y=0.9)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    # dirname is "" for a bare filename (cwd) — makedirs("") would raise, so only create
    # a parent directory when output_path actually has one
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
        description="Print a stage-timing tree, write a benchmark report PNG, and flag "
        "likely bottlenecks for one Aetherscan run."
    )
    parser.add_argument("--save-tag", required=True, help="Run tag to report on")
    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to aetherscan.db (default: {AETHERSCAN_OUTPUT_PATH}/db/aetherscan.db)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the report PNG (default: <db parent>/../plots)",
    )
    args = parser.parse_args(argv)

    db_path = args.db_path or default_db_path()
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}", file=sys.stderr)
        return 1

    rows = load_rows(db_path, args.save_tag)
    if not rows:
        print(f"No pipeline_stages rows found for tag {args.save_tag!r} in {db_path}")
        return 1

    root = build_stage_tree(rows)

    print(f"Benchmark report for tag {args.save_tag!r} ({len(rows)} stage spans)")
    print()
    for line in format_tree(root):
        print(line)

    print()
    total_wall = max(r["end_time"] for r in rows) - min(r["start_time"] for r in rows)
    print(f"Run wall-clock (first stage start -> last stage end): {format_duration(total_wall)}")
    print()
    print("Top 10 slowest stages (by self time):")
    for node in top_slowest(root, k=10):
        print(
            f"  {node.full_name:<52} total {format_duration(node.total_duration):>10}"
            f"   self {format_duration(node.self_duration):>10}"
        )

    print()
    print("Suggestions:")
    suggestions = build_suggestions(root, db_path, args.save_tag)
    if suggestions:
        for suggestion in suggestions:
            print(f"  - {suggestion}")
    else:
        print("  - No bottleneck heuristics triggered.")

    output_dir = args.output_dir or os.path.join(os.path.dirname(os.path.dirname(db_path)), "plots")
    output_path = os.path.join(output_dir, f"benchmark_report_{args.save_tag}.png")
    render_report_png(root, rows, args.save_tag, output_path)
    print()
    print(f"Report figure written to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
