#!/usr/bin/env python3
"""
Candidate RFI-triage report for one streaming-CSV inference run (#396).

Reads a run's candidates (inference_results, prediction=1) straight from the SQLite DB and
prints the three RFI signatures a reviewer wants before opening a single waterfall:

1. **Multi-target frequency coincidence** — candidates whose frequency bin lights up
   across >= N distinct targets. The sky does not put the same narrowband signal on many
   independent pointings; a terrestrial transmitter does. This catches transmitters that
   appear in no static allocation table.
2. **Known-RFI allocation flags** — a built-in (deliberately coarse, easily extended)
   table of allocations that dominated past runs' false positives (GPS, GLONASS, Iridium).
3. **Per-target concentration** — a handful of targets holding most of the candidates
   (e.g. NGC1172 + MESSIER87 held 49% in inf_blpc3_20260807_011509) reads as localized
   RFI/receiver artifacts, not uniform detections.

Also mirrors the pipeline's report-time frequency-exclusion accounting
(--exclude-frequency-range, same semantics as the pipeline's
--report-exclude-frequency-range) so off-cluster reviews see the same original vs
excluded vs reported split.

Everything here FLAGS, nothing deletes: the DB is opened read-only and candidates are
never dropped — a genuine technosignature re-observed at one target is, by construction,
not multi-target-coincident, but the human stays the judge.

Reads the SQLite file directly (stdlib sqlite3 + csv + numpy only — no aetherscan
imports), so it also runs against a database fetched from a cluster with
utils/fetch_run_outputs.sh:

    python utils/candidate_rfi_report.py --save-tag inf_20260807_011509
    python utils/candidate_rfi_report.py --save-tag inf_20260807_011509 \
        --db-path /path/to/aetherscan.db --csv triage.csv \
        --exclude-frequency-range 1616 1626.5
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import os
import sqlite3
import sys
import urllib.parse
from collections import Counter, defaultdict

import numpy as np

# Known-RFI allocations (MHz). Coarse by design: edges are the nominal allocation plus a
# little out-of-band spill where past runs showed it (Iridium OOB emission is notorious).
# Extend freely — every entry only ever FLAGS candidates.
KNOWN_RFI_BANDS: list[tuple[str, float, float]] = [
    ("GPS L1", 1574.4, 1576.4),
    ("GPS L2", 1226.6, 1228.6),
    ("GPS L5", 1175.4, 1177.5),
    ("GLONASS L1", 1598.0, 1606.0),
    ("GLONASS L2", 1242.0, 1249.0),
    ("Iridium", 1616.0, 1626.5),
]

DEFAULT_COINCIDENCE_BIN_MHZ = 0.2
DEFAULT_MIN_TARGETS = 3


def default_db_path() -> str:
    output_path = os.environ.get(
        "AETHERSCAN_OUTPUT_PATH", "/datax/scratch/zachy/outputs/aetherscan"
    )
    return os.path.join(output_path, "db", "aetherscan.db")


def load_candidates(db_path: str, tag: str) -> list[dict]:
    """This tag's live candidate rows (prediction=1, superseded=0), one dict per row."""
    # quote() the path: sqlite's URI parser treats raw ? and # as parameter/fragment
    # markers, which would truncate the path and silently open (or CREATE) a different
    # file read-write — breaking the read-only guarantee this script advertises
    ro_uri = f"file:{urllib.parse.quote(db_path)}?mode=ro"
    with contextlib.closing(sqlite3.connect(ro_uri, uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT target, band, frequency_mhz, confidence, mc_mean, mc_std, "
            "npy_path, snippet_index FROM inference_results "
            "WHERE tag = ? AND prediction = 1 AND superseded = 0",
            (tag,),
        ).fetchall()
    return [dict(row) for row in rows]


def known_band_hits(candidates: list[dict]) -> dict[str, list[dict]]:
    """{allocation name: candidate rows inside it} for every allocation that was hit."""
    hits: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        freq = row.get("frequency_mhz")
        if freq is None:
            continue
        for name, start, end in KNOWN_RFI_BANDS:
            if start <= float(freq) <= end:
                hits[name].append(row)
    return dict(hits)


def coincidence_bins(candidates: list[dict], bin_mhz: float, min_targets: int) -> list[dict]:
    """
    Frequency bins (width bin_mhz) hit by >= min_targets distinct targets, most targets
    first. Neighboring-bin straddle is deliberately not merged: a transmitter wide or
    drifty enough to straddle bins lights up both, and both get flagged.
    """
    per_bin_targets: dict[int, set] = defaultdict(set)
    per_bin_rows: dict[int, list[dict]] = defaultdict(list)
    for row in candidates:
        freq = row.get("frequency_mhz")
        if freq is None:
            continue
        bin_index = int(float(freq) // bin_mhz)
        per_bin_targets[bin_index].add(row.get("target") or "?")
        per_bin_rows[bin_index].append(row)

    flagged = []
    for bin_index, targets in per_bin_targets.items():
        if len(targets) < min_targets:
            continue
        flagged.append(
            {
                "start_mhz": bin_index * bin_mhz,
                "end_mhz": (bin_index + 1) * bin_mhz,
                "n_targets": len(targets),
                "n_candidates": len(per_bin_rows[bin_index]),
                "targets": sorted(targets),
                "rows": per_bin_rows[bin_index],
            }
        )
    flagged.sort(key=lambda b: (-b["n_targets"], -b["n_candidates"], b["start_mhz"]))
    return flagged


def parse_exclusion_ranges(pairs: list[list[float]] | None) -> list[tuple[float, float]]:
    """Validate [[start, end], ...] MHz pairs (finite, 0 < start < end) — the same rules
    the pipeline's --report-exclude-frequency-range enforces."""
    if not pairs:
        return []
    ranges = []
    for pair in pairs:
        start, end = float(pair[0]), float(pair[1])
        if not (np.isfinite(start) and np.isfinite(end) and 0 < start < end):
            raise SystemExit(
                f"--exclude-frequency-range invalid: {pair!r} (need finite 0 < start < end)"
            )
        ranges.append((start, end))
    return sorted(ranges)


def in_ranges(freq, ranges: list[tuple[float, float]]) -> bool:
    return freq is not None and any(start <= float(freq) <= end for start, end in ranges)


def build_report(
    candidates: list[dict],
    bin_mhz: float,
    min_targets: int,
    exclusion_ranges: list[tuple[float, float]],
) -> tuple[str, list[dict]]:
    """(report text, per-candidate flagged rows in DB order)."""
    lines: list[str] = []
    total = len(candidates)
    lines.append(f"candidates: {total}")
    band_counts = Counter(r.get("band") or "?" for r in candidates)
    lines.append(
        "per band: " + ", ".join(f"{band} {count}" for band, count in sorted(band_counts.items()))
    )

    allocation_hits = known_band_hits(candidates)
    lines.append("")
    lines.append(f"known-RFI allocation hits ({len(allocation_hits)} allocation(s)):")
    if allocation_hits:
        for name, rows in sorted(allocation_hits.items(), key=lambda kv: -len(kv[1])):
            targets = sorted({r.get("target") or "?" for r in rows})
            lines.append(f"  {name}: {len(rows)} candidate(s) across {len(targets)} target(s)")
    else:
        lines.append("  none")

    bins = coincidence_bins(candidates, bin_mhz, min_targets)
    lines.append("")
    lines.append(
        f"multi-target coincidence bins (width {bin_mhz:g} MHz, >= {min_targets} targets): "
        f"{len(bins)}"
    )
    for entry in bins[:15]:
        sample = ", ".join(entry["targets"][:5])
        if entry["n_targets"] > 5:
            sample += ", ..."
        lines.append(
            f"  {entry['start_mhz']:.1f}-{entry['end_mhz']:.1f} MHz: "
            f"{entry['n_candidates']} candidate(s) across {entry['n_targets']} target(s) "
            f"({sample})"
        )
    if len(bins) > 15:
        lines.append(f"  ... and {len(bins) - 15} more bin(s)")

    lines.append("")
    lines.append("per-target concentration (top 10):")
    target_counts = Counter(r.get("target") or "?" for r in candidates)
    cumulative = 0
    for target, count in target_counts.most_common(10):
        cumulative += count
        lines.append(
            f"  {target}: {count} ({100 * count / total:.1f}%; cumulative "
            f"{100 * cumulative / total:.1f}%)"
        )

    if exclusion_ranges:
        excluded = [r for r in candidates if in_ranges(r.get("frequency_mhz"), exclusion_ranges)]
        range_label = ", ".join(f"{start:g}-{end:g}" for start, end in exclusion_ranges)
        lines.append("")
        lines.append(f"exclusion accounting ({range_label} MHz):")
        lines.append(f"  excluded: {len(excluded)}")
        lines.append(f"  reported after exclusion: {total - len(excluded)}")

    # Per-candidate flag rows for the optional CSV
    coincident_keys = {
        (row.get("npy_path"), row.get("snippet_index")) for entry in bins for row in entry["rows"]
    }
    allocation_by_key: dict[tuple, str] = {}
    for name, rows in allocation_hits.items():
        for row in rows:
            key = (row.get("npy_path"), row.get("snippet_index"))
            allocation_by_key[key] = (
                f"{allocation_by_key[key]};{name}" if key in allocation_by_key else name
            )
    flagged_rows = []
    for row in candidates:
        key = (row.get("npy_path"), row.get("snippet_index"))
        flagged_rows.append(
            {
                **row,
                "known_rfi_allocation": allocation_by_key.get(key, ""),
                "multi_target_coincident": int(key in coincident_keys),
                "excluded_by_range": int(in_ranges(row.get("frequency_mhz"), exclusion_ranges)),
            }
        )
    return "\n".join(lines), flagged_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Flag likely-RFI candidates for one Aetherscan inference run: "
        "multi-target frequency coincidence, known allocations, per-target concentration. "
        "Flags only — the database is opened read-only and nothing is deleted."
    )
    parser.add_argument("--save-tag", required=True, help="Run tag to report on")
    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to aetherscan.db (default: {AETHERSCAN_OUTPUT_PATH}/db/aetherscan.db)",
    )
    parser.add_argument(
        "--coincidence-bin-mhz",
        type=float,
        default=DEFAULT_COINCIDENCE_BIN_MHZ,
        help=f"Frequency bin width for the coincidence check "
        f"(default: {DEFAULT_COINCIDENCE_BIN_MHZ} MHz)",
    )
    parser.add_argument(
        "--min-targets",
        type=int,
        default=DEFAULT_MIN_TARGETS,
        help=f"Distinct targets a bin needs to be flagged coincident "
        f"(default: {DEFAULT_MIN_TARGETS})",
    )
    parser.add_argument(
        "--exclude-frequency-range",
        action="append",
        nargs=2,
        type=float,
        default=None,
        metavar=("START_MHZ", "END_MHZ"),
        help="Report-time exclusion accounting (repeatable) — same semantics as the "
        "pipeline's --report-exclude-frequency-range",
    )
    parser.add_argument(
        "--csv", default=None, help="Optional path for a per-candidate flagged-rows CSV"
    )
    args = parser.parse_args(argv)

    db_path = args.db_path or default_db_path()
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}", file=sys.stderr)
        return 1
    if args.coincidence_bin_mhz <= 0:
        print("--coincidence-bin-mhz must be > 0", file=sys.stderr)
        return 1
    if args.min_targets < 2:
        print("--min-targets must be >= 2 (one target cannot be coincident)", file=sys.stderr)
        return 1

    candidates = load_candidates(db_path, args.save_tag)
    if not candidates:
        print(f"No live candidates for tag {args.save_tag!r} in {db_path}")
        return 1

    exclusion_ranges = parse_exclusion_ranges(args.exclude_frequency_range)
    report, flagged_rows = build_report(
        candidates, args.coincidence_bin_mhz, args.min_targets, exclusion_ranges
    )
    print(f"Candidate RFI-triage report for tag {args.save_tag!r}")
    print()
    print(report)

    if args.csv:
        out_dir = os.path.dirname(args.csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flagged_rows[0].keys()))
            writer.writeheader()
            writer.writerows(flagged_rows)
        print()
        print(f"Per-candidate flags written to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
