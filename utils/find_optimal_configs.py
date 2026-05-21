#!/usr/bin/env python3
"""
Validate the Aetherscan config (defaults from src/aetherscan/config.py, plus any pipeline
CLI flags supplied as overrides) and propose optimal values for every violated constraint.

The script accepts the same flag surface as the pipeline's `train` and `inference`
subcommands — overrides applied here merge into the singleton config exactly as they would
in a real pipeline run. The only flag unique to this script is `--num-gpus`, which takes a
comma-separated list of replica counts to validate against (e.g., `--num-gpus 4,6` finds a
config valid under both 4-GPU and 6-GPU training).

Examples:
    # Check default config against a 4-GPU setup
    %(prog)s --num-gpus 4 train

    # Check defaults overridden by --effective-batch-size and search across 4 & 6 GPUs
    %(prog)s --num-gpus 4,6 train --effective-batch-size 3072

    # Check inference defaults with an invalid overlap-fraction
    %(prog)s inference --overlap-fraction 1.5
"""

import argparse
import os
import sys
from functools import reduce
from itertools import product
from math import gcd

# Make src/ importable when running directly: utils/find_optimal_configs.py
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_SCRIPT_DIR), "src"))

from aetherscan.cli import (  # noqa: E402
    ValidationError,
    _add_inference_flags_to,
    _add_train_flags_to,
    apply_args_to_config,
    collect_validation_errors,
)
from aetherscan.config import get_config, init_config  # noqa: E402

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the script's parser: top-level --num-gpus + train/inference subcommands that
    mirror the pipeline's flag surface via :func:`_add_train_flags_to` /
    :func:`_add_inference_flags_to`."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate Aetherscan config (defaults + CLI overrides) and propose optimal "
            "values for any violations."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--num-gpus",
        type=str,
        default="1",
        help=(
            "Comma-separated GPU/replica counts to validate against (e.g., '4' or '4,6'). "
            "Cross-replica constraints are checked for each value and the proposed fix must "
            "satisfy all of them simultaneously."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_sub = subparsers.add_parser("train", help="Validate training config")
    _add_train_flags_to(train_sub)
    inf_sub = subparsers.add_parser("inference", help="Validate inference config")
    _add_inference_flags_to(inf_sub)
    return parser


# ---------------------------------------------------------------------------
# Cross-parameter constraint solver (batch / sample divisibility)
# ---------------------------------------------------------------------------
#
# Six interdependent training parameters control how data is divided across
# replicas, batches, and train/val splits. Most violations are not fixable by
# changing a single value — we need to find a coordinated patch. We do this with
# a bounded grid search minimizing L1 distance to the current values.

_SEARCH_RANGES: dict[str, tuple[int, int, int]] = {
    # (min, max, step)
    "num_samples_beta_vae": (400_000, 600_000, 10_240),
    "num_samples_rf": (80_000, 120_000, 2_048),
    "per_replica_batch_size": (64, 256, 1),
    "effective_batch_size": (2_048, 6_144, 64),
    "per_replica_val_batch_size": (256, 768, 1),
}

_CROSS_PARAM_FIELDS = (
    "num_samples_beta_vae",
    "num_samples_rf",
    "per_replica_batch_size",
    "effective_batch_size",
    "per_replica_val_batch_size",
)


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // gcd(a, b)


def lcm_multiple(numbers: list[int]) -> int:
    return reduce(_lcm, numbers)


def _check_cross_constraints(
    nsb: int,
    nsr: int,
    tvs: float,
    prb: int,
    eb: int,
    prvb: int,
    num_replicas_list: list[int],
) -> bool:
    """Return True iff the six-tuple satisfies all cross-replica constraints for every
    num_replicas in the list. Mirrors the cross-replica checks in
    :func:`aetherscan.cli.collect_validation_errors`."""
    if not (0 <= tvs <= 1):
        return False
    if nsr % 2 != 0:
        return False
    for nr in num_replicas_list:
        gtrain = prb * nr
        gval = prvb * nr
        # round() rather than int() — `nsb * (1 - tvs)` suffers IEEE-754 noise (e.g.
        # 499200 * 0.2 = 99839.999...) and int() would truncate to 99839.
        train_samples = round(nsb * tvs)
        val_samples = round(nsb * (1 - tvs))
        if not (gtrain <= eb <= nsb * tvs):
            return False
        if gval > nsb * (1 - tvs):
            return False
        if gval > nsr:
            return False
        if eb % gtrain != 0:
            return False
        if train_samples % eb != 0:
            return False
        if val_samples % gval != 0:
            return False
        if nsr % gval != 0:
            return False
    return True


def _solve_cross_param_constraints(
    base: dict[str, int | float],
    num_replicas_list: list[int],
    max_candidates: int = 10_000_000,
) -> dict[str, int | float] | None:
    """Grid-search the six batch/sample params for a configuration satisfying all
    cross-replica constraints, minimizing L1 distance from ``base``. Returns None if no
    solution found within the search ranges or the candidate budget."""
    if _check_cross_constraints(
        base["num_samples_beta_vae"],
        base["num_samples_rf"],
        base["train_val_split"],
        base["per_replica_batch_size"],
        base["effective_batch_size"],
        base["per_replica_val_batch_size"],
        num_replicas_list,
    ):
        return dict(base)

    ranges: dict[str, list[int]] = {}
    for field in _CROSS_PARAM_FIELDS:
        lo, hi, step = _SEARCH_RANGES[field]
        ranges[field] = list(range(lo, hi + 1, step))

    total = 1
    for r in ranges.values():
        total *= len(r)
    if total > max_candidates:
        print(
            f"  [solver] Search space ({total:,}) exceeds budget ({max_candidates:,}); "
            f"narrow --num-gpus list or accept a coarser fix."
        )
        return None

    tvs = base["train_val_split"]  # held fixed across the search
    best: dict[str, int | float] | None = None
    best_dist = float("inf")
    for nsb, nsr, prb, eb, prvb in product(
        ranges["num_samples_beta_vae"],
        ranges["num_samples_rf"],
        ranges["per_replica_batch_size"],
        ranges["effective_batch_size"],
        ranges["per_replica_val_batch_size"],
    ):
        if not _check_cross_constraints(nsb, nsr, tvs, prb, eb, prvb, num_replicas_list):
            continue
        dist = (
            abs(nsb - base["num_samples_beta_vae"])
            + abs(nsr - base["num_samples_rf"])
            + abs(prb - base["per_replica_batch_size"])
            + abs(eb - base["effective_batch_size"])
            + abs(prvb - base["per_replica_val_batch_size"])
        )
        if dist < best_dist:
            best_dist = dist
            best = {
                "num_samples_beta_vae": nsb,
                "num_samples_rf": nsr,
                "train_val_split": tvs,
                "per_replica_batch_size": prb,
                "effective_batch_size": eb,
                "per_replica_val_batch_size": prvb,
            }
    return best


# ---------------------------------------------------------------------------
# Per-error fix proposer
# ---------------------------------------------------------------------------


def _round_to_multiple(val: int, divisor: int) -> int:
    """Round ``val`` to the nearest multiple of ``divisor`` (ties round up)."""
    q, r = divmod(val, divisor)
    return divisor * (q if r * 2 < divisor else q + 1)


def propose_simple_fix(err: ValidationError) -> object | None:
    """Suggest a single replacement value for a non-cross-param violation. Returns None when
    a violation cannot be auto-fixed (e.g. a missing file)."""
    if err.fix_kind == "clamp_low" and err.min_val is not None:
        return err.min_val
    if err.fix_kind == "clamp_high" and err.max_val is not None:
        return err.max_val
    if err.fix_kind == "range" and err.min_val is not None and err.max_val is not None:
        if isinstance(err.current, (int, float)):
            return max(err.min_val, min(err.current, err.max_val))
        return err.min_val
    if err.fix_kind == "enum" and err.allowed:
        return err.allowed[0]
    if err.fix_kind == "divisibility" and err.divisor and isinstance(err.current, int):
        return _round_to_multiple(err.current, err.divisor)
    return None  # format / file_exists / cross_param handled separately


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

_BAR = "=" * 80
_RULE = "-" * 80


def _print_header(mode: str, num_replicas_list: list[int]) -> None:
    print(_BAR)
    print(f"Aetherscan Config Sanity Check  —  mode: {mode}  —  num_replicas: {num_replicas_list}")
    print(_BAR)


def _format_value(val: object) -> str:
    if isinstance(val, float):
        return f"{val:g}"
    if isinstance(val, list):
        return "[" + ", ".join(str(x) for x in val) + "]"
    return str(val)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_num_gpus(raw: str) -> list[int]:
    try:
        out = [int(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError as exc:
        raise SystemExit(
            f"--num-gpus must be a comma-separated list of integers, got {raw!r}"
        ) from exc
    if not out:
        raise SystemExit("--num-gpus must list at least one value")
    if any(n < 1 for n in out):
        raise SystemExit(f"--num-gpus values must be >= 1, got {out}")
    return out


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    num_replicas_list = _parse_num_gpus(args.num_gpus)
    # If user explicitly passed the subcommand --num-replicas, prefer it
    if getattr(args, "num_replicas", None) is not None:
        num_replicas_list = [int(args.num_replicas)]

    init_config()
    apply_args_to_config(args)
    config = get_config()

    _print_header(args.command, num_replicas_list)

    # Collect violations across every replica count, dedup by message
    all_errors: list[ValidationError] = []
    seen: set[str] = set()
    for nr in num_replicas_list:
        for err in collect_validation_errors(args, nr):
            if err.message in seen:
                continue
            seen.add(err.message)
            all_errors.append(err)

    if not all_errors:
        print("All config values valid for every requested replica count.")
        return 0

    # Partition: cross-param (solver) vs simple (clamp / enum / etc.)
    cross_param_errors = [e for e in all_errors if e.fix_kind == "cross_param"]
    simple_errors = [e for e in all_errors if e.fix_kind != "cross_param"]

    proposals: dict[str, object] = {}

    if simple_errors:
        print(f"\n{len(simple_errors)} simple violation(s):")
        for err in simple_errors:
            fix = propose_simple_fix(err)
            fix_str = (
                f"  →  suggested: {_format_value(fix)}" if fix is not None else "  (no auto-fix)"
            )
            print(f"  ✗ {err.message}{fix_str}")
            if fix is not None:
                proposals[err.field] = fix

    if cross_param_errors:
        print(f"\n{len(cross_param_errors)} cross-parameter violation(s):")
        for err in cross_param_errors:
            print(f"  ✗ {err.message}")
        # Run the grid search once for all cross-param violations
        base = {
            "num_samples_beta_vae": config.training.num_samples_beta_vae,
            "num_samples_rf": config.training.num_samples_rf,
            "train_val_split": config.training.train_val_split,
            "per_replica_batch_size": config.training.per_replica_batch_size,
            "effective_batch_size": config.training.effective_batch_size,
            "per_replica_val_batch_size": config.training.per_replica_val_batch_size,
        }
        print(
            f"\n  Searching for a coordinated patch across {len(num_replicas_list)} replica count(s)..."
        )
        solution = _solve_cross_param_constraints(base, num_replicas_list)
        if solution is None:
            print("  [solver] No valid configuration found within search ranges.")
        else:
            print(f"\n  {_RULE}")
            print(f"  {'Parameter':<35} {'Current':<15} {'Proposed':<15} {'Delta'}")
            print(f"  {_RULE}")
            for field in (
                "num_samples_beta_vae",
                "num_samples_rf",
                "train_val_split",
                "per_replica_batch_size",
                "effective_batch_size",
                "per_replica_val_batch_size",
            ):
                cur = base[field]
                new = solution[field]
                if field == "train_val_split":
                    delta = f"{new - cur:+.4f}"
                    cur_s, new_s = f"{cur:.4f}", f"{new:.4f}"
                else:
                    delta = f"{new - cur:+d}"
                    cur_s, new_s = str(cur), str(new)
                marker = "  " if cur == new else " *"
                print(f"  {field:<35} {cur_s:<15} {new_s:<15} {delta}{marker}")
                if cur != new:
                    proposals[f"training.{field}"] = new

    print(f"\n{_RULE}")
    if proposals:
        print("\nProposed pipeline CLI flags to apply all fixes:")
        flag_parts = []
        for field, value in proposals.items():
            # Convert dotted config field → CLI flag form. Most fields drop the
            # config-section prefix and use kebab-case.
            short = field.split(".", 1)[-1]
            flag = "--" + short.replace("_", "-")
            flag_parts.append(f"{flag} {_format_value(value)}")
        print("  " + " \\\n  ".join(flag_parts))
    else:
        print("\nNo automated fixes proposed (violations require manual resolution).")

    return 1


if __name__ == "__main__":
    sys.exit(main())
