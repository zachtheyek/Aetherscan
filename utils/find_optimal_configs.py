#!/usr/bin/env python3

# TODO: come back to this later
"""
Optimize training parameters to satisfy divisibility constraints across multiple replica counts.

Usage:
  # Use config values as defaults, vary all parameters
  ./utils/find_optimal_configs.py

  # Hold certain parameters constant
  ./utils/find_optimal_configs.py --hold-per-replica-batch-size --hold-train-val-split

  # Only allow certain parameters to increase
  ./utils/find_optimal_configs.py --only-increase-effective-batch-size --only-decrease-num-samples-beta-vae
"""

import argparse
import os
import sys
from functools import reduce
from itertools import product
from math import gcd

# Add parent directory to path to import from src/
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_root, "src"))

from aetherscan.config import init_config  # noqa: E402


def lcm(a, b):
    """Compute least common multiple of two numbers."""
    return abs(a * b) // gcd(a, b)


def lcm_multiple(numbers):
    """Compute LCM of multiple numbers."""
    return reduce(lcm, numbers)


def check_constraints(
    num_samples_beta_vae: int,
    num_samples_rf: int,
    train_val_split: float,
    per_replica_batch_size: int,
    effective_batch_size: int,
    per_replica_val_batch_size: int,
    num_replicas_list: list[int],
) -> bool:
    """Check if all constraints are satisfied for all replica counts."""

    # Constraint 1: 0 <= train_val_split <= 1
    if not (0 <= train_val_split <= 1):
        return False

    # Constraint 9: num_samples_rf is divisible by 2
    if num_samples_rf % 2 != 0:
        return False

    for num_replicas in num_replicas_list:
        # Constraint 2: per_replica_batch_size * num_replicas <= effective_batch_size <= num_samples_beta_vae * train_val_split
        if not (
            per_replica_batch_size * num_replicas
            <= effective_batch_size
            <= num_samples_beta_vae * train_val_split
        ):
            return False

        # Constraint 3: per_replica_val_batch_size * num_replicas <= num_samples_beta_vae * (1 - train_val_split)
        if not (
            per_replica_val_batch_size * num_replicas
            <= num_samples_beta_vae * (1 - train_val_split)
        ):
            return False

        # Constraint 4: per_replica_val_batch_size * num_replicas <= num_samples_rf
        if not (per_replica_val_batch_size * num_replicas <= num_samples_rf):
            return False

        # Constraint 5: effective_batch_size is divisible by per_replica_batch_size * num_replicas
        if effective_batch_size % (per_replica_batch_size * num_replicas) != 0:
            return False

        # Constraint 6: num_samples_beta_vae * train_val_split is divisible by effective_batch_size
        train_samples = int(num_samples_beta_vae * train_val_split)
        if train_samples % effective_batch_size != 0:
            return False

        # Constraint 7: num_samples_beta_vae * (1 - train_val_split) is divisible by per_replica_val_batch_size * num_replicas
        val_samples = int(num_samples_beta_vae * (1 - train_val_split))
        if val_samples % (per_replica_val_batch_size * num_replicas) != 0:
            return False

        # Constraint 8: num_samples_rf is divisible by per_replica_val_batch_size * num_replicas
        if num_samples_rf % (per_replica_val_batch_size * num_replicas) != 0:
            return False

    return True


def calculate_distance(original: dict, candidate: dict) -> int:
    """Calculate L1 distance between original and candidate values."""
    distance = 0
    for key, value in original.items():
        # NOTE: why do we omit train_val_split?
        if key != "train_val_split":  # Don't include train_val_split in distance
            distance += abs(value - candidate[key])
    return distance


def generate_candidates(
    base_values: dict,
    hold_params: dict[str, bool],
    direction_constraints: dict[str, str],
    search_ranges: dict[str, tuple],
) -> list[dict]:
    """Generate all candidate parameter combinations."""
    candidates = []

    # Build ranges for each parameter
    ranges = {}
    for param in [
        "num_samples_beta_vae",
        "num_samples_rf",
        "per_replica_batch_size",
        "effective_batch_size",
        "per_replica_val_batch_size",
    ]:
        if hold_params.get(param, False):
            # Parameter is held constant
            ranges[param] = [base_values[param]]
        else:
            # Parameter can vary
            min_val, max_val, step = search_ranges[param]
            base_val = base_values[param]
            direction = direction_constraints.get(param, "both")

            if direction == "increase":
                # Only allow values >= base_value
                min_val = max(min_val, base_val)
            elif direction == "decrease":
                # Only allow values <= base_value
                max_val = min(max_val, base_val)
            # else direction == 'both', use full range

            ranges[param] = list(range(min_val, max_val + 1, step))

    # train_val_split handling
    if hold_params.get("train_val_split", False):
        ranges["train_val_split"] = [base_values["train_val_split"]]
    else:
        # For simplicity, train_val_split is typically held constant
        # If you want to vary it, you'd need to define a range
        ranges["train_val_split"] = [base_values["train_val_split"]]

    # Generate all combinations
    keys = list(ranges.keys())
    for values in product(*[ranges[k] for k in keys]):
        candidate = dict(zip(keys, values, strict=True))
        candidates.append(candidate)

    return candidates


def optimize_parameters(
    base_values: dict,
    num_replicas_list: list[int],
    hold_params: dict[str, bool],
    direction_constraints: dict[str, str],
    search_ranges: dict[str, tuple],
    max_candidates: int = 10000000,
) -> dict | None:
    """Find optimal parameters that satisfy all constraints."""

    print("Searching for optimal parameters...")
    print("Base values:")
    for k, v in base_values.items():
        print(f"  {k}: {v}")

    held = [k for k, v in hold_params.items() if v]
    if held:
        print(f"\nHolding constant: {held}")
    else:
        print("\nAll parameters allowed to vary")

    directional = [(k, v) for k, v in direction_constraints.items() if v != "both"]
    if directional:
        print("Directional constraints:")
        for k, v in directional:
            print(f"  {k}: only {v}")

    print(f"\nTarget num_replicas: {num_replicas_list}")
    print()

    # Check if current parameters already satisfy all constraints
    if check_constraints(
        base_values["num_samples_beta_vae"],
        base_values["num_samples_rf"],
        base_values["train_val_split"],
        base_values["per_replica_batch_size"],
        base_values["effective_batch_size"],
        base_values["per_replica_val_batch_size"],
        num_replicas_list,
    ):
        print("Current parameters already satisfy all constraints!")
        print("No optimization needed.\n")
        return base_values

    candidates = generate_candidates(base_values, hold_params, direction_constraints, search_ranges)

    print(f"Total candidates to check: {len(candidates):,}")
    if len(candidates) > max_candidates:
        print(f"Warning: This exceeds the limit of {max_candidates:,} candidates.")
        print("Consider reducing search ranges or increasing step sizes.")
        return None

    best_candidate = None
    best_distance = float("inf")
    valid_count = 0

    for i, candidate in enumerate(candidates):
        if (i + 1) % 100000 == 0:
            print(
                f"Checked {i + 1:,}/{len(candidates):,} candidates... (found {valid_count} valid)"
            )

        if check_constraints(
            candidate["num_samples_beta_vae"],
            candidate["num_samples_rf"],
            candidate["train_val_split"],
            candidate["per_replica_batch_size"],
            candidate["effective_batch_size"],
            candidate["per_replica_val_batch_size"],
            num_replicas_list,
        ):
            valid_count += 1
            distance = calculate_distance(base_values, candidate)
            if distance < best_distance:
                best_distance = distance
                best_candidate = candidate

    print(f"\nTotal valid candidates found: {valid_count:,}/{len(candidates):,}")

    if best_candidate:
        print(f"Best L1 distance: {best_distance}")

    return best_candidate


def main():
    parser = argparse.ArgumentParser(
        description="Optimize training parameters to satisfy divisibility constraints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use config defaults, vary all parameters
  %(prog)s --num-replicas 4,6

  # Hold batch sizes constant, only vary sample counts
  %(prog)s --hold-per-replica-batch-size --hold-effective-batch-size \\
           --hold-per-replica-val-batch-size --num-replicas 4,6

  # Only allow effective_batch_size to decrease
  %(prog)s --only-decrease-effective-batch-size --num-replicas 4,6
        """,
    )

    # Parameter values
    parser.add_argument("--num-samples-beta-vae", type=int, help="Number of beta-VAE samples")
    parser.add_argument("--num-samples-rf", type=int, help="Number of RF samples")
    parser.add_argument("--train-val-split", type=float, help="Train/val split ratio")
    parser.add_argument("--per-replica-batch-size", type=int, help="Per-replica batch size")
    parser.add_argument("--effective-batch-size", type=int, help="Effective batch size")
    parser.add_argument(
        "--per-replica-val-batch-size", type=int, help="Per-replica validation batch size"
    )

    # Which parameters to hold constant
    parser.add_argument(
        "--hold-num-samples-beta-vae",
        action="store_true",
        help="Hold num_samples_beta_vae constant",
    )
    parser.add_argument(
        "--hold-num-samples-rf", action="store_true", help="Hold num_samples_rf constant"
    )
    parser.add_argument(
        "--hold-train-val-split", action="store_true", help="Hold train_val_split constant"
    )
    parser.add_argument(
        "--hold-per-replica-batch-size",
        action="store_true",
        help="Hold per_replica_batch_size constant",
    )
    parser.add_argument(
        "--hold-effective-batch-size",
        action="store_true",
        help="Hold effective_batch_size constant",
    )
    parser.add_argument(
        "--hold-per-replica-val-batch-size",
        action="store_true",
        help="Hold per_replica_val_batch_size constant",
    )

    # Directional constraints (only increase or only decrease)
    parser.add_argument(
        "--only-increase-num-samples-beta-vae",
        action="store_true",
        help="Only allow num_samples_beta_vae to increase",
    )
    parser.add_argument(
        "--only-decrease-num-samples-beta-vae",
        action="store_true",
        help="Only allow num_samples_beta_vae to decrease",
    )
    parser.add_argument(
        "--only-increase-num-samples-rf",
        action="store_true",
        help="Only allow num_samples_rf to increase",
    )
    parser.add_argument(
        "--only-decrease-num-samples-rf",
        action="store_true",
        help="Only allow num_samples_rf to decrease",
    )
    parser.add_argument(
        "--only-increase-per-replica-batch-size",
        action="store_true",
        help="Only allow per_replica_batch_size to increase",
    )
    parser.add_argument(
        "--only-decrease-per-replica-batch-size",
        action="store_true",
        help="Only allow per_replica_batch_size to decrease",
    )
    parser.add_argument(
        "--only-increase-effective-batch-size",
        action="store_true",
        help="Only allow effective_batch_size to increase",
    )
    parser.add_argument(
        "--only-decrease-effective-batch-size",
        action="store_true",
        help="Only allow effective_batch_size to decrease",
    )
    parser.add_argument(
        "--only-increase-per-replica-val-batch-size",
        action="store_true",
        help="Only allow per_replica_val_batch_size to increase",
    )
    parser.add_argument(
        "--only-decrease-per-replica-val-batch-size",
        action="store_true",
        help="Only allow per_replica_val_batch_size to decrease",
    )

    # Search ranges (format: min,max,step)
    parser.add_argument(
        "--range-num-samples-beta-vae",
        type=str,
        default="400000,600000,10240",
        help="Search range for num_samples_beta_vae (min,max,step)",
    )
    parser.add_argument(
        "--range-num-samples-rf",
        type=str,
        default="80000,120000,2048",
        help="Search range for num_samples_rf (min,max,step)",
    )
    parser.add_argument(
        "--range-per-replica-batch-size",
        type=str,
        default="64,256,1",
        help="Search range for per_replica_batch_size (min,max,step)",
    )
    parser.add_argument(
        "--range-effective-batch-size",
        type=str,
        default="2048,6144,64",
        help="Search range for effective_batch_size (min,max,step)",
    )
    parser.add_argument(
        "--range-per-replica-val-batch-size",
        type=str,
        default="256,768,1",
        help="Search range for per_replica_val_batch_size (min,max,step)",
    )

    # Replica counts
    parser.add_argument(
        "--num-replicas",
        type=str,
        default="4,6",
        help='Comma-separated list of num_replicas values to satisfy (e.g., "4,6")',
    )

    args = parser.parse_args()

    # Determine if we should load from config
    use_config = all(
        [
            args.num_samples_beta_vae is None,
            args.num_samples_rf is None,
            args.train_val_split is None,
            args.per_replica_batch_size is None,
            args.effective_batch_size is None,
            args.per_replica_val_batch_size is None,
        ]
    )

    if use_config:
        try:
            config = init_config()

            base_values = {
                "num_samples_beta_vae": config.training.num_samples_beta_vae,
                "num_samples_rf": config.training.num_samples_rf,
                "train_val_split": config.training.train_val_split,
                "per_replica_batch_size": config.training.per_replica_batch_size,
                "effective_batch_size": config.training.effective_batch_size,
                "per_replica_val_batch_size": config.training.per_replica_val_batch_size,
            }
            print("Loaded base values from config")
        except (ImportError, AttributeError) as e:
            print(f"Error loading from config: {e}")
            print("Please provide parameters via command line arguments.")
            sys.exit(1)
    else:
        # Use command line arguments
        base_values = {
            "num_samples_beta_vae": args.num_samples_beta_vae,
            "num_samples_rf": args.num_samples_rf,
            "train_val_split": args.train_val_split,
            "per_replica_batch_size": args.per_replica_batch_size,
            "effective_batch_size": args.effective_batch_size,
            "per_replica_val_batch_size": args.per_replica_val_batch_size,
        }

        # Check that all values are provided
        if any(v is None for v in base_values.values()):
            print("Error: When providing any parameter, all parameters must be specified.")
            parser.print_help()
            sys.exit(1)

    # Parse num_replicas list
    num_replicas_list = [int(x.strip()) for x in args.num_replicas.split(",")]

    # Parse hold flags
    hold_params = {
        "num_samples_beta_vae": args.hold_num_samples_beta_vae,
        "num_samples_rf": args.hold_num_samples_rf,
        "train_val_split": args.hold_train_val_split,
        "per_replica_batch_size": args.hold_per_replica_batch_size,
        "effective_batch_size": args.hold_effective_batch_size,
        "per_replica_val_batch_size": args.hold_per_replica_val_batch_size,
    }

    # Parse directional constraints
    direction_constraints = {}
    param_list = [
        "num_samples_beta_vae",
        "num_samples_rf",
        "per_replica_batch_size",
        "effective_batch_size",
        "per_replica_val_batch_size",
    ]

    for param in param_list:
        increase_flag = getattr(args, f"only_increase_{param}")
        decrease_flag = getattr(args, f"only_decrease_{param}")

        if increase_flag and decrease_flag:
            print(
                f"Error: Cannot specify both --only-increase-{param.replace('_', '-')} "
                f"and --only-decrease-{param.replace('_', '-')}"
            )
            sys.exit(1)

        if increase_flag:
            direction_constraints[param] = "increase"
        elif decrease_flag:
            direction_constraints[param] = "decrease"
        else:
            direction_constraints[param] = "both"

    # Parse search ranges
    def parse_range(s):
        parts = [int(x.strip()) for x in s.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Range must be in format 'min,max,step', got: {s}")
        return tuple(parts)

    search_ranges = {
        "num_samples_beta_vae": parse_range(args.range_num_samples_beta_vae),
        "num_samples_rf": parse_range(args.range_num_samples_rf),
        "per_replica_batch_size": parse_range(args.range_per_replica_batch_size),
        "effective_batch_size": parse_range(args.range_effective_batch_size),
        "per_replica_val_batch_size": parse_range(args.range_per_replica_val_batch_size),
    }

    # Run optimization
    result = optimize_parameters(
        base_values, num_replicas_list, hold_params, direction_constraints, search_ranges
    )

    if result:
        print("\n" + "=" * 80)
        print("OPTIMAL SOLUTION FOUND")
        print("=" * 80)
        print(f"{'Parameter':<35} {'Original':<15} {'Optimized':<15} {'Delta'}")
        print("-" * 80)
        for key in [
            "num_samples_beta_vae",
            "num_samples_rf",
            "train_val_split",
            "per_replica_batch_size",
            "effective_batch_size",
            "per_replica_val_batch_size",
        ]:
            orig = base_values[key]
            opt = result[key]
            if key == "train_val_split":
                delta = f"{opt - orig:.6f}"
                orig_str = f"{orig:.2f}"
                opt_str = f"{opt:.2f}"
            else:
                delta = f"{opt - orig:+d}"
                orig_str = str(orig)
                opt_str = str(opt)
            print(f"{key:<35} {orig_str:<15} {opt_str:<15} {delta}")
        print("=" * 80)

        # Verify constraints
        print("\nVerifying constraints for all num_replicas values:")
        all_valid = True
        for nr in num_replicas_list:
            valid = check_constraints(
                result["num_samples_beta_vae"],
                result["num_samples_rf"],
                result["train_val_split"],
                result["per_replica_batch_size"],
                result["effective_batch_size"],
                result["per_replica_val_batch_size"],
                [nr],
            )
            status = "✓ VALID" if valid else "✗ INVALID"
            print(f"  num_replicas={nr}: {status}")
            all_valid = all_valid and valid

        if all_valid:
            print("\n✓ All constraints satisfied!")
        else:
            print("\n✗ Some constraints not satisfied (this should not happen)")

    else:
        print("\n" + "=" * 80)
        print("NO VALID SOLUTION FOUND")
        print("=" * 80)
        print("Suggestions:")
        print("  1. Increase search ranges")
        print("  2. Decrease step sizes (but beware of computational cost)")
        print("  3. Allow more parameters to vary")
        print("  4. Try different num_replicas values")
        print("  5. Remove or relax directional constraints")


if __name__ == "__main__":
    main()
