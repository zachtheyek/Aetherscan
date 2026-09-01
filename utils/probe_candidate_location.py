#!/usr/bin/env python3
"""Trace specified sky-frequency locations through Aetherscan's inference cascade.

Heavy scientific imports happen only after argument parsing, so ``--help`` works in a
stdlib-only environment. The probe is read-only except for explicitly requested plots and CSV.

Example (container path, explicit artifacts — the cluster-standard trio):

    ./utils/run_container.sh python utils/probe_candidate_location.py \\
        --catalog /path/to/catalog.csv --target NGC1172 --band C \\
        --frequency-mhz 4026.4159 7499.0 \\
        --encoder-path /path/to/vae_encoder_<tag>.keras \\
        --rf-path /path/to/random_forest_<tag>.joblib \\
        --config-path /path/to/config_<tag>.json \\
        --csv probe_results.csv --plot-dir probe_plots
"""

from __future__ import annotations

import argparse
import csv
import functools
import math
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FrequencyAxis:
    """Frequency-axis and data geometry read from one observation's ``data`` dataset."""

    fch1: float
    foff: float
    nchans: int
    shape: tuple[int, ...]


@dataclass
class ProbeResult:
    """All preprocessing and scoring results for one requested frequency."""

    requested_frequency_mhz: float
    resolved_frequency_mhz: float
    frequency_offset_hz: float
    absolute_bin: int
    coarse_channel: int
    bin_in_coarse: int
    stamp_start_bin: int
    stamp_end_bin: int
    stamp_clamped: bool
    on_max_k2: list[float]
    ed_max_k2: float
    ed_would_propose: bool
    normalized_stamp: Any
    status: str = "ok"
    error: str = ""
    raw_rf_probability: float = float("nan")
    screening_probability: float = float("nan")
    screen_pass: bool = False
    mc_mean: float = float("nan")
    mc_std: float = float("nan")
    mc_pass: bool = False
    plot_path: str = ""


@dataclass
class ScoringAssets:
    """Loaded encoder, forest wrapper, and optional probability calibrator."""

    encoder_runner: Any
    rf_model: Any
    calibrator: dict | None


@dataclass
class ProbeContext:
    """Read-only cadence/config state shared across requested frequencies."""

    h5_paths: list[str]
    axis: FrequencyAxis
    config: Any
    bandpass_flatten: Callable[[Any], Any]
    bandpass_method: str
    energy_cache: dict[tuple[str, int], Any] = field(default_factory=dict)


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone probe CLI without importing the scientific stack."""
    parser = argparse.ArgumentParser(
        description=(
            "Probe one or more frequencies through Aetherscan energy detection, stamp "
            "preprocessing, deterministic screening, and seeded Monte Carlo scoring."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--h5-files",
        nargs=6,
        metavar=("OBS1", "OBS2", "OBS3", "OBS4", "OBS5", "OBS6"),
        help="Six observation .h5 paths in cadence row order (ABACAD)",
    )
    source.add_argument(
        "--catalog",
        help="Catalog CSV to group using the saved inference cadence-grouping config",
    )
    parser.add_argument("--target", help="Exact Target value for --catalog resolution")
    parser.add_argument("--band", help="Exact Band value for --catalog resolution")
    parser.add_argument("--session", help="Optional exact Session value for --catalog resolution")
    parser.add_argument(
        "--cadence-id",
        help="Optional exact Cadence ID value for --catalog resolution",
    )
    parser.add_argument(
        "--frequency-mhz",
        type=float,
        nargs="+",
        action="extend",
        required=True,
        metavar="MHZ",
        help="One or more target frequencies in MHz; the option may be repeated",
    )
    parser.add_argument("--encoder-path", required=True, help="Saved VAE encoder (.keras)")
    parser.add_argument("--rf-path", required=True, help="Saved Random Forest (.joblib)")
    parser.add_argument(
        "--config-path",
        required=True,
        help="Saved config JSON from the same training run",
    )
    parser.add_argument(
        "--plot-dir",
        help="Optional directory for one six-panel normalized waterfall PNG per frequency",
    )
    parser.add_argument("--csv", help="Optional path for the probe results CSV")
    parser.add_argument(
        "--mc-draws",
        type=int,
        help="Override the saved inference Monte Carlo draw count",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help=(
            "Root seed for the MC draws. apply_saved_config deliberately does NOT layer the "
            "saved run's reproducibility section (#303 allowlist), so without this flag the "
            "probe uses the process-default seed — pass the run's --seed to mirror it"
        ),
    )
    parser.add_argument(
        "--cadence-seed-key",
        type=int,
        default=0,
        help=(
            "Cadence-level MC SeedSequence sub-key. Each location is further sub-keyed by its "
            "absolute frequency bin (the non-zero trailing key also means the default does NOT "
            "alias catalog cadence 0's stream), so a location's draws are reproducible "
            "regardless of batch composition"
        ),
    )
    parser.add_argument(
        "--traceback",
        action="store_true",
        help="Print full tracebacks in addition to the one-line ERROR summaries",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.catalog and (args.target is None or args.band is None):
        parser.error("--catalog requires both --target and --band")
    if args.h5_files and any(
        value is not None for value in (args.target, args.band, args.session, args.cadence_id)
    ):
        parser.error("--target/--band/--session/--cadence-id are only valid with --catalog")
    if args.mc_draws is not None and args.mc_draws < 1:
        parser.error(f"--mc-draws must be a positive integer, got {args.mc_draws}")
    if args.seed is not None and args.seed < 0:
        parser.error(f"--seed must be >= 0, got {args.seed}")
    if args.cadence_seed_key < 0:
        parser.error(f"--cadence-seed-key must be >= 0, got {args.cadence_seed_key}")
    non_finite = [value for value in args.frequency_mhz if not math.isfinite(value)]
    if non_finite:
        parser.error(f"--frequency-mhz values must be finite, got {non_finite}")


def _make_src_importable() -> None:
    """Make the checkout's ``src`` tree importable for direct script invocation."""
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "src"))


def _load_config(args: argparse.Namespace) -> Any:
    # Local imports are deliberate: argparse --help must not require TensorFlow or scipy.
    from aetherscan.cli import apply_saved_config  # noqa: PLC0415
    from aetherscan.config import init_config  # noqa: PLC0415

    config = init_config()
    # NOTE: reuses cli.py:1147-1214, including the #303 result-field allowlist layering.
    apply_saved_config(args.config_path)
    if args.mc_draws is not None:
        config.inference.mc_draws = args.mc_draws
    if args.seed is not None:
        config.reproducibility.seed = args.seed

    if config.data.num_observations != 6 or config.data.time_bins != 16:
        raise ValueError(
            "This probe requires the production six-observation, 16-time-bin model geometry; "
            f"saved config declares ({config.data.num_observations}, {config.data.time_bins})"
        )
    if config.inference.stamp_width != config.data.width_bin:
        raise ValueError(
            f"Saved config mismatch: inference.stamp_width={config.inference.stamp_width} "
            f"but data.width_bin={config.data.width_bin}"
        )
    if config.inference.stamp_width % config.data.downsample_factor:
        raise ValueError(
            f"stamp_width={config.inference.stamp_width} is not divisible by "
            f"downsample_factor={config.data.downsample_factor}"
        )
    final_width = config.data.width_bin // config.data.downsample_factor
    if final_width != 512:
        raise ValueError(
            f"Saved config produces {final_width} frequency bins, but the deployed encoder "
            "contract requires 512"
        )
    if config.inference.screening_threshold > config.inference.classification_threshold:
        raise ValueError(
            f"screening_threshold={config.inference.screening_threshold} exceeds "
            f"classification_threshold={config.inference.classification_threshold}"
        )
    return config


def _catalog_key_map(group: Any, group_by_cols: list[str]) -> dict[str, str]:
    return {
        str(column).strip().lower(): str(value)
        for column, value in zip(group_by_cols, group.key, strict=True)
    }


def _format_catalog_match(group: Any) -> str:
    paths = ", ".join(group.h5_paths)
    return f"key={group.key!r}, observations={len(group.h5_paths)}, h5_paths=[{paths}]"


def _resolve_catalog(args: argparse.Namespace, config: Any) -> list[str]:
    from aetherscan.preprocessing import group_observations_from_csv  # noqa: PLC0415

    group_by_cols = list(config.inference.cadence_group_by_cols)
    # NOTE: reuses preprocessing.py:967-1044, preserving ordered grouping and flagged groups.
    valid, flagged = group_observations_from_csv(
        csv_path=args.catalog,
        group_by_cols=group_by_cols,
        h5_path_col=config.inference.cadence_h5_path_col,
        expected_obs=config.inference.cadence_expected_obs,
    )

    wanted = {
        "target": args.target,
        "band": args.band,
        "session": args.session,
        "cadence id": args.cadence_id,
    }
    matches = []
    for group in [*valid, *flagged]:
        key_map = _catalog_key_map(group, group_by_cols)
        if all(value is None or key_map.get(name) == value for name, value in wanted.items()):
            matches.append(group)

    if not matches:
        filters = ", ".join(f"{name}={value!r}" for name, value in wanted.items() if value)
        raise ValueError(f"Catalog filter matched no cadence group: {filters}")
    if len(matches) > 1:
        listing = "\n".join(
            f"  {index}. {_format_catalog_match(group)}" for index, group in enumerate(matches, 1)
        )
        raise ValueError(
            f"Catalog filter is ambiguous; matched {len(matches)} cadence groups:\n{listing}"
        )
    match = matches[0]
    if not match.is_valid:
        raise ValueError(
            "Catalog filter selected a flagged cadence instead of a six-observation group: "
            f"{_format_catalog_match(match)}"
        )
    return list(match.h5_paths)


def _read_frequency_axis(h5_path: str, time_bins: int) -> FrequencyAxis:
    import h5py  # noqa: PLC0415

    with h5py.File(h5_path, "r") as handle:
        if "data" not in handle:
            raise ValueError(f"{h5_path} has no 'data' dataset")
        data = handle["data"]
        shape = tuple(int(value) for value in data.shape)
        if len(shape) != 3:
            raise ValueError(
                f"{h5_path} has rank-{len(shape)} 'data' shape {shape}; expected "
                "(time, polarization, frequency)"
            )
        if shape[0] < time_bins or shape[1] < 1:
            raise ValueError(
                f"{h5_path} shape {shape} cannot supply {time_bins} time bins and polarization 0"
            )
        missing = [name for name in ("fch1", "foff") if name not in data.attrs]
        if missing:
            raise ValueError(f"{h5_path} data header is missing {missing}")
        nchans = int(data.attrs.get("nchans", shape[-1]))
        if nchans < 1 or shape[-1] < nchans:
            raise ValueError(
                f"{h5_path} header nchans={nchans} is incompatible with data width {shape[-1]}"
            )
        foff = float(data.attrs["foff"])
        if not math.isfinite(foff) or foff == 0.0:
            raise ValueError(f"{h5_path} has invalid foff={foff!r}")
        fch1 = float(data.attrs["fch1"])
        if not math.isfinite(fch1):
            raise ValueError(f"{h5_path} has invalid fch1={fch1!r}")
    return FrequencyAxis(fch1=fch1, foff=foff, nchans=nchans, shape=shape)


def _validate_cadence_axes(h5_paths: list[str], time_bins: int) -> FrequencyAxis:
    missing = [path for path in h5_paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Observation file(s) not found: {missing}")
    axes = [_read_frequency_axis(path, time_bins) for path in h5_paths]
    reference = axes[0]
    mismatches = [
        f"{path}: fch1={axis.fch1}, foff={axis.foff}, nchans={axis.nchans}"
        for path, axis in zip(h5_paths[1:], axes[1:], strict=True)
        if (axis.fch1, axis.foff, axis.nchans) != (reference.fch1, reference.foff, reference.nchans)
    ]
    if mismatches:
        details = "\n".join(f"  {line}" for line in mismatches)
        raise ValueError(
            "The six observations do not share one frequency axis. Production stamp indexing "
            f"uses the first observation's axis, so this probe refuses a misaligned cadence:\n{details}"
        )
    return reference


def _select_bandpass_flattener(config: Any, n_coarse_channels: int) -> tuple[Callable, str]:
    import numpy as np  # noqa: PLC0415

    from aetherscan.pfb import gen_coarse_channel_response  # noqa: PLC0415
    from aetherscan.preprocessing import (  # noqa: PLC0415
        _pfb_flatten_bandpass,
        _spline_flatten_bandpass,
    )

    method = config.inference.bandpass_method
    # NOTE: mirrors preprocessing.py:2488-2522. Unlike _ensure_pfb_response_file at
    # preprocessing.py:2524-2570, this read-only probe never creates or repairs the cache.
    if method == "pfb" and n_coarse_channels >= 2:
        width = config.inference.coarse_channel_width
        taps = config.inference.pfb_taps_per_channel
        response_path = os.path.join(
            config.data_path,
            "cache",
            "pfb",
            f"pfb_response_w{width}_c{n_coarse_channels}_t{taps}.npy",
        )
        try:
            cached = np.load(response_path)
            expected = gen_coarse_channel_response(width, n_coarse_channels, taps)
            if not np.array_equal(cached, expected):
                raise ValueError("cached response differs from the configured response")
            return (
                functools.partial(_pfb_flatten_bandpass, response_path=response_path),
                "pfb",
            )
        except Exception as exc:
            print(
                f"WARNING: PFB response is not usable at {response_path} ({exc}); "
                "falling back to spline without writing the cache",
                file=sys.stderr,
            )
            return (
                functools.partial(
                    _spline_flatten_bandpass,
                    spl_order=config.inference.spline_order,
                ),
                "spline (PFB fallback)",
            )
    if method == "pfb":
        print(
            "WARNING: PFB flattening needs at least two complete coarse channels; "
            "falling back to spline",
            file=sys.stderr,
        )
    elif method != "spline":
        raise ValueError(f"Unknown bandpass_method {method!r}; expected 'pfb' or 'spline'")
    return (
        functools.partial(
            _spline_flatten_bandpass,
            spl_order=config.inference.spline_order,
        ),
        "spline",
    )


def _frequency_bin(axis: FrequencyAxis, frequency_mhz: float) -> tuple[int, float, float]:
    fractional_bin = (frequency_mhz - axis.fch1) / axis.foff
    absolute_bin = int(math.floor(fractional_bin + 0.5))
    if not 0 <= absolute_bin < axis.nchans:
        low = min(axis.fch1, axis.fch1 + axis.foff * (axis.nchans - 1))
        high = max(axis.fch1, axis.fch1 + axis.foff * (axis.nchans - 1))
        raise ValueError(
            f"Frequency {frequency_mhz:.9f} MHz lies outside the h5 bin-center range "
            f"[{low:.9f}, {high:.9f}] MHz"
        )
    resolved = axis.fch1 + axis.foff * absolute_bin
    return absolute_bin, resolved, (resolved - frequency_mhz) * 1e6


def _stamp_bounds(center_bin: int, stamp_width: int, nchans: int) -> tuple[int, int, bool]:
    if nchans < stamp_width:
        raise ValueError(f"nchans={nchans} is smaller than configured stamp_width={stamp_width}")
    requested_start = center_bin - stamp_width // 2
    start = min(max(requested_start, 0), nchans - stamp_width)
    return start, start + stamp_width, start != requested_start


def _energy_k2(context: ProbeContext, h5_path: str, coarse_channel: int) -> Any:
    import h5py  # noqa: PLC0415

    from aetherscan.preprocessing import (  # noqa: PLC0415
        _remove_dc_spike,
        _sliding_normality_k2,
    )

    cache_key = (h5_path, coarse_channel)
    if cache_key in context.energy_cache:
        return context.energy_cache[cache_key]

    config = context.config
    width = config.inference.coarse_channel_width
    start = coarse_channel * width
    end = start + width
    with h5py.File(h5_path, "r") as handle:
        channel = handle["data"][: config.data.time_bins, 0, start:end]
    if channel.shape != (config.data.time_bins, width):
        raise ValueError(
            f"{h5_path} coarse channel {coarse_channel} yielded shape {channel.shape}; "
            f"expected ({config.data.time_bins}, {width})"
        )

    # NOTE: mirrors preprocessing.py:732-754: read, in-place DC removal, configured
    # bandpass flattening, then the vectorized sliding normality statistic.
    _remove_dc_spike(channel, width, 1)
    flattened = context.bandpass_flatten(channel)
    k2 = _sliding_normality_k2(
        flattened,
        config.inference.detection_window_size,
        config.inference.detection_step_size,
    )
    context.energy_cache[cache_key] = k2
    return k2


def _max_k2_in_stamp(
    k2: Any,
    coarse_channel: int,
    coarse_width: int,
    step_size: int,
    stamp_start: int,
    stamp_end: int,
) -> float:
    import numpy as np  # noqa: PLC0415

    window_starts = coarse_channel * coarse_width + np.arange(len(k2)) * step_size
    in_stamp = (window_starts >= stamp_start) & (window_starts < stamp_end) & np.isfinite(k2)
    if not np.any(in_stamp):
        return float("nan")
    return float(np.max(k2[in_stamp]))


def _extract_raw_stamp(context: ProbeContext, start: int, end: int) -> Any:
    import h5py  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    config = context.config
    raw = np.empty(
        (config.data.num_observations, config.data.time_bins, end - start),
        dtype=np.float32,
    )
    # NOTE: mirrors preprocessing.py:842-855 for one stamp: first time rows, polarization 0,
    # all six observations, and a float32 destination. This probe keeps the full-width row
    # in memory instead of publishing a cache memmap.
    for observation, h5_path in enumerate(context.h5_paths):
        with h5py.File(h5_path, "r") as handle:
            stamp = handle["data"][: config.data.time_bins, 0, start:end]
        if stamp.shape != (config.data.time_bins, end - start):
            raise ValueError(f"{h5_path} yielded a short stamp with shape {stamp.shape}")
        raw[observation] = stamp
    return raw


def _normalize_stamp(raw_stamp: Any, config: Any) -> Any:
    import numpy as np  # noqa: PLC0415

    from aetherscan.preprocessing import (  # noqa: PLC0415
        _downsample_cadence,
        _log_norm_chunk_vectorized,
    )

    final_width = config.data.width_bin // config.data.downsample_factor
    # NOTE: reuses preprocessing.py:224-250 and :305-336, matching the legacy full-width
    # inference loader's downsample-then-log-normalize semantics at :1401-1410/:1537-1549.
    downsampled = _downsample_cadence(raw_stamp, config.data.downsample_factor, final_width)
    if downsampled is None:
        raise ValueError(
            "Stamp failed the production validity filter (NaN/Inf, non-positive max, or negative)"
        )
    normalized, valid = _log_norm_chunk_vectorized(downsampled[np.newaxis, ...])
    if len(normalized) != 1 or not bool(valid[0]):
        raise ValueError("Stamp failed production log-normalization validation")
    return normalized[0]


def _prepare_location(context: ProbeContext, frequency_mhz: float) -> ProbeResult:
    import numpy as np  # noqa: PLC0415

    config = context.config
    absolute_bin, resolved_frequency, offset_hz = _frequency_bin(context.axis, frequency_mhz)
    coarse_width = config.inference.coarse_channel_width
    coarse_channel, bin_in_coarse = divmod(absolute_bin, coarse_width)
    complete_coarse_channels = context.axis.nchans // coarse_width
    if coarse_channel >= complete_coarse_channels:
        raise ValueError(
            f"Frequency {frequency_mhz:.9f} MHz falls in incomplete trailing coarse channel "
            f"{coarse_channel}; production processes only {complete_coarse_channels} complete channels"
        )

    stamp_start, stamp_end, stamp_clamped = _stamp_bounds(
        absolute_bin,
        config.inference.stamp_width,
        context.axis.nchans,
    )
    # Scan every complete coarse channel the stamp window overlaps, not just the target
    # frequency's own channel — a stamp within stamp_width // 2 of a channel boundary can
    # carry ED-visible power from the neighbor. Dedup/overlap placement is still not
    # replayed (documented simplification: this verdict is an upper bound on proposal).
    overlapped_channels = [
        channel
        for channel in range(stamp_start // coarse_width, (stamp_end - 1) // coarse_width + 1)
        if channel < complete_coarse_channels
    ]
    on_max_k2 = []
    for observation in (0, 2, 4):
        channel_maxima = []
        for channel in overlapped_channels:
            k2 = _energy_k2(context, context.h5_paths[observation], channel)
            channel_maxima.append(
                _max_k2_in_stamp(
                    k2,
                    channel,
                    coarse_width,
                    config.inference.detection_step_size,
                    stamp_start,
                    stamp_end,
                )
            )
        finite = [value for value in channel_maxima if math.isfinite(value)]
        on_max_k2.append(max(finite, default=float("nan")))
    finite_maxima = [value for value in on_max_k2 if math.isfinite(value)]
    ed_max_k2 = max(finite_maxima, default=float("nan"))
    ed_would_propose = bool(
        math.isfinite(ed_max_k2) and ed_max_k2 > config.inference.stat_threshold
    )

    raw_stamp = _extract_raw_stamp(context, stamp_start, stamp_end)
    normalized_stamp = _normalize_stamp(raw_stamp, config)
    if not np.isfinite(normalized_stamp).all():
        raise ValueError("Normalized stamp contains non-finite values")
    return ProbeResult(
        requested_frequency_mhz=frequency_mhz,
        resolved_frequency_mhz=resolved_frequency,
        frequency_offset_hz=offset_hz,
        absolute_bin=absolute_bin,
        coarse_channel=coarse_channel,
        bin_in_coarse=bin_in_coarse,
        stamp_start_bin=stamp_start,
        stamp_end_bin=stamp_end,
        stamp_clamped=stamp_clamped,
        on_max_k2=on_max_k2,
        ed_max_k2=ed_max_k2,
        ed_would_propose=ed_would_propose,
        normalized_stamp=normalized_stamp,
    )


def _validate_rf_layout(rf_model: Any, config: Any) -> None:
    from aetherscan.latent_variants import variant_feature_count  # noqa: PLC0415

    declared_variant = config.rf.latent_variant
    expected_features = variant_feature_count(
        declared_variant,
        config.data.num_observations,
        config.beta_vae.latent_dim,
        config.rf.active_dims,
    )
    forest = rf_model.model
    actual_features = getattr(forest, "n_features_in_", None)
    if actual_features is not None and actual_features != expected_features:
        raise ValueError(
            f"RF feature-count mismatch: forest expects {actual_features}, but config variant "
            f"{declared_variant!r} with active_dims={config.rf.active_dims} builds "
            f"{expected_features}"
        )

    # NOTE: mirrors inference.py:413-458, including the persisted variant/active-dim stamps
    # that catch same-width config/forest mismatches which sklearn cannot detect.
    recorded_variant = getattr(forest, "aetherscan_latent_variant_", None)
    if recorded_variant is not None and recorded_variant != declared_variant:
        raise ValueError(
            f"RF latent-variant mismatch: forest stamp is {recorded_variant!r}, but saved "
            f"config declares {declared_variant!r}"
        )
    recorded_active_dims = getattr(forest, "aetherscan_active_dims_", None)
    if (
        recorded_active_dims is not None
        and declared_variant == "z_mean_logvar_active"
        and sorted(recorded_active_dims) != sorted(config.rf.active_dims or [])
    ):
        raise ValueError(
            f"RF active-dims mismatch: forest stamp is {list(recorded_active_dims)}, but "
            f"saved config declares {list(config.rf.active_dims or [])}"
        )


def _load_scoring_assets(args: argparse.Namespace, config: Any) -> ScoringAssets:
    import joblib  # noqa: PLC0415
    import tensorflow as tf  # noqa: PLC0415

    # Importing the models package registers the serialized Sampling layer before keras load.
    from aetherscan import models as aetherscan_models  # noqa: PLC0415
    from aetherscan.inference import (  # noqa: PLC0415
        InferencePipeline,
        _derive_encode_model,
    )

    missing = [
        path
        for path in (args.encoder_path, args.rf_path, args.config_path)
        if not os.path.isfile(path)
    ]
    if missing:
        raise FileNotFoundError(f"Artifact file(s) not found: {missing}")

    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        encoder = tf.keras.models.load_model(args.encoder_path)
        encode_model = _derive_encode_model(encoder)

    # Build only the attributes used by _distributed_encode. Calling InferencePipeline's
    # constructor would initialize the DB and reference-cloud state, which this read-only
    # utility deliberately excludes. The unmodified production encoder method still owns
    # bucketing, replica distribution, padding, and output ordering.
    encoder_runner = InferencePipeline.__new__(InferencePipeline)
    encoder_runner.config = config
    encoder_runner.strategy = strategy
    encoder_runner.num_replicas = strategy.num_replicas_in_sync
    encoder_runner.encoder = encoder
    encoder_runner._encode_model = encode_model
    encoder_runner.latent_dim = config.beta_vae.latent_dim
    encoder_runner.num_observations = config.data.num_observations
    encoder_runner.per_replica_inf_batch_size = config.inference.per_replica_batch_size
    encoder_runner._encode_step = None

    rf_model = aetherscan_models.RandomForestModel()
    rf_model.load(args.rf_path)
    _validate_rf_layout(rf_model, config)

    calibrator = None
    if config.rf.calibration_active:
        calibrator_path = os.path.join(
            os.path.dirname(args.rf_path),
            os.path.basename(args.rf_path).replace("random_forest", "rf_calibrator"),
        )
        if not os.path.isfile(calibrator_path):
            raise FileNotFoundError(
                f"Saved config requires {config.rf.calibration_method!r} calibration, but "
                f"{calibrator_path} does not exist"
            )
        calibrator = joblib.load(calibrator_path)
    return ScoringAssets(
        encoder_runner=encoder_runner,
        rf_model=rf_model,
        calibrator=calibrator,
    )


def _score_locations(
    results: list[ProbeResult], assets: ScoringAssets, config: Any, cadence_seed_key: int
) -> None:
    import numpy as np  # noqa: PLC0415

    from aetherscan.inference import _batched_mc_scores  # noqa: PLC0415
    from aetherscan.latent_variants import (  # noqa: PLC0415
        apply_probability_calibrator,
        build_variant_features,
    )
    from aetherscan.models import prepare_latent_features  # noqa: PLC0415
    from aetherscan.seeding import (  # noqa: PLC0415
        STREAM_INFERENCE_MC,
        derive_rng,
    )

    cadence_batch = np.stack([result.normalized_stamp for result in results])
    # NOTE: reuses inference.py:709-839 wholesale: the same bucketed batch geometry,
    # cadence-start padding, strategy dispatch, and deterministic z_mean/z_log_var outputs.
    z_mean, z_log_var = assets.encoder_runner._distributed_encode(cadence_batch)
    expected_shape = (
        len(results) * config.data.num_observations,
        config.beta_vae.latent_dim,
    )
    if z_mean.shape != expected_shape or z_log_var.shape != expected_shape:
        raise ValueError(
            f"Encoder returned z_mean={z_mean.shape}, z_log_var={z_log_var.shape}; "
            f"expected {expected_shape}"
        )

    num_observations = config.data.num_observations
    latent_dim = config.beta_vae.latent_dim
    variant = config.rf.latent_variant
    active_dims = config.rf.active_dims
    # NOTE: reuses inference.py:545-563 exactly: float32 obs-major flattening, saved latent
    # variant features, raw forest probability, then the persisted calibrator when active.
    mean_flat = prepare_latent_features(z_mean, num_observations, dtype=np.float32)
    logvar_flat = prepare_latent_features(z_log_var, num_observations, dtype=np.float32)
    deterministic_features = build_variant_features(
        variant,
        mean_flat,
        logvar_flat,
        num_observations,
        latent_dim,
        active_dims,
    )
    raw_probabilities = assets.rf_model.model.predict_proba(deterministic_features)[:, 1]
    screening_probabilities = apply_probability_calibrator(
        assets.calibrator,
        raw_probabilities,
    ).astype(np.float32)

    survivors = screening_probabilities > config.inference.screening_threshold
    mc_means = np.full(len(results), np.nan, dtype=np.float32)
    mc_stds = np.full(len(results), np.nan, dtype=np.float32)

    # NOTE: deliberate delta from production MC seeding, twice over. (1) There is no
    # catalog cadence index for an ad-hoc probe, so the cadence-level sub-key comes from
    # --cadence-seed-key. (2) Production draws one noise block per cadence submatrix, so a
    # snippet's epsilons depend on its position in the batch; a diagnostic tool must not
    # change its answer for a location because a second location was probed alongside it,
    # so each location is scored ALONE with an extra sub-key of its absolute frequency bin
    # — reproducible under any batch composition. Screen rejects get the same forced
    # diagnostic pass; production stops before MC for them.
    for index, result in enumerate(results):
        mc_rng = derive_rng(
            config.reproducibility.seed,
            STREAM_INFERENCE_MC,
            cadence_seed_key,
            result.absolute_bin,
        )
        draw_scores = _batched_mc_scores(
            assets.rf_model,
            assets.calibrator,
            variant,
            mean_flat[index : index + 1],
            logvar_flat[index : index + 1],
            num_observations,
            latent_dim,
            active_dims,
            config.inference.mc_draws,
            mc_rng,
        )
        mc_means[index] = draw_scores.mean(axis=0)[0]
        mc_stds[index] = draw_scores.std(axis=0)[0]

    for index, result in enumerate(results):
        result.raw_rf_probability = float(raw_probabilities[index])
        result.screening_probability = float(screening_probabilities[index])
        result.screen_pass = bool(survivors[index])
        result.mc_mean = float(mc_means[index])
        result.mc_std = float(mc_stds[index])
        result.mc_pass = bool(mc_means[index] > config.inference.classification_threshold)


def _write_plot(
    result: ProbeResult,
    index: int,
    plot_dir: str,
    axis: FrequencyAxis,
) -> str:
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    destination = Path(plot_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / (f"probe_{index:02d}_{result.requested_frequency_mhz:.6f}_MHz.png")

    # Six small multiples use one scale and one sequential map. The target is the only accent.
    fig, panels = plt.subplots(6, 1, figsize=(10, 10), sharex=True, sharey=True)
    factor = (result.stamp_end_bin - result.stamp_start_bin) // result.normalized_stamp.shape[-1]
    # imshow's extent wants pixel EDGES; downsampled pixel i spans raw bins
    # [start + i*factor, start + (i+1)*factor), so the edges sit half a raw bin outside
    # the first/last bin centers.
    edge_bins = result.stamp_start_bin + np.array(
        [-0.5, result.normalized_stamp.shape[-1] * factor - 0.5]
    )
    frequencies = axis.fch1 + axis.foff * edge_bins
    image = None
    for observation, panel in enumerate(panels):
        image = panel.imshow(
            result.normalized_stamp[observation],
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            extent=(frequencies[0], frequencies[-1], 0, result.normalized_stamp.shape[1] - 1),
        )
        panel.axvline(result.requested_frequency_mhz, color="#35b5d8", linewidth=0.9)
        source = "ON" if observation % 2 == 0 else "OFF"
        pair = observation // 2 + 1
        panel.text(
            0.006,
            0.84,
            f"{source} {pair}",
            transform=panel.transAxes,
            color="white",
            fontsize=9,
            ha="left",
            va="top",
        )
        panel.set_ylabel("time bin", fontsize=8)
        panel.tick_params(length=0, labelsize=8)
        for spine in panel.spines.values():
            spine.set_visible(False)
    panels[-1].set_xlabel("frequency (MHz)", fontsize=9)
    panels[0].set_title(
        f"{result.requested_frequency_mhz:.9f} MHz | h5 bin {result.absolute_bin} | "
        "cyan: requested frequency",
        loc="left",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.09, right=0.88, top=0.95, bottom=0.07, hspace=0.08)
    colorbar_axis = fig.add_axes((0.9, 0.12, 0.012, 0.76))
    fig.colorbar(image, cax=colorbar_axis, label="normalized log power")
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    result.plot_path = str(output)
    return str(output)


def _format_number(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.6g}"


def _print_table(results: list[ProbeResult], config: Any) -> None:
    rows = []
    for result in results:
        if result.status != "ok":
            rows.append(
                [
                    f"{result.requested_frequency_mhz:.9f}",
                    "ERROR",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                ]
            )
            continue
        frequency_cell = f"{result.requested_frequency_mhz:.9f}"
        if result.stamp_clamped:
            frequency_cell += " (clamped)"
        rows.append(
            [
                frequency_cell,
                str(result.absolute_bin),
                f"{result.coarse_channel}:{result.bin_in_coarse}",
                _format_number(result.ed_max_k2),
                "yes" if result.ed_would_propose else "no",
                f"{result.raw_rf_probability:.6f}",
                f"{result.screening_probability:.6f}",
                "PASS" if result.screen_pass else "FAIL",
                f"{result.mc_mean:.6f}",
                f"{result.mc_std:.6f}",
                "PASS" if result.mc_pass else "FAIL",
            ]
        )
    headers = [
        "Frequency MHz",
        "Bin",
        "Coarse:fine",
        "max k^2",
        "ED proposes",
        "RF raw",
        "Screen P",
        "Screen gate",
        "MC mean",
        "MC std",
        "MC gate",
    ]
    widths = [
        max(len(header), *(len(row[column]) for row in rows))
        for column, header in enumerate(headers)
    ]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))

    print()
    for result in results:
        if result.status != "ok":
            print(
                f"VERDICT {result.requested_frequency_mhz:.9f} MHz: "
                f"preprocessing failed — {result.error}"
            )
            continue
        if not result.ed_would_propose:
            prefix = "ED would not propose this location; forced scoring says: "
        else:
            prefix = "ED would propose this location; scoring says: "
        if result.stamp_clamped:
            prefix = "[stamp clamped off-center at a band edge] " + prefix
        if not result.screen_pass:
            cascade = (
                f"FAIL screen P>{config.inference.screening_threshold:g}; production stops "
                "before MC"
            )
        elif result.mc_pass:
            cascade = (
                f"PASS screen P>{config.inference.screening_threshold:g}, PASS MC mean>"
                f"{config.inference.classification_threshold:g}: candidate"
            )
        else:
            cascade = (
                f"PASS screen P>{config.inference.screening_threshold:g}, FAIL MC mean>"
                f"{config.inference.classification_threshold:g}: not a candidate"
            )
        print(f"VERDICT {result.requested_frequency_mhz:.9f} MHz: {prefix}{cascade}.")


def _csv_row(result: ProbeResult, context: ProbeContext) -> dict[str, Any]:
    config = context.config
    row = {
        "status": result.status,
        "error": result.error,
        "requested_frequency_mhz": result.requested_frequency_mhz,
        "resolved_frequency_mhz": result.resolved_frequency_mhz,
        "frequency_offset_hz": result.frequency_offset_hz,
        "absolute_bin": result.absolute_bin,
        "coarse_channel": result.coarse_channel,
        "bin_in_coarse": result.bin_in_coarse,
        "stamp_start_bin": result.stamp_start_bin,
        "stamp_end_bin_exclusive": result.stamp_end_bin,
        "stamp_clamped": result.stamp_clamped,
        "on1_max_k2": result.on_max_k2[0],
        "on2_max_k2": result.on_max_k2[1],
        "on3_max_k2": result.on_max_k2[2],
        "ed_max_k2": result.ed_max_k2,
        "stat_threshold": config.inference.stat_threshold,
        "ed_would_propose": result.ed_would_propose,
        "bandpass_method_used": context.bandpass_method,
        "latent_variant": config.rf.latent_variant,
        "raw_rf_probability": result.raw_rf_probability,
        "screening_probability": result.screening_probability,
        "screening_threshold": config.inference.screening_threshold,
        "screen_pass": result.screen_pass,
        "mc_mean": result.mc_mean,
        "mc_std": result.mc_std,
        "mc_draws": config.inference.mc_draws,
        "mc_scoring_mode": (
            ""
            if result.status != "ok"
            else ("production pass-2" if result.screen_pass else "forced diagnostic")
        ),
        "classification_threshold": config.inference.classification_threshold,
        "mc_pass": result.mc_pass,
        "plot_path": result.plot_path,
    }
    if result.status != "ok":
        # Booleans have no honest value for a never-scored row — blank beats a default False
        # that an aggregation would count as a real verdict.
        for key in ("stamp_clamped", "ed_would_propose", "screen_pass", "mc_pass"):
            row[key] = ""
    return row


def _write_csv(path: str, results: list[ProbeResult], context: ProbeContext) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [_csv_row(result, context) for result in results]
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run(args: argparse.Namespace) -> None:
    config = _load_config(args)
    h5_paths = list(args.h5_files) if args.h5_files else _resolve_catalog(args, config)
    if len(h5_paths) != 6:
        raise ValueError(f"Expected exactly six observation paths, got {len(h5_paths)}")
    axis = _validate_cadence_axes(h5_paths, config.data.time_bins)

    n_coarse_channels = axis.nchans // config.inference.coarse_channel_width
    bandpass_flatten, bandpass_method = _select_bandpass_flattener(
        config,
        n_coarse_channels,
    )
    context = ProbeContext(
        h5_paths=h5_paths,
        axis=axis,
        config=config,
        bandpass_flatten=bandpass_flatten,
        bandpass_method=bandpass_method,
    )
    # One bad location must not abort a batch over a published event table — capture the
    # failure per frequency and keep going; error rows carry the reason in table and CSV.
    results = []
    for frequency in args.frequency_mhz:
        try:
            results.append(_prepare_location(context, frequency))
        except Exception as exc:
            if args.traceback:
                import traceback  # noqa: PLC0415

                traceback.print_exc()
            results.append(
                ProbeResult(
                    requested_frequency_mhz=frequency,
                    resolved_frequency_mhz=float("nan"),
                    frequency_offset_hz=float("nan"),
                    absolute_bin=-1,
                    coarse_channel=-1,
                    bin_in_coarse=-1,
                    stamp_start_bin=-1,
                    stamp_end_bin=-1,
                    stamp_clamped=False,
                    on_max_k2=[float("nan")] * 3,
                    ed_max_k2=float("nan"),
                    ed_would_propose=False,
                    normalized_stamp=None,
                    status="error",
                    error=str(exc),
                )
            )
    scorable = [result for result in results if result.status == "ok"]
    if scorable:
        assets = _load_scoring_assets(args, config)
        _score_locations(scorable, assets, config, args.cadence_seed_key)
    if args.plot_dir and scorable:
        for index, result in enumerate(results, start=1):
            if result.status == "ok":
                _write_plot(result, index, args.plot_dir, axis)
    if args.csv:
        _write_csv(args.csv, results, context)

    print("Cadence observations (row order):")
    for index, path in enumerate(h5_paths, start=1):
        print(f"  {index}: {path}")
    print(f"Axis: fch1={axis.fch1:.9f} MHz, foff={axis.foff:.12g} MHz/bin, nchans={axis.nchans}")
    print(
        f"Preprocessing: bandpass={bandpass_method}, stamp_width="
        f"{config.inference.stamp_width}, downsample_factor={config.data.downsample_factor}, "
        f"ED window/step={config.inference.detection_window_size}/"
        f"{config.inference.detection_step_size}, stat_threshold="
        f"{config.inference.stat_threshold:g}"
    )
    print(
        "ED column semantics: max k^2 over detection windows STARTING inside the stamp "
        "(production's hit-placement convention) — an upper bound on proposal; dedup/overlap "
        "placement is not replayed"
    )
    print(
        f"Scoring: latent_variant={config.rf.latent_variant}, mc_draws="
        f"{config.inference.mc_draws}, MC SeedSequence root={config.reproducibility.seed}, "
        f"stream=STREAM_INFERENCE_MC, cadence_key={args.cadence_seed_key}, plus a per-location "
        "sub-key of the absolute frequency bin (MC draws are batch-composition independent — a "
        "documented delta from production's per-cadence draw blocks; encoder latents stay "
        "bit-exact only within one padding bucket, so very large probe batches can shift "
        "low-order bits); pass-1 rejects get a forced diagnostic MC pass production would "
        "never run"
    )
    print()
    _print_table(results, config)
    if args.csv:
        print(f"CSV: {args.csv}")
    if args.plot_dir and scorable:
        print(f"Plots: {args.plot_dir}")
    if not scorable:
        raise ValueError("Every requested frequency failed preprocessing; see the error rows above")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)
    _make_src_importable()
    try:
        _run(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        if args.traceback:
            import traceback  # noqa: PLC0415

            traceback.print_exc()
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
