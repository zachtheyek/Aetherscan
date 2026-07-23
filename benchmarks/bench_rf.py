#!/usr/bin/env python3
"""
CPU benchmark: the Random Forest stage — latent-feature prep, RF fit, and RF inference.

The Beta-VAE runs on the GPU (see bench_gpu.py), but the second-stage classifier is
scikit-learn on the CPU and was previously unmeasured. This times the three pieces of
`train.py::train_random_forest` that follow encoding:

    prep    -- prepare_latent_features: reshape per-observation latents
               (n_cadences*num_observations, latent_dim) into per-cadence rows
               (n_cadences, num_observations*latent_dim) — the 48-feature layout the RF sees.
    fit     -- RandomForestClassifier.fit on the (n_cadences, 48) feature matrix.
    predict -- predict_proba on the held-out split (drives the RF eval plots + inference).

Defaults mirror production (config.rf + config.training): n_estimators=1000, bootstrap=True,
max_features="sqrt", n_jobs=-1, seed=11, latent_dim=8, num_observations=6, and
num_samples_rf=99840 cadences split train/val at 0.8. This is framework-free (numpy + sklearn,
no TensorFlow), so it runs anywhere the other CPU benchmarks do — the RF hyperparameters and
the prep reshape are inlined here rather than imported so the script never pulls the TF-backed
`aetherscan.models` package.

Labels are random binary (0/1). With no signal the trees grow until pure, so the fit time is a
*conservative upper bound*: real (separable) latents yield shallower trees that fit — and, later,
SHAP-explain — faster. Fit is measured once (expensive, low-variance); prep and predict use the
best of --repeats.

    python benchmarks/bench_rf.py                       # production sizes
    python benchmarks/bench_rf.py --n-estimators 200    # sweep tree count
    ./utils/run_container.sh python benchmarks/bench_rf.py   # on a cluster, matching the pipeline host
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from _common import machine_info, summarize, time_repeats, write_result
from sklearn.ensemble import RandomForestClassifier

# Production defaults (config.rf / config.beta_vae / config.data / config.training). Kept here as
# argparse defaults, matching the "size knobs default to production shapes" convention of the other
# benchmarks; override to sweep.
_N_ESTIMATORS = 1000
_BOOTSTRAP = True
_MAX_FEATURES = "sqrt"
_N_JOBS = -1
_SEED = 11
_LATENT_DIM = 8
_NUM_OBSERVATIONS = 6
_NUM_SAMPLES = 99840  # num_samples_rf (cadences)
_TRAIN_VAL_SPLIT = 0.8


def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value!r}")
    return ivalue


def _prepare_latent_features(latent_vectors: np.ndarray, num_observations: int) -> np.ndarray:
    """Concatenate each cadence's `num_observations` per-observation latents into one feature row.
    Mirrors aetherscan.models.random_forest.prepare_latent_features (inlined to stay TF-free) —
    the same explicit per-cadence loop, so the timing reflects the production implementation."""
    num_latents = latent_vectors.shape[0]
    if num_latents % num_observations != 0:
        raise ValueError(f"{num_latents} latent vectors not divisible by {num_observations}")
    num_cadences = num_latents // num_observations
    latent_dim = latent_vectors.shape[1]
    features = np.zeros((num_cadences, num_observations * latent_dim))
    for i in range(num_cadences):
        features[i, :] = latent_vectors[
            i * num_observations : (i + 1) * num_observations, :
        ].ravel()
    return features


def _build_model(args) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=args.n_estimators,
        bootstrap=_BOOTSTRAP,
        max_features=_MAX_FEATURES,
        n_jobs=args.n_jobs,
        random_state=_SEED,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-estimators", type=_positive_int, default=_N_ESTIMATORS)
    parser.add_argument("--n-samples", type=_positive_int, default=_NUM_SAMPLES, help="Cadences.")
    parser.add_argument("--latent-dim", type=_positive_int, default=_LATENT_DIM)
    parser.add_argument("--num-observations", type=_positive_int, default=_NUM_OBSERVATIONS)
    parser.add_argument(
        "--n-jobs", type=int, default=_N_JOBS, help="sklearn n_jobs (-1 = all cores)."
    )
    parser.add_argument(
        "--repeats", type=_positive_int, default=3, help="Best-of repeats for prep + predict."
    )
    parser.add_argument("--output", default=None, help="Result JSON path.")
    args = parser.parse_args()

    n_cadences = args.n_samples
    n_features = args.num_observations * args.latent_dim
    n_train = int(n_cadences * _TRAIN_VAL_SPLIT)
    rng = np.random.default_rng(_SEED)

    # Per-observation latents (n_cadences*num_observations, latent_dim), as the encoder emits them.
    latents = rng.standard_normal((n_cadences * args.num_observations, args.latent_dim)).astype(
        np.float32
    )
    labels = rng.integers(0, 2, size=n_cadences)

    print(
        f"n_cadences={n_cadences}  features={n_features}  n_estimators={args.n_estimators}  "
        f"n_jobs={args.n_jobs}  train/val={n_train}/{n_cadences - n_train}"
    )

    # prep: per-observation latents -> per-cadence 48-feature rows
    prep_durations = time_repeats(
        lambda: _prepare_latent_features(latents, args.num_observations), args.repeats
    )
    features = _prepare_latent_features(latents, args.num_observations)
    train_x, val_x = features[:n_train], features[n_train:]
    train_y = labels[:n_train]

    # fit: measured once (a full 1000-tree fit is expensive and low-variance)
    model = _build_model(args)
    fit_start = time.perf_counter()
    model.fit(train_x, train_y)
    fit_elapsed = time.perf_counter() - fit_start

    # predict: predict_proba over the held-out split
    predict_durations = time_repeats(lambda: model.predict_proba(val_x), args.repeats)

    prep = summarize(prep_durations, n_cadences)
    predict = summarize(predict_durations, val_x.shape[0])
    fit = {
        "elapsed_s": fit_elapsed,
        "cadences_per_s": n_train / fit_elapsed if fit_elapsed > 0 else float("inf"),
        "n_train": n_train,
    }

    print(f"{'stage':>8}  {'cadences/s':>14}  {'seconds':>10}")
    print(f"{'prep':>8}  {prep['ops_per_s']:>14,.0f}  {prep['best_s']:>10.3f}")
    print(f"{'fit':>8}  {fit['cadences_per_s']:>14,.0f}  {fit['elapsed_s']:>10.3f}")
    print(f"{'predict':>8}  {predict['ops_per_s']:>14,.0f}  {predict['best_s']:>10.3f}")

    path = write_result(
        "bench_rf",
        {
            "n_cadences": n_cadences,
            "n_features": n_features,
            "n_estimators": args.n_estimators,
            "n_jobs": args.n_jobs,
            "num_observations": args.num_observations,
            "latent_dim": args.latent_dim,
            "train_val_split": _TRAIN_VAL_SPLIT,
            "repeats": args.repeats,
            "cpu": machine_info()["hostname"],
        },
        {"prep": prep, "fit": fit, "predict": predict},
        args.output,
    )
    print(f"Result written to {path}")


if __name__ == "__main__":
    main()
