"""
Latent-representation variants, selection metrics, and probability calibration (#282).

The RF historically consumed the stochastic sampled `z` — nobody ever tested whether that is
the best representation, and the encoder's `z_mean`/`z_log_var` were computed and thrown away
at the RF stage. This module defines the 8-variant catalogue trained in one sweep on the same
generated data and split, the feature builders that inference reuses (driven by the saved
config's rf.latent_variant — never hardcoded), the selection metrics (recall at a fixed low
FPR is primary; AUC averages over operating points the pipeline never uses), the bootstrap
minimum-margin tie-break toward simpler variants, and the ECE-gated probability calibrator.

Everything operates on FLATTENED per-cadence feature blocks of shape
(n_cadences, num_observations * latent_dim) — the exact output of
models.random_forest.prepare_latent_features — so builders compose by hstack and the RF
feature layout stays `[z_mean block | extras]`, documented for SHAP readability.

Deliberately TF-free (mirrors rf_metrics.py): imported by both train.py and inference.py, and
unit-testable without the scientific stack.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

# Variant catalogue, ordered SIMPLE -> COMPLEX (by feature count, deterministic-first among
# equals). The minimum-margin selection walks this order, so ties break toward the earlier
# (simpler / cheaper-SHAP) variant. Names are persisted in config_{tag}.json (rf.latent_variant)
# and in artifact filenames — treat them as a stable contract.
VARIANT_ORDER = [
    "z_mean",  # deterministic baseline: per-obs posterior means (F = 48 at defaults)
    "z",  # legacy baseline: one stochastic sample (F = 48)
    "z_aug",  # trained on z_mean + K sampled draws as extra ROWS, evaluated on z_mean (F = 48)
    "z_mean_total_kl",  # z_mean + total KL per cadence (F = 49)
    "z_mean_obs_logvar",  # z_mean + per-observation mean log_var (F = 48 + num_obs = 54)
    "z_mean_dim_logvar",  # z_mean + per-dim mean log_var over obs (F = 48 + latent_dim = 56)
    "z_mean_logvar_active",  # z_mean + log_var restricted to ACTIVE dims (F = 48..96)
    "z_mean_logvar",  # full per-dimension uncertainty (F = 96)
]


def _reshape_blocks(flat: np.ndarray, num_observations: int, latent_dim: int) -> np.ndarray:
    return np.asarray(flat).reshape(-1, num_observations, latent_dim)


def per_cadence_kl(z_mean_flat: np.ndarray, z_log_var_flat: np.ndarray) -> np.ndarray:
    """Total KL divergence per cadence: -0.5*(1 + lv - mu^2 - e^lv) summed over every
    (observation, dim) — the same closed form as models/vae.py's loss term."""
    z_mean = np.asarray(z_mean_flat, dtype=np.float64)
    z_log_var = np.asarray(z_log_var_flat, dtype=np.float64)
    kl = -0.5 * (1.0 + z_log_var - np.square(z_mean) - np.exp(z_log_var))
    return kl.sum(axis=1).astype(np.float32)


def latent_dim_variances(
    z_mean_flat: np.ndarray, num_observations: int, latent_dim: int
) -> np.ndarray:
    """Per-dim z_mean variance across samples (pooling all observations) — the raw quantity
    the Active Units threshold cuts on. Exposed so callers can log the margins: a dim a hair
    below the cutoff and a dim that went dark read identically in the AU count but mean
    opposite things for an A/B parity decision (#288's fp16 verdict hinged on exactly this)."""
    per_obs = _reshape_blocks(z_mean_flat, num_observations, latent_dim)
    pooled = per_obs.reshape(-1, latent_dim)
    return pooled.var(axis=0)


def active_latent_dims(
    z_mean_flat: np.ndarray, num_observations: int, latent_dim: int, threshold: float
) -> list[int]:
    """
    Active Units (Burda et al.): dims whose z_mean variance across samples (pooling all
    observations) exceeds `threshold`. Collapsed dims carry near-constant posteriors and
    contribute dead-weight log_var features — this gates the z_mean_logvar_active variant
    and feeds check_posterior_collapse.
    """
    variances = latent_dim_variances(z_mean_flat, num_observations, latent_dim)
    return [int(d) for d in np.nonzero(variances > threshold)[0]]


def _active_logvar_columns(
    active_dims: list[int], num_observations: int, latent_dim: int
) -> list[int]:
    """Column indices of the active dims inside a flattened per-cadence block (obs-major)."""
    return [
        obs * latent_dim + dim for obs in range(num_observations) for dim in sorted(active_dims)
    ]


def variant_feature_count(
    variant: str, num_observations: int, latent_dim: int, active_dims: list[int] | None = None
) -> int:
    base = num_observations * latent_dim
    if variant in ("z", "z_mean", "z_aug"):
        return base
    if variant == "z_mean_total_kl":
        return base + 1
    if variant == "z_mean_obs_logvar":
        return base + num_observations
    if variant == "z_mean_dim_logvar":
        return base + latent_dim
    if variant == "z_mean_logvar_active":
        return base + num_observations * len(active_dims or [])
    if variant == "z_mean_logvar":
        return 2 * base
    raise ValueError(f"Unknown latent variant {variant!r}; expected one of {VARIANT_ORDER}")


def build_variant_features(
    variant: str,
    lead_flat: np.ndarray,
    z_log_var_flat: np.ndarray,
    num_observations: int,
    latent_dim: int,
    active_dims: list[int] | None = None,
) -> np.ndarray:
    """
    Assemble one variant's feature matrix from a LEAD block plus the log_var block.

    `lead_flat` fills the first num_observations*latent_dim columns: z_mean for the
    deterministic (pass-1 / evaluation) form, a sampled draw for MC scoring, or the sampled
    z for the legacy "z" variant's training rows. The uncertainty extras always come from
    the DETERMINISTIC z_log_var block. Layout is [lead | extras], obs-major within blocks —
    the RF is permutation-invariant across features, so ordering only affects SHAP/feature
    naming readability.
    """
    lead = np.asarray(lead_flat, dtype=np.float32)
    if variant in ("z", "z_mean", "z_aug"):
        return lead

    z_log_var = np.asarray(z_log_var_flat, dtype=np.float32)
    if variant == "z_mean_logvar":
        return np.hstack([lead, z_log_var])
    if variant == "z_mean_total_kl":
        kl = per_cadence_kl(lead, z_log_var)
        return np.hstack([lead, kl[:, None]])
    if variant == "z_mean_obs_logvar":
        per_obs_mean = _reshape_blocks(z_log_var, num_observations, latent_dim).mean(axis=2)
        return np.hstack([lead, per_obs_mean.astype(np.float32)])
    if variant == "z_mean_dim_logvar":
        per_dim_mean = _reshape_blocks(z_log_var, num_observations, latent_dim).mean(axis=1)
        return np.hstack([lead, per_dim_mean.astype(np.float32)])
    if variant == "z_mean_logvar_active":
        columns = _active_logvar_columns(active_dims or [], num_observations, latent_dim)
        if not columns:
            # Every dim collapsed (or none measured active) — degenerate to plain z_mean
            return lead
        return np.hstack([lead, z_log_var[:, columns]])
    raise ValueError(f"Unknown latent variant {variant!r}; expected one of {VARIANT_ORDER}")


def variant_feature_names(
    variant: str,
    num_observations: int,
    latent_dim: int,
    active_dims: list[int] | None = None,
) -> list[str]:
    """
    Human-readable column names mirroring build_variant_features' exact layout —
    [lead | extras], obs-major within blocks. Even-indexed observations are ON, odd are OFF,
    pairs numbered 1..3 (the data_generation cadence convention). MUST stay column-for-column
    in lockstep with build_variant_features: the SHAP plots pair these names with the feature
    matrix, and a length mismatch is an IndexError inside shap (a 54-feature
    z_mean_obs_logvar winner against the old hardcoded 48-name list is exactly how this
    function came to exist).
    """

    def obs_label(o: int) -> str:
        return f"{'ON' if o % 2 == 0 else 'OFF'}-{o // 2 + 1}"

    lead = [f"{obs_label(o)}_dim-{d}" for o in range(num_observations) for d in range(latent_dim)]
    if variant in ("z", "z_mean", "z_aug"):
        return lead
    if variant == "z_mean_logvar":
        return lead + [
            f"logvar_{obs_label(o)}_dim-{d}"
            for o in range(num_observations)
            for d in range(latent_dim)
        ]
    if variant == "z_mean_total_kl":
        return lead + ["total_kl"]
    if variant == "z_mean_obs_logvar":
        return lead + [f"logvar_mean_{obs_label(o)}" for o in range(num_observations)]
    if variant == "z_mean_dim_logvar":
        return lead + [f"logvar_mean_dim-{d}" for d in range(latent_dim)]
    if variant == "z_mean_logvar_active":
        columns = _active_logvar_columns(active_dims or [], num_observations, latent_dim)
        return lead + [f"logvar_{obs_label(c // latent_dim)}_dim-{c % latent_dim}" for c in columns]
    raise ValueError(f"Unknown latent variant {variant!r}; expected one of {VARIANT_ORDER}")


def sample_z_flat(
    z_mean_flat: np.ndarray, z_log_var_flat: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """One reparameterized draw z = mu + exp(lv/2) * eps on the flattened blocks (numpy —
    seeded via aetherscan.seeding, independent of TF's RNG)."""
    z_mean = np.asarray(z_mean_flat, dtype=np.float32)
    z_log_var = np.asarray(z_log_var_flat, dtype=np.float32)
    epsilon = rng.standard_normal(size=z_mean.shape).astype(np.float32)
    return z_mean + np.exp(0.5 * z_log_var) * epsilon


def build_z_aug_training_set(
    z_mean_flat: np.ndarray,
    z_log_var_flat: np.ndarray,
    labels: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """The z_aug variant's training rows: the deterministic z_mean row plus `draws` sampled
    rows per cadence (sampling-as-augmentation), labels replicated to match. Evaluation
    stays on plain z_mean — that is the whole point of the variant."""
    blocks = [np.asarray(z_mean_flat, dtype=np.float32)]
    blocks += [sample_z_flat(z_mean_flat, z_log_var_flat, rng) for _ in range(draws)]
    features = np.vstack(blocks)
    stacked_labels = np.concatenate([np.asarray(labels)] * (draws + 1))
    return features, stacked_labels


# ------------------------------------------------------------------ selection metrics


def recall_at_fpr(labels: np.ndarray, scores: np.ndarray, max_fpr: float) -> float:
    """
    Recall of the positive class at the score threshold whose false-positive rate is
    `max_fpr`: the primary #282 selection metric (don't miss signals while keeping the
    candidate list reviewable). Returns NaN for single-class labels.
    """
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores)
    negatives = scores[~labels]
    positives = scores[labels]
    if len(negatives) == 0 or len(positives) == 0:
        return float("nan")
    threshold = np.quantile(negatives, 1.0 - max_fpr)
    return float(np.mean(positives > threshold))


def expected_calibration_error(labels: np.ndarray, probas: np.ndarray, n_bins: int = 10) -> float:
    """ECE over quantile bins — the same binning strategy as plot_rf_calibration_curve, so
    the gate and the plotted number agree."""
    labels = np.asarray(labels, dtype=np.float64)
    probas = np.asarray(probas, dtype=np.float64)
    edges = np.quantile(probas, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=False):
        mask = (probas > low) & (probas <= high)
        if not mask.any():
            continue
        ece += (mask.mean()) * abs(labels[mask].mean() - probas[mask].mean())
    return float(ece)


def recall_margin_lower_bound(
    labels: np.ndarray,
    scores_best: np.ndarray,
    scores_simpler: np.ndarray,
    max_fpr: float,
    rounds: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> float:
    """Lower bound of the bootstrap CI on recall@FPR(best) - recall@FPR(simpler). A bound
    <= 0 means the best variant does not beat the simpler one by more than noise."""
    labels = np.asarray(labels)
    n = len(labels)
    diffs = np.empty(rounds)
    for i in range(rounds):
        idx = rng.integers(0, n, size=n)
        diffs[i] = recall_at_fpr(labels[idx], scores_best[idx], max_fpr) - recall_at_fpr(
            labels[idx], scores_simpler[idx], max_fpr
        )
    return float(np.nanquantile(diffs, alpha))


def select_winner(
    labels: np.ndarray,
    scores_by_variant: dict[str, np.ndarray],
    max_fpr: float,
    bootstrap_rounds: int,
    rng: np.random.Generator,
) -> tuple[str, dict[str, float]]:
    """
    Minimum-margin selection (#282): compute recall@FPR for every variant, find the best,
    then walk VARIANT_ORDER (simple -> complex) and return the FIRST variant that is
    statistically tied with the best (bootstrap lower bound of the best-minus-variant
    margin <= 0). Only a variant the best beats beyond noise is passed over.
    """
    recalls = {
        name: recall_at_fpr(labels, scores, max_fpr) for name, scores in scores_by_variant.items()
    }
    ordered = [name for name in VARIANT_ORDER if name in scores_by_variant]
    best = max(ordered, key=lambda name: (recalls[name], -ordered.index(name)))
    for name in ordered:
        if name == best:
            break
        margin_bound = recall_margin_lower_bound(
            labels,
            scores_by_variant[best],
            scores_by_variant[name],
            max_fpr,
            bootstrap_rounds,
            rng,
        )
        if margin_bound <= 0:
            logger.info(
                f"Variant selection: '{best}' (recall {recalls[best]:.4f}) does not beat "
                f"simpler '{name}' (recall {recalls[name]:.4f}) beyond noise "
                f"(bootstrap lower margin {margin_bound:+.4f}) — tie broken toward '{name}'"
            )
            return name, recalls
    return best, recalls


# ------------------------------------------------------------------ probability calibration


def fit_probability_calibrator(probas: np.ndarray, labels: np.ndarray, min_isotonic: int) -> dict:
    """
    Fit a probability calibrator on HELD-OUT calibration rows (#282): isotonic when the set
    is large enough (>= min_isotonic rows), else sigmoid/Platt (isotonic overfits small
    sets). Returned dict {method, model} is joblib-persistable as rf_calibrator_{tag}.joblib
    and applied identically at inference — an unapplied calibrator would be a silent
    train/serve mismatch.
    """
    probas = np.asarray(probas, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(probas) >= min_isotonic:
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(probas, labels)
        method = "isotonic"
    else:
        model = LogisticRegression(solver="lbfgs")
        model.fit(probas.reshape(-1, 1), labels)
        method = "sigmoid"
    return {"method": method, "model": model}


def apply_probability_calibrator(calibrator: dict | None, probas: np.ndarray) -> np.ndarray:
    """Map raw RF probabilities through the calibrator (identity when None). Monotonic, so
    rank metrics (AUC, recall@FPR) are unchanged — only probability VALUES move."""
    if calibrator is None:
        return np.asarray(probas)
    probas = np.asarray(probas, dtype=np.float64)
    if calibrator["method"] == "isotonic":
        return calibrator["model"].predict(probas)
    return calibrator["model"].predict_proba(probas.reshape(-1, 1))[:, 1]
