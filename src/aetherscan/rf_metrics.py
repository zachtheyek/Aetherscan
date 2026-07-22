"""
Pure Random Forest eval-metric computation for Aetherscan Pipeline.

Deliberately TF-free (numpy + sklearn.metrics only; no aetherscan.models import — that
package's __init__ pulls in TensorFlow via vae.py) so the helper stays unit-testable
without the GPU stack. train.py calls compute_rf_eval_metrics() at the tail of
train_random_forest() and persists each returned scalar to training_stats via
db.write_training_stat(model_name="rf", ...) for the live dashboard's RF tab.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# Persisted quantiles of the val P(true) distribution (stat_name val_proba_q{QQ})
_CONFIDENCE_QUANTILES = (5, 25, 50, 75, 95)


def compute_rf_eval_metrics(
    val_binary_labels: np.ndarray,
    val_subtype_labels: np.ndarray,
    val_probas: np.ndarray,
    val_preds: np.ndarray,
) -> dict[str, float]:
    """
    Compute the scalar RF validation metrics persisted to training_stats (model_name='rf').

    Inputs are the aligned per-cadence arrays already built by train_random_forest():
    binary ground truth (0/1), sub-type ground truth (false_no_signal / false_with_rfi /
    true_only_eti / true_eti_rfi), ensemble P(true), and thresholded predictions (at
    inference.classification_threshold). Returns a flat {stat_name: float} dict:

      val_accuracy / val_roc_auc / val_average_precision / val_brier_score
      val_accuracy_{subtype}                  binary correctness within each sub-type
      confusion_{tn,fp,fn,tp}                 binary 2x2 confusion cell counts
      confusion_{subtype}_pred_{false,true}   sub-type x predicted-class cell counts
      val_proba_q{05,...,95}                  confidence-distribution quantiles

    The ranking metrics (ROC-AUC / average precision) are threshold-free; accuracy and the
    confusion cells reflect val_preds' operating point (the deployment threshold). On a
    degenerate single-class val split the ranking metrics are undefined, so their two keys
    are omitted (the dashboard renders absent tiles as an em dash) and everything else
    still lands.
    """
    val_binary_labels = np.asarray(val_binary_labels)
    val_subtype_labels = np.asarray(val_subtype_labels)
    val_probas = np.asarray(val_probas, dtype=np.float64)
    val_preds = np.asarray(val_preds)

    metrics: dict[str, float] = {
        "val_accuracy": float(np.mean(val_preds == val_binary_labels)),
        "val_brier_score": float(brier_score_loss(val_binary_labels, val_probas)),
    }

    # Ranking metrics are undefined when only one class is present: roc_auc_score raises
    # ValueError and average_precision_score silently degenerates (all-negative labels
    # -> -0.0 with a warning). Omit both keys in that case rather than letting the raise
    # reach the caller's blanket best-effort guard and lose the whole dict.
    if np.unique(val_binary_labels).size > 1:
        metrics["val_roc_auc"] = float(roc_auc_score(val_binary_labels, val_probas))
        metrics["val_average_precision"] = float(
            average_precision_score(val_binary_labels, val_probas)
        )
    else:
        logger.warning(
            "Single-class val split: omitting undefined val_roc_auc / val_average_precision"
        )

    # Binary 2x2 cells (explicit label order keeps the cell layout stable regardless of
    # which classes appear in the val split)
    tn, fp, fn, tp = confusion_matrix(val_binary_labels, val_preds, labels=[0, 1]).ravel()
    metrics["confusion_tn"] = float(tn)
    metrics["confusion_fp"] = float(fp)
    metrics["confusion_fn"] = float(fn)
    metrics["confusion_tp"] = float(tp)

    # Per-sub-type binary accuracy + sub-type x predicted-class cell counts
    correct = val_preds == val_binary_labels
    for subtype in np.unique(val_subtype_labels):
        mask = val_subtype_labels == subtype
        metrics[f"val_accuracy_{subtype}"] = float(np.mean(correct[mask]))
        n_pred_true = int(np.sum(val_preds[mask] == 1))
        metrics[f"confusion_{subtype}_pred_true"] = float(n_pred_true)
        metrics[f"confusion_{subtype}_pred_false"] = float(int(np.sum(mask)) - n_pred_true)

    # Confidence-distribution quantiles of val P(true)
    for q in _CONFIDENCE_QUANTILES:
        metrics[f"val_proba_q{q:02d}"] = float(np.quantile(val_probas, q / 100))

    return metrics
