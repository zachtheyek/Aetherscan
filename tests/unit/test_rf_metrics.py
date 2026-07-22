"""Unit tests for aetherscan/rf_metrics.py — the pure (TF-free) RF eval-metric helper whose
output train.py persists to training_stats (model_name='rf') for the dashboard's RF tab."""

from __future__ import annotations

import numpy as np
import pytest

from aetherscan.rf_metrics import compute_rf_eval_metrics


def _tiny_eval_arrays():
    """One cadence per sub-type, with one binary error in each class (threshold 0.7)."""
    val_binary = np.array([0, 0, 1, 1], dtype=np.int64)
    val_subtype = np.array(
        ["false_no_signal", "false_with_rfi", "true_only_eti", "true_eti_rfi"], dtype="U20"
    )
    val_probas = np.array([0.1, 0.8, 0.9, 0.4], dtype=np.float32)
    val_preds = (val_probas >= 0.7).astype(np.int64)  # -> [0, 1, 1, 0]
    return val_binary, val_subtype, val_probas, val_preds


def test_scalar_metrics_hand_computed():
    metrics = compute_rf_eval_metrics(*_tiny_eval_arrays())
    assert metrics["val_accuracy"] == pytest.approx(0.5)
    # AUC = fraction of (neg, pos) pairs ranked correctly = 3/4
    assert metrics["val_roc_auc"] == pytest.approx(0.75)
    # ranked desc: 0.9(1), 0.8(0), 0.4(1), 0.1(0) -> AP = 0.5*1 + 0.5*(2/3)
    assert metrics["val_average_precision"] == pytest.approx(5 / 6, abs=1e-6)
    # mean((p - y)^2) = (0.01 + 0.64 + 0.01 + 0.36) / 4
    assert metrics["val_brier_score"] == pytest.approx(0.255, abs=1e-6)


def test_confusion_cells():
    metrics = compute_rf_eval_metrics(*_tiny_eval_arrays())
    assert (
        metrics["confusion_tn"],
        metrics["confusion_fp"],
        metrics["confusion_fn"],
        metrics["confusion_tp"],
    ) == (1.0, 1.0, 1.0, 1.0)
    # sub-type x predicted-class cells: preds are [0, 1, 1, 0] in sub-type order
    assert metrics["confusion_false_no_signal_pred_false"] == 1.0
    assert metrics["confusion_false_no_signal_pred_true"] == 0.0
    assert metrics["confusion_false_with_rfi_pred_true"] == 1.0
    assert metrics["confusion_true_only_eti_pred_true"] == 1.0
    assert metrics["confusion_true_eti_rfi_pred_false"] == 1.0


def test_subtype_accuracies():
    metrics = compute_rf_eval_metrics(*_tiny_eval_arrays())
    assert metrics["val_accuracy_false_no_signal"] == pytest.approx(1.0)
    assert metrics["val_accuracy_false_with_rfi"] == pytest.approx(0.0)
    assert metrics["val_accuracy_true_only_eti"] == pytest.approx(1.0)
    assert metrics["val_accuracy_true_eti_rfi"] == pytest.approx(0.0)


def test_confidence_quantiles_monotonic():
    metrics = compute_rf_eval_metrics(*_tiny_eval_arrays())
    quantiles = [metrics[f"val_proba_q{q:02d}"] for q in (5, 25, 50, 75, 95)]
    assert metrics["val_proba_q50"] == pytest.approx(0.6, abs=1e-6)
    assert quantiles == sorted(quantiles)


def test_all_values_are_floats_and_key_count():
    metrics = compute_rf_eval_metrics(*_tiny_eval_arrays())
    assert all(isinstance(v, float) for v in metrics.values())
    # 4 scalars + 4 binary cells + 4 sub-type accuracies + 8 sub-type cells + 5 quantiles
    assert len(metrics) == 25


def test_single_class_split_omits_ranking_metrics():
    """A degenerate all-one-class val split must still yield the non-ranking metrics
    (ranking metrics are undefined there: roc_auc_score raises, AP degenerates)."""
    val_binary = np.array([0, 0, 0], dtype=np.int64)
    val_subtype = np.array(["false_no_signal", "false_with_rfi", "false_no_signal"], dtype="U20")
    val_probas = np.array([0.1, 0.8, 0.3], dtype=np.float32)
    val_preds = (val_probas >= 0.7).astype(np.int64)  # -> [0, 1, 0]
    metrics = compute_rf_eval_metrics(val_binary, val_subtype, val_probas, val_preds)
    assert "val_roc_auc" not in metrics
    assert "val_average_precision" not in metrics
    assert metrics["val_accuracy"] == pytest.approx(2 / 3)
    # mean((p - y)^2) = (0.01 + 0.64 + 0.09) / 3
    assert metrics["val_brier_score"] == pytest.approx(0.74 / 3, abs=1e-6)
    assert (metrics["confusion_tn"], metrics["confusion_fp"]) == (2.0, 1.0)
    assert metrics["confusion_fn"] == 0.0 and metrics["confusion_tp"] == 0.0
    assert metrics["val_proba_q50"] == pytest.approx(0.3, abs=1e-6)
    assert all(isinstance(v, float) for v in metrics.values())


def test_perfect_predictions():
    val_binary = np.array([0, 0, 1, 1], dtype=np.int64)
    val_subtype = np.array(
        ["false_no_signal", "false_with_rfi", "true_only_eti", "true_eti_rfi"], dtype="U20"
    )
    val_probas = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
    val_preds = (val_probas >= 0.5).astype(np.int64)
    metrics = compute_rf_eval_metrics(val_binary, val_subtype, val_probas, val_preds)
    assert metrics["val_accuracy"] == pytest.approx(1.0)
    assert metrics["val_roc_auc"] == pytest.approx(1.0)
    assert metrics["val_average_precision"] == pytest.approx(1.0)
    assert metrics["confusion_fp"] == 0.0
    assert metrics["confusion_fn"] == 0.0
    assert metrics["confusion_tn"] == 2.0
    assert metrics["confusion_tp"] == 2.0
