"""
Unit tests for aetherscan.shap_parallel (TF-free): the positive-class shape normalization and the
process-pool wrapper, whose output must be byte-identical to the single-threaded computation.
"""

from __future__ import annotations

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from aetherscan.shap_parallel import parallel_shap, select_positive_class_shap


class TestSelectPositiveClassShap:
    N, F = 5, 8

    def test_list_of_class_arrays_selects_positive(self):
        neg = np.zeros((self.N, self.F))
        pos = np.ones((self.N, self.F))
        result = select_positive_class_shap([neg, pos])
        np.testing.assert_array_equal(result, pos)

    def test_trailing_class_axis_values(self):
        values = np.stack(
            [np.zeros((self.N, self.F)), np.ones((self.N, self.F))], axis=-1
        )  # (N,F,2)
        result = select_positive_class_shap(values)
        assert result.shape == (self.N, self.F)
        assert np.all(result == 1.0)

    def test_trailing_class_axis_interactions(self):
        values = np.stack(
            [np.zeros((self.N, self.F, self.F)), np.ones((self.N, self.F, self.F))], axis=-1
        )  # (N, F, F, 2)
        result = select_positive_class_shap(values)
        assert result.shape == (self.N, self.F, self.F)
        assert np.all(result == 1.0)

    def test_single_output_passthrough(self):
        values = np.arange(self.N * self.F, dtype=float).reshape(self.N, self.F)
        np.testing.assert_array_equal(select_positive_class_shap(values), values)

    def test_log_loss_list_selects_first(self):
        first = np.ones((self.N, self.F))
        result = select_positive_class_shap([first, np.zeros((self.N, self.F))], log_loss=True)
        np.testing.assert_array_equal(result, first)

    def test_log_loss_trailing_class_axis(self):
        values = np.stack([np.zeros((self.N, self.F)), np.ones((self.N, self.F))], axis=-1)
        result = select_positive_class_shap(values, log_loss=True)
        assert result.shape == (self.N, self.F)
        assert np.all(result == 1.0)

    def test_log_loss_passthrough(self):
        values = np.arange(self.N * self.F, dtype=float).reshape(self.N, self.F)
        np.testing.assert_array_equal(select_positive_class_shap(values, log_loss=True), values)


@pytest.fixture(scope="module")
def small_rf(tmp_path_factory):
    """A tiny persisted binary RF + a held-out set to explain + a background set."""
    rng = np.random.default_rng(0)
    n_features = 6
    train_x = rng.standard_normal((128, n_features))
    train_y = (train_x[:, 0] + 0.3 * rng.standard_normal(128) > 0).astype(int)
    rf = RandomForestClassifier(n_estimators=8, max_depth=4, random_state=0, n_jobs=1)
    rf.fit(train_x, train_y)
    rf_path = tmp_path_factory.mktemp("shap_parallel") / "rf.joblib"
    joblib.dump(rf, rf_path)
    val_x = rng.standard_normal((24, n_features))
    val_y = (val_x[:, 0] > 0).astype(int)
    background = rng.standard_normal((16, n_features))
    return str(rf_path), val_x, val_y, background


@pytest.mark.parametrize("kind", ["summary", "interaction", "logloss"])
def test_parallel_matches_single_process(small_rf, kind):
    """The pooled result must equal the single-process result for every pass, in sample order."""
    pytest.importorskip("shap")  # the compute path needs shap in the workers
    rf_path, val_x, val_y, background = small_rf
    extra = {"background": background, "y": val_y} if kind == "logloss" else {}
    serial = parallel_shap(rf_path, val_x, kind, 1, **extra)
    pooled = parallel_shap(rf_path, val_x, kind, 4, **extra)
    assert serial.shape == pooled.shape
    assert serial.shape[0] == len(val_x)  # sample axis preserved
    np.testing.assert_allclose(pooled, serial, atol=1e-8)


def test_more_workers_than_samples_is_safe(small_rf):
    """n_workers is clamped to the sample count; a 2-row input with 4 workers still works."""
    pytest.importorskip("shap")  # the compute path needs shap in the workers
    rf_path, val_x, _, _ = small_rf
    out = parallel_shap(rf_path, val_x[:2], "summary", 4)
    assert out.shape[0] == 2


def test_unknown_kind_raises(small_rf):
    rf_path, val_x, _, _ = small_rf
    with pytest.raises(ValueError):
        parallel_shap(rf_path, val_x, "bogus", 2)
