# NOTE: come back to this later

"""Unit tests for aetherscan.data_generation: log-norm, intersection checks, signal injection,
create_* cadence generators, and intensity statistics."""

from __future__ import annotations

import itertools
import math
import random

import numpy as np
import pytest

from aetherscan import data_generation
from aetherscan.data_generation import (
    MAX_INTERSECTION_RETRIES,
    _compute_intensity_stats,
    check_valid_intersection,
    create_false,
    create_true_double,
    create_true_single,
    log_norm,
    new_cadence,
    write_segment_stats,
)

# Keep injection fast: small frequency axis, real-ish resolutions.
_WIDTH_BIN = 128
_FREQ_RES = 2.7939677238464355  # Hz
_TIME_RES = 18.25361108  # seconds

_SIGNAL_INFO_KEYS = {
    "snr",
    "drift_rate",
    "signal_width",
    "starting_bin",
    "slope_pixel",
    "y_intercept",
}

_STAT_KEYS = {
    "global_mean",
    "global_median",
    "global_std",
    "global_mad",
    "global_skew",
    "global_kurtosis",
}


@pytest.fixture(autouse=True)
def _seed_rngs():
    # NOTE: np.random.seed() intentionally seeds the legacy global RandomState — the
    # production code under test (new_cadence, create_*) draws from the legacy np.random.*
    # API, so switching this to a Generator (np.random.default_rng) would NOT make those
    # draws deterministic. Don't "modernize" this without migrating data_generation.py first.
    random.seed(11)
    np.random.seed(11)


@pytest.fixture
def plate(make_background_npy):
    """A tiny background plate loaded from a factory-written .npy (round-trips the factory)."""
    path = make_background_npy("plate.npy", n_cadences=4, width_bin=_WIDTH_BIN)
    loaded = np.load(path)
    assert loaded.shape == (4, 6, 16, _WIDTH_BIN)
    assert loaded.dtype == np.float32
    return loaded


class TestLogNorm:
    def test_output_bounded_in_unit_interval(self):
        rng = np.random.default_rng(0)
        data = rng.chisquare(df=4, size=(16, 64))
        result = log_norm(data)
        assert result.shape == data.shape
        assert result.min() == 0.0
        assert result.max() == pytest.approx(1.0)

    def test_preserves_ordering(self):
        data = np.array([[1.0, 10.0, 100.0, 1000.0]])
        result = log_norm(data)
        assert np.all(np.diff(result[0]) > 0)

    def test_constant_input_maps_to_zeros(self):
        result = log_norm(np.full((4, 4), 3.0))
        assert np.all(result == 0.0)

    def test_idempotence_on_normalized_scale(self):
        # Re-normalizing an already [0, 1] array must stay within [0, 1] with the same
        # endpoints — the transform is a monotone squash onto the unit interval.
        rng = np.random.default_rng(1)
        once = log_norm(rng.chisquare(df=4, size=(8, 32)))
        twice = log_norm(once)
        assert twice.min() == 0.0
        assert twice.max() == pytest.approx(1.0)
        assert np.array_equal(np.argsort(once, axis=None), np.argsort(twice, axis=None))

    def test_return_params_matches_default_output(self):
        rng = np.random.default_rng(2)
        data = rng.chisquare(df=4, size=(16, 64))
        with_params, params = log_norm(data, return_params=True)
        np.testing.assert_array_equal(with_params, log_norm(data))
        assert len(params) == 2

    def test_return_params_inverts_transform(self):
        # exp(normalized * range_log + min_log) must recover the epsilon-shifted input.
        rng = np.random.default_rng(3)
        data = rng.chisquare(df=4, size=(16, 64))
        normalized, (min_log, range_log) = log_norm(data, return_params=True)
        recovered = np.exp(normalized * range_log + min_log)
        np.testing.assert_allclose(recovered, data + 1e-10, rtol=1e-6)

    def test_return_params_constant_input_degenerate_range(self):
        # Constant input has zero dynamic range: range_log == 0 flags "no inversion possible".
        normalized, (min_log, range_log) = log_norm(np.full((4, 4), 3.0), return_params=True)
        assert np.all(normalized == 0.0)
        assert range_log == 0.0
        assert min_log == pytest.approx(np.log(3.0), abs=1e-9)


class TestCheckValidIntersection:
    def test_parallel_lines_are_valid(self):
        assert check_valid_intersection(2.0, 2.0, 0.0, 50.0) is True

    def test_intersection_inside_on_region_invalid(self):
        # y = x and y = -x + 20 intersect at (10, 10): inside the first ON block [0, 16].
        assert check_valid_intersection(1.0, -1.0, 0.0, 20.0) is False

    def test_intersection_between_on_regions_valid(self):
        # y = x and y = -x + 40 intersect at (20, 20): inside the first OFF block (16, 32).
        assert check_valid_intersection(1.0, -1.0, 0.0, 40.0) is True

    @pytest.mark.parametrize("y_target", [0.0, 16.0, 32.0, 48.0, 64.0, 80.0])
    def test_on_region_boundaries_are_inclusive(self, y_target):
        # Intersection exactly on an ON boundary counts as inside (invalid).
        # y = x and y = -x + 2*y_target intersect at (y_target, y_target).
        assert check_valid_intersection(1.0, -1.0, 0.0, 2 * y_target) is False

    @pytest.mark.parametrize(
        "b1, b2, positive_drift",
        [
            # Pairs whose float intersection lands at y* ~ -1e-14 instead of the exact 0:
            # the un-padded inclusive comparison accepted all of these by rounding luck (#118).
            (1, 187, True),
            (173, 434, True),
            (383, 511, True),
            (1, 75, False),
            (102, 500, False),
            (239, 511, False),
        ],
    )
    def test_boundary_intersections_rejected_despite_float_noise(self, b1, b2, positive_drift):
        # Production geometry (new_cadence): positive-drift trajectories all pass through
        # (0, 0) and negative-drift ones through (width_bin, 0), so same-sign pairs intersect
        # exactly ON the first ON-band edge (y* = 0) and must be rejected (boundaries are
        # inclusive) regardless of float rounding in slope/intercept/intersection arithmetic.
        total_time, width_bin = 96, 512
        slopes_intercepts = []
        for b in (b1, b2):
            slope = total_time / b if positive_drift else total_time / (b - width_bin)
            slopes_intercepts.append((slope, total_time - slope * b))
        (s1, e1), (s2, e2) = slopes_intercepts
        assert check_valid_intersection(s1, s2, e1, e2) is False


class TestNewCadence:
    def _background(self, rng=None):
        rng = rng or np.random.default_rng(3)
        return rng.chisquare(df=4, size=(96, _WIDTH_BIN))

    def test_shapes_and_signal_info(self):
        data = self._background()
        modified, signal_info, clamped = new_cadence(
            data.copy(),
            snr=20.0,
            width_bin=_WIDTH_BIN,
            freq_resolution=_FREQ_RES,
            time_resolution=_TIME_RES,
        )
        assert modified.shape == data.shape
        assert set(signal_info.keys()) == _SIGNAL_INFO_KEYS
        assert all(isinstance(v, float) for v in signal_info.values())
        assert signal_info["snr"] == 20.0
        assert 1 <= signal_info["starting_bin"] <= _WIDTH_BIN - 1
        assert isinstance(clamped, bool)

    def test_injection_adds_power(self):
        data = self._background()
        modified, _, _ = new_cadence(
            data.copy(),
            snr=50.0,
            width_bin=_WIDTH_BIN,
            freq_resolution=_FREQ_RES,
            time_resolution=_TIME_RES,
        )
        # A signal only adds intensity — total power must strictly increase.
        assert modified.sum() > data.sum()

    def test_drift_rate_inverse_of_slope(self):
        _, signal_info, clamped = new_cadence(
            self._background(),
            snr=20.0,
            width_bin=_WIDTH_BIN,
            freq_resolution=_FREQ_RES,
            time_resolution=_TIME_RES,
        )
        if not clamped:
            slope_physical = signal_info["slope_pixel"] * (_TIME_RES / _FREQ_RES)
            # drift_rate = -1/slope_physical up to the additive noise term on the slope,
            # so only the sign relationship is exact.
            assert math.copysign(1, signal_info["drift_rate"]) == -math.copysign(1, slope_physical)

    def test_near_zero_slope_is_clamped(self, monkeypatch):
        # Force the degenerate geometry: time/freq resolution ratio ~0 makes the physical
        # slope collapse below the clamp threshold when the additive noise term is 0.
        # new_cadence draws random.random() in order: starting_bin, slope noise, signal width.
        draws = iter([0.5, 0.0])
        monkeypatch.setattr(random, "random", lambda: next(draws, 0.25))
        monkeypatch.setattr(np.random, "choice", lambda *a, **k: 1)
        _, signal_info, clamped = new_cadence(
            self._background(),
            snr=20.0,
            width_bin=_WIDTH_BIN,
            freq_resolution=1.0,
            time_resolution=1e-12,
        )
        assert clamped is True
        # Clamped slope preserves direction but pins magnitude at the floor: |drift| = 1e6.
        assert abs(signal_info["drift_rate"]) == pytest.approx(1e6)


class TestCreateCadences:
    def _kwargs(self):
        return {
            "snr_base": 10.0,
            "snr_range": 5.0,
            "width_bin": _WIDTH_BIN,
            "freq_resolution": _FREQ_RES,
            "time_resolution": _TIME_RES,
        }

    def _assert_common(self, final, sample_info, plate):
        assert final.shape == (6, 16, _WIDTH_BIN)
        assert 0 <= sample_info["background_index"] < plate.shape[0]
        assert set(sample_info["intensity_stats"].keys()) == {"A", "B", "C"}
        for stage_stats in sample_info["intensity_stats"].values():
            assert set(stage_stats.keys()) == _STAT_KEYS
        assert isinstance(sample_info["slope_was_clamped"], bool)
        # Output is log-normalized per observation.
        assert final.min() >= 0.0
        assert final.max() <= 1.0
        # Per-observation log-norm params are threaded through for display inversion,
        # with a positive dynamic range for noisy inputs.
        assert sample_info["lognorm_params"].shape == (6, 2)
        assert np.all(sample_info["lognorm_params"][:, 1] > 0)

    def test_create_false_injected(self, plate):
        final, sample_info = create_false(plate, inject=True, **self._kwargs())
        self._assert_common(final, sample_info, plate)
        keys = set(sample_info["signal_info"].keys())
        assert keys == {f"rfi_{k}" for k in _SIGNAL_INFO_KEYS}

    def test_create_false_no_injection(self, plate):
        final, sample_info = create_false(plate, inject=False, **self._kwargs())
        self._assert_common(final, sample_info, plate)
        assert sample_info["signal_info"] == {}
        # Without injection, Stage B mirrors Stage A exactly.
        assert sample_info["intensity_stats"]["B"] == sample_info["intensity_stats"]["A"]
        # And every observation is just the log-normalized background (float32 path: the
        # generator normalizes the raw float32 plate slices directly).
        base = plate[sample_info["background_index"]]
        for obs in range(6):
            np.testing.assert_allclose(final[obs], log_norm(base[obs]))
        # The recorded per-observation params invert the normalization back to the raw
        # (epsilon-shifted) background.
        for obs in range(6):
            min_log, range_log = sample_info["lognorm_params"][obs]
            recovered = np.exp(final[obs] * range_log + min_log)
            np.testing.assert_allclose(recovered, base[obs] + 1e-10, rtol=1e-5)

    def test_create_true_single_injects_on_only(self, plate):
        final, sample_info = create_true_single(plate, **self._kwargs())
        self._assert_common(final, sample_info, plate)
        keys = set(sample_info["signal_info"].keys())
        assert keys == {f"eti_{k}" for k in _SIGNAL_INFO_KEYS}
        # OFF observations (1, 3, 5) carry no injection: they equal the log-normed background.
        base = plate[sample_info["background_index"]]
        for obs in (1, 3, 5):
            np.testing.assert_allclose(final[obs], log_norm(base[obs].astype(np.float64)))

    def test_create_true_double_injects_both(self, plate):
        final, sample_info = create_true_double(plate, **self._kwargs())
        self._assert_common(final, sample_info, plate)
        keys = set(sample_info["signal_info"].keys())
        expected = {f"rfi_{k}" for k in _SIGNAL_INFO_KEYS} | {f"eti_{k}" for k in _SIGNAL_INFO_KEYS}
        assert keys == expected
        # The two injected trajectories must not intersect inside an ON region.
        assert check_valid_intersection(
            sample_info["signal_info"]["rfi_slope_pixel"],
            sample_info["signal_info"]["eti_slope_pixel"],
            sample_info["signal_info"]["rfi_y_intercept"],
            sample_info["signal_info"]["eti_y_intercept"],
        )
        # Retry-cap telemetry (#118): real geometry accepts well before the cap.
        assert sample_info["intersection_retries"] >= 1
        assert sample_info["intersection_retry_capped"] is False

    def _stub_injections(self, monkeypatch):
        # Replace the expensive draw+materialize pair with shape-preserving stubs whose slope
        # differs on every draw, so the slope_1 != slope_2 guard never causes a retry and
        # the retry count is driven purely by the (patched) intersection check. Since the
        # draw-first split, create_true_double calls _draw_signal_params per attempt and
        # _inject_drawn_signal only for the final pair — patch both.
        calls = itertools.count(1)

        def fake_draw_signal_params(total_time, width_bin, freq_resolution, time_resolution):
            params = {
                "drift_rate": 1.0,
                "signal_width": 1.0,
                "starting_bin": 1.0,
                "slope_pixel": float(next(calls)),
                "y_intercept": 0.0,
            }
            return params, False

        monkeypatch.setattr(data_generation, "_draw_signal_params", fake_draw_signal_params)
        monkeypatch.setattr(
            data_generation,
            "_inject_drawn_signal",
            lambda data, snr, params, freq_resolution, time_resolution: data,
        )

    def test_true_double_retry_cap_clamps_and_flags(self, plate, monkeypatch):
        # Always-rejecting geometry must terminate at the cap, keep the last drawn pair,
        # and flag the sample — not raise and not spin forever (#118).
        self._stub_injections(monkeypatch)
        monkeypatch.setattr(data_generation, "check_valid_intersection", lambda *a: False)
        final, sample_info = create_true_double(plate, **self._kwargs())
        assert final.shape == (6, 16, _WIDTH_BIN)
        assert sample_info["intersection_retries"] == MAX_INTERSECTION_RETRIES
        assert sample_info["intersection_retry_capped"] is True

    def test_true_double_first_attempt_accept_records_one_retry(self, plate, monkeypatch):
        self._stub_injections(monkeypatch)
        monkeypatch.setattr(data_generation, "check_valid_intersection", lambda *a: True)
        _, sample_info = create_true_double(plate, **self._kwargs())
        assert sample_info["intersection_retries"] == 1
        assert sample_info["intersection_retry_capped"] is False


class TestDrawFirstEquivalence:
    """Byte-compatibility pins for the draw-first split of new_cadence (#TBD).

    The split moved setigen materialization out of create_true_double's retry loop; these
    tests pin the two contracts that make that byte-identical: materialization consumes no
    RNG, and create_true_double's draw order matches the pre-split inject-then-test loop
    exactly.
    """

    def test_inject_drawn_signal_consumes_no_rng(self):
        data = np.abs(np.random.randn(96, _WIDTH_BIN)) + 1.0
        params, _ = data_generation._draw_signal_params(96, _WIDTH_BIN, _FREQ_RES, _TIME_RES)
        py_state = random.getstate()
        np_state = np.random.get_state()
        data_generation._inject_drawn_signal(data, 20.0, params, _FREQ_RES, _TIME_RES)
        assert random.getstate() == py_state
        np_state_after = np.random.get_state()
        assert np_state[0] == np_state_after[0]
        np.testing.assert_array_equal(np_state[1], np_state_after[1])
        assert np_state[2:] == np_state_after[2:]

    def test_new_cadence_is_draw_then_inject_composition(self):
        data = np.abs(np.random.randn(96, _WIDTH_BIN)) + 1.0
        state_py = random.getstate()
        state_np = np.random.get_state()
        modified, signal_info, clamped = new_cadence(data, 20.0, _WIDTH_BIN, _FREQ_RES, _TIME_RES)
        random.setstate(state_py)
        np.random.set_state(state_np)
        params, clamped_ref = data_generation._draw_signal_params(
            data.shape[0], _WIDTH_BIN, _FREQ_RES, _TIME_RES
        )
        modified_ref = data_generation._inject_drawn_signal(
            data, 20.0, params, _FREQ_RES, _TIME_RES
        )
        np.testing.assert_array_equal(modified, modified_ref)
        assert signal_info == data_generation._signal_info_from_params(20.0, params)
        assert clamped == clamped_ref

    @staticmethod
    def _pre_split_true_double_reference(
        plate, snr_base, snr_range, width_bin, freq_resolution, time_resolution, dynamic_range=1.0
    ):
        """The pre-split create_true_double loop (inject both, then test), reproduced verbatim
        via the public new_cadence so the equivalence test compares against the exact legacy
        RNG-consumption and materialization order."""
        background_index = int(plate.shape[0] * random.random())
        base = plate[background_index, :, :, :]
        n_obs = plate.shape[1]
        n_time = plate.shape[2]
        final = np.zeros((n_obs, n_time, width_bin))
        stats_a = _compute_intensity_stats(base)
        data = np.zeros((n_obs * n_time, width_bin))
        for i in range(n_obs):
            data[i * n_time : (i + 1) * n_time, :] = base[i, :, :]
        snr = random.random() * snr_range + snr_base
        intersection_retries = 0
        intersection_retry_capped = False
        while True:
            intersection_retries += 1
            cadence_1, rfi_signal_info, rfi_slope_clamped = new_cadence(
                data, snr, width_bin, freq_resolution, time_resolution
            )
            cadence_2, eti_signal_info, eti_slope_clamped = new_cadence(
                cadence_1, snr * dynamic_range, width_bin, freq_resolution, time_resolution
            )
            slope_1 = rfi_signal_info["slope_pixel"]
            intercept_1 = rfi_signal_info["y_intercept"]
            slope_2 = eti_signal_info["slope_pixel"]
            intercept_2 = eti_signal_info["y_intercept"]
            if slope_1 != slope_2 and check_valid_intersection(
                slope_1, slope_2, intercept_1, intercept_2
            ):
                break
            if intersection_retries >= MAX_INTERSECTION_RETRIES:
                intersection_retry_capped = True
                break
        slope_was_clamped = rfi_slope_clamped or eti_slope_clamped
        signal_info = {
            **{f"rfi_{k}": v for k, v in rfi_signal_info.items()},
            **{f"eti_{k}": v for k, v in eti_signal_info.items()},
        }
        stats_b = _compute_intensity_stats(cadence_2)
        lognorm_params = np.zeros((n_obs, 2), dtype=np.float32)
        for i in range(n_obs):
            source = cadence_2 if i % 2 == 0 else cadence_1
            final[i, :, :], lognorm_params[i] = log_norm(
                source[i * n_time : (i + 1) * n_time, :], return_params=True
            )
        stats_c = _compute_intensity_stats(final)
        return final, {
            "background_index": background_index,
            "intensity_stats": {"A": stats_a, "B": stats_b, "C": stats_c},
            "signal_info": signal_info,
            "slope_was_clamped": slope_was_clamped,
            "intersection_retries": intersection_retries,
            "intersection_retry_capped": intersection_retry_capped,
            "lognorm_params": lognorm_params,
        }

    @pytest.mark.parametrize("seed", [11, 12, 13, 14])
    def test_true_double_matches_pre_split_reference(self, plate, seed):
        kwargs = {
            "snr_base": 10.0,
            "snr_range": 5.0,
            "width_bin": _WIDTH_BIN,
            "freq_resolution": _FREQ_RES,
            "time_resolution": _TIME_RES,
        }
        random.seed(seed)
        np.random.seed(seed)
        final_ref, info_ref = self._pre_split_true_double_reference(plate, **kwargs)
        state_py_ref = random.getstate()
        state_np_ref = np.random.get_state()
        random.seed(seed)
        np.random.seed(seed)
        final_new, info_new = create_true_double(plate, **kwargs)
        np.testing.assert_array_equal(final_new, final_ref)
        np.testing.assert_array_equal(
            info_new.pop("lognorm_params"), info_ref.pop("lognorm_params")
        )
        assert info_new == info_ref
        # Both arms must also leave the global RNG streams in the same state (no extra or
        # missing draws) — the property the per-task seeding contract depends on.
        assert random.getstate() == state_py_ref
        state_np_new = np.random.get_state()
        assert state_np_new[0] == state_np_ref[0]
        np.testing.assert_array_equal(state_np_new[1], state_np_ref[1])
        assert state_np_new[2:] == state_np_ref[2:]


class TestComputeIntensityStats:
    def test_normal_case_matches_numpy(self):
        rng = np.random.default_rng(5)
        data = rng.chisquare(df=4, size=(6, 16, 32)).astype(np.float32)
        stats = _compute_intensity_stats(data)
        assert set(stats.keys()) == _STAT_KEYS
        flat = data.ravel().astype(np.float64)
        assert stats["global_mean"] == pytest.approx(np.mean(flat))
        assert stats["global_median"] == pytest.approx(np.median(flat))
        assert stats["global_std"] == pytest.approx(np.std(flat))
        assert stats["global_mad"] == pytest.approx(np.median(np.abs(flat - np.median(flat))))
        assert all(math.isfinite(v) for v in stats.values())

    def test_empty_array_returns_nan_for_every_key(self):
        stats = _compute_intensity_stats(np.array([]))
        assert set(stats.keys()) == _STAT_KEYS
        assert all(math.isnan(v) for v in stats.values())

    def test_constant_array_higher_moments_degenerate(self):
        stats = _compute_intensity_stats(np.full((4, 4), 7.0))
        assert stats["global_mean"] == 7.0
        assert stats["global_std"] == 0.0
        assert stats["global_mad"] == 0.0
        # Skew/kurtosis of a zero-variance array are NaN — the DB layer records these with
        # is_finite=0 rather than rejecting the write.
        assert math.isnan(stats["global_skew"])
        assert math.isnan(stats["global_kurtosis"])


class _RecordingDB:
    """Minimal stand-in for the DB singleton: records injection-stat rows. Since #277
    write_segment_stats batches everything through write_injection_stats_bulk; each row dict
    is recorded individually so assertions read the same as for the old per-row API."""

    def __init__(self):
        self.calls = []

    def write_injection_stat(self, **kwargs):
        self.calls.append(kwargs)

    def write_injection_stats_bulk(self, stats, tag=None):
        for row in stats:
            self.calls.append({**row, "tag": tag})


class TestWriteSegmentStats:
    def _sample_info(self, **extra):
        info = {
            "background_index": 0,
            "intensity_stats": {s: dict.fromkeys(_STAT_KEYS, 0.0) for s in ("A", "B", "C")},
            "signal_info": {},
            "slope_was_clamped": False,
        }
        info.update(extra)
        return info

    def _segment(self, stats_list):
        return {
            "round_number": 0,
            "chunk_number": 0,
            "signal_class": "true",
            "signal_type": "true_eti_rfi",
            "timestamp": 123.0,
            "stats_list": stats_list,
            "snr_range_floor": 10.0,
            "snr_range_ceil": 15.0,
            "num_samples": len(stats_list),
            "inject_duration": 1.0,
        }

    def test_emits_intersection_retry_rows_for_true_double_samples(self):
        db = _RecordingDB()
        sample = self._sample_info(intersection_retries=7, intersection_retry_capped=True)
        write_segment_stats(db, "test_tag", self._segment([sample]))
        rows = {c["stat_name"]: c for c in db.calls if c["stat_name"].startswith("intersection_")}
        assert set(rows) == {"intersection_retries", "intersection_retry_capped"}
        assert rows["intersection_retries"]["value"] == 7.0
        assert rows["intersection_retry_capped"]["value"] == 1.0
        for row in rows.values():
            assert row["sample_index"] == 0
            assert row["injection_stage"] is None
            assert row["signal_type"] == "true_eti_rfi"

    def test_no_retry_rows_for_samples_without_retry_keys(self):
        # create_false / create_true_single samples carry no retry telemetry — no rows.
        db = _RecordingDB()
        write_segment_stats(db, "test_tag", self._segment([self._sample_info()]))
        assert not [c for c in db.calls if c["stat_name"].startswith("intersection_")]
