# NOTE: come back to this later

"""Unit tests for aetherscan.preprocessing: hit deduplication, CSV cadence grouping, filename
sanitization, JSON coercion, DC-spike removal, and spline bandpass fitting."""

from __future__ import annotations

import numpy as np
import pytest

from aetherscan.config import get_config
from aetherscan.preprocessing import (
    DataPreprocessor,
    _fit_channel_bandpass,
    _read_coarse_channel_worker,
    _remove_dc_spike,
    group_observations_from_csv,
)


@pytest.fixture
def group_cols():
    """The real grouping columns from InferenceConfig — read from the config rather than
    duplicated, so these tests track the defaults the pipeline actually uses."""
    return list(get_config().inference.cadence_group_by_cols)


@pytest.fixture
def h5_col():
    return get_config().inference.cadence_h5_path_col


class TestDeduplicateHits:
    def test_empty_input(self):
        assert DataPreprocessor._deduplicate_hits([], stamp_width=512) == []

    def test_far_apart_hits_survive(self):
        hits = [(0, 5.0, 0.01), (300, 4.0, 0.01)]
        assert DataPreprocessor._deduplicate_hits(hits, stamp_width=512) == hits

    def test_nearby_hits_merge_keeping_higher_statistic(self):
        hits = [(0, 5.0, 0.01), (100, 9.0, 0.02)]
        assert DataPreprocessor._deduplicate_hits(hits, stamp_width=512) == [(100, 9.0, 0.02)]

    def test_merge_is_greedy_left_to_right(self):
        # 0 and 100 merge (keep 100 @ 9.0); 300 is then within 256 of 100 but weaker,
        # so it merges away too — one survivor.
        hits = [(0, 5.0, 0.01), (100, 9.0, 0.02), (300, 2.0, 0.03)]
        assert DataPreprocessor._deduplicate_hits(hits, stamp_width=512) == [(100, 9.0, 0.02)]

    def test_input_order_does_not_matter(self):
        hits = [(300, 2.0, 0.03), (0, 5.0, 0.01), (100, 9.0, 0.02)]
        assert DataPreprocessor._deduplicate_hits(hits, stamp_width=512) == [(100, 9.0, 0.02)]

    def test_boundary_distance_exactly_half_width_kept(self):
        # Merge condition is strict (<): a gap of exactly stamp_width // 2 keeps both.
        hits = [(0, 5.0, 0.01), (256, 4.0, 0.02)]
        assert DataPreprocessor._deduplicate_hits(hits, stamp_width=512) == hits


class TestGroupObservationsFromCsv:
    def test_valid_and_flagged_groups(self, make_inference_csv, group_cols, h5_col):
        key_a = {
            "Target": "HIP110750",
            "Session": "AGBT21B_999_31",
            "Band": "L",
            "Cadence ID": "0",
            "Frequency": "1400",
        }
        key_b = {**key_a, "Cadence ID": "1"}
        csv_path = make_inference_csv(
            "two_groups.csv",
            groups=[
                (key_a, [f"/data/a_{i}.h5" for i in range(6)]),
                (key_b, [f"/data/b_{i}.h5" for i in range(3)]),  # short cadence -> flagged
            ],
        )
        valid, flagged = group_observations_from_csv(
            str(csv_path), group_cols, h5_col, expected_obs=6
        )
        assert len(valid) == 1
        assert len(flagged) == 1
        assert valid[0].is_valid is True
        assert valid[0].key == tuple(key_a[c] for c in group_cols)
        # Row order within the group is preserved.
        assert valid[0].h5_paths == [f"/data/a_{i}.h5" for i in range(6)]
        assert flagged[0].is_valid is False
        assert len(flagged[0].h5_paths) == 3

    def test_missing_column_raises_keyerror(self, make_inference_csv, group_cols, h5_col):
        csv_path = make_inference_csv("ok.csv")
        with pytest.raises(KeyError, match="missing required column"):
            group_observations_from_csv(
                str(csv_path), [*group_cols, "Nonexistent"], h5_col, expected_obs=6
            )

    def test_missing_file_raises(self, group_cols, h5_col):
        with pytest.raises(FileNotFoundError):
            group_observations_from_csv("/nope.csv", group_cols, h5_col)

    def test_expected_obs_parameter(self, make_inference_csv, group_cols, h5_col):
        key = {
            "Target": "T",
            "Session": "S",
            "Band": "L",
            "Cadence ID": "0",
            "Frequency": "1",
        }
        csv_path = make_inference_csv("three.csv", groups=[(key, ["/a.h5", "/b.h5", "/c.h5"])])
        valid, flagged = group_observations_from_csv(
            str(csv_path), group_cols, h5_col, expected_obs=3
        )
        assert len(valid) == 1
        assert flagged == []


class TestCadenceNpyFilename:
    def test_clean_key_passes_through(self):
        name = DataPreprocessor._cadence_npy_filename("catalog", ("HIP110750", "L", "0"))
        assert name == "catalog_HIP110750_L_0.npy"

    def test_hostile_characters_collapse_to_underscore(self):
        name = DataPreprocessor._cadence_npy_filename(
            "catalog", ("HIP 110750", "AGBT/18A", "a;b|c", "$(rm -rf)")
        )
        assert "/" not in name
        assert " " not in name
        assert ";" not in name
        assert "$" not in name
        assert name.startswith("catalog_")
        assert name.endswith(".npy")

    def test_dash_and_dot_survive(self):
        name = DataPreprocessor._cadence_npy_filename("catalog", ("L-band", "1.5"))
        assert name == "catalog_L-band_1.5.npy"

    def test_non_string_key_components(self):
        name = DataPreprocessor._cadence_npy_filename("catalog", (0, 1400.5))
        assert name == "catalog_0_1400.5.npy"


class TestToJsonSafe:
    def test_bytes_decode(self):
        assert DataPreprocessor._to_json_safe(b"HIP110750") == "HIP110750"

    def test_numpy_scalars(self):
        assert DataPreprocessor._to_json_safe(np.float32(1.5)) == 1.5
        assert DataPreprocessor._to_json_safe(np.int64(7)) == 7
        assert DataPreprocessor._to_json_safe(np.bool_(True)) is True

    def test_ndarray_to_list(self):
        result = DataPreprocessor._to_json_safe(np.arange(3, dtype=np.int32))
        assert result == [0, 1, 2]
        assert all(isinstance(v, int) for v in result)

    def test_nested_structures_and_keys(self):
        obj = {
            1: {"vals": np.array([1.0, 2.0], dtype=np.float64)},
            "meta": (b"x", np.int16(2)),
        }
        result = DataPreprocessor._to_json_safe(obj)
        assert result == {"1": {"vals": [1.0, 2.0]}, "meta": ["x", 2]}

    def test_plain_values_pass_through(self):
        assert DataPreprocessor._to_json_safe("s") == "s"
        assert DataPreprocessor._to_json_safe(3.5) == 3.5
        assert DataPreprocessor._to_json_safe(None) is None


class TestRemoveDcSpike:
    def test_spike_replaced_by_neighbor_interpolation(self):
        rng = np.random.default_rng(9)
        coarse_width, n_coarse = 32, 3
        block = rng.chisquare(df=4, size=(4, n_coarse * coarse_width))
        original = block.copy()

        _remove_dc_spike(block, coarse_width, n_coarse)

        half = coarse_width // 2
        for i in range(n_coarse):
            dc = i * coarse_width + half
            np.testing.assert_allclose(
                block[:, dc], (original[:, dc + 1] + original[:, dc - 3]) / 2
            )
            np.testing.assert_allclose(
                block[:, dc - 1], (original[:, dc + 2] + original[:, dc - 2]) / 2
            )
            # Everything else in the channel is untouched.
            untouched = [
                j for j in range(i * coarse_width, (i + 1) * coarse_width) if j not in (dc, dc - 1)
            ]
            np.testing.assert_array_equal(block[:, untouched], original[:, untouched])

    def test_visible_spike_is_flattened(self):
        coarse_width = 64
        block = np.ones((4, coarse_width))
        dc = coarse_width // 2
        block[:, dc] = 100.0
        block[:, dc - 1] = 100.0
        _remove_dc_spike(block, coarse_width, 1)
        np.testing.assert_allclose(block, np.ones((4, coarse_width)))


class TestFitChannelBandpass:
    def test_smooth_bandpass_recovered(self):
        channel_width, spl_order = 1024, 16
        x = np.arange(channel_width)
        bandpass = 100.0 + 10.0 * np.sin(2 * np.pi * x / channel_width)
        fit = _fit_channel_bandpass(bandpass, channel_width, spl_order)
        assert fit.shape == bandpass.shape
        # A smooth curve must be recovered nearly exactly by the spline.
        assert np.max(np.abs(fit - bandpass)) < 0.1

    def test_subtracting_fit_flattens_channel(self):
        rng = np.random.default_rng(2)
        channel_width, spl_order = 1024, 16
        x = np.arange(channel_width)
        bandpass = 100.0 + 20.0 * np.cos(2 * np.pi * x / channel_width)
        noisy = bandpass + rng.normal(0, 0.5, size=channel_width)
        residual = noisy - _fit_channel_bandpass(noisy, channel_width, spl_order)
        # Residuals lose the 20-unit bandpass structure and keep only ~noise-scale variance.
        assert np.abs(residual.mean()) < 0.5
        assert residual.std() < 1.0


class TestReadCoarseChannelWorker:
    def test_reads_one_coarse_channel(self, make_h5_observation):
        n_chans, coarse_width = 2048, 512
        h5_path = make_h5_observation("obs.h5", n_chans=n_chans)
        channel = _read_coarse_channel_worker((str(h5_path), 2, coarse_width))
        assert channel.shape == (16, coarse_width)

        import h5py  # noqa: PLC0415

        with h5py.File(h5_path, "r") as hf:
            expected = hf["data"][:, 0, 2 * coarse_width : 3 * coarse_width]
        np.testing.assert_array_equal(channel, expected)
