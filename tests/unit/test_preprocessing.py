# NOTE: come back to this later

"""Unit tests for aetherscan.preprocessing: hit deduplication, CSV cadence grouping, filename
sanitization, JSON coercion, DC-spike removal, spline bandpass fitting, the vectorized
normality test (equivalence-gated against scipy.stats.normaltest), the fused energy-detection
worker, downsample-at-extraction, provenance derivation, and the legacy-vs-downsampled
load_inference_data paths."""

from __future__ import annotations

import functools
import json
import os

import numpy as np
import pytest
from scipy import stats
from skimage.transform import downscale_local_mean

from aetherscan.config import get_config
from aetherscan.data_generation import log_norm
from aetherscan.preprocessing import (
    DataPreprocessor,
    PendingCadence,
    _energy_detect_channel_worker,
    _extract_stamps_worker,
    _fit_channel_bandpass,
    _remove_dc_spike,
    _sliding_normality_k2,
    _spline_flatten_bandpass,
    derive_cadence_provenance,
    group_observations_from_csv,
)

_TESTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES_DIR = os.path.join(_TESTS_ROOT, "fixtures")


def _scipy_window_loop(
    channel: np.ndarray, window_size: int, step_size: int
) -> tuple[list[int], list[float]]:
    """The historical per-window normaltest loop (indices + statistics), used as the oracle."""
    indices, statistics = [], []
    for i in range(0, channel.shape[1] - window_size, step_size):
        s, _ = stats.normaltest(channel[:, i : i + window_size].flatten())
        indices.append(i)
        statistics.append(float(s))
    return indices, statistics


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


class TestDeriveCadenceProvenance:
    GROUP_COLS = ["Target", "Session", "Band", "Cadence ID", "Frequency"]

    def _metadata(self, **overrides):
        metadata = {
            "h5_paths": [f"/data/obs_{i}.h5" for i in range(6)],
            "header": {"tstart": 58306.36766203704, "fch1": 2251.46, "foff": -2.79e-06},
            "stamp_frequencies_mhz": [2250.0, 2249.5, 2249.0],
        }
        metadata.update(overrides)
        return metadata

    def test_full_mapping(self):
        key = ("DDO210", "AGBT18A_999_103", "L", "24777", "2251")
        prov = derive_cadence_provenance(key, self.GROUP_COLS, self._metadata())
        assert prov["target"] == "DDO210"
        assert prov["session"] == "AGBT18A_999_103"
        assert prov["band"] == "L"
        assert prov["cadence_id"] == 24777
        assert prov["timestamp_observed"] == pytest.approx(58306.36766203704)
        assert prov["h5_path"] == "/data/obs_0.h5"
        assert prov["stamp_frequencies_mhz"] == [2250.0, 2249.5, 2249.0]

    def test_column_matching_is_case_insensitive(self):
        cols = ["target", "SESSION", " Band ", "cadence id", "frequency"]
        key = ("T1", "S1", "X", "3", "1400")
        prov = derive_cadence_provenance(key, cols, self._metadata())
        assert prov["target"] == "T1"
        assert prov["session"] == "S1"
        assert prov["band"] == "X"
        assert prov["cadence_id"] == 3

    def test_unparseable_cadence_id_becomes_none(self):
        key = ("T", "S", "L", "not-a-number", "1400")
        prov = derive_cadence_provenance(key, self.GROUP_COLS, self._metadata())
        assert prov["cadence_id"] is None

    def test_missing_metadata_degrades_to_sparse(self):
        key = ("T", "S", "L", "1", "1400")
        prov = derive_cadence_provenance(key, self.GROUP_COLS, {})
        assert prov["target"] == "T"
        assert prov["timestamp_observed"] is None
        assert prov["h5_path"] is None
        assert prov["stamp_frequencies_mhz"] is None

    def test_unknown_columns_yield_none_fields(self):
        prov = derive_cadence_provenance(("a", "b"), ["ColX", "ColY"], self._metadata())
        assert prov["target"] is None
        assert prov["session"] is None
        assert prov["band"] is None
        assert prov["cadence_id"] is None


class TestExtractStampsWorkerDownsample:
    def _run_worker(self, tmp_path, make_h5_observation, downsample_factor):
        import h5py  # noqa: PLC0415

        time_bins, stamp_width = 16, 64
        stored_width = stamp_width // downsample_factor
        stamp_starts = [0, 512, 1000]
        h5_path = make_h5_observation("obs.h5", n_chans=2048)
        npy_path = str(tmp_path / f"stamps_x{downsample_factor}.npy")

        out = np.lib.format.open_memmap(
            npy_path, mode="w+", dtype=np.float32, shape=(3, 6, time_bins, stored_width)
        )
        out.flush()
        del out

        _extract_stamps_worker(
            (npy_path, 1, str(h5_path), stamp_starts, 0, time_bins, stamp_width, downsample_factor)
        )

        written = np.load(npy_path)
        with h5py.File(h5_path, "r") as hf:
            raw = [hf["data"][:time_bins, 0, s : s + stamp_width] for s in stamp_starts]
        return written, raw

    def test_downsampled_stamps_match_downscale_local_mean(self, tmp_path, make_h5_observation):
        factor = 8
        written, raw = self._run_worker(tmp_path, make_h5_observation, factor)
        for i, stamp in enumerate(raw):
            expected = downscale_local_mean(stamp, (1, factor)).astype(np.float32)
            np.testing.assert_array_equal(written[i, 1], expected)
        # Untouched obs slots stay zero
        assert np.all(written[:, 0] == 0)

    def test_factor_one_stores_raw_stamps(self, tmp_path, make_h5_observation):
        written, raw = self._run_worker(tmp_path, make_h5_observation, 1)
        for i, stamp in enumerate(raw):
            np.testing.assert_array_equal(written[i, 1], stamp)


@pytest.fixture
def initialized_runtime():
    """DataPreprocessor needs live db + manager singletons; conftest tears them down."""
    from aetherscan.db import init_db  # noqa: PLC0415
    from aetherscan.manager import init_manager  # noqa: PLC0415

    init_manager()
    init_db()


class TestProcessCadenceEndToEnd:
    """Sequential (no pool) end-to-end run of _process_cadence on synthetic .h5 observations
    with an injected non-Gaussian feature: exercises fused detection, dedup, overlap stamps,
    downsample-at-extraction, metadata, and the resume path."""

    def _make_cadence(self, tmp_path, n_chans=2048, inject_at=768):
        import h5py  # noqa: PLC0415

        rng = np.random.default_rng(23)
        h5_paths = []
        for obs in range(6):
            # Gaussian background (k2 stays far below threshold) with a strong, narrow
            # non-Gaussian spur injected into the ON files only
            data = rng.normal(1000.0, 10.0, size=(16, 1, n_chans)).astype(np.float32)
            if obs in (0, 2, 4):
                data[:, 0, inject_at : inject_at + 4] *= 50.0
            path = tmp_path / f"cad_obs_{obs}.h5"
            with h5py.File(path, "w") as hf:
                dset = hf.create_dataset("data", data=data)
                dset.attrs["fch1"] = 2251.46
                dset.attrs["foff"] = -2.7939677238464355e-06
                dset.attrs["nchans"] = n_chans
                dset.attrs["tstart"] = 58306.3676
            h5_paths.append(str(path))
        return h5_paths

    def _configure(self):
        config = get_config()
        config.manager.n_processes = 1  # sequential: no pools/shm in unit tests
        config.data.time_bins = 16
        config.data.width_bin = 256
        config.data.downsample_factor = 8
        config.inference.coarse_channel_width = 512
        config.inference.spline_order = 4
        config.inference.detection_window_size = 64
        config.inference.detection_step_size = 32
        config.inference.stat_threshold = 500.0
        config.inference.stamp_width = 256
        config.inference.overlap_search = True
        config.inference.overlap_fraction = 0.5
        return config

    def test_process_and_resume(self, tmp_path, initialized_runtime):
        from aetherscan.preprocessing import CadenceGroup  # noqa: PLC0415

        config = self._configure()
        h5_paths = self._make_cadence(tmp_path)
        group = CadenceGroup(
            key=("T1", "S1", "L", "7", "2251"),
            h5_paths=h5_paths,
            csv_path="unused.csv",
            expected_obs=6,
            is_valid=True,
        )
        npy_path = str(tmp_path / "out" / "cadence.npy")
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)

        preprocessor = DataPreprocessor()
        result = preprocessor.process_pending_cadence(PendingCadence(group, npy_path))

        assert result is not None
        assert result.npy_path == npy_path
        stamps = np.load(npy_path)
        stored_width = config.inference.stamp_width // config.data.downsample_factor
        assert stamps.ndim == 4
        assert stamps.shape[1:] == (6, 16, stored_width)
        assert stamps.shape[0] == result.n_hits
        assert np.all(np.isfinite(stamps))
        # Every observation slot was filled (positive noise -> strictly positive means)
        assert np.all(stamps > 0)

        with open(result.metadata_path) as f:
            metadata = json.load(f)
        assert metadata["stored_width"] == stored_width
        assert metadata["downsample_factor_applied"] == config.data.downsample_factor
        assert metadata["stamp_width"] == config.inference.stamp_width
        assert len(metadata["stamp_frequencies_mhz"]) == result.n_hits
        assert len(metadata["stamp_starts"]) == result.n_hits

        # Resume path: a second call must skip reprocessing and report the same hit count
        resumed = preprocessor.process_pending_cadence(PendingCadence(group, npy_path))
        assert resumed is not None
        assert resumed.n_hits == result.n_hits
        assert resumed.npy_path == npy_path

    def test_full_width_stamps_when_disabled(self, tmp_path, initialized_runtime):
        from aetherscan.preprocessing import CadenceGroup  # noqa: PLC0415

        config = self._configure()
        config.inference.store_downsampled_stamps = False
        h5_paths = self._make_cadence(tmp_path)
        group = CadenceGroup(
            key=("T2", "S1", "L", "8", "2251"),
            h5_paths=h5_paths,
            csv_path="unused.csv",
            expected_obs=6,
            is_valid=True,
        )
        npy_path = str(tmp_path / "out" / "cadence_full.npy")
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)

        result = DataPreprocessor().process_pending_cadence(PendingCadence(group, npy_path))

        assert result is not None
        stamps = np.load(npy_path)
        assert stamps.shape[1:] == (6, 16, config.inference.stamp_width)
        with open(result.metadata_path) as f:
            metadata = json.load(f)
        assert metadata["stored_width"] == config.inference.stamp_width
        assert metadata["downsample_factor_applied"] == 1


class TestLoadInferenceDataPaths:
    """load_inference_data must branch on the stored width: already-downsampled cadence .npy
    files get log-norm only; legacy full-width files keep downsample + log-norm."""

    def _configure(self):
        config = get_config()
        config.manager.n_processes = 1
        config.data.width_bin = 512
        config.data.downsample_factor = 8
        return config

    def test_downsampled_path_lognorm_only(self, tmp_path, initialized_runtime):
        config = self._configure()
        final_width = config.data.width_bin // config.data.downsample_factor
        rng = np.random.default_rng(31)
        arr = rng.chisquare(df=4, size=(3, 6, 16, final_width)).astype(np.float32)
        path = tmp_path / "downsampled.npy"
        np.save(path, arr)

        loaded = DataPreprocessor().load_inference_data(override_filepaths=[str(path)])

        assert loaded.shape == (3, 6, 16, final_width)
        for i in range(3):
            np.testing.assert_allclose(loaded[i], log_norm(arr[i]), rtol=1e-6)
        assert loaded.min() >= 0.0 and loaded.max() <= 1.0

    def test_legacy_full_width_path_downsamples_then_lognorms(self, tmp_path, initialized_runtime):
        config = self._configure()
        width_bin = config.data.width_bin
        factor = config.data.downsample_factor
        final_width = width_bin // factor
        rng = np.random.default_rng(37)
        arr = rng.chisquare(df=4, size=(2, 6, 16, width_bin)).astype(np.float32)
        path = tmp_path / "legacy.npy"
        np.save(path, arr)

        loaded = DataPreprocessor().load_inference_data(override_filepaths=[str(path)])

        assert loaded.shape == (2, 6, 16, final_width)
        for i in range(2):
            downsampled = np.stack(
                [
                    downscale_local_mean(arr[i, obs], (1, factor)).astype(np.float32)
                    for obs in range(6)
                ]
            )
            np.testing.assert_allclose(loaded[i], log_norm(downsampled), rtol=1e-6)

    def test_unrecognized_width_is_skipped(self, tmp_path, initialized_runtime):
        self._configure()
        arr = np.abs(np.random.default_rng(41).normal(1, 0.1, (2, 6, 16, 100))).astype(np.float32)
        path = tmp_path / "weird.npy"
        np.save(path, arr)

        with pytest.raises(ValueError, match="No data loaded successfully"):
            DataPreprocessor().load_inference_data(override_filepaths=[str(path)])

    def test_mixed_legacy_and_downsampled_files(self, tmp_path, initialized_runtime):
        config = self._configure()
        final_width = config.data.width_bin // config.data.downsample_factor
        rng = np.random.default_rng(43)
        legacy = rng.chisquare(df=4, size=(1, 6, 16, config.data.width_bin)).astype(np.float32)
        modern = rng.chisquare(df=4, size=(2, 6, 16, final_width)).astype(np.float32)
        legacy_path = tmp_path / "legacy.npy"
        modern_path = tmp_path / "modern.npy"
        np.save(legacy_path, legacy)
        np.save(modern_path, modern)

        loaded = DataPreprocessor().load_inference_data(
            override_filepaths=[str(legacy_path), str(modern_path)]
        )
        assert loaded.shape == (3, 6, 16, final_width)


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


class TestSlidingNormalityK2:
    """The §8.1 correctness gates: the vectorized k2 must match scipy.stats.normaltest
    per window to rtol=1e-9 across distributions, scales, and window/step geometries."""

    WINDOW_STEP_COMBOS = [
        (256, 128),  # default config: step divides window (fast path)
        (256, 256),  # window == step
        (96, 64),  # step does not divide window -> gcd (32) general path
        (100, 30),  # gcd 10 general path, ragged tail blocks
    ]

    @pytest.mark.parametrize(
        "name,dist",
        [
            ("normal", lambda rng, shape: rng.normal(0.0, 1.0, shape)),
            ("normal_offset", lambda rng, shape: rng.normal(5.0, 0.01, shape)),
            ("normal_tiny_scale", lambda rng, shape: rng.normal(0.0, 1e-3, shape)),
            ("laplace_large_scale", lambda rng, shape: rng.laplace(0.0, 1e3, shape)),
            ("uniform", lambda rng, shape: rng.uniform(-1.0, 1.0, shape)),
            ("chisquare_skewed", lambda rng, shape: rng.chisquare(4, shape)),
        ],
    )
    @pytest.mark.parametrize("window_size,step_size", WINDOW_STEP_COMBOS)
    def test_matches_scipy_normaltest_per_window(self, name, dist, window_size, step_size):
        rng = np.random.default_rng(11)
        channel = dist(rng, (16, 4096))

        k2 = _sliding_normality_k2(channel, window_size, step_size)
        indices, expected = _scipy_window_loop(channel, window_size, step_size)

        assert len(k2) == len(indices)
        np.testing.assert_allclose(k2, expected, rtol=1e-9)

    def test_float32_input_matches_scipy_on_float64_view(self):
        # The h5 data is float32, but the pipeline always tests the bandpass-subtracted
        # residuals, which are float64 (float32 channel - float64 spline fit). The oracle
        # therefore consumes the float64 view of the same values — scipy fed raw float32
        # would itself compute in float32 and diverge from its own float64 result.
        rng = np.random.default_rng(3)
        channel = rng.normal(0, 1, (16, 2048)).astype(np.float32)
        k2 = _sliding_normality_k2(channel, 256, 128)
        _, expected = _scipy_window_loop(channel.astype(np.float64), 256, 128)
        np.testing.assert_allclose(k2, expected, rtol=1e-9)

    def test_window_indices_match_historical_loop(self):
        # n_windows = len(range(0, width - window, step)): the window starting exactly at
        # width - window is excluded, like the historical loop.
        rng = np.random.default_rng(5)
        channel = rng.normal(0, 1, (16, 1024))
        k2 = _sliding_normality_k2(channel, 256, 128)
        assert len(k2) == len(range(0, 1024 - 256, 128))

    def test_zero_variance_window_yields_nan_not_hit(self):
        channel = np.ones((16, 1024))
        k2 = _sliding_normality_k2(channel, 256, 128)
        assert np.all(np.isnan(k2))
        # NaN never exceeds any threshold -> no spurious hits on degenerate data
        assert not np.any(k2 > 0.0)

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match=">= 8 samples"):
            _sliding_normality_k2(np.zeros((1, 16)), 4, 2)

    def test_empty_when_window_exceeds_width(self):
        assert _sliding_normality_k2(np.zeros((16, 128)), 256, 128).size == 0


@pytest.mark.slow
class TestRealSliceEquivalence:
    """Hit-set identity on recorded real data (a 16384-bin slice of one coarse channel of a
    Breakthrough Listen GBT .h5, centered on the channel's DC spike; the region is
    RFI-contaminated so it produces real hits)."""

    def test_identical_hits_on_real_slice(self):
        fixture = np.load(os.path.join(FIXTURES_DIR, "ed_real_slice.npz"))
        data = fixture["data"]
        assert data.shape == (16, 16384)

        width = data.shape[1]
        window_size, step_size, stat_threshold = 256, 128, 2048.0

        # Reproduce the per-channel pipeline: DC-spike removal (the slice is centered on the
        # spike, so treating it as one coarse channel puts the spike at width // 2, exactly
        # where _remove_dc_spike expects it) -> spline bandpass flatten -> threshold.
        channel = data.copy()
        _remove_dc_spike(channel, width, 1)
        residuals = _spline_flatten_bandpass(channel, spl_order=16)

        k2 = _sliding_normality_k2(residuals, window_size, step_size)
        vec_hits = {int(j) * step_size: float(k2[j]) for j in np.nonzero(k2 > stat_threshold)[0]}

        indices, statistics = _scipy_window_loop(residuals, window_size, step_size)
        scipy_hits = {i: s for i, s in zip(indices, statistics, strict=True) if s > stat_threshold}

        # The fixture region was chosen to contain real hits — an empty set would make this
        # test vacuous.
        assert len(scipy_hits) > 0
        assert set(vec_hits) == set(scipy_hits)
        np.testing.assert_allclose(
            [vec_hits[i] for i in sorted(vec_hits)],
            [scipy_hits[i] for i in sorted(scipy_hits)],
            rtol=1e-9,
        )


class TestEnergyDetectChannelWorker:
    def test_fused_worker_matches_composed_pipeline(self, make_h5_observation):
        """The fused read -> DC spike -> bandpass -> threshold worker must reproduce the
        explicitly composed per-stage pipeline on the same coarse channel."""
        import h5py  # noqa: PLC0415

        n_chans, coarse_width = 2048, 512
        channel_index = 2
        window_size, step_size = 64, 32
        stat_threshold = 20.0  # low threshold so chi-square noise produces hits
        spl_order = 4

        h5_path = make_h5_observation("obs.h5", n_chans=n_chans)
        bandpass_flatten = functools.partial(_spline_flatten_bandpass, spl_order=spl_order)

        hits = _energy_detect_channel_worker(
            (
                str(h5_path),
                channel_index,
                coarse_width,
                16,
                bandpass_flatten,
                window_size,
                step_size,
                stat_threshold,
            )
        )

        # Compose the same chain by hand
        with h5py.File(h5_path, "r") as hf:
            channel = hf["data"][
                :16, 0, channel_index * coarse_width : (channel_index + 1) * coarse_width
            ]
        _remove_dc_spike(channel, coarse_width, 1)
        residuals = bandpass_flatten(channel)
        indices, statistics = _scipy_window_loop(residuals, window_size, step_size)
        expected = {
            channel_index * coarse_width + i: s
            for i, s in zip(indices, statistics, strict=True)
            if s > stat_threshold
        }

        assert len(hits) > 0  # chi-square noise vs a low threshold must produce hits
        assert {idx for idx, _, _ in hits} == set(expected)
        for idx, stat_val, pval in hits:
            np.testing.assert_allclose(stat_val, expected[idx], rtol=1e-9)
            np.testing.assert_allclose(pval, stats.chi2.sf(stat_val, 2), rtol=1e-9)

    def test_absolute_indices_offset_by_channel_start(self, make_h5_observation):
        h5_path = make_h5_observation("obs.h5", n_chans=2048)
        coarse_width = 512
        bandpass_flatten = functools.partial(_spline_flatten_bandpass, spl_order=4)
        for channel_index in (0, 3):
            hits = _energy_detect_channel_worker(
                (str(h5_path), channel_index, coarse_width, 16, bandpass_flatten, 64, 32, 0.0)
            )
            starts = [idx for idx, _, _ in hits]
            assert all(
                channel_index * coarse_width <= s < (channel_index + 1) * coarse_width
                for s in starts
            )
