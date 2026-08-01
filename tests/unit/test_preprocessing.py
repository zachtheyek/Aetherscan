# NOTE: come back to this later

"""Unit tests for aetherscan.preprocessing: hit deduplication, CSV cadence grouping, filename
sanitization, JSON coercion, DC-spike removal, spline bandpass fitting, the vectorized
normality test (equivalence-gated against scipy.stats.normaltest), the fused energy-detection
worker, downsample-at-extraction, provenance derivation, and the legacy-vs-downsampled
load_inference_data paths."""

from __future__ import annotations

import functools
import json
import logging
import math
import os

import numpy as np
import pytest
from scipy import stats
from skimage.transform import downscale_local_mean

from aetherscan.config import get_config
from aetherscan.data_generation import log_norm
from aetherscan.pfb import edge_mid_power_ratio, equalize_passband, gen_coarse_channel_response
from aetherscan.preprocessing import (
    ED_STAT_HIST_EDGES,
    DataPreprocessor,
    PendingCadence,
    _chunk_cache_kwargs,
    _decimate_for_plot,
    _energy_detect_channel_worker,
    _extract_stamps_worker,
    _fit_channel_bandpass,
    _log_norm_chunk_vectorized,
    _lognorm_worker,
    _pfb_flatten_bandpass,
    _remove_dc_spike,
    _sliding_normality_k2,
    _spline_flatten_bandpass,
    derive_cadence_provenance,
    group_observations_from_csv,
)
from aetherscan.run_state import preprocessing_config_fingerprint

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

    def test_empty_catalog_returns_no_groups(self, make_inference_csv, group_cols, h5_col):
        # A header-only catalog is not a hard error at this layer: grouping degrades to two
        # empty lists (the loud "no work" failure is raised later, in the inference command).
        key = {
            "Target": "T",
            "Session": "S",
            "Band": "L",
            "Cadence ID": "0",
            "Frequency": "1",
        }
        csv_path = make_inference_csv("header_only.csv", groups=[(key, [])])
        valid, flagged = group_observations_from_csv(
            str(csv_path), group_cols, h5_col, expected_obs=6
        )
        assert valid == []
        assert flagged == []

    def test_headerless_catalog_raises_keyerror(self, tmp_path, group_cols, h5_col):
        # A catalog with no parseable header (e.g. an empty/truncated file) has none of the
        # required columns, so grouping fails loudly rather than silently yielding nothing.
        empty = tmp_path / "headerless.csv"
        empty.write_text("")
        with pytest.raises(KeyError, match="missing required column"):
            group_observations_from_csv(str(empty), group_cols, h5_col)

    def test_missing_path_row_skipped_with_warning(self, tmp_path, group_cols, h5_col, caplog):
        """A row whose h5-path cell is missing (ragged row -> csv.DictReader None) or blank is
        skipped with a warning and never grouped under a None/empty path, while the rest of the
        catalog is grouped normally (skip-and-continue)."""
        import csv as _csv  # noqa: PLC0415

        header = [*group_cols, h5_col]
        good_vals = [f"good_{i}" for i in range(len(group_cols))]
        bad_vals = [f"bad_{i}" for i in range(len(group_cols))]

        csv_path = tmp_path / "with_missing_path.csv"
        with open(csv_path, "w", newline="") as f:
            writer = _csv.writer(f)
            writer.writerow(header)
            for i in range(6):  # a complete, valid cadence
                writer.writerow([*good_vals, f"/data/good_{i}.h5"])
            writer.writerow([*bad_vals, ""])  # blank h5-path cell
            writer.writerow(bad_vals)  # ragged row: no h5-path field -> DictReader None

        with caplog.at_level(logging.WARNING, logger="aetherscan.preprocessing"):
            valid, flagged = group_observations_from_csv(
                str(csv_path), group_cols, h5_col, expected_obs=6
            )

        # The valid cadence survives intact; the bad key never entered the grouping at all.
        assert len(valid) == 1
        assert valid[0].key == tuple(good_vals)
        assert valid[0].h5_paths == [f"/data/good_{i}.h5" for i in range(6)]
        assert tuple(bad_vals) not in {g.key for g in (*valid, *flagged)}
        # Both malformed rows were warned about.
        missing_warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "missing/empty" in r.getMessage()
        ]
        assert len(missing_warnings) == 2


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


class TestLogNormChunkVectorized:
    """#298 I5: the vectorized chunk log-norm must reproduce _lognorm_worker bit-for-bit —
    same validity decisions, same float32 arithmetic, same zero-range guard."""

    def _chunk(self, dtype=np.float32):
        rng = np.random.default_rng(31)
        chunk = rng.chisquare(df=4, size=(7, 6, 4, 16)).astype(dtype)
        chunk[1, 0, 0, 0] = np.nan  # invalid: NaN
        chunk[3, 2, 1, 5] = np.inf  # invalid: Inf
        chunk[5] = 0.0  # invalid: non-positive max
        chunk[6] = 2.5  # valid but constant: zero range after the log shift
        return chunk

    def _worker_reference(self, chunk, monkeypatch):
        import aetherscan.preprocessing as preprocessing_module  # noqa: PLC0415

        monkeypatch.setattr(preprocessing_module, "_GLOBAL_CHUNK_DATA", chunk)
        rows, valid = [], []
        for i in range(len(chunk)):
            result = _lognorm_worker((i,))
            valid.append(result is not None)
            if result is not None:
                rows.append(result)
        return np.array(rows, dtype=np.float32), np.array(valid)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_matches_lognorm_worker_bitwise(self, dtype, monkeypatch):
        chunk = self._chunk(dtype)
        expected_rows, expected_valid = self._worker_reference(chunk, monkeypatch)

        normalized, valid = _log_norm_chunk_vectorized(chunk)

        np.testing.assert_array_equal(valid, expected_valid)
        np.testing.assert_array_equal(normalized, expected_rows)
        assert normalized.dtype == np.float32

    def test_all_invalid_chunk_returns_empty(self):
        chunk = np.zeros((3, 6, 4, 16), dtype=np.float32)
        normalized, valid = _log_norm_chunk_vectorized(chunk)
        assert len(normalized) == 0
        assert not valid.any()


class TestChunkCacheKwargs:
    def test_chunked_compressed_dataset_gets_sized_cache(self, make_h5_observation):
        # 16 chunks of (1, 1, 256) float32 = 1 KiB each -> stripe far below the 1 MiB
        # default, so force visibility by checking the sizing arithmetic directly
        path = make_h5_observation("chunked.h5", n_chans=2048, chunks=(1, 1, 256))
        kwargs = _chunk_cache_kwargs(str(path), time_bins=16)
        # Tiny chunks: default cache already covers the stripe -> no override
        assert kwargs == {}

    def test_large_chunks_get_cache_override(self, make_h5_observation):
        # One full-width chunk per time row (256 KiB each x 16 rows = 4 MiB stripe):
        # exceeds the 1 MiB default, so the cache must be sized up
        path = make_h5_observation("bigchunks.h5", n_chans=65536, chunks=(1, 1, 65536))
        kwargs = _chunk_cache_kwargs(str(path), time_bins=16)
        stripe = 16 * 65536 * 4
        assert kwargs["rdcc_nbytes"] >= stripe
        assert kwargs["rdcc_nbytes"] <= 512 * 1024**2
        assert kwargs["rdcc_w0"] == 0.0
        assert kwargs["rdcc_nslots"] > 0

    def test_contiguous_dataset_returns_defaults(self, make_h5_observation):
        path = make_h5_observation("contig.h5", n_chans=2048)
        assert _chunk_cache_kwargs(str(path), time_bins=16) == {}

    def test_missing_file_returns_defaults(self, tmp_path):
        assert _chunk_cache_kwargs(str(tmp_path / "nope.h5"), time_bins=16) == {}


class TestExtractStampsWorkerDownsample:
    def _run_worker(
        self, tmp_path, make_h5_observation, downsample_factor, stamp_starts=None, **h5_kwargs
    ):
        import h5py  # noqa: PLC0415

        time_bins, stamp_width = 16, 64
        stored_width = stamp_width // downsample_factor
        stamp_starts = stamp_starts if stamp_starts is not None else [0, 512, 1000]
        h5_path = make_h5_observation("obs.h5", n_chans=2048, **h5_kwargs)
        npy_path = str(tmp_path / f"stamps_x{downsample_factor}.npy")

        out = np.lib.format.open_memmap(
            npy_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(stamp_starts), 6, time_bins, stored_width),
        )
        out.flush()
        del out

        # open_kwargs is computed once per obs file by the parent and shipped in the task
        # args (#301) — mirror that here
        _extract_stamps_worker(
            (
                npy_path,
                1,
                str(h5_path),
                stamp_starts,
                0,
                time_bins,
                stamp_width,
                downsample_factor,
                _chunk_cache_kwargs(str(h5_path), time_bins),
            )
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

    def test_chunked_compressed_input_is_byte_identical(self, tmp_path, make_h5_observation):
        # The chunk-cache sizing is pure cache tuning: a chunked+compressed input (gzip
        # stands in for bitshuffle) must produce byte-identical stamps to the raw reads
        written, raw = self._run_worker(
            tmp_path, make_h5_observation, 8, chunks=(1, 1, 256), compression="gzip"
        )
        for i, stamp in enumerate(raw):
            expected = downscale_local_mean(stamp, (1, 8)).astype(np.float32)
            np.testing.assert_array_equal(written[i, 1], expected)

    def test_overlapping_windows_coalesce_byte_identical(self, tmp_path, make_h5_observation):
        """#301 fetches overlapping/abutting windows (the overlap_search triplet shape:
        gaps of stamp_width // 2) as one wide read sliced per stamp — the written stamps
        must be byte-identical to independent per-stamp reads, for both raw and
        downsampled storage."""
        overlapping = [0, 32, 64, 512, 544, 1000]  # two triplet-shaped chains + a loner
        for factor in (1, 8):
            written, raw = self._run_worker(
                tmp_path, make_h5_observation, factor, stamp_starts=overlapping
            )
            for i, stamp in enumerate(raw):
                expected = (
                    stamp
                    if factor == 1
                    else downscale_local_mean(stamp, (1, factor)).astype(np.float32)
                )
                np.testing.assert_array_equal(written[i, 1], expected)

    def test_stale_tmp_sweep_covers_all_tmp_suffixes(self, tmp_path, monkeypatch):
        """#305: the age-gated sweep must catch the metadata '.json.<pid>.<hex>.tmp' and
        candidate-sidecar '.candidates.npz.<pid>.<hex>.tmp' orphans, not just the stamp
        '.tmp.npy' — a crash between np.savez/json.dump and os.replace otherwise leaks them
        forever in the shared cache dir."""
        import glob as _glob  # noqa: PLC0415
        import time as _time  # noqa: PLC0415

        from aetherscan.preprocessing import _STALE_TMP_MAX_AGE_S  # noqa: PLC0415

        stem = str(tmp_path / "cad")
        old = _time.time() - _STALE_TMP_MAX_AGE_S - 100
        fresh = _time.time()
        tmps = {
            f"{stem}.12345.abcdef.tmp.npy": old,  # stamp memmap tmp (already swept)
            f"{stem}.json.12345.abcdef.tmp": old,  # metadata tmp (was leaked)
            f"{stem}.candidates.npz.12345.abcdef.tmp": old,  # #302 sidecar tmp (was leaked)
            f"{stem}.99999.fedcba.tmp": fresh,  # a live concurrent run's tmp — must survive
        }
        for path, mtime in tmps.items():
            open(path, "wb").close()
            os.utime(path, (mtime, mtime))

        # Exercise the exact sweep expression from _process_cadence
        stale_cutoff = _time.time() - _STALE_TMP_MAX_AGE_S
        escaped = _glob.escape(stem)
        swept = (
            _glob.glob(f"{escaped}.*.tmp.npy")
            + _glob.glob(f"{escaped}.*.tmp")
            + [f"{stem}.tmp.npy"]
        )
        for stale in swept:
            if os.path.exists(stale) and os.path.getmtime(stale) < stale_cutoff:
                os.remove(stale)

        assert not os.path.exists(f"{stem}.json.12345.abcdef.tmp")
        assert not os.path.exists(f"{stem}.candidates.npz.12345.abcdef.tmp")
        assert not os.path.exists(f"{stem}.12345.abcdef.tmp.npy")
        assert os.path.exists(f"{stem}.99999.fedcba.tmp")  # fresh tmp untouched

    def test_coalesce_grouping_rules(self):
        from aetherscan.preprocessing import (  # noqa: PLC0415
            _COALESCE_MAX_BINS,
            _coalesce_stamp_groups,
        )

        # Empty input hardening (review note on #305): no [[0]] seed group
        assert _coalesce_stamp_groups([], 64) == []
        # Overlap/abut chains group; disjoint windows stay singletons
        assert _coalesce_stamp_groups([0, 32, 64, 512], 64) == [[0, 1, 2], [3]]
        # Abutting exactly (next start == previous end) still groups
        assert _coalesce_stamp_groups([0, 64, 128], 64) == [[0, 1, 2]]
        # A gap of one bin splits
        assert _coalesce_stamp_groups([0, 65], 64) == [[0], [1]]
        # The span cap breaks an otherwise-endless chain
        starts = list(range(0, _COALESCE_MAX_BINS + 4096, 32))
        groups = _coalesce_stamp_groups(starts, 64)
        assert len(groups) > 1
        for group in groups:
            assert starts[group[-1]] + 64 - starts[group[0]] <= _COALESCE_MAX_BINS


@pytest.fixture
def initialized_runtime():
    """DataPreprocessor needs live db + manager singletons; conftest tears them down.
    Returns the Database so tests can flush/query the run manifest."""
    from aetherscan.db import init_db  # noqa: PLC0415
    from aetherscan.manager import init_manager  # noqa: PLC0415

    init_manager()
    return init_db()


class TestProcessCadenceEndToEnd:
    """Sequential (no pool) end-to-end run of _process_cadence on synthetic .h5 observations
    with an injected non-Gaussian feature: exercises fused detection, both bandpass
    flatteners (spline on flat data, PFB on PFB-shaped data), dedup, overlap stamps,
    downsample-at-extraction, metadata, the debug overlay plot, and the resume path."""

    def _make_cadence(self, tmp_path, n_chans=2048, inject_at=768, pfb_shaped=False):
        import h5py  # noqa: PLC0415

        rng = np.random.default_rng(23)
        h5_paths = []
        for obs in range(6):
            # Gaussian background (k2 stays far below threshold) with a strong, narrow
            # non-Gaussian spur injected into the ON files only
            data = rng.normal(1000.0, 10.0, size=(16, 1, n_chans)).astype(np.float32)
            if obs in (0, 2, 4):
                data[:, 0, inject_at : inject_at + 4] *= 50.0
            if pfb_shaped:
                # Imprint the instrument passband the PFB flattener expects to divide out
                # (a flat recording would look like a static-response mismatch instead)
                response = gen_coarse_channel_response(512, n_chans // 512, 12)
                data *= np.tile(response, n_chans // 512).astype(np.float32)
            path = tmp_path / f"cad_obs_{obs}.h5"
            with h5py.File(path, "w") as hf:
                dset = hf.create_dataset("data", data=data)
                dset.attrs["fch1"] = 2251.46
                dset.attrs["foff"] = -2.7939677238464355e-06
                dset.attrs["nchans"] = n_chans
                dset.attrs["tstart"] = 58306.3676
            h5_paths.append(str(path))
        return h5_paths

    def _configure(self, bandpass_method="spline"):
        config = get_config()
        # In production main() resolves save_tag to {command}_{datetime} before any stage runs
        # (the dataclass default is None); mirror that here so tag-scoped output paths (e.g. the
        # bandpass debug overlay under plots/inference/{save_tag}/) resolve to a real directory.
        config.checkpoint.save_tag = "inf_20260101_120000"
        config.manager.n_processes = 1  # sequential: no pools/shm in unit tests
        config.data.time_bins = 16
        config.data.width_bin = 256
        config.data.downsample_factor = 8
        config.inference.coarse_channel_width = 512
        config.inference.bandpass_method = bandpass_method
        config.inference.spline_order = 4
        config.inference.detection_window_size = 64
        config.inference.detection_step_size = 32
        config.inference.stat_threshold = 500.0
        config.inference.stamp_width = 256
        config.inference.overlap_search = True
        config.inference.overlap_fraction = 0.5
        return config

    @pytest.mark.parametrize(("bandpass_method", "pfb_shaped"), [("spline", False), ("pfb", True)])
    def test_process_and_resume(self, tmp_path, initialized_runtime, bandpass_method, pfb_shaped):
        from aetherscan.preprocessing import CadenceGroup  # noqa: PLC0415

        config = self._configure(bandpass_method)
        h5_paths = self._make_cadence(tmp_path, pfb_shaped=pfb_shaped)
        group = CadenceGroup(
            key=("T1", "S1", "L", "7", "2251"),
            h5_paths=h5_paths,
            csv_path="unused.csv",
            expected_obs=6,
            is_valid=True,
        )
        npy_path = str(tmp_path / "out" / "cadence.npy")
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)

        # Abandoned partial outputs are age-swept (#298 I3: tmp names are per-run-unique in
        # the shared cache dir, so only AGE proves abandonment): an expired tmp — legacy
        # fixed name or unique-name — is removed; a fresh unique-name tmp may be a live
        # concurrent run's in-progress write and must survive.
        import time as time_module  # noqa: PLC0415

        expired = (time_module.time() - 25 * 3600,) * 2
        stale_tmp = os.path.splitext(npy_path)[0] + ".tmp.npy"
        with open(stale_tmp, "wb") as f:
            f.write(b"junk from a SIGKILLed pre-#298 run")
        os.utime(stale_tmp, expired)
        stale_unique_tmp = os.path.splitext(npy_path)[0] + ".12345.deadbeef.tmp.npy"
        with open(stale_unique_tmp, "wb") as f:
            f.write(b"junk from a SIGKILLed run")
        os.utime(stale_unique_tmp, expired)
        fresh_tmp = os.path.splitext(npy_path)[0] + ".67890.cafebabe.tmp.npy"
        with open(fresh_tmp, "wb") as f:
            f.write(b"a live concurrent run's in-progress write")

        preprocessor = DataPreprocessor()
        result = preprocessor.process_pending_cadence(PendingCadence(group, npy_path))

        assert not os.path.exists(stale_tmp)
        assert not os.path.exists(stale_unique_tmp)
        assert os.path.exists(fresh_tmp)
        os.remove(fresh_tmp)
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

        # Viz-suite provenance: per-ON-file all-window statistic histograms on the shared
        # bins, the pre-binned hit-frequency histograms (#301 — the raw per-hit list is no
        # longer stored), the post-dedup merged list, and the bandpass envelopes
        ed_hist = metadata["ed_stat_hist"]
        assert ed_hist["bin_edges"] == pytest.approx(list(ED_STAT_HIST_EDGES))
        assert len(ed_hist["counts_per_on_file"]) == 3  # one histogram per ON file
        assert all(sum(counts) > 0 for counts in ed_hist["counts_per_on_file"])
        assert metadata["n_raw_hits"] >= metadata["n_merged_hits"] > 0
        assert "raw_hit_frequencies_mhz" not in metadata
        hit_hist = metadata["hit_spectrum_hist"]
        assert sum(hit_hist["raw_counts"]) == metadata["n_raw_hits"]
        assert sum(hit_hist["merged_counts"]) == metadata["n_merged_hits"]
        assert hit_hist["freq_lo"] <= hit_hist["raw_freq_min"] <= hit_hist["raw_freq_max"]
        assert len(metadata["merged_hit_frequencies_mhz"]) == metadata["n_merged_hits"]
        # Envelopes: one entry per sampled channel, three decimated lines each, exact
        # per the commuting-mean argument (raw/H for pfb, raw - fit for spline)
        envelopes = metadata["bandpass_envelopes"]
        assert envelopes and all(
            set(entry) >= {"channel", "overlay_label", "raw", "flat", "overlay"}
            for entry in envelopes
        )

        # Preprocessing completion is recorded in the inference_cadences run manifest
        db = initialized_runtime
        assert db.flush(timeout=10) is True
        manifest = db.query_inference_cadences(tag=config.checkpoint.save_tag, npy_path=npy_path)
        assert [r["status"] for r in manifest] == ["preprocessed"]
        assert manifest[0]["n_stamps"] == result.n_hits
        assert manifest[0]["csv_path"] == "unused.csv"
        assert json.loads(manifest[0]["cadence_key"]) == ["T1", "S1", "L", "7", "2251"]
        assert manifest[0]["duration_s"] > 0

        # Fresh extraction is flagged prunable (#302)
        assert result.freshly_extracted is True

        # Resume path: a second call must skip reprocessing and report the same hit count,
        # without duplicating the manifest row. Because THIS preprocessor extracted the
        # .npy, its resume is still prunable (#305 disk-leak fix: a failed-then-retried
        # cadence of the same run must not escape pruning forever).
        resumed = preprocessor.process_pending_cadence(PendingCadence(group, npy_path))
        assert resumed is not None
        assert resumed.n_hits == result.n_hits
        assert resumed.npy_path == npy_path
        assert resumed.freshly_extracted is True

        # A DIFFERENT preprocessor (a genuinely handed cache — separate process/operator)
        # resuming the same .npy must NOT be prunable: its extracted-set is empty.
        handed = DataPreprocessor().process_pending_cadence(PendingCadence(group, npy_path))
        assert handed is not None
        assert handed.freshly_extracted is False
        assert db.flush(timeout=10) is True
        manifest = db.query_inference_cadences(tag=config.checkpoint.save_tag, npy_path=npy_path)
        assert len(manifest) == 1

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

    @pytest.mark.parametrize("missing_attr", ["foff", "fch1"])
    def test_missing_header_attr_raises_keyerror(self, tmp_path, initialized_runtime, missing_attr):
        """A cadence whose primary ON-source .h5 lacks a required header attr (fch1/foff)
        fails loudly: _process_cadence reads them unguarded to derive stamp frequencies."""
        import h5py  # noqa: PLC0415

        from aetherscan.preprocessing import CadenceGroup  # noqa: PLC0415

        self._configure()
        h5_paths = self._make_cadence(tmp_path)
        # Tamper the primary ON-source file (position 0 in ABACAD), whose header is read.
        with h5py.File(h5_paths[0], "r+") as hf:
            del hf["data"].attrs[missing_attr]
        group = CadenceGroup(
            key=("T3", "S1", "L", "9", "2251"),
            h5_paths=h5_paths,
            csv_path="unused.csv",
            expected_obs=6,
            is_valid=True,
        )
        npy_path = str(tmp_path / "out" / f"cad_no_{missing_attr}.npy")
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)

        with pytest.raises(KeyError, match=missing_attr):
            DataPreprocessor()._process_cadence(group, npy_path)

    @staticmethod
    def _rewrite_rank(h5_path, shape):
        """Replace one observation file's 'data' with an array of the given (wrong) shape,
        keeping the original header attrs so only the rank is invalid."""
        import h5py  # noqa: PLC0415

        with h5py.File(h5_path, "r+") as hf:
            attrs = dict(hf["data"].attrs)
            del hf["data"]
            dset = hf.create_dataset("data", data=np.ones(shape, dtype=np.float32))
            for k, v in attrs.items():
                dset.attrs[k] = v

    @pytest.mark.parametrize("bad_shape", [(16, 2048), (16, 1, 2048, 1)])
    def test_wrong_rank_data_raises_valueerror(self, tmp_path, initialized_runtime, bad_shape):
        """A primary ON-source .h5 whose 'data' dataset is not rank-3 is rejected up front with a
        clear ValueError (file path + expected-vs-actual rank), instead of a cryptic IndexError
        deep inside an energy-detection worker."""
        from aetherscan.preprocessing import CadenceGroup  # noqa: PLC0415

        self._configure()
        h5_paths = self._make_cadence(tmp_path)
        self._rewrite_rank(h5_paths[0], bad_shape)
        group = CadenceGroup(
            key=("T4", "S1", "L", "10", "2251"),
            h5_paths=h5_paths,
            csv_path="unused.csv",
            expected_obs=6,
            is_valid=True,
        )
        npy_path = str(tmp_path / "out" / "cad_bad_rank.npy")
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)

        with pytest.raises(ValueError, match=r"expected rank 3"):
            DataPreprocessor()._process_cadence(group, npy_path)

    def test_wrong_rank_in_non_primary_file_rejected_up_front(self, tmp_path, initialized_runtime):
        """The rank check covers all 6 observation files, not just the primary: a wrong-rank
        OFF file (index 3) is rejected with the same up-front ValueError naming the file."""
        from aetherscan.preprocessing import CadenceGroup  # noqa: PLC0415

        self._configure()
        h5_paths = self._make_cadence(tmp_path)
        self._rewrite_rank(h5_paths[3], (16, 2048))
        group = CadenceGroup(
            key=("T7", "S1", "L", "13", "2251"),
            h5_paths=h5_paths,
            csv_path="unused.csv",
            expected_obs=6,
            is_valid=True,
        )
        npy_path = str(tmp_path / "out" / "cad_bad_rank_off.npy")
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)

        with pytest.raises(ValueError, match=r"expected rank 3") as exc_info:
            DataPreprocessor()._process_cadence(group, npy_path)
        assert h5_paths[3] in str(exc_info.value)

    @pytest.mark.parametrize("short_obs_idx", [1, 2])
    def test_short_time_bins_in_any_file_skips_whole_cadence(
        self, tmp_path, initialized_runtime, caplog, short_obs_idx
    ):
        """One observation file with fewer than time_bins rows — an OFF file (index 1) or a
        non-primary ON file (index 2) — skips the whole cadence up front with a warning
        naming the file, instead of failing mid-extraction with a broadcast ValueError (and,
        for a short ON file, silently degrading the k2 statistic first). The cadence is
        dropped whole because the (n, 6, time_bins, width) stamp tensor and the downstream
        num_observations=6 contract have no representation for a 5-observation cadence."""
        import h5py  # noqa: PLC0415

        from aetherscan.preprocessing import CadenceGroup  # noqa: PLC0415

        config = self._configure()
        h5_paths = self._make_cadence(tmp_path)
        short_path = h5_paths[short_obs_idx]
        with h5py.File(short_path, "r+") as hf:
            attrs = dict(hf["data"].attrs)
            short_data = hf["data"][: config.data.time_bins - 4]
            del hf["data"]
            dset = hf.create_dataset("data", data=short_data)
            for k, v in attrs.items():
                dset.attrs[k] = v
        group = CadenceGroup(
            key=("T6", "S1", "L", "12", "2251"),
            h5_paths=h5_paths,
            csv_path="unused.csv",
            expected_obs=6,
            is_valid=True,
        )
        npy_path = str(tmp_path / "out" / "cad_short.npy")
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)

        with caplog.at_level(logging.WARNING, logger="aetherscan.preprocessing"):
            result = DataPreprocessor().process_pending_cadence(PendingCadence(group, npy_path))

        assert result is None
        assert not os.path.exists(npy_path)
        # The skip happens before extraction, so no partial .tmp.npy is ever created
        assert not os.path.exists(os.path.splitext(npy_path)[0] + ".tmp.npy")
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any(short_path in m and "time bins" in m for m in warnings)

    @pytest.mark.parametrize("channel_deficit", [64, 1])
    def test_narrow_frequency_width_in_any_file_skips_whole_cadence(
        self, tmp_path, initialized_runtime, caplog, channel_deficit
    ):
        """An observation file with FEWER channels than the [0:n_chans] read window would
        truncate into a short stamp — a broadcast ValueError deep in a pool worker — so the
        up-front geometry check turns it into a clean skip-and-warn naming the file."""
        import h5py  # noqa: PLC0415

        from aetherscan.preprocessing import CadenceGroup  # noqa: PLC0415

        self._configure()
        h5_paths = self._make_cadence(tmp_path)
        off_path = h5_paths[1]
        with h5py.File(off_path, "r+") as hf:
            attrs = dict(hf["data"].attrs)
            t, p, w = hf["data"].shape
            del hf["data"]
            dset = hf.create_dataset(
                "data", data=np.ones((t, p, w - channel_deficit), dtype=np.float32)
            )
            for k, v in attrs.items():
                dset.attrs[k] = v
        group = CadenceGroup(
            key=("T8", "S1", "L", "14", "2251"),
            h5_paths=h5_paths,
            csv_path="unused.csv",
            expected_obs=6,
            is_valid=True,
        )
        npy_path = str(tmp_path / "out" / "cad_narrow_width.npy")
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)

        with caplog.at_level(logging.WARNING, logger="aetherscan.preprocessing"):
            result = DataPreprocessor().process_pending_cadence(PendingCadence(group, npy_path))

        assert result is None
        assert not os.path.exists(npy_path)
        assert not os.path.exists(os.path.splitext(npy_path)[0] + ".tmp.npy")
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any(off_path in m and "read window" in m for m in warnings)

    def test_wider_off_file_is_not_rejected(self, tmp_path, initialized_runtime, caplog):
        """A file WIDER than the n_chans read window reads cleanly ([0:n_chans]) and processed
        on master, so the width guard deliberately leaves it alone — the check is `< n_chans`,
        not strict equality. Only genuinely narrow files (the crash case) are skipped."""
        import h5py  # noqa: PLC0415

        from aetherscan.preprocessing import CadenceGroup  # noqa: PLC0415

        self._configure()
        h5_paths = self._make_cadence(tmp_path)
        off_path = h5_paths[1]
        with h5py.File(off_path, "r+") as hf:
            attrs = dict(hf["data"].attrs)  # nchans stays 2048 -> read window unchanged
            data = hf["data"][:]
            del hf["data"]
            pad = np.random.default_rng(7).normal(
                1000.0, 10.0, size=(data.shape[0], data.shape[1], 64)
            )
            wider = np.concatenate([data, pad.astype(np.float32)], axis=-1)
            dset = hf.create_dataset("data", data=wider)
            for k, v in attrs.items():
                dset.attrs[k] = v
        group = CadenceGroup(
            key=("T9", "S1", "L", "15", "2251"),
            h5_paths=h5_paths,
            csv_path="unused.csv",
            expected_obs=6,
            is_valid=True,
        )
        npy_path = str(tmp_path / "out" / "cad_wider.npy")
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)

        with caplog.at_level(logging.WARNING, logger="aetherscan.preprocessing"):
            DataPreprocessor().process_pending_cadence(PendingCadence(group, npy_path))

        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("read window" in m for m in warnings)  # width guard did NOT fire
        assert os.path.exists(npy_path)  # processed through to writing stamps, like master

    def test_wrong_rank_data_is_skipped_not_fatal(self, tmp_path, initialized_runtime):
        """process_pending_cadence swallows the rank ValueError into a logged skip (returns
        None, no .npy written), so one malformed cadence can't abort a large-catalog run."""
        from aetherscan.preprocessing import CadenceGroup  # noqa: PLC0415

        self._configure()
        h5_paths = self._make_cadence(tmp_path)
        self._rewrite_rank(h5_paths[0], (16, 2048))
        group = CadenceGroup(
            key=("T5", "S1", "L", "11", "2251"),
            h5_paths=h5_paths,
            csv_path="unused.csv",
            expected_obs=6,
            is_valid=True,
        )
        npy_path = str(tmp_path / "out" / "cad_bad_rank_skip.npy")
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)

        result = DataPreprocessor().process_pending_cadence(PendingCadence(group, npy_path))
        assert result is None
        assert not os.path.exists(npy_path)

    def test_bandpass_debug_plot_saved(self, tmp_path, initialized_runtime):
        from aetherscan.preprocessing import CadenceGroup  # noqa: PLC0415

        config = self._configure("pfb")
        config.inference.bandpass_debug_plot = True
        h5_paths = self._make_cadence(tmp_path, pfb_shaped=True)
        group = CadenceGroup(
            key=("T3", "S1", "L", "9", "2251"),
            h5_paths=h5_paths,
            csv_path="unused.csv",
            expected_obs=6,
            is_valid=True,
        )
        npy_path = str(tmp_path / "out" / "cadence_dbg.npy")
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)

        result = DataPreprocessor().process_pending_cadence(PendingCadence(group, npy_path))

        assert result is not None
        plot_path = os.path.join(
            config.output_path,
            "plots",
            "inference",
            config.checkpoint.save_tag,
            f"bandpass_overlay_cadence_dbg_{config.checkpoint.save_tag}.png",
        )
        assert os.path.exists(plot_path)
        assert os.path.getsize(plot_path) > 0


class TestBandpassFlattenerSelection:
    """_get_bandpass_flattener must route on config.inference.bandpass_method (pfb default),
    materialize the parent-computed response as a sidecar .npy carried by path in the PFB
    partial (workers must never generate the response themselves — at GBT scale that is an
    ~n_chans-point FFT per worker, concurrently), fall back to spline when the fold is
    impossible, and reject unknown methods."""

    def test_default_is_pfb(self, initialized_runtime):
        config = get_config()
        config.inference.coarse_channel_width = 512
        assert config.inference.bandpass_method == "pfb"
        flattener = DataPreprocessor()._get_bandpass_flattener(num_coarse_channels=4)
        assert flattener.func is _pfb_flatten_bandpass
        response_path = flattener.keywords["response_path"]
        # The sidecar file lives under the run's output path and holds exactly the response
        # the parent computed for (width, coarse count, taps)
        assert response_path.startswith(os.path.join(config.output_path, "cache", "pfb"))
        expected = gen_coarse_channel_response(512, 4, config.inference.pfb_taps_per_channel)
        np.testing.assert_array_equal(np.load(response_path), expected)

    def test_response_file_reused_across_calls(self, initialized_runtime):
        config = get_config()
        config.inference.coarse_channel_width = 512
        preprocessor = DataPreprocessor()
        first = preprocessor._get_bandpass_flattener(num_coarse_channels=4)
        mtime = os.path.getmtime(first.keywords["response_path"])
        second = preprocessor._get_bandpass_flattener(num_coarse_channels=4)
        assert second.keywords["response_path"] == first.keywords["response_path"]
        assert os.path.getmtime(second.keywords["response_path"]) == mtime  # not rewritten

    def test_corrupt_response_file_is_rewritten(self, initialized_runtime):
        config = get_config()
        config.inference.coarse_channel_width = 512
        preprocessor = DataPreprocessor()
        response_path = preprocessor._get_bandpass_flattener(num_coarse_channels=4).keywords[
            "response_path"
        ]
        with open(response_path, "wb") as f:
            f.write(b"garbage")
        response_path_2 = preprocessor._get_bandpass_flattener(num_coarse_channels=4).keywords[
            "response_path"
        ]
        assert response_path_2 == response_path
        expected = gen_coarse_channel_response(512, 4, config.inference.pfb_taps_per_channel)
        np.testing.assert_array_equal(np.load(response_path), expected)

    def test_spline_selected_via_config(self, initialized_runtime):
        config = get_config()
        config.inference.bandpass_method = "spline"
        flattener = DataPreprocessor()._get_bandpass_flattener(num_coarse_channels=4)
        assert flattener.func is _spline_flatten_bandpass
        assert flattener.keywords == {"spl_order": config.inference.spline_order}

    def test_pfb_single_coarse_channel_falls_back_to_spline(self, initialized_runtime):
        config = get_config()
        config.inference.bandpass_method = "pfb"
        flattener = DataPreprocessor()._get_bandpass_flattener(num_coarse_channels=1)
        assert flattener.func is _spline_flatten_bandpass

    def test_unknown_method_raises(self, initialized_runtime):
        config = get_config()
        config.inference.bandpass_method = "median"
        with pytest.raises(ValueError, match="bandpass_method"):
            DataPreprocessor()._get_bandpass_flattener(num_coarse_channels=4)

    def test_pfb_flattener_is_picklable(self, initialized_runtime):
        # The flattener ships to pool workers inside task tuples
        import pickle  # noqa: PLC0415

        config = get_config()
        config.inference.coarse_channel_width = 512
        flattener = DataPreprocessor()._get_bandpass_flattener(num_coarse_channels=4)
        restored = pickle.loads(pickle.dumps(flattener))
        channel = np.full((4, 512), 2.0)
        np.testing.assert_array_equal(restored(channel), flattener(channel))

    def test_pfb_flatten_divides_out_response(self, initialized_runtime):
        config = get_config()
        config.inference.coarse_channel_width = 512
        config.inference.pfb_taps_per_channel = 12
        flattener = DataPreprocessor()._get_bandpass_flattener(num_coarse_channels=4)
        response = gen_coarse_channel_response(512, 4, 12)
        shaped = np.ones((16, 1)) * response
        np.testing.assert_allclose(flattener(shaped), 1.0, rtol=1e-12)


class TestPfbResidualFlatnessWarning:
    """_warn_on_pfb_response_mismatch flattens sampled channels with the active response H and
    warns once per file when the median flattened edge/mid power ratio deviates from 1.0 by
    more than _PFB_RESIDUAL_FLATNESS_TOL."""

    def _make_obs(self, tmp_path, taps: int | None, n_chans=1024, width=512):
        # taps=None -> flat noise (no PFB shaping); taps=k -> noise x H(k taps)
        import h5py  # noqa: PLC0415

        rng = np.random.default_rng(51)
        data = rng.normal(1000.0, 5.0, size=(16, 1, n_chans)).astype(np.float32)
        if taps is not None:
            response = gen_coarse_channel_response(width, n_chans // width, taps)
            data *= np.tile(response, n_chans // width).astype(np.float32)
        path = tmp_path / f"obs_taps_{taps}.h5"
        with h5py.File(path, "w") as hf:
            hf.create_dataset("data", data=data)
        return str(path)

    def _configure(self):
        config = get_config()
        config.data.time_bins = 16
        config.inference.coarse_channel_width = 512
        config.inference.pfb_taps_per_channel = 12
        return config

    @staticmethod
    def _flatness_warnings(caplog):
        return [r for r in caplog.records if "flattened edge/mid power ratio" in r.message]

    def test_true_taps_recording_does_not_warn(self, tmp_path, initialized_runtime, caplog):
        # noise x H(true taps): dividing by the active H flattens it, ratio ~1.0
        self._configure()
        h5_path = self._make_obs(tmp_path, taps=12)
        with caplog.at_level("WARNING", logger="aetherscan.preprocessing"):
            DataPreprocessor()._warn_on_pfb_response_mismatch(h5_path, n_coarse_total=2)
        assert self._flatness_warnings(caplog) == []

    def test_wrong_taps_recording_warns(self, tmp_path, initialized_runtime, caplog):
        # noise x H(4 taps) flattened by H(12 taps): the residual keeps a strong scallop
        self._configure()
        h5_path = self._make_obs(tmp_path, taps=4)
        with caplog.at_level("WARNING", logger="aetherscan.preprocessing"):
            DataPreprocessor()._warn_on_pfb_response_mismatch(h5_path, n_coarse_total=2)
        assert len(self._flatness_warnings(caplog)) == 1  # once per file, not per channel

    def test_flat_recording_warns(self, tmp_path, initialized_runtime, caplog):
        # Unshaped noise: dividing by H imprints the inverse scallop, ratio far from 1.0
        self._configure()
        h5_path = self._make_obs(tmp_path, taps=None)
        with caplog.at_level("WARNING", logger="aetherscan.preprocessing"):
            DataPreprocessor()._warn_on_pfb_response_mismatch(h5_path, n_coarse_total=2)
        assert len(self._flatness_warnings(caplog)) == 1

    def test_band_slice_reader_matches_full_read(self, tmp_path, initialized_runtime):
        # The banded float32 reader must agree with the despiked full-channel float64
        # reference (read, despike, time-integrate, divide by H, edge/mid ratio)
        self._configure()
        h5_path = self._make_obs(tmp_path, taps=12)
        response = gen_coarse_channel_response(512, 2, 12)
        preprocessor = DataPreprocessor()
        for ch in (0, 1):
            channel = preprocessor._read_despiked_channel(h5_path, ch)
            reference = edge_mid_power_ratio(equalize_passband(channel, response).mean(axis=0))
            banded = preprocessor._flattened_edge_mid_ratio(h5_path, ch, response)
            assert banded == pytest.approx(reference, rel=1e-4)

    def test_flatness_statistic_separates_true_from_wrong_taps(self):
        # The underlying statistic the check relies on (sanity of the ~0.10 tolerance)
        h_true = gen_coarse_channel_response(512, 2, 12)
        h_wrong = gen_coarse_channel_response(512, 2, 4)
        assert abs(edge_mid_power_ratio(h_true / h_true) - 1.0) < 0.10
        assert abs(edge_mid_power_ratio(h_wrong / h_true) - 1.0) > 0.10


class TestDecimateForPlot:
    """_decimate_for_plot reduces long spectrum lines to a per-bin min/max envelope so the
    bandpass debug/viz figures don't render ~1M-point matplotlib lines at GBT scale."""

    def test_short_input_passes_through(self):
        y = np.arange(100, dtype=np.float64)
        x, out = _decimate_for_plot(y, max_points=64)
        np.testing.assert_array_equal(x, np.arange(100))
        np.testing.assert_array_equal(out, y)

    def test_long_input_respects_point_budget(self):
        rng = np.random.default_rng(11)
        y = rng.normal(size=100_000)
        x, out = _decimate_for_plot(y, max_points=1024)
        assert out.shape[0] <= 2 * 1024
        assert x.shape == out.shape

    def test_envelope_preserves_narrowband_spike(self):
        # The property a plain stride lacks: a single-bin spike (RFI/hit) must survive.
        y = np.zeros(100_000)
        y[54_321] = 7.0
        x, out = _decimate_for_plot(y, max_points=1024)
        assert out.max() == 7.0
        assert 54_321 in x

    def test_indices_are_sorted_and_in_range(self):
        rng = np.random.default_rng(12)
        y = rng.normal(size=99_991)  # prime length: exercises the padded tail bin
        x, out = _decimate_for_plot(y, max_points=1024)
        assert np.all(np.diff(x) >= 0)
        assert x.min() >= 0 and x.max() <= y.shape[0] - 1

    def test_constant_bins_are_deduped(self):
        # A constant line makes every bin's argmin == argmax; the indices must be deduped to
        # strictly increasing (one point per bin), not emit each index twice.
        y = np.full(100_000, 3.14)
        x, out = _decimate_for_plot(y, max_points=1024)
        assert np.all(np.diff(x) > 0)  # strictly increasing → no duplicate points
        assert x.shape[0] <= 1024
        np.testing.assert_array_equal(out, y[x])


class TestPlanCadencesOutputDir:
    """plan_cadences resolves the .npy output dir per CSV: ED-fingerprint-scoped default
    under {data_path}/inference/preprocessed/ (#298 I3 — runs sharing an ED config share
    stamps), with an explicit --preprocess-output-dir shared across CSVs."""

    def test_default_is_fingerprint_scoped_not_tag_scoped(
        self, initialized_runtime, make_inference_csv
    ):
        config = get_config()
        make_inference_csv("subset.csv")
        config.data.inference_files = ["subset.csv"]
        config.checkpoint.save_tag = "test_v1"

        units = DataPreprocessor().plan_cadences()

        assert len(units) == 1
        fingerprint = preprocessing_config_fingerprint(config.to_dict())
        expected_dir = os.path.join(
            config.data_path, "inference", "preprocessed", f"subset_ed{fingerprint[:12]}"
        )
        assert os.path.dirname(units[0].npy_path) == expected_dir
        assert os.path.isdir(expected_dir)
        assert "test_v1" not in os.path.basename(expected_dir)

    def test_scoring_change_reuses_directory(self, initialized_runtime, make_inference_csv):
        # The point of #298 I3: a new threshold / model must land in the SAME stamp dir
        config = get_config()
        make_inference_csv("subset.csv")
        config.data.inference_files = ["subset.csv"]

        config.checkpoint.save_tag = "test_v1"
        first = DataPreprocessor().plan_cadences()
        config.checkpoint.save_tag = "test_v2"
        config.inference.classification_threshold = 0.5
        config.inference.encoder_path = "/some/other/encoder.keras"
        second = DataPreprocessor().plan_cadences()

        assert os.path.dirname(first[0].npy_path) == os.path.dirname(second[0].npy_path)

    def test_ed_change_gets_clean_directory(self, initialized_runtime, make_inference_csv):
        config = get_config()
        make_inference_csv("subset.csv")
        config.data.inference_files = ["subset.csv"]

        first = DataPreprocessor().plan_cadences()
        # Plant a stale stamp at exactly the path resume keys on
        with open(first[0].npy_path, "wb") as f:
            f.write(b"stale")
        config.inference.stat_threshold = 4096.0  # an ED-affecting change
        second = DataPreprocessor().plan_cadences()

        assert os.path.dirname(first[0].npy_path) != os.path.dirname(second[0].npy_path)
        # The new config's unit must not see the old stamp — process_pending_cadence would
        # otherwise resume-skip the cadence off stamps produced under a different ED config.
        assert not os.path.exists(second[0].npy_path)

    def test_duplicate_csv_basenames_rejected(self, initialized_runtime, make_inference_csv):
        # Same basename in different subdirectories would collide into one tag-scoped
        # output dir (and shared stamp filenames) — plan_cadences fails fast instead.
        config = get_config()
        make_inference_csv("run_a/subset.csv")
        make_inference_csv("run_b/subset.csv")
        config.data.inference_files = ["run_a/subset.csv", "run_b/subset.csv"]

        with pytest.raises(ValueError, match="subset"):
            DataPreprocessor().plan_cadences()

    def test_explicit_override_shared_across_csvs(
        self, initialized_runtime, make_inference_csv, tmp_path
    ):
        config = get_config()
        make_inference_csv("a.csv")
        make_inference_csv("b.csv")
        config.data.inference_files = ["a.csv", "b.csv"]
        override = str(tmp_path / "shared_preproc")
        config.inference.preprocess_output_dir = override

        units = DataPreprocessor().plan_cadences()

        assert len(units) == 2
        assert {os.path.dirname(u.npy_path) for u in units} == {override}


class TestResumeProvenanceGuard:
    """#298 I3: an existing stamp .npy is only reused when its metadata sidecar's h5_paths
    and recorded ED fingerprint match; missing/legacy metadata degrades to warn-and-reuse."""

    def _group_and_paths(self, tmp_path, h5_paths=None):
        from aetherscan.preprocessing import CadenceGroup  # noqa: PLC0415

        group = CadenceGroup(
            key=("T", "S", "L", "1", "1400"),
            h5_paths=h5_paths or [f"/data/obs_{i}.h5" for i in range(6)],
            csv_path="catalog.csv",
            expected_obs=6,
            is_valid=True,
        )
        npy_path = str(tmp_path / "cadence.npy")
        metadata_path = DataPreprocessor.cadence_metadata_path(npy_path)
        return group, npy_path, metadata_path

    def _write_metadata(self, metadata_path, **fields):
        with open(metadata_path, "w") as f:
            json.dump(fields, f)

    def test_matching_metadata_reuses(self, tmp_path, initialized_runtime):
        preprocessor = DataPreprocessor()
        group, npy_path, metadata_path = self._group_and_paths(tmp_path)
        self._write_metadata(
            metadata_path,
            h5_paths=group.h5_paths,
            ed_config_fingerprint=preprocessing_config_fingerprint(get_config().to_dict()),
        )
        assert preprocessor._resume_provenance_ok(group, npy_path, metadata_path)

    def test_h5_path_mismatch_reprocesses(self, tmp_path, initialized_runtime):
        preprocessor = DataPreprocessor()
        group, npy_path, metadata_path = self._group_and_paths(tmp_path)
        self._write_metadata(metadata_path, h5_paths=[f"/other/obs_{i}.h5" for i in range(6)])
        assert not preprocessor._resume_provenance_ok(group, npy_path, metadata_path)

    def test_fingerprint_mismatch_reprocesses(self, tmp_path, initialized_runtime):
        preprocessor = DataPreprocessor()
        group, npy_path, metadata_path = self._group_and_paths(tmp_path)
        self._write_metadata(
            metadata_path, h5_paths=group.h5_paths, ed_config_fingerprint="deadbeef"
        )
        assert not preprocessor._resume_provenance_ok(group, npy_path, metadata_path)

    def test_missing_metadata_reuses_with_warning(self, tmp_path, initialized_runtime, caplog):
        preprocessor = DataPreprocessor()
        group, npy_path, metadata_path = self._group_and_paths(tmp_path)
        with caplog.at_level("WARNING", logger="aetherscan.preprocessing"):
            assert preprocessor._resume_provenance_ok(group, npy_path, metadata_path)
        assert any("unguarded" in r.message for r in caplog.records)

    def test_legacy_metadata_without_fingerprint_reuses(self, tmp_path, initialized_runtime):
        preprocessor = DataPreprocessor()
        group, npy_path, metadata_path = self._group_and_paths(tmp_path)
        self._write_metadata(metadata_path, h5_paths=group.h5_paths)
        assert preprocessor._resume_provenance_ok(group, npy_path, metadata_path)


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

    def test_unrecognized_width_skipped_but_valid_file_still_loads(
        self, tmp_path, initialized_runtime
    ):
        # A single .npy at an unsupported stored width is logged and skipped; a valid file in
        # the same batch must still load, so one malformed file can't sink the whole run.
        config = self._configure()
        final_width = config.data.width_bin // config.data.downsample_factor
        rng = np.random.default_rng(53)
        bad = rng.chisquare(df=4, size=(2, 6, 16, final_width + 3)).astype(np.float32)
        good = rng.chisquare(df=4, size=(2, 6, 16, final_width)).astype(np.float32)
        bad_path = tmp_path / "bad_width.npy"
        good_path = tmp_path / "good.npy"
        np.save(bad_path, bad)
        np.save(good_path, good)

        loaded = DataPreprocessor().load_inference_data(
            override_filepaths=[str(bad_path), str(good_path)]
        )
        assert loaded.shape == (2, 6, 16, final_width)
        for i in range(2):
            np.testing.assert_allclose(loaded[i], log_norm(good[i]), rtol=1e-6)

    def test_non_float_dtype_coerced_with_warning(self, tmp_path, initialized_runtime, caplog):
        """A non-float cadence plate still loads (values preserved through log-norm) but the
        previously-silent float32 coercion is surfaced as a warning naming the file and dtype."""
        config = self._configure()
        final_width = config.data.width_bin // config.data.downsample_factor
        rng = np.random.default_rng(61)
        arr = rng.integers(1, 500, size=(2, 6, 16, final_width)).astype(np.int32)
        path = tmp_path / "int_counts.npy"
        np.save(path, arr)

        with caplog.at_level(logging.WARNING, logger="aetherscan.preprocessing"):
            loaded = DataPreprocessor().load_inference_data(override_filepaths=[str(path)])

        assert loaded.shape == (2, 6, 16, final_width)
        for i in range(2):
            np.testing.assert_allclose(loaded[i], log_norm(arr[i].astype(np.float32)), rtol=1e-6)
        coercion_warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "non-float dtype" in r.getMessage()
        ]
        assert len(coercion_warnings) == 1
        assert "int32" in coercion_warnings[0].getMessage()

    def test_wrong_ndim_is_skipped_but_valid_file_still_loads(self, tmp_path, initialized_runtime):
        """A cadence plate with the wrong rank is logged and skipped (only the trailing width was
        validated before); a valid file in the same batch still loads (skip-and-continue)."""
        config = self._configure()
        final_width = config.data.width_bin // config.data.downsample_factor
        rng = np.random.default_rng(67)
        bad = rng.chisquare(df=4, size=(6, 16, final_width)).astype(np.float32)  # 3-D, wrong rank
        good = rng.chisquare(df=4, size=(2, 6, 16, final_width)).astype(np.float32)
        bad_path = tmp_path / "wrong_ndim.npy"
        good_path = tmp_path / "good.npy"
        np.save(bad_path, bad)
        np.save(good_path, good)

        loaded = DataPreprocessor().load_inference_data(
            override_filepaths=[str(bad_path), str(good_path)]
        )
        assert loaded.shape == (2, 6, 16, final_width)
        for i in range(2):
            np.testing.assert_allclose(loaded[i], log_norm(good[i]), rtol=1e-6)

    def test_parallel_false_uses_no_pool_or_shared_memory(
        self, tmp_path, initialized_runtime, monkeypatch
    ):
        """parallel=False (the streaming per-cadence path in main._infer_cadence) must route
        through the sequential in-process branch even when n_processes > 1 — no chunk pool,
        no shared memory — so the loader can't double-subscribe the CPU against the
        persistent energy-detection pool. The output is pinned to the same log-norm oracle
        the parallel-path tests above use, so both paths provably agree."""
        config = self._configure()
        config.manager.n_processes = 4  # would fork a chunk pool on the parallel path
        final_width = config.data.width_bin // config.data.downsample_factor
        rng = np.random.default_rng(71)
        arr = rng.chisquare(df=4, size=(3, 6, 16, final_width)).astype(np.float32)
        path = tmp_path / "streaming_cadence.npy"
        np.save(path, arr)

        preprocessor = DataPreprocessor()

        def _fail(*args, **kwargs):
            raise AssertionError("no pool/shared memory may be created when parallel=False")

        monkeypatch.setattr(preprocessor.manager, "create_pool", _fail)
        monkeypatch.setattr(preprocessor.manager, "create_shared_memory", _fail)

        loaded = preprocessor.load_inference_data(override_filepaths=[str(path)], parallel=False)

        assert loaded.shape == (3, 6, 16, final_width)
        for i in range(3):
            np.testing.assert_allclose(loaded[i], log_norm(arr[i]), rtol=1e-6)

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

    @pytest.mark.parametrize(
        "width,window_size,step_size",
        [
            (1234, 100, 30),  # gcd 10; 1234 % 10 == 4 -> ragged short trailing block
            (1000, 96, 36),  # gcd 12; 1000 % 12 == 4 -> ragged short trailing block
        ],
    )
    def test_non_power_of_2_width_short_block_matches_scipy(self, width, window_size, step_size):
        # Guards the reduceat trailing-block truncation: when block does not divide the coarse
        # channel width, the final block spans a short ragged tail. No in-range window may
        # consume it, so k2 must still match scipy exactly. Assert the geometry actually
        # triggers the short block so a future refactor can't quietly make this test vacuous.
        block = step_size if window_size % step_size == 0 else math.gcd(window_size, step_size)
        assert width % block != 0  # this geometry must exercise the short-block path
        rng = np.random.default_rng(17)
        channel = rng.normal(0.0, 1.0, (16, width))

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


class TestSlidingNormalityK2AffineInvariance:
    """k2 is a function of the standardized 3rd/4th moments, so it is invariant under any affine
    remap x -> a*x + b (a != 0): the skew Z enters squared (sign-flip safe) and the kurtosis Z
    is even. This location/scale invariance is what PR-07's spline-vs-PFB detection-equivalence
    argument relies on, so pin it to the same rtol=1e-9 as the scipy-equivalence gates above."""

    WINDOW, STEP = 256, 128

    def _channel(self, seed=101):
        # Skewed and heavy-tailed enough that k2 is large and finite in every window.
        return np.random.default_rng(seed).chisquare(4, (16, 4096))

    @pytest.mark.parametrize("shift", [2.5, -100.0, 1e4])
    def test_additive_shift_leaves_k2_unchanged(self, shift):
        channel = self._channel()
        base = _sliding_normality_k2(channel, self.WINDOW, self.STEP)
        shifted = _sliding_normality_k2(channel + shift, self.WINDOW, self.STEP)
        np.testing.assert_allclose(shifted, base, rtol=1e-9)

    @pytest.mark.parametrize("scale", [2.0, 0.5, 1e3])
    def test_positive_scale_leaves_k2_unchanged(self, scale):
        channel = self._channel()
        base = _sliding_normality_k2(channel, self.WINDOW, self.STEP)
        scaled = _sliding_normality_k2(scale * channel, self.WINDOW, self.STEP)
        np.testing.assert_allclose(scaled, base, rtol=1e-9)

    @pytest.mark.parametrize("scale", [-1.0, -4.0])
    def test_negative_scale_leaves_k2_unchanged(self, scale):
        # A sign flip negates the skew statistic Z1, but k2 = Z1**2 + Z2**2 squares it and the
        # kurtosis Z2 is untouched, so k2 is invariant for a < 0 as well as a > 0.
        channel = self._channel()
        base = _sliding_normality_k2(channel, self.WINDOW, self.STEP)
        flipped = _sliding_normality_k2(scale * channel, self.WINDOW, self.STEP)
        np.testing.assert_allclose(flipped, base, rtol=1e-9)

    def test_full_affine_matches_scipy_on_transformed_data(self):
        # Cross-check against scipy on the *transformed* samples, so the invariance is a real
        # property of the statistic and not a tautology of reusing the same input array.
        channel = self._channel()
        a, b = 2.5, -7.0
        transformed = a * channel + b
        k2 = _sliding_normality_k2(transformed, self.WINDOW, self.STEP)
        _, expected = _scipy_window_loop(transformed, self.WINDOW, self.STEP)
        np.testing.assert_allclose(k2, expected, rtol=1e-9)
        base = _sliding_normality_k2(channel, self.WINDOW, self.STEP)
        np.testing.assert_allclose(k2, base, rtol=1e-9)

    def test_hit_set_identical_under_affine(self):
        # Half-Gaussian (low k2) + half strongly-skewed (high k2) so a mid threshold yields a
        # proper, non-trivial hit set; the surviving windows must be identical after an affine
        # remap with either sign of scale.
        rng = np.random.default_rng(7)
        gaussian = rng.normal(0.0, 1.0, (16, 2048))
        skewed = rng.chisquare(2, (16, 2048))
        channel = np.hstack([gaussian, skewed])
        threshold = 50.0

        base = _sliding_normality_k2(channel, self.WINDOW, self.STEP)
        base_hits = set(np.nonzero(base > threshold)[0].tolist())
        assert 0 < len(base_hits) < int(np.sum(np.isfinite(base)))

        for a, b in ((3.0, 10.0), (-2.0, -5.0)):
            k2 = _sliding_normality_k2(a * channel + b, self.WINDOW, self.STEP)
            assert set(np.nonzero(k2 > threshold)[0].tolist()) == base_hits


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

        hits, stat_hist, integrated = _energy_detect_channel_worker(
            (
                str(h5_path),
                channel_index,
                coarse_width,
                16,
                bandpass_flatten,
                window_size,
                step_size,
                stat_threshold,
                True,  # want_spectrum (#301): the despiked integrated spectrum rides along
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

        # want_spectrum: the integrated spectrum is the despiked channel's float64 time
        # mean — the exact quantity the persisted bandpass envelopes are built from (#301)
        np.testing.assert_array_equal(integrated, channel.astype(np.float64).mean(axis=0))

        # The summary histogram covers every finite window statistic, not just hits, on the
        # fixed shared bins — one count per window
        assert stat_hist.shape == (len(ED_STAT_HIST_EDGES) - 1,)
        assert stat_hist.sum() == len(statistics)
        # Hits land in the bins above the threshold
        hit_bin_counts = stat_hist[np.searchsorted(ED_STAT_HIST_EDGES, stat_threshold) - 1 :]
        assert hit_bin_counts.sum() >= len(hits)

    def test_absolute_indices_offset_by_channel_start(self, make_h5_observation):
        h5_path = make_h5_observation("obs.h5", n_chans=2048)
        coarse_width = 512
        bandpass_flatten = functools.partial(_spline_flatten_bandpass, spl_order=4)
        for channel_index in (0, 3):
            hits, _, integrated = _energy_detect_channel_worker(
                (
                    str(h5_path),
                    channel_index,
                    coarse_width,
                    16,
                    bandpass_flatten,
                    64,
                    32,
                    0.0,
                    False,
                )
            )
            assert integrated is None  # want_spectrum=False returns no spectrum
            starts = [idx for idx, _, _ in hits]
            assert len(starts) > 0  # threshold 0.0 must produce hits; else all(...) is vacuous
            assert all(
                channel_index * coarse_width <= s < (channel_index + 1) * coarse_width
                for s in starts
            )

    def test_no_hits_still_returns_histogram(self, make_h5_observation):
        """An impossibly high threshold yields no hits but the all-window statistic
        histogram must still be populated (it feeds the viz suite regardless of hits)."""
        h5_path = make_h5_observation("obs.h5", n_chans=2048)
        bandpass_flatten = functools.partial(_spline_flatten_bandpass, spl_order=4)
        hits, stat_hist, _ = _energy_detect_channel_worker(
            (str(h5_path), 0, 512, 16, bandpass_flatten, 64, 32, 1e12, False)
        )
        assert hits == []
        assert stat_hist.sum() > 0
