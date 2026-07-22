"""Unit tests for aetherscan.round_data (paths, manifests, startup dir cleanup, producer
protocol) and the batched memmap generation in aetherscan.data_generation."""

from __future__ import annotations

import os
import queue
import random
import threading
import time
from multiprocessing.shared_memory import SharedMemory
from types import SimpleNamespace

import numpy as np
import psutil
import pytest

from aetherscan.data_generation import (
    build_chunk_segments,
    build_segment_tasks,
    generate_round_to_memmap,
)
from aetherscan.round_data import (
    RoundDataPaths,
    RoundDataProducer,
    _producer_main,
    _reap_stale_producer,
    build_manifest,
    load_round_arrays,
    prepare_round_data_dir,
    validate_done_manifest,
    write_done_manifest,
)

# Keep injection fast: small frequency axis, real-ish resolutions (mirrors
# tests/unit/test_data_generation.py).
_WIDTH_BIN = 128
_FREQ_RES = 2.7939677238464355  # Hz
_TIME_RES = 18.25361108  # seconds

_SIGNAL_TYPES = ["false_no_signal", "false_with_rfi", "true_only_eti", "true_eti_rfi"]


@pytest.fixture(autouse=True)
def _seed_rngs():
    random.seed(11)
    np.random.seed(11)


def _write_small_round(paths: RoundDataPaths, n_samples=8, width_bin=16, snr_base=10.0):
    """Write a tiny but structurally-complete round dataset + validated manifest."""
    os.makedirs(paths.round_dir, exist_ok=True)
    rng = np.random.default_rng(paths.round_idx + 1)
    shape = (n_samples, 6, 4, width_bin)
    for path in paths.array_paths.values():
        arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
        arr[:] = rng.random(shape, dtype=np.float32)
        del arr
    lognorm_shape = (n_samples, 6, 2)
    for path in paths.lognorm_paths.values():
        arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=lognorm_shape)
        arr[:] = rng.random(lognorm_shape, dtype=np.float32)
        del arr
    labels = np.array(
        [_SIGNAL_TYPES[i % 4] for i in range(n_samples)],
        dtype="U20",
    )
    np.save(paths.labels_path, labels)
    manifest = build_manifest(
        paths,
        n_samples=n_samples,
        snr_base=snr_base,
        snr_range=40.0,
        wall_time_s=1.0,
        chunk_count=1,
    )
    write_done_manifest(paths, manifest)
    return manifest


class TestRoundDataPaths:
    def test_for_round_naming(self, tmp_path):
        paths = RoundDataPaths.for_round(str(tmp_path), 3)
        assert paths.round_dir == str(tmp_path / "round_03")
        assert paths.round_idx == 3
        assert paths.done_path.endswith("round_03.done")

    def test_array_and_label_paths(self, tmp_path):
        paths = RoundDataPaths.for_round(str(tmp_path), 12)
        assert set(paths.array_paths.keys()) == {"main", "true", "false"}
        for name, path in paths.array_paths.items():
            assert path == os.path.join(paths.round_dir, f"{name}.npy")
        assert set(paths.lognorm_paths.keys()) == {"main", "true", "false"}
        for name, path in paths.lognorm_paths.items():
            assert path == os.path.join(paths.round_dir, f"{name}_lognorm.npy")
        assert paths.labels_path == os.path.join(paths.round_dir, "labels.npy")


class TestManifest:
    def test_roundtrip_validates(self, tmp_path):
        paths = RoundDataPaths.for_round(str(tmp_path), 1)
        written = _write_small_round(paths)
        manifest = validate_done_manifest(paths, expected_n_samples=8)
        assert manifest is not None
        assert manifest["n_samples"] == written["n_samples"] == 8
        assert manifest["round_idx"] == 1
        assert set(manifest["checksums"].keys()) == {
            "main",
            "true",
            "false",
            "main_lognorm",
            "true_lognorm",
            "false_lognorm",
            "labels",
        }

    def test_missing_done_file_invalid(self, tmp_path):
        paths = RoundDataPaths.for_round(str(tmp_path), 1)
        _write_small_round(paths)
        os.remove(paths.done_path)
        assert validate_done_manifest(paths) is None

    def test_expected_n_samples_mismatch_invalid(self, tmp_path):
        paths = RoundDataPaths.for_round(str(tmp_path), 1)
        _write_small_round(paths, n_samples=8)
        assert validate_done_manifest(paths, expected_n_samples=16) is None

    def test_round_idx_mismatch_invalid(self, tmp_path):
        # A manifest written for round 1 dropped into a round-2 dir must not validate.
        paths_1 = RoundDataPaths.for_round(str(tmp_path), 1)
        manifest = _write_small_round(paths_1)
        paths_2 = RoundDataPaths.for_round(str(tmp_path), 2)
        _write_small_round(paths_2)
        write_done_manifest(paths_2, manifest)
        assert validate_done_manifest(paths_2) is None

    def test_missing_array_invalid(self, tmp_path):
        paths = RoundDataPaths.for_round(str(tmp_path), 1)
        _write_small_round(paths)
        os.remove(paths.true_path)
        assert validate_done_manifest(paths) is None

    def test_missing_lognorm_sibling_invalid(self, tmp_path):
        # A round dir predating the lognorm-sibling feature fails validation & regenerates.
        paths = RoundDataPaths.for_round(str(tmp_path), 1)
        _write_small_round(paths)
        os.remove(paths.lognorm_paths["main"])
        assert validate_done_manifest(paths) is None

    def test_shape_mismatch_invalid(self, tmp_path):
        paths = RoundDataPaths.for_round(str(tmp_path), 1)
        _write_small_round(paths)
        np.save(paths.main_path, np.zeros((2, 2), dtype=np.float32))
        assert validate_done_manifest(paths) is None

    def test_corrupted_values_invalid(self, tmp_path):
        paths = RoundDataPaths.for_round(str(tmp_path), 1)
        _write_small_round(paths)
        # Same shape, different content: the sampled-value checksum must catch it.
        arr = np.load(paths.false_path, mmap_mode="r+")
        arr[:] = arr[:] + 7.0
        arr.flush()
        del arr
        assert validate_done_manifest(paths) is None

    def test_corrupted_manifest_json_invalid(self, tmp_path):
        paths = RoundDataPaths.for_round(str(tmp_path), 1)
        _write_small_round(paths)
        with open(paths.done_path, "w") as f:
            f.write("{not json")
        assert validate_done_manifest(paths) is None

    def test_load_round_arrays_returns_memmaps(self, tmp_path):
        paths = RoundDataPaths.for_round(str(tmp_path), 1)
        _write_small_round(paths, n_samples=8, width_bin=16)
        data = load_round_arrays(paths)
        assert set(data.keys()) == {"concatenated", "true", "false", "labels", "lognorm"}
        for key in ("concatenated", "true", "false"):
            assert isinstance(data[key], np.memmap)
            assert data[key].shape == (8, 6, 4, 16)
            assert data[key].dtype == np.float32
        assert data["labels"].shape == (8,)
        assert not isinstance(data["labels"], np.memmap)
        # The main array's log-norm params are tiny and loaded eagerly
        assert data["lognorm"].shape == (8, 6, 2)
        assert not isinstance(data["lognorm"], np.memmap)


class TestPrepareRoundDataDir:
    def test_creates_missing_base_dir(self, tmp_path):
        base = tmp_path / "round_data" / "test_v1"
        prepare_round_data_dir(str(base), start_round=1)
        assert base.is_dir()

    def test_resume_semantics(self, tmp_path):
        base = str(tmp_path)
        # round 1: valid manifest (< start_round -> kept)
        _write_small_round(RoundDataPaths.for_round(base, 1))
        # round 2: no .done (< start_round -> deleted)
        paths_2 = RoundDataPaths.for_round(base, 2)
        _write_small_round(paths_2)
        os.remove(paths_2.done_path)
        # round 3: valid manifest but >= start_round -> deleted (regenerated by this run)
        _write_small_round(RoundDataPaths.for_round(base, 3))
        # non-round entry (stale RF dataset) -> deleted
        os.makedirs(os.path.join(base, "rf"))
        # loose file -> untouched
        with open(os.path.join(base, "stray.txt"), "w") as f:
            f.write("x")

        prepare_round_data_dir(base, start_round=3)

        assert os.path.isdir(os.path.join(base, "round_01"))
        assert not os.path.exists(os.path.join(base, "round_02"))
        assert not os.path.exists(os.path.join(base, "round_03"))
        assert not os.path.exists(os.path.join(base, "rf"))
        assert os.path.isfile(os.path.join(base, "stray.txt"))

    def test_fresh_run_deletes_everything(self, tmp_path):
        base = str(tmp_path)
        _write_small_round(RoundDataPaths.for_round(base, 1))
        _write_small_round(RoundDataPaths.for_round(base, 2))
        prepare_round_data_dir(base, start_round=1)
        assert not os.path.exists(os.path.join(base, "round_01"))
        assert not os.path.exists(os.path.join(base, "round_02"))


def _segment_coverage(segments, array_name):
    """Row indices covered by `array_name`'s segments (with multiplicity)."""
    covered = []
    for segment in segments:
        if segment.array_name == array_name:
            covered.extend(range(segment.start_idx, segment.start_idx + segment.count))
    return covered


class TestChunkSegmentsAndTasks:
    def test_segments_cover_each_array_exactly_once(self):
        segments = build_chunk_segments(chunk_start=8, chunk_size=8)
        assert len(segments) == 8
        for array_name in ("main", "false", "true"):
            covered = _segment_coverage(segments, array_name)
            assert sorted(covered) == list(range(8, 16))  # exactly once, no gaps/overlaps

    def test_main_segment_order_matches_labels(self):
        segments = build_chunk_segments(chunk_start=0, chunk_size=8)
        main_types = [s.signal_type for s in segments if s.array_name == "main"]
        assert main_types == _SIGNAL_TYPES

    def test_chunk_size_not_divisible_by_4_raises(self):
        with pytest.raises(ValueError, match="divisible by 4"):
            build_chunk_segments(chunk_start=0, chunk_size=6)

    def test_task_partitioning_covers_every_row_exactly_once(self):
        segments = build_chunk_segments(chunk_start=0, chunk_size=40)
        seed_rng = np.random.default_rng(0)
        for segment in segments:
            tasks = build_segment_tasks(
                segment, "arr.npy", 3, 10.0, 40.0, _WIDTH_BIN, _FREQ_RES, _TIME_RES, seed_rng
            )
            covered = []
            for _, start_idx, count, *_rest in tasks:
                covered.extend(range(start_idx, start_idx + count))
            assert sorted(covered) == list(
                range(segment.start_idx, segment.start_idx + segment.count)
            )
            # Every task respects the max size and carries the segment's generation params
            assert all(task[2] <= 3 for task in tasks)
            assert all(task[3] == segment.create_fn_name for task in tasks)

    def test_task_size_below_one_raises(self):
        segment = build_chunk_segments(0, 4)[0]
        with pytest.raises(ValueError, match="task_size"):
            build_segment_tasks(
                segment,
                "arr.npy",
                0,
                10.0,
                40.0,
                _WIDTH_BIN,
                _FREQ_RES,
                _TIME_RES,
                np.random.default_rng(0),
            )


class TestGenerateRoundToMemmap:
    @pytest.fixture
    def plate(self, make_background_npy):
        path = make_background_npy("plate.npy", n_cadences=4, width_bin=_WIDTH_BIN)
        return np.load(path)

    def test_sequential_generation_end_to_end(self, tmp_path, plate):
        paths = RoundDataPaths.for_round(str(tmp_path), 1)
        stats_segments = []
        progress = []

        manifest = generate_round_to_memmap(
            paths,
            n_samples=8,
            snr_base=10.0,
            snr_range=5.0,
            width_bin=_WIDTH_BIN,
            num_observations=6,
            time_bins=16,
            chunk_size=4,
            task_size=3,
            freq_resolution=_FREQ_RES,
            time_resolution=_TIME_RES,
            pool=None,
            backgrounds=plate,
            round_num=1,
            stats_cb=stats_segments.append,
            progress_cb=lambda chunk, n_chunks: progress.append((chunk, n_chunks)),
        )

        # Arrays on disk with the right shape/dtype, every row populated & log-normalized
        data = load_round_arrays(paths)
        for key in ("concatenated", "true", "false"):
            arr = data[key]
            assert arr.shape == (8, 6, 16, _WIDTH_BIN)
            assert arr.dtype == np.float32
            row_maxes = arr.max(axis=(1, 2, 3))
            assert np.all(row_maxes > 0)  # no row was left unwritten
            assert float(arr.max()) <= 1.0
            assert float(arr.min()) >= 0.0

        # Labels mirror the contiguous per-chunk layout (chunk_size=4 -> quarter=1)
        assert list(data["labels"]) == _SIGNAL_TYPES + _SIGNAL_TYPES

        # Per-observation log-norm params were recorded for every array & every row
        # (range_log > 0 for chi-squared-noise inputs). This asserts the PLUMBING — params
        # populated, right shape, and a finite inversion — not an exact roundtrip: the raw
        # pre-normalization data isn't retained at generate time, so the exact
        # exp(x*range+min) == data check lives at unit level (test_create_false_not_injected).
        for name, lognorm_path in paths.lognorm_paths.items():
            params = np.load(lognorm_path)
            assert params.shape == (8, 6, 2)
            assert np.all(params[..., 1] > 0), f"{name} lognorm range_log not populated"
        main_params = np.load(paths.lognorm_paths["main"])
        recovered = np.exp(data["concatenated"][0, 0] * main_params[0, 0, 1] + main_params[0, 0, 0])
        assert np.all(np.isfinite(recovered))  # finite inversion (exp of a finite value is > 0)

        # Manifest validates and matches the generation request
        assert validate_done_manifest(paths, expected_n_samples=8) is not None
        assert manifest["chunk_count"] == 2
        assert manifest["snr_base"] == 10.0

        # Stats: 8 class-segments per chunk x 2 chunks; per-segment sample counts add up
        assert len(stats_segments) == 16
        for segment in stats_segments:
            assert segment["num_samples"] == len(segment["stats_list"])
            assert segment["round_number"] == 1
            assert segment["snr_range_ceil"] == 15.0
        main_total = sum(s["num_samples"] for s in stats_segments if s["signal_class"] == "main")
        false_total = sum(s["num_samples"] for s in stats_segments if s["signal_class"] == "false")
        true_total = sum(s["num_samples"] for s in stats_segments if s["signal_class"] == "true")
        assert main_total == false_total == true_total == 8

        # Signal-info keys match the signal type
        by_type = {(s["signal_class"], s["signal_type"]): s for s in stats_segments}
        assert by_type[("main", "false_no_signal")]["stats_list"][0]["signal_info"] == {}
        double_keys = set(by_type[("true", "true_eti_rfi")]["stats_list"][0]["signal_info"])
        assert any(k.startswith("eti_") for k in double_keys)
        assert any(k.startswith("rfi_") for k in double_keys)

        assert progress == [(1, 2), (2, 2)]

    def test_regeneration_clears_stale_dir(self, tmp_path, plate):
        paths = RoundDataPaths.for_round(str(tmp_path), 1)
        os.makedirs(paths.round_dir)
        stale = os.path.join(paths.round_dir, "leftover.npy")
        with open(stale, "w") as f:
            f.write("stale")
        generate_round_to_memmap(
            paths,
            n_samples=4,
            snr_base=10.0,
            snr_range=5.0,
            width_bin=_WIDTH_BIN,
            num_observations=6,
            time_bins=16,
            chunk_size=4,
            task_size=2,
            freq_resolution=_FREQ_RES,
            time_resolution=_TIME_RES,
            backgrounds=plate,
        )
        assert not os.path.exists(stale)
        assert validate_done_manifest(paths, expected_n_samples=4) is not None

    def test_invalid_inputs_raise(self, tmp_path, plate):
        paths = RoundDataPaths.for_round(str(tmp_path), 1)
        common = {
            "width_bin": _WIDTH_BIN,
            "num_observations": 6,
            "time_bins": 16,
            "task_size": 2,
            "freq_resolution": _FREQ_RES,
            "time_resolution": _TIME_RES,
            "backgrounds": plate,
        }
        with pytest.raises(ValueError, match="n_samples"):
            generate_round_to_memmap(paths, 6, 10.0, 5.0, chunk_size=4, **common)
        with pytest.raises(ValueError, match="chunk_size"):
            generate_round_to_memmap(paths, 8, 10.0, 5.0, chunk_size=6, **common)
        with pytest.raises(ValueError, match="backgrounds"):
            generate_round_to_memmap(
                paths,
                8,
                10.0,
                5.0,
                chunk_size=4,
                **{**common, "backgrounds": None},
            )


def _stub_generate_ok(paths, round_idx, snr_base, snr_range, pool, params, stats_cb, progress_cb):
    """Producer-protocol stub: emits one stats + one progress message, returns a manifest."""
    stats_cb({"signal_class": "main", "stats_list": []})
    progress_cb(1, 1)
    return {"round_idx": round_idx, "n_samples": params["n_samples"], "stub": True}


def _stub_generate_boom(paths, round_idx, snr_base, snr_range, pool, params, stats_cb, progress_cb):
    raise RuntimeError("stub generation exploded")


class TestProducerProtocol:
    """Drive _producer_main in a thread with plain queues and a stub generate_fn — validates
    the message protocol without child processes or the setigen generation stack."""

    def _run(self, tmp_path, generate_fn, requests):
        request_queue: queue.Queue = queue.Queue()
        result_queue: queue.Queue = queue.Queue()
        params = {"base_dir": str(tmp_path), "n_samples": 8}
        for request in requests:
            request_queue.put(request)
        request_queue.put(("shutdown",))
        thread = threading.Thread(
            target=_producer_main,
            args=(request_queue, result_queue, params),
            kwargs={"generate_fn": generate_fn},
            daemon=True,
        )
        thread.start()
        thread.join(timeout=30)
        assert not thread.is_alive()
        messages = []
        while True:
            try:
                messages.append(result_queue.get_nowait())
            except queue.Empty:
                break
        return messages

    def test_generate_emits_stats_progress_timing_done(self, tmp_path):
        messages = self._run(tmp_path, _stub_generate_ok, [("generate", 1, 10, 40)])
        kinds = [m[0] for m in messages]
        # Timing precedes done (queue FIFO: the drainer records the stage span before
        # await_round unblocks)
        assert kinds == ["stats", "progress", "timing", "done", "shutdown_ack"]
        timing = messages[2]
        assert timing[1] == 1
        assert timing[2] <= timing[3]  # start_ts <= end_ts
        done = messages[3]
        assert done[1] == 1
        assert done[2]["stub"] is True

    def test_error_is_reported_and_producer_keeps_serving(self, tmp_path):
        messages = self._run(
            tmp_path,
            _stub_generate_boom,
            [("generate", 1, 10, 40), ("generate", 2, 10, 40)],
        )
        kinds = [m[0] for m in messages]
        assert kinds == ["error", "error", "shutdown_ack"]
        assert messages[0][1] == 1
        assert "stub generation exploded" in messages[0][2]
        assert messages[1][1] == 2

    def test_existing_valid_round_short_circuits(self, tmp_path):
        # A validated on-disk round must be reused: done comes back without generate_fn firing.
        _write_small_round(RoundDataPaths.for_round(str(tmp_path), 1), n_samples=8)
        messages = self._run(tmp_path, _stub_generate_boom, [("generate", 1, 10, 40)])
        kinds = [m[0] for m in messages]
        assert kinds == ["done", "shutdown_ack"]
        assert messages[0][2]["n_samples"] == 8

    def test_unknown_message_ignored(self, tmp_path):
        messages = self._run(tmp_path, _stub_generate_ok, [("bogus",)])
        assert [m[0] for m in messages] == ["shutdown_ack"]


class TestProducerParentDeathWatch:
    """The request loop's ppid watch: an ungraceful parent death (kill -9 / OOM) never sends
    "shutdown", so the producer must notice the reparenting itself and exit (issue #141)."""

    def _start(self, tmp_path, request_queue, result_queue, parent_pid):
        thread = threading.Thread(
            target=_producer_main,
            args=(request_queue, result_queue, {"base_dir": str(tmp_path), "n_samples": 8}),
            kwargs={
                "generate_fn": _stub_generate_ok,
                "parent_pid": parent_pid,
                "poll_interval": 0.05,
            },
            daemon=True,
        )
        thread.start()
        return thread

    def test_producer_exits_when_parent_is_gone(self, tmp_path):
        result_queue: queue.Queue = queue.Queue()
        # No message ever arrives and the captured parent PID never matches the real ppid —
        # exactly the reparented-orphan state after a parent SIGKILL.
        thread = self._start(tmp_path, queue.Queue(), result_queue, parent_pid=-1)
        thread.join(timeout=10)
        assert not thread.is_alive()
        # Parent death is not a graceful shutdown: no shutdown_ack is emitted.
        assert result_queue.empty()

    def test_producer_keeps_serving_across_empty_polls_while_parent_lives(self, tmp_path):
        request_queue: queue.Queue = queue.Queue()
        result_queue: queue.Queue = queue.Queue()
        # Matches what _producer_main would capture itself — the parent is "alive".
        thread = self._start(tmp_path, request_queue, result_queue, parent_pid=os.getppid())
        time.sleep(0.3)  # several empty polls — none may trigger an exit
        request_queue.put(("generate", 1, 10, 40))
        request_queue.put(("shutdown",))
        thread.join(timeout=10)
        assert not thread.is_alive()
        kinds = []
        while True:
            try:
                kinds.append(result_queue.get_nowait()[0])
            except queue.Empty:
                break
        assert kinds == ["stats", "progress", "timing", "done", "shutdown_ack"]


class _FakeSpawnedProcess:
    """Stand-in for _MP_CONTEXT.Process in pidfile tests — records nothing, spawns nothing."""

    def __init__(self, *args, **kwargs):
        self.pid = 31337

    def start(self):
        pass

    def join(self, timeout=None):
        pass

    def is_alive(self):
        return False


class TestProducerPidfile:
    def test_pidfile_written_on_start_and_removed_on_shutdown(self, tmp_path, monkeypatch):
        producer = RoundDataProducer(
            base_dir=str(tmp_path),
            n_samples=8,
            shm_name="unused",
            background_shape=(1, 6, 4, 16),
            background_dtype="float32",
            n_processes=1,
            width_bin=16,
            num_observations=6,
            time_bins=4,
            chunk_size=4,
            task_size=2,
            freq_resolution=_FREQ_RES,
            time_resolution=_TIME_RES,
            db=_FakeDB(),
            tag="test_v1",
        )
        # Swap the spawn context for a no-op process AFTER construction (the queues in
        # __init__ must stay real) and keep the manager out of the way.
        monkeypatch.setattr(
            "aetherscan.round_data._MP_CONTEXT", SimpleNamespace(Process=_FakeSpawnedProcess)
        )
        monkeypatch.setattr("aetherscan.round_data.get_manager", lambda: None)

        producer.start()
        pidfile = tmp_path / "producer.pid"
        assert pidfile.read_text() == "31337"

        producer.shutdown(timeout=2.0)
        assert not pidfile.exists()


class _FakeReapChild:
    def __init__(self):
        self.killed = False

    def kill(self):
        self.killed = True


class _FakeReapProc:
    def __init__(self, pid, create_time, children=()):
        self.pid = pid
        self._create_time = create_time
        self._children = list(children)
        self.terminated = False
        self.killed = False

    def create_time(self):
        return self._create_time

    def children(self, recursive=False):
        return self._children

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        pass

    def kill(self):
        self.killed = True


class TestReapStaleProducer:
    """The restart-race guard run by prepare_round_data_dir before any rmtree (psutil mocked
    — no real signals are sent)."""

    def _write_pidfile(self, base_dir, pid=424242):
        path = os.path.join(base_dir, "producer.pid")
        with open(path, "w") as f:
            f.write(str(pid))
        return path

    def _patch_psutil(self, monkeypatch, process_cls):
        monkeypatch.setattr(
            "aetherscan.round_data.psutil",
            SimpleNamespace(
                Process=process_cls,
                NoSuchProcess=psutil.NoSuchProcess,
                TimeoutExpired=psutil.TimeoutExpired,
            ),
        )

    def test_missing_pidfile_is_a_noop(self, tmp_path):
        _reap_stale_producer(str(tmp_path))  # must not raise

    def test_dead_recorded_pid_removes_pidfile(self, tmp_path, monkeypatch):
        path = self._write_pidfile(str(tmp_path))

        def _no_such_process(pid):
            raise psutil.NoSuchProcess(pid)

        self._patch_psutil(monkeypatch, _no_such_process)
        _reap_stale_producer(str(tmp_path))
        assert not os.path.exists(path)

    def test_live_recorded_producer_tree_is_reaped(self, tmp_path, monkeypatch):
        path = self._write_pidfile(str(tmp_path))
        children = [_FakeReapChild(), _FakeReapChild()]
        # Created before the pidfile was written -> genuinely the recorded producer.
        proc = _FakeReapProc(424242, os.path.getmtime(path) - 60.0, children=children)
        self._patch_psutil(monkeypatch, lambda pid: proc)
        _reap_stale_producer(str(tmp_path), term_timeout=0.1)
        assert proc.terminated and proc.killed
        assert all(child.killed for child in children)
        assert not os.path.exists(path)

    def test_recycled_pid_is_left_alone(self, tmp_path, monkeypatch):
        path = self._write_pidfile(str(tmp_path))
        # Created after the pidfile was written -> a PID recycled by an unrelated process.
        proc = _FakeReapProc(424242, os.path.getmtime(path) + 60.0)
        self._patch_psutil(monkeypatch, lambda pid: proc)
        _reap_stale_producer(str(tmp_path))
        assert not proc.terminated and not proc.killed
        assert not os.path.exists(path)  # the stale pidfile is still cleared


class _FakeProcess:
    """Stand-in for the producer multiprocessing.Process on the main-side handle."""

    def __init__(self):
        self.alive = True
        self.pid = 4242

    def is_alive(self):
        return self.alive


class _FakeDB:
    def __init__(self):
        self.writes = []

    def write_injection_stat(self, **kwargs):
        self.writes.append(kwargs)


def _minimal_sample_info():
    stage = dict.fromkeys(
        [
            "global_mean",
            "global_median",
            "global_std",
            "global_mad",
            "global_skew",
            "global_kurtosis",
        ],
        0.5,
    )
    return {
        "background_index": 0,
        "intensity_stats": {"A": dict(stage), "B": dict(stage), "C": dict(stage)},
        "signal_info": {},
        "slope_was_clamped": False,
    }


def _minimal_segment():
    return {
        "round_number": 1,
        "chunk_number": 1,
        "signal_class": "main",
        "signal_type": "false_no_signal",
        "snr_range_floor": 10,
        "snr_range_ceil": 50,
        "num_samples": 1,
        "inject_duration": 0.1,
        "timestamp": time.time(),
        "stats_list": [_minimal_sample_info()],
    }


class TestRoundDataProducerDrainer:
    """Exercise the main-side drainer thread + await_round against a fake process, feeding
    the result queue directly (no child process involved)."""

    @pytest.fixture
    def producer(self, tmp_path):
        producer = RoundDataProducer(
            base_dir=str(tmp_path),
            n_samples=8,
            shm_name="unused",
            background_shape=(1, 6, 4, 16),
            background_dtype="float32",
            n_processes=1,
            width_bin=16,
            num_observations=6,
            time_bins=4,
            chunk_size=4,
            task_size=2,
            freq_resolution=_FREQ_RES,
            time_resolution=_TIME_RES,
            db=_FakeDB(),
            tag="test_v1",
        )
        producer._process = _FakeProcess()
        producer._drainer = threading.Thread(target=producer._drain_results, daemon=True)
        producer._drainer.start()
        yield producer
        producer._process.alive = False
        producer._drainer.join(timeout=10)

    def test_done_unblocks_await_round(self, producer):
        producer._result_queue.put(("done", 1, {"n_samples": 8}))
        assert producer.await_round(1) == {"n_samples": 8}

    def test_error_raises_with_producer_traceback(self, producer):
        producer._result_queue.put(("error", 2, "Traceback: boom"))
        with pytest.raises(RuntimeError, match="boom"):
            producer.await_round(2)

    def test_stats_are_written_to_db_from_main_process(self, producer):
        producer._result_queue.put(("stats", 1, _minimal_segment()))
        producer._result_queue.put(("done", 1, {"n_samples": 8}))
        producer.await_round(1)
        # 18 per-sample intensity rows (6 stats x 3 stages) + 0 signal rows + 4 metadata rows
        deadline = time.time() + 10
        while len(producer._db.writes) < 22 and time.time() < deadline:
            time.sleep(0.05)
        assert len(producer._db.writes) == 22
        metadata_names = {w["stat_name"] for w in producer._db.writes if w["sample_index"] is None}
        assert metadata_names == {
            "snr_range_floor",
            "snr_range_ceil",
            "num_samples",
            "inject_duration",
        }
        assert all(w["tag"] == "test_v1" for w in producer._db.writes)

    def test_timing_message_records_stage_span(self, producer, monkeypatch):
        recorded = []

        def fake_record_stage(*args, **kwargs):
            # FIFO guard: the span must be recorded BEFORE the done message resolves the
            # round — a reversed (done-before-timing) drain order would trip this.
            assert 3 not in producer._results
            recorded.append((args, kwargs))

        monkeypatch.setattr("aetherscan.round_data.record_stage", fake_record_stage)
        producer._result_queue.put(("timing", 3, 100.0, 160.0))
        producer._result_queue.put(("done", 3, {"n_samples": 8}))
        producer.await_round(3)  # FIFO: the timing message was handled before done
        assert recorded == [
            (
                ("train.round_03.data_generation", 100.0, 160.0),
                {"tag": "test_v1", "metadata": {"source": "producer"}},
            )
        ]

    def test_producer_death_unblocks_await_round(self, producer):
        producer._process.alive = False
        with pytest.raises(RuntimeError, match="exited before producing"):
            producer.await_round(5)


@pytest.mark.slow
class TestRoundDataProducerSpawnEndToEnd:
    """Real spawn-started producer process + real (tiny) generation against real shared
    memory — exercises the spawn pickling boundary, the child's import chain, its pool
    creation, and the stats/done/shutdown protocol for real. Both cluster-smoke failures
    this PR hit (a fork-inherited deadlocked lock; a fork-context SemLock crossing the
    spawn boundary) were only reachable through a real child process.

    COVERAGE LIMIT: this validates that spawn WORKS, not the rationale for choosing it. The
    original fork deadlock needs a thread-laden TF/NCCL parent, which pytest does not have —
    so a revert to a fork start method would still PASS here (verified: the e2e body passes
    with _MP_CONTEXT patched to fork in an idle parent). The only guard against re-introducing
    the fork deadlock is a cluster smoke; gate any change to _MP_CONTEXT / the start method on
    one. The SemLock/log-relay regression, by contrast, IS caught here (it raises at start)."""

    def test_spawned_producer_generates_round(self, tmp_path):
        rng = np.random.default_rng(11)
        plate = rng.chisquare(df=4, size=(4, 6, 16, _WIDTH_BIN)).astype(np.float32)

        shm = SharedMemory(create=True, size=plate.nbytes)
        producer = None
        try:
            shared = np.ndarray(plate.shape, dtype=plate.dtype, buffer=shm.buf)
            shared[:] = plate

            db = _FakeDB()
            producer = RoundDataProducer(
                base_dir=str(tmp_path),
                n_samples=8,
                shm_name=shm.name,
                background_shape=plate.shape,
                background_dtype=str(plate.dtype),
                n_processes=2,
                width_bin=_WIDTH_BIN,
                num_observations=6,
                time_bins=16,
                chunk_size=4,
                task_size=3,
                freq_resolution=_FREQ_RES,
                time_resolution=_TIME_RES,
                db=db,
                tag="test_v1",
            )
            producer.start()
            producer.request_generation(1, 10, 5)
            manifest = producer.await_round(1)
            assert manifest["n_samples"] == 8

            paths = RoundDataPaths.for_round(str(tmp_path), 1)
            assert validate_done_manifest(paths, expected_n_samples=8) is not None

            # Streamed stats land in the main-process drainer (DB writes stay in main)
            deadline = time.time() + 30
            while not db.writes and time.time() < deadline:
                time.sleep(0.2)
            assert db.writes

            # A second request for the already-generated round short-circuits via the manifest
            producer.request_generation(1, 10, 5)
            assert producer.await_round(1)["n_samples"] == 8
        finally:
            if producer is not None:
                producer.shutdown()
            shm.close()
            shm.unlink()
