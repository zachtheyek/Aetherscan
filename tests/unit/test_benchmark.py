"""Unit tests for aetherscan.benchmark (stage_timer nesting / DB rows / failure recording),
the monitor's stage-band span selection, and utils/benchmark_report.py's tree math and
suggestion rules — all against tmp-path SQLite files."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sqlite3
import sys
import threading
from pathlib import Path

import pytest

from aetherscan.benchmark import current_stage, record_stage, stage_timer
from aetherscan.config import get_config
from aetherscan.db.db import Database
from aetherscan.monitor.monitor import select_annotation_spans

# utils/ is not a package — load the report tool straight from its file. The sys.modules
# registration must precede exec_module: the module's @dataclass resolves its own module
# by name at class-creation time (PEP 563 string annotations).
_REPORT_PATH = Path(__file__).resolve().parents[2] / "utils" / "benchmark_report.py"
_spec = importlib.util.spec_from_file_location("benchmark_report", _REPORT_PATH)
benchmark_report = importlib.util.module_from_spec(_spec)
sys.modules["benchmark_report"] = benchmark_report
_spec.loader.exec_module(benchmark_report)


@pytest.fixture
def db(tmp_path):
    """A started Database against the tmp output path with periodic flushing effectively
    disabled, so flush() is the only thing that drains writes (same setup as test_db)."""
    config = get_config()
    config.db.write_interval = 300.0
    config.db.write_buffer_max_size = 100_000
    database = Database()
    assert database.db_path.startswith(str(tmp_path))
    database.start()
    yield database
    database.stop()


class TestStageTimer:
    def test_context_manager_records_row(self, db):
        with stage_timer("train.load_backgrounds", tag="test_v1"):
            pass
        assert db.flush(timeout=10) is True

        rows = db.query_pipeline_stages(tag="test_v1")
        assert len(rows) == 1
        row = rows[0]
        assert row["stage"] == "train.load_backgrounds"
        assert row["end_time"] >= row["start_time"]
        assert row["duration_s"] == pytest.approx(row["end_time"] - row["start_time"])
        assert row["metadata"] is None

    def test_tag_defaults_to_save_tag(self, db):
        get_config().checkpoint.save_tag = "test_v9"
        with stage_timer("inference.viz"):
            pass
        assert db.flush(timeout=10) is True
        assert len(db.query_pipeline_stages(tag="test_v9")) == 1

    def test_nested_timers_produce_dotted_names(self, db):
        with stage_timer("train.round_01", tag="test_v1"):
            assert current_stage() == "train.round_01"
            with stage_timer("data_generation", tag="test_v1"):
                assert current_stage() == "train.round_01.data_generation"
            with stage_timer("epochs", tag="test_v1"):
                pass
        assert current_stage() is None
        assert db.flush(timeout=10) is True

        names = [r["stage"] for r in db.query_pipeline_stages(tag="test_v1")]
        # All resolve to full dot-names; rows come back in start_time order, so the
        # umbrella (which started first) precedes its children
        assert names == [
            "train.round_01",
            "train.round_01.data_generation",
            "train.round_01.epochs",
        ]

    def test_deep_nesting(self, db):
        with (
            stage_timer("inference.preprocess_cadence_001", tag="test_v1"),
            stage_timer("read_ed_on1", tag="test_v1"),
        ):
            pass
        assert db.flush(timeout=10) is True
        names = {r["stage"] for r in db.query_pipeline_stages(tag="test_v1")}
        assert "inference.preprocess_cadence_001.read_ed_on1" in names

    def test_decorator_usage(self, db):
        @stage_timer("train.final_save", tag="test_v1")
        def do_save():
            return 42

        assert do_save() == 42
        assert do_save() == 42  # Reusable across calls (fresh enter/exit each time)
        assert db.flush(timeout=10) is True
        rows = db.query_pipeline_stages(stage="train.final_save", tag="test_v1")
        assert len(rows) == 2

    def test_exception_recorded_and_propagates(self, db):
        with pytest.raises(RuntimeError, match="boom"), stage_timer("train.rf", tag="test_v1"):
            raise RuntimeError("boom")
        assert current_stage() is None  # Stack unwound despite the exception
        assert db.flush(timeout=10) is True

        rows = db.query_pipeline_stages(tag="test_v1")
        assert len(rows) == 1
        metadata = json.loads(rows[0]["metadata"])
        assert metadata["status"] == "failed"
        assert "boom" in metadata["error"]

    def test_metadata_passthrough(self, db):
        with stage_timer(
            "train.round_01.data_generation", tag="test_v1", metadata={"source": "in-process"}
        ):
            pass
        assert db.flush(timeout=10) is True
        rows = db.query_pipeline_stages(tag="test_v1")
        assert json.loads(rows[0]["metadata"]) == {"source": "in-process"}

    def test_threads_do_not_share_nesting(self, db):
        recorded = []

        def worker():
            with stage_timer("inference.preprocess_cadence_001", tag="test_v1"):
                recorded.append(current_stage())

        with stage_timer("train.round_01", tag="test_v1"):
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()

        # The worker thread's timer must NOT nest under the main thread's umbrella
        assert recorded == ["inference.preprocess_cadence_001"]

    def test_no_db_is_a_noop(self, caplog):
        # No Database instance exists (conftest resets singletons): timing must degrade
        # to a debug log, never an exception
        assert Database._instance is None
        with caplog.at_level(logging.DEBUG, logger="aetherscan.benchmark"):
            with stage_timer("train.round_01"):
                pass
            record_stage("train.round_02", 1.0, 2.0)
        # The missing-DB path logs the documented debug message (one per dropped span)...
        debug_messages = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        assert sum("No database instance" in m for m in debug_messages) == 2
        # ...and nothing louder: no WARNING+ from any logger (INFO+ can reach Slack)
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

    def test_record_stage_explicit_timestamps(self, db):
        record_stage(
            "train.round_03.data_generation",
            50.0,
            80.0,
            tag="test_v1",
            metadata={"source": "producer"},
        )
        assert db.flush(timeout=10) is True
        rows = db.query_pipeline_stages(tag="test_v1")
        assert len(rows) == 1
        assert rows[0]["start_time"] == 50.0
        assert rows[0]["end_time"] == 80.0
        assert rows[0]["duration_s"] == 30.0
        assert json.loads(rows[0]["metadata"]) == {"source": "producer"}


class TestMonitorAnnotationSpans:
    def test_depth_filter_and_sort(self):
        rows = [
            {"stage": "train.round_01.epochs", "start_time": 10.0, "end_time": 20.0},
            {"stage": "train.round_01", "start_time": 5.0, "end_time": 25.0},
            {"stage": "train.load_backgrounds", "start_time": 1.0, "end_time": 4.0},
            {"stage": "inference.infer_cadence_001.encode", "start_time": 30.0, "end_time": 31.0},
            {"stage": "inference.viz", "start_time": 40.0, "end_time": 41.0},
        ]
        spans = select_annotation_spans(rows)
        assert [s["stage"] for s in spans] == [
            "train.load_backgrounds",
            "train.round_01",
            "inference.viz",
        ]

    def test_empty_input(self):
        assert select_annotation_spans([]) == []


def _row(stage, start, end, metadata=None):
    return {
        "stage": stage,
        "start_time": start,
        "end_time": end,
        "duration_s": end - start,
        "tag": "test_v1",
        "metadata": json.dumps(metadata) if metadata else None,
    }


@pytest.fixture
def synthetic_rows():
    """A small two-round training run: round 1's data generation dominates (60 of 100 s)."""
    return [
        _row("train.load_backgrounds", 0.0, 10.0),
        _row("train.round_01", 10.0, 110.0),
        _row("train.round_01.data_generation", 10.0, 70.0, {"source": "in-process"}),
        _row("train.round_01.epochs", 70.0, 100.0),
        _row("train.round_01.plots", 100.0, 105.0),
        _row("train.round_01.checkpoint_save", 105.0, 110.0),
        _row("train.round_02", 110.0, 160.0),
        _row("train.round_02.epochs", 110.0, 155.0),
        _row("train.final_save", 160.0, 170.0),
    ]


class TestReportTreeMath:
    def test_tree_structure_and_totals(self, synthetic_rows):
        root = benchmark_report.build_stage_tree(synthetic_rows)
        train = root.children["train"]

        # Pure grouping node: total = sum of children (10 + 100 + 50 + 10)
        assert train.spans == []
        assert train.total_duration == pytest.approx(170.0)

        round_01 = train.children["round_01"]
        # Umbrella node with its own span: total = own span, not the children's sum
        assert round_01.total_duration == pytest.approx(100.0)
        assert round_01.children["data_generation"].total_duration == pytest.approx(60.0)
        # Self time = umbrella minus children (100 - 60 - 30 - 5 - 5)
        assert round_01.self_duration == pytest.approx(0.0)
        assert train.children["round_02"].self_duration == pytest.approx(5.0)

    def test_multiple_spans_accumulate_on_one_node(self):
        rows = [
            _row("inference.infer_cadence_001.encode", 0.0, 2.0),
            _row("inference.infer_cadence_001.encode", 5.0, 6.0),
        ]
        root = benchmark_report.build_stage_tree(rows)
        encode = root.children["inference"].children["infer_cadence_001"].children["encode"]
        assert len(encode.spans) == 2
        assert encode.total_duration == pytest.approx(3.0)

    def test_concurrent_children_never_go_negative(self):
        # Producer-style overlap: the child's span exceeds the parent's wall-clock
        rows = [
            _row("train.round_02", 100.0, 110.0),
            _row("train.round_02.data_generation", 50.0, 108.0),
        ]
        root = benchmark_report.build_stage_tree(rows)
        node = root.children["train"].children["round_02"]
        assert node.self_duration == 0.0

    def test_top_slowest_ranked_by_self_time(self, synthetic_rows):
        root = benchmark_report.build_stage_tree(synthetic_rows)
        slowest = benchmark_report.top_slowest(root, k=3)
        # data_generation (60s, leaf) beats epochs (45s round 2) beats epochs (30s round 1)
        assert slowest[0].full_name == "train.round_01.data_generation"
        assert slowest[0].self_duration == pytest.approx(60.0)
        assert slowest[1].full_name == "train.round_02.epochs"

    def test_format_tree_lines(self, synthetic_rows):
        root = benchmark_report.build_stage_tree(synthetic_rows)
        lines = benchmark_report.format_tree(root)
        text = "\n".join(lines)
        assert "train" in text
        assert "round_01" in text
        assert "data_generation" in text
        # data_generation is 60% of round_01's 100 s
        datagen_line = next(line for line in lines if "data_generation" in line)
        assert "60.0%" in datagen_line

    def test_format_duration(self):
        assert benchmark_report.format_duration(45.3) == "45.3s"
        assert benchmark_report.format_duration(125) == "2m 05s"
        assert benchmark_report.format_duration(8010) == "2h 13m"


class TestReportSuggestions:
    def _make_db(self, tmp_path, stage_rows, resource_rows=()):
        db_path = str(tmp_path / "aetherscan.db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE pipeline_stages (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " stage TEXT NOT NULL, start_time REAL NOT NULL, end_time REAL NOT NULL,"
            " duration_s REAL NOT NULL, tag TEXT, metadata TEXT)"
        )
        conn.execute(
            "CREATE TABLE system_resources (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " timestamp REAL NOT NULL, resource_type TEXT NOT NULL,"
            " resource_name TEXT NOT NULL, value REAL NOT NULL, unit TEXT, tag TEXT,"
            " metadata TEXT)"
        )
        conn.executemany(
            "INSERT INTO pipeline_stages (stage, start_time, end_time, duration_s, tag,"
            " metadata) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    r["stage"],
                    r["start_time"],
                    r["end_time"],
                    r["duration_s"],
                    r["tag"],
                    r["metadata"],
                )
                for r in stage_rows
            ],
        )
        conn.executemany(
            "INSERT INTO system_resources (timestamp, resource_type, resource_name, value,"
            " unit, tag, metadata) VALUES (?, ?, ?, ?, 'percent', 'test_v1', NULL)",
            resource_rows,
        )
        conn.commit()
        conn.close()
        return db_path

    def test_data_generation_rule_fires(self, tmp_path, synthetic_rows):
        db_path = self._make_db(tmp_path, synthetic_rows)
        root = benchmark_report.build_stage_tree(synthetic_rows)
        suggestions = benchmark_report.build_suggestions(root, db_path, "test_v1")
        datagen = [s for s in suggestions if "data_generation" in s]
        assert len(datagen) == 1
        assert "round_01" in datagen[0]
        assert "overlap-data-generation" in datagen[0]  # in-process source -> enable overlap

    def test_fully_overlapped_producer_generation_is_quiet(self, tmp_path):
        # Producer generated round 2's data entirely during round 1: the span is large
        # relative to round 2's wall-clock, but round 2 never waited — no suggestion
        rows = [
            _row("train.round_01", 0.0, 100.0),
            _row("train.round_01.epochs", 0.0, 100.0),
            _row("train.round_02", 100.0, 150.0),
            _row("train.round_02.data_generation", 10.0, 95.0, {"source": "producer"}),
            _row("train.round_02.epochs", 100.0, 145.0),
        ]
        db_path = self._make_db(tmp_path, rows)
        root = benchmark_report.build_stage_tree(rows)
        suggestions = benchmark_report.build_suggestions(root, db_path, "test_v1")
        assert not [s for s in suggestions if "data_generation" in s]

    def test_producer_generation_that_stalled_the_round_fires(self, tmp_path):
        # Generation ran past the point epochs could start: the round waited on it
        rows = [
            _row("train.round_02", 100.0, 200.0),
            _row("train.round_02.data_generation", 50.0, 160.0, {"source": "producer"}),
            _row("train.round_02.epochs", 160.0, 195.0),
        ]
        db_path = self._make_db(tmp_path, rows)
        root = benchmark_report.build_stage_tree(rows)
        suggestions = benchmark_report.build_suggestions(root, db_path, "test_v1")
        stalled = [s for s in suggestions if "data_generation" in s]
        assert len(stalled) == 1
        assert "waited on background generation" in stalled[0]

    def test_gpu_util_rule_fires(self, tmp_path, synthetic_rows):
        # GPU samples inside round 1's epochs span (70-100 s) averaging ~10%
        resource_rows = [(t, "gpu", "GPU:0_utilization", 10.0) for t in range(70, 100, 5)]
        db_path = self._make_db(tmp_path, synthetic_rows, resource_rows)
        root = benchmark_report.build_stage_tree(synthetic_rows)
        suggestions = benchmark_report.build_suggestions(root, db_path, "test_v1")
        assert any("input-bound" in s and "round_01.epochs" in s for s in suggestions)

    def test_ram_rule_fires(self, tmp_path, synthetic_rows):
        resource_rows = [(30.0, "ram", "system_total", 95.0)]
        db_path = self._make_db(tmp_path, synthetic_rows, resource_rows)
        root = benchmark_report.build_stage_tree(synthetic_rows)
        suggestions = benchmark_report.build_suggestions(root, db_path, "test_v1")
        ram = [s for s in suggestions if "RAM" in s]
        assert len(ram) == 1
        assert "95.0%" in ram[0]
        # Attributed to the deepest span covering t=30: round_01.data_generation
        assert "train.round_01.data_generation" in ram[0]

    def test_preprocess_domination_rule_fires(self, tmp_path):
        rows = [
            _row("inference.preprocess_cadence_001", 0.0, 70.0),
            _row("inference.infer_cadence_001", 70.0, 100.0),
        ]
        db_path = self._make_db(tmp_path, rows)
        root = benchmark_report.build_stage_tree(rows)
        suggestions = benchmark_report.build_suggestions(root, db_path, "test_v1")
        assert any("energy-detection preprocessing" in s for s in suggestions)

    def test_quiet_run_yields_no_suggestions(self, tmp_path):
        rows = [
            _row("train.round_01", 0.0, 100.0),
            _row("train.round_01.data_generation", 0.0, 10.0),  # 10% < 30% threshold
            _row("train.round_01.epochs", 10.0, 100.0),
        ]
        # Healthy GPU utilization during epochs
        resource_rows = [(t, "gpu", "GPU:0_utilization", 85.0) for t in range(10, 100, 10)]
        db_path = self._make_db(tmp_path, rows, resource_rows)
        root = benchmark_report.build_stage_tree(rows)
        assert benchmark_report.build_suggestions(root, db_path, "test_v1") == []


class TestReportEndToEnd:
    def test_load_rows_and_render_png(self, tmp_path, synthetic_rows):
        db_path = TestReportSuggestions()._make_db(tmp_path, synthetic_rows)
        rows = benchmark_report.load_rows(db_path, "test_v1")
        assert len(rows) == len(synthetic_rows)
        assert rows[0]["stage"] == "train.load_backgrounds"  # ORDER BY start_time

        root = benchmark_report.build_stage_tree(rows)
        png = str(tmp_path / "plots" / "benchmark_report_test_v1.png")
        benchmark_report.render_report_png(root, rows, "test_v1", png)
        assert os.path.exists(png)
        assert os.path.getsize(png) > 0

    def test_main_missing_tag_returns_nonzero(self, tmp_path, synthetic_rows, capsys):
        db_path = TestReportSuggestions()._make_db(tmp_path, synthetic_rows)
        assert benchmark_report.main(["--save-tag", "no_such_tag", "--db-path", db_path]) == 1

    def test_main_end_to_end(self, tmp_path, synthetic_rows, capsys):
        db_path = TestReportSuggestions()._make_db(tmp_path, synthetic_rows)
        out_dir = str(tmp_path / "plots")
        assert (
            benchmark_report.main(
                ["--save-tag", "test_v1", "--db-path", db_path, "--output-dir", out_dir]
            )
            == 0
        )
        captured = capsys.readouterr().out
        assert "Benchmark report for tag 'test_v1'" in captured
        assert "Suggestions:" in captured
        assert os.path.exists(os.path.join(out_dir, "benchmark_report_test_v1.png"))
