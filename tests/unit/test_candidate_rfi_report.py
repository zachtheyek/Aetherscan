"""Unit tests for utils/candidate_rfi_report.py (#396) — the standalone candidate
RFI-triage report. Builds a tiny synthetic inference_results SQLite DB and exercises the
coincidence binning, known-allocation flags, exclusion accounting, and the CSV emit.
Stdlib + numpy only (no TensorFlow, no aetherscan imports — mirroring the tool itself)."""

from __future__ import annotations

import csv
import importlib.util
import sqlite3
import sys
from pathlib import Path

# utils/ is not a package — load the tool straight from its file (same pattern as
# tests/unit/test_perband_report.py).
_TOOL_PATH = Path(__file__).resolve().parents[2] / "utils" / "candidate_rfi_report.py"
_spec = importlib.util.spec_from_file_location("candidate_rfi_report", _TOOL_PATH)
candidate_rfi_report = importlib.util.module_from_spec(_spec)
sys.modules["candidate_rfi_report"] = candidate_rfi_report
_spec.loader.exec_module(candidate_rfi_report)

TAG = "inf_test"


def _make_db(db_path: Path, rows: list[tuple]):
    """Minimal inference_results DB: (target, band, frequency_mhz, confidence, superseded,
    prediction) per row; npy_path/snippet_index synthesized."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE inference_results ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, npy_path TEXT, snippet_index INTEGER, "
        "prediction INTEGER, confidence REAL, target TEXT, band TEXT, frequency_mhz REAL, "
        "mc_mean REAL, mc_std REAL, tag TEXT, superseded INTEGER DEFAULT 0)"
    )
    for i, (target, band, freq, conf, superseded, prediction) in enumerate(rows):
        conn.execute(
            "INSERT INTO inference_results (npy_path, snippet_index, prediction, confidence, "
            "target, band, frequency_mhz, mc_mean, mc_std, tag, superseded) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"/x/{target}.npy",
                i,
                prediction,
                conf,
                target,
                band,
                freq,
                conf,
                0.001,
                TAG,
                superseded,
            ),
        )
    conn.commit()
    conn.close()


def _standard_rows():
    """Three targets on GPS L1 (coincident + known allocation), one Iridium hit, two clean
    candidates on distinct frequencies, plus one superseded and one negative row that must
    both be invisible to the report."""
    return [
        ("T1", "L", 1575.40, 1.0, 0, 1),
        ("T2", "L", 1575.45, 1.0, 0, 1),
        ("T3", "L", 1575.42, 1.0, 0, 1),
        ("T4", "L", 1620.0, 0.999, 0, 1),
        ("T1", "C", 8438.0, 0.995, 0, 1),
        ("T5", "X", 9500.0, 0.992, 0, 1),
        ("GHOST", "L", 1575.42, 1.0, 1, 1),  # superseded -> ignored
        ("NEG", "L", 1575.42, 0.2, 0, 0),  # not a candidate -> ignored
    ]


class TestLoadAndFlags:
    def test_load_candidates_filters_superseded_and_negatives(self, tmp_path):
        db = tmp_path / "a.db"
        _make_db(db, _standard_rows())
        candidates = candidate_rfi_report.load_candidates(str(db), TAG)
        assert len(candidates) == 6
        assert all(r["target"] not in ("GHOST", "NEG") for r in candidates)

    def test_known_band_hits(self, tmp_path):
        db = tmp_path / "a.db"
        _make_db(db, _standard_rows())
        candidates = candidate_rfi_report.load_candidates(str(db), TAG)
        hits = candidate_rfi_report.known_band_hits(candidates)
        assert len(hits["GPS L1"]) == 3
        assert len(hits["Iridium"]) == 1
        assert "GLONASS L1" not in hits

    def test_coincidence_bins_require_min_targets(self, tmp_path):
        db = tmp_path / "a.db"
        _make_db(db, _standard_rows())
        candidates = candidate_rfi_report.load_candidates(str(db), TAG)
        bins = candidate_rfi_report.coincidence_bins(candidates, 0.2, 3)
        # Only the GPS L1 cluster spans >= 3 distinct targets in one 0.2 MHz bin
        assert len(bins) == 1
        assert bins[0]["n_targets"] == 3
        assert bins[0]["targets"] == ["T1", "T2", "T3"]
        # Raising the floor above the cluster size flags nothing
        assert candidate_rfi_report.coincidence_bins(candidates, 0.2, 4) == []


class TestReportAndCli:
    def test_build_report_sections_and_flag_rows(self, tmp_path):
        db = tmp_path / "a.db"
        _make_db(db, _standard_rows())
        candidates = candidate_rfi_report.load_candidates(str(db), TAG)
        report, flagged = candidate_rfi_report.build_report(candidates, 0.2, 3, [(1616.0, 1626.5)])
        assert "candidates: 6" in report
        assert "GPS L1: 3 candidate(s) across 3 target(s)" in report
        assert "excluded: 1" in report
        assert "reported after exclusion: 5" in report
        by_target = {(r["target"], r["frequency_mhz"]): r for r in flagged}
        assert by_target[("T1", 1575.40)]["multi_target_coincident"] == 1
        assert by_target[("T1", 1575.40)]["known_rfi_allocation"] == "GPS L1"
        assert by_target[("T4", 1620.0)]["excluded_by_range"] == 1
        assert by_target[("T5", 9500.0)]["multi_target_coincident"] == 0
        assert by_target[("T5", 9500.0)]["known_rfi_allocation"] == ""

    def test_cli_end_to_end_with_csv(self, tmp_path, capsys):
        db = tmp_path / "a.db"
        _make_db(db, _standard_rows())
        out_csv = tmp_path / "flags.csv"
        rc = candidate_rfi_report.main(
            [
                "--save-tag",
                TAG,
                "--db-path",
                str(db),
                "--csv",
                str(out_csv),
                "--exclude-frequency-range",
                "1616",
                "1626.5",
            ]
        )
        assert rc == 0
        printed = capsys.readouterr().out
        assert "multi-target coincidence bins" in printed
        with open(out_csv) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 6
        assert {r["multi_target_coincident"] for r in rows} == {"0", "1"}

    def test_cli_rejects_bad_args(self, tmp_path, capsys):
        db = tmp_path / "a.db"
        _make_db(db, _standard_rows())
        assert (
            candidate_rfi_report.main(
                ["--save-tag", TAG, "--db-path", str(db), "--min-targets", "1"]
            )
            == 1
        )
        assert (
            candidate_rfi_report.main(
                ["--save-tag", TAG, "--db-path", str(db), "--coincidence-bin-mhz", "0"]
            )
            == 1
        )
        assert candidate_rfi_report.main(["--save-tag", "absent", "--db-path", str(db)]) == 1

    def test_db_opened_read_only(self, tmp_path):
        db = tmp_path / "a.db"
        _make_db(db, _standard_rows())
        before = db.read_bytes()
        candidate_rfi_report.main(["--save-tag", TAG, "--db-path", str(db)])
        assert db.read_bytes() == before
