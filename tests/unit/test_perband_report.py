"""Unit tests for utils/perband_report.py — the per-band inference-performance plot.

Builds a tiny synthetic pipeline_stages SQLite DB (a few `inference.preprocess_cadence_<N>`
umbrella spans plus `.extract`/`.read_ed`/`.dedup` children that must be excluded) and a tiny
catalog CSV (mixed bands, one flagged cadence with the wrong obs count), then exercises the
plan-index join, the per-band aggregation, and the PNG render. Stdlib + matplotlib + numpy only
(no TensorFlow), so it runs fast; skipped cleanly when matplotlib isn't installed."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

# The report tool imports matplotlib at module load; skip the whole module without it.
pytest.importorskip("matplotlib")

# utils/ is not a package — load the tool straight from its file (same pattern as
# tests/unit/test_benchmark.py loads benchmark_report). The sys.modules registration must
# precede exec_module: the module's @dataclass resolves its own module by name at class
# creation (PEP 563 string annotations).
_TOOL_PATH = Path(__file__).resolve().parents[2] / "utils" / "perband_report.py"
_spec = importlib.util.spec_from_file_location("perband_report", _TOOL_PATH)
perband_report = importlib.util.module_from_spec(_spec)
sys.modules["perband_report"] = perband_report
_spec.loader.exec_module(perband_report)


def _make_db(db_path: Path, tag: str, umbrella: dict[int, float], children: dict[str, float]):
    """Create a minimal pipeline_stages DB and insert umbrella + child spans for `tag`.

    `umbrella` maps N -> duration for `inference.preprocess_cadence_<N:03d>` spans; `children`
    maps a full child stage name -> duration (e.g. `inference.preprocess_cadence_001.extract`).
    """
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE pipeline_stages ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, stage TEXT NOT NULL, start_time REAL NOT NULL, "
        "end_time REAL NOT NULL, duration_s REAL NOT NULL, tag TEXT, metadata TEXT)"
    )
    t = 1000.0
    rows = [(f"inference.preprocess_cadence_{n:03d}", dur) for n, dur in sorted(umbrella.items())]
    rows += list(children.items())
    for stage, dur in rows:
        conn.execute(
            "INSERT INTO pipeline_stages (stage, start_time, end_time, duration_s, tag, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (stage, t, t + dur, dur, tag, None),
        )
        t += dur
    conn.commit()
    conn.close()


# The four valid cadences (planner order) + one flagged cadence (3 obs) the planner skips.
_VALID_GROUPS = [
    ({"Target": "HIP1", "Session": "S1", "Band": "L", "Cadence ID": "1", "Frequency": "1400.0"}, 6),
    ({"Target": "HIP2", "Session": "S1", "Band": "C", "Cadence ID": "2", "Frequency": "5000.0"}, 6),
    ({"Target": "HIP3", "Session": "S1", "Band": "L", "Cadence ID": "3", "Frequency": "1500.0"}, 6),
    ({"Target": "HIP4", "Session": "S1", "Band": "X", "Cadence ID": "4", "Frequency": "9000.0"}, 6),
]
_FLAGGED_GROUP = (
    {"Target": "HIP5", "Session": "S1", "Band": "S", "Cadence ID": "5", "Frequency": "2500.0"},
    3,
)
# N -> preprocess wall (seconds): L={100, 300}, C={200}, X={50}
_UMBRELLA = {1: 100.0, 2: 200.0, 3: 300.0, 4: 50.0}
_CHILDREN = {
    "inference.preprocess_cadence_001.extract": 40.0,
    "inference.preprocess_cadence_001.read_ed": 30.0,
    "inference.preprocess_cadence_001.dedup": 5.0,
}


def _catalog(make_inference_csv, groups):
    """Build a catalog CSV from (key_dict, n_obs) pairs via the repo's make_inference_csv."""
    return make_inference_csv(
        filename="perband_catalog.csv",
        groups=[(key, [f"/data/{key['Target']}_{i}.h5" for i in range(n)]) for key, n in groups],
    )


def test_umbrella_durations_exclude_children(tmp_path):
    db_path = tmp_path / "aetherscan.db"
    _make_db(db_path, "test_v1", _UMBRELLA, _CHILDREN)

    durations = perband_report.load_umbrella_preprocess_durations(str(db_path), "test_v1")

    # Only the four umbrella spans survive; the .extract/.read_ed/.dedup children are dropped.
    assert durations == {1: 100.0, 2: 200.0, 3: 300.0, 4: 50.0}


def test_map_catalog_cadences_skips_flagged_group(make_inference_csv):
    csv_path = _catalog(make_inference_csv, _VALID_GROUPS + [_FLAGGED_GROUP])

    mapping = perband_report.map_catalog_cadences(str(csv_path))

    # The 3-obs S cadence is flagged (no planner index / span), so only the four valid cadences
    # map, with global 1-based planner indices in first-appearance order.
    assert mapping == [
        (1, "L", 1400.0),
        (2, "C", 5000.0),
        (3, "L", 1500.0),
        (4, "X", 9000.0),
    ]


def test_aggregate_by_band_counts_and_order(make_inference_csv, tmp_path):
    db_path = tmp_path / "aetherscan.db"
    _make_db(db_path, "test_v1", _UMBRELLA, _CHILDREN)
    csv_path = _catalog(make_inference_csv, _VALID_GROUPS + [_FLAGGED_GROUP])

    durations = perband_report.load_umbrella_preprocess_durations(str(db_path), "test_v1")
    mapping = perband_report.map_catalog_cadences(str(csv_path))
    rows = [
        perband_report.CadenceBandRow(n=n, band=band, frequency_mhz=freq, preprocess_s=durations[n])
        for n, band, freq in mapping
    ]

    agg = perband_report.aggregate_by_band(rows)

    # Per-band counts, plus [L, S, C, X] ordering with S absent.
    assert list(agg.keys()) == ["L", "C", "X"]
    assert {band: agg[band]["n"] for band in agg} == {"L": 2, "C": 1, "X": 1}
    # L = {100, 300}: median 200, p90 280, max 300.
    assert agg["L"]["median"] == pytest.approx(200.0)
    assert agg["L"]["p90"] == pytest.approx(280.0)
    assert agg["L"]["max"] == pytest.approx(300.0)


def test_render_writes_nonempty_png(make_inference_csv, tmp_path):
    db_path = tmp_path / "aetherscan.db"
    _make_db(db_path, "test_v1", _UMBRELLA, _CHILDREN)
    csv_path = _catalog(make_inference_csv, _VALID_GROUPS + [_FLAGGED_GROUP])
    out_png = tmp_path / "plots" / "perband_inference_perf_test_v1.png"

    result = perband_report.render_perband_report(
        str(db_path), "test_v1", str(csv_path), str(out_png), "testhost"
    )

    assert result == str(out_png)
    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_render_accepts_display_tag_without_affecting_db_query(make_inference_csv, tmp_path):
    # The display tag scopes only the on-figure title; `tag` still keys the pipeline_stages
    # query. If render mistakenly queried on the display tag it would find no umbrella spans and
    # return None — so a successful render here proves the two are decoupled.
    db_path = tmp_path / "aetherscan.db"
    _make_db(db_path, "test_v1", _UMBRELLA, _CHILDREN)
    csv_path = _catalog(make_inference_csv, _VALID_GROUPS + [_FLAGGED_GROUP])
    out_png = tmp_path / "plots" / "perband_inference_perf_inf_testhost_20260101_120000.png"

    result = perband_report.render_perband_report(
        str(db_path),
        "test_v1",
        str(csv_path),
        str(out_png),
        "testhost",
        display_tag="inf_testhost_20260101_120000",
    )

    assert result == str(out_png)
    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_render_skips_on_count_mismatch(make_inference_csv, tmp_path):
    # DB has only three umbrella spans, but the catalog maps four valid cadences: the plan-index
    # guard must skip (return None) rather than render a misleading plot.
    db_path = tmp_path / "aetherscan.db"
    _make_db(db_path, "test_v1", {1: 100.0, 2: 200.0, 3: 300.0}, {})
    csv_path = _catalog(make_inference_csv, _VALID_GROUPS)
    out_png = tmp_path / "plots" / "perband_inference_perf_test_v1.png"

    result = perband_report.render_perband_report(
        str(db_path), "test_v1", str(csv_path), str(out_png), "testhost"
    )

    assert result is None
    assert not out_png.exists()


def test_map_catalog_cadences_missing_required_column(tmp_path):
    # A catalog missing a required grouping column (here 'Band') can't reproduce the planner's
    # cadence grouping, so the map degrades to None (-> the plot is skipped) rather than
    # silently regrouping on a subset of columns and mis-counting cadences.
    csv_path = tmp_path / "no_band.csv"
    csv_path.write_text(
        "Target,Session,Cadence ID,Frequency,.h5 path\nHIP1,S1,1,1400.0,/data/a.h5\n"
    )
    assert perband_report.map_catalog_cadences(str(csv_path)) is None
