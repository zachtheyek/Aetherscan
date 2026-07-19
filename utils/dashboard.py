#!/usr/bin/env python3
"""
Live dashboard for an Aetherscan pipeline run, read straight from its SQLite DB.

Renders, in one auto-refreshing page: resource utilization (CPU/RAM/GPU over time),
training-loss curves, and the stage timeline (pipeline_stages, from the #134 benchmarking
layer). Like utils/benchmark_report.py it is STANDALONE — no `aetherscan` imports, only
stdlib sqlite3 + pandas + plotly + streamlit — so it runs against a live run's DB or one
fetched from a cluster with utils/fetch_run_outputs.sh.

    pip install streamlit plotly pandas          # not pipeline/container deps
    streamlit run utils/dashboard.py -- --db-path /path/to/aetherscan.db --tag final_v1

To watch a run on a cluster, SSH-forward the port and read the DB where it lives:
    ssh -L 8501:localhost:8501 blpc3
    # then on the cluster: streamlit run utils/dashboard.py -- --db-path .../aetherscan.db

Read-only: opens the DB with mode=ro; the pipeline's writer-thread journaling makes
concurrent reads safe. The data layer (load_* functions) is pure and unit-tested against a
synthetic DB; the Streamlit UI is a thin rendering shell on top.
"""

from __future__ import annotations

import argparse
import sqlite3

import pandas as pd

# --------------------------------------------------------------------------- #
# Data layer — pure, unit-testable, no Streamlit                              #
# --------------------------------------------------------------------------- #

# Non-superseded filter reused across the supersede-aware tables
_ALIVE = "COALESCE(superseded, 0) = 0"


def connect_ro(db_path: str) -> sqlite3.Connection:
    """Open the DB strictly read-only (won't create/lock a missing file for writing)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def list_tags(conn: sqlite3.Connection) -> list[str]:
    """All run tags present anywhere in the DB, most-recent-looking first."""
    tags: set[str] = set()
    for table in ("pipeline_stages", "system_resources", "training_stats"):
        try:
            rows = conn.execute(
                f"SELECT DISTINCT tag FROM {table} WHERE tag IS NOT NULL"  # noqa: S608 (fixed table names)
            ).fetchall()
            tags.update(r[0] for r in rows)
        except sqlite3.OperationalError:
            continue  # table absent in an older schema
    return sorted(tags)


def load_resources(conn: sqlite3.Connection, tag: str) -> pd.DataFrame:
    """system_resources rows for `tag` as a tidy frame (timestamp, resource_type,
    resource_name, value, unit), ordered by time."""
    return pd.read_sql_query(
        "SELECT timestamp, resource_type, resource_name, value, unit "
        "FROM system_resources WHERE tag = ? ORDER BY timestamp",
        conn,
        params=(tag,),
    )


def load_training_stats(conn: sqlite3.Connection, tag: str) -> pd.DataFrame:
    """Non-superseded training_stats for `tag` (model_name, stat_name, value, round_number,
    epoch_number, timestamp), ordered so loss curves plot in run order."""
    return pd.read_sql_query(
        "SELECT timestamp, model_name, stat_name, value, round_number, epoch_number "
        f"FROM training_stats WHERE tag = ? AND {_ALIVE} "
        "ORDER BY round_number, epoch_number, timestamp",
        conn,
        params=(tag,),
    )


def load_stages(conn: sqlite3.Connection, tag: str) -> pd.DataFrame:
    """pipeline_stages spans for `tag` (stage, start_time, end_time, duration_s, metadata,
    plus a derived `depth` = dot-name component count), ordered by start."""
    df = pd.read_sql_query(
        "SELECT stage, start_time, end_time, duration_s, metadata "
        "FROM pipeline_stages WHERE tag = ? ORDER BY start_time",
        conn,
        params=(tag,),
    )
    if not df.empty:
        df["depth"] = df["stage"].str.split(".").str.len()
    return df


def run_summary(resources: pd.DataFrame, stages: pd.DataFrame) -> dict:
    """Small headline dict: wall-clock span, #stages, latest stage, peak system RAM %."""
    starts = []
    if not stages.empty:
        starts = [stages["start_time"].min(), stages["end_time"].max()]
    elif not resources.empty:
        starts = [resources["timestamp"].min(), resources["timestamp"].max()]
    wall = (starts[1] - starts[0]) if starts else 0.0

    latest_stage = None
    if not stages.empty:
        latest_stage = stages.sort_values("start_time")["stage"].iloc[-1]

    peak_ram = None
    if not resources.empty:
        ram = resources[
            (resources["resource_type"] == "ram") & (resources["resource_name"] == "system_total")
        ]
        if not ram.empty:
            peak_ram = float(ram["value"].max())

    return {
        "wall_s": float(wall),
        "n_stages": int(len(stages)),
        "latest_stage": latest_stage,
        "peak_ram_pct": peak_ram,
    }


# --------------------------------------------------------------------------- #
# Streamlit UI — thin shell over the data layer                               #
# --------------------------------------------------------------------------- #


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Live Aetherscan run dashboard")
    ap.add_argument("--db-path", required=True, help="Path to aetherscan.db")
    ap.add_argument("--tag", default=None, help="Run tag (else pick in the sidebar)")
    ap.add_argument("--refresh", type=int, default=10, help="Auto-refresh seconds (0 = off)")
    # Streamlit passes script args after `--`; ignore anything it injects
    args, _ = ap.parse_known_args()
    return args


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def render(args: argparse.Namespace) -> None:  # pragma: no cover - requires Streamlit runtime
    import plotly.express as px  # noqa: PLC0415 (kept out of module scope so the data layer imports without plotly/streamlit)
    import plotly.graph_objects as go  # noqa: PLC0415
    import streamlit as st  # noqa: PLC0415

    st.set_page_config(page_title="Aetherscan live", layout="wide")

    with st.sidebar:
        st.header("Aetherscan run")
        db_path = st.text_input("DB path", value=args.db_path)
        try:
            conn = connect_ro(db_path)
        except sqlite3.OperationalError as e:
            st.error(f"Cannot open DB read-only: {e}")
            st.stop()
        tags = list_tags(conn)
        if not tags:
            st.warning("No run tags found in this DB yet.")
            st.stop()
        default_idx = tags.index(args.tag) if args.tag in tags else len(tags) - 1
        tag = st.selectbox("Tag", tags, index=default_idx)
        refresh = st.number_input("Auto-refresh (s, 0=off)", 0, 600, value=args.refresh)

    resources = load_resources(conn, tag)
    stages = load_stages(conn, tag)
    training = load_training_stats(conn, tag)
    summary = run_summary(resources, stages)

    st.title(f"Aetherscan — {tag}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Wall clock", _fmt_duration(summary["wall_s"]))
    c2.metric("Stage spans", summary["n_stages"])
    c3.metric("Latest stage", summary["latest_stage"] or "—")
    c4.metric("Peak sys RAM", f"{summary['peak_ram_pct']:.0f}%" if summary["peak_ram_pct"] else "—")

    # --- Training loss -----------------------------------------------------
    st.subheader("Training loss")
    if training.empty:
        st.caption("No training_stats yet.")
    else:
        training = training.copy()
        training["step"] = range(len(training))
        for stat in sorted(training["stat_name"].unique()):
            sub = training[training["stat_name"] == stat]
            fig = px.line(
                sub,
                x="step",
                y="value",
                color="round_number",
                title=stat,
                markers=False,
            )
            fig.update_layout(height=260, margin={"l": 10, "r": 10, "t": 40, "b": 10})
            st.plotly_chart(fig, use_container_width=True)

    # --- Resource utilization ---------------------------------------------
    st.subheader("Resource utilization")
    if resources.empty:
        st.caption("No system_resources yet.")
    else:
        r = resources.copy()
        r["t_min"] = (r["timestamp"] - r["timestamp"].min()) / 60.0
        r["series"] = r["resource_type"] + ":" + r["resource_name"]
        for rtype in ("cpu", "ram", "gpu"):
            sub = r[r["resource_type"] == rtype]
            if sub.empty:
                continue
            fig = px.line(sub, x="t_min", y="value", color="series", title=rtype.upper())
            fig.update_layout(
                height=260,
                margin={"l": 10, "r": 10, "t": 40, "b": 10},
                xaxis_title="minutes",
                yaxis_title="%",
            )
            st.plotly_chart(fig, use_container_width=True)

    # --- Stage timeline ----------------------------------------------------
    st.subheader("Stage timeline")
    if stages.empty:
        st.caption("No pipeline_stages yet.")
    else:
        s = stages.copy()
        t0 = s["start_time"].min()
        s["start_min"] = (s["start_time"] - t0) / 60.0
        s["dur_min"] = (s["end_time"] - s["start_time"]) / 60.0
        s["family"] = s["stage"].str.split(".").str[0]
        # Numeric Gantt via horizontal bars with an explicit base (px.timeline assumes
        # datetime x-axes, which mis-renders float "minutes"). One trace per family for color.
        palette = px.colors.qualitative.Plotly
        fig = go.Figure()
        for i, fam in enumerate(sorted(s["family"].unique())):
            sub = s[s["family"] == fam]
            fig.add_bar(
                x=sub["dur_min"],
                base=sub["start_min"],
                y=sub["stage"],
                orientation="h",
                name=fam,
                marker_color=palette[i % len(palette)],
                customdata=sub[["duration_s", "depth"]],
                hovertemplate="%{y}<br>%{customdata[0]:.1f}s (depth %{customdata[1]})<extra></extra>",
            )
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="minutes since first stage")
        fig.update_layout(
            height=max(300, 22 * s["stage"].nunique()),
            margin={"l": 10, "r": 10, "t": 20, "b": 10},
            barmode="overlay",
        )
        st.plotly_chart(fig, use_container_width=True)

    conn.close()

    if refresh and refresh > 0:
        # Re-run the whole script every `refresh` seconds for a live view. sleep()+rerun() is
        # the version-robust way (works on any Streamlit); for a smoother non-blocking refresh,
        # `pip install streamlit-autorefresh` and swap in st_autorefresh().
        import time  # noqa: PLC0415

        time.sleep(refresh)
        st.rerun()


def main() -> None:  # pragma: no cover
    render(_parse_args())


if __name__ == "__main__":
    main()
