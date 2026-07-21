#!/usr/bin/env python3
"""
Live dashboard for an Aetherscan pipeline run, read straight from its SQLite DB.

Renders, in one auto-refreshing page, every run metric that is reconstructable from the DB —
resource utilization (CPU/RAM/GPU), beta-VAE loss + stability curves, the signal-injection
stats suite, a live latent-space scatter, the stage timeline (PR #134), and inference candidate
stats — plus a gallery of every saved plot PNG (RF diagnostics, latent traversals, inference
figures) as it appears on disk.

It is STANDALONE — no `aetherscan` imports (like utils/benchmark_report.py), only stdlib sqlite3 +
numpy + pandas + plotly + streamlit — so streamlit can execute it directly and it runs against a
live run's DB or one fetched from a cluster with utils/fetch_run_outputs.sh. It ships inside the
package (src/aetherscan/dashboard.py) so every install method (pip `aetherscan[dashboard]`, the
container, or a source checkout) can auto-launch it. main.py auto-launches it (--no-dashboard to
opt out); to run it by hand against a saved DB, point streamlit at the installed module file:

    streamlit run "$(python -c 'import aetherscan, os; \
        print(os.path.join(os.path.dirname(aetherscan.__file__), "dashboard.py"))')" \
        -- --db-path /path/to/aetherscan.db --tag final_v1

To watch a run on a cluster, SSH-forward the port:  ssh -L 8501:localhost:8501 blpc3

Read-only (mode=ro); the pipeline's writer-thread journaling makes concurrent reads safe. The data
layer (load_* / parse / pca helpers) is pure and unit-tested against a synthetic DB; the Streamlit
UI (render*) is a thin rendering shell that imports plotly/streamlit lazily.

Auto-refresh uses a blocking `time.sleep(refresh) + st.rerun()` — deliberately version-robust (works
on any Streamlit), at the cost of freezing sidebar/tooltip interaction during the sleep window. For
a smoother non-blocking refresh, `pip install streamlit-autorefresh` (or use `st.fragment(run_every=)`
on Streamlit >= 1.33) and swap it in.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Data layer — pure, unit-testable, no Streamlit                              #
# --------------------------------------------------------------------------- #

# Non-superseded filter reused across the supersede-aware tables
_ALIVE = "COALESCE(superseded, 0) = 0"

# beta-VAE stat_names split into the two training figures (train.py loss curves / stability)
_LOSS_STATS = [
    "total_loss",
    "reconstruction_loss",
    "kl_loss",
    "true_loss",
    "false_loss",
    "learning_rate",
]
_STABILITY_STATS = ["clipping_rate", "gradient_norm_mean", "gradient_norm_std", "gradient_norm_max"]


def connect_ro(db_path: str) -> sqlite3.Connection:
    """Open the DB strictly read-only (won't create/lock a missing file for writing)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        is not None
    )


def list_tags(conn: sqlite3.Connection) -> list[str]:
    """All run tags present anywhere in the DB, sorted."""
    tags: set[str] = set()
    for table in ("pipeline_stages", "system_resources", "training_stats", "inference_cadences"):
        try:
            rows = conn.execute(
                f"SELECT DISTINCT tag FROM {table} WHERE tag IS NOT NULL"  # noqa: S608 (fixed names)
            ).fetchall()
            tags.update(r[0] for r in rows)
        except sqlite3.OperationalError:
            continue  # table absent in an older schema
    return sorted(tags)


def load_resources(conn: sqlite3.Connection, tag: str) -> pd.DataFrame:
    """system_resources rows for `tag` (timestamp, resource_type, resource_name, value, unit)."""
    return pd.read_sql_query(
        "SELECT timestamp, resource_type, resource_name, value, unit "
        "FROM system_resources WHERE tag = ? ORDER BY timestamp",
        conn,
        params=(tag,),
    )


def load_training_stats(conn: sqlite3.Connection, tag: str, stats: list[str]) -> pd.DataFrame:
    """Non-superseded beta_vae training_stats for `tag` limited to `stats` (and their val_
    counterparts), ordered so curves plot in run order (round, epoch, timestamp)."""
    wanted = list(stats) + [f"val_{s}" for s in stats]
    placeholders = ",".join("?" * len(wanted))
    df = pd.read_sql_query(
        "SELECT timestamp, stat_name, value, round_number, epoch_number "
        f"FROM training_stats WHERE tag = ? AND model_name = 'beta_vae' AND {_ALIVE} "
        f"AND stat_name IN ({placeholders}) "  # noqa: S608 (placeholders are bound params)
        "ORDER BY round_number, epoch_number, timestamp",
        conn,
        params=(tag, *wanted),
    )
    return df


def load_injection_stats(conn: sqlite3.Connection, tag: str) -> pd.DataFrame:
    """Non-superseded injection_stats for `tag` (stat_name, value, signal_type, injection_stage,
    round_number, is_finite, slope_clamped, timestamp)."""
    return pd.read_sql_query(
        "SELECT timestamp, stat_name, value, signal_type, injection_stage, round_number, "
        "is_finite, slope_clamped "
        f"FROM injection_stats WHERE tag = ? AND {_ALIVE} ORDER BY timestamp",
        conn,
        params=(tag,),
    )


def load_latent_snapshots_latest(conn: sqlite3.Connection, tag: str) -> pd.DataFrame:
    """The most recent latent_snapshots frame for `tag` (all rows sharing the max
    round/epoch/step), as (signal_type, latent_vector[JSON]). Empty if none."""
    if not _table_exists(conn, "latent_snapshots"):
        return pd.DataFrame(columns=["signal_type", "latent_vector"])
    key = conn.execute(
        "SELECT round_number, epoch_number, step_number FROM latent_snapshots "
        f"WHERE tag = ? AND {_ALIVE} "
        "ORDER BY round_number DESC, epoch_number DESC, step_number DESC LIMIT 1",
        (tag,),
    ).fetchone()
    if key is None:
        return pd.DataFrame(columns=["signal_type", "latent_vector"])
    return pd.read_sql_query(
        "SELECT signal_type, latent_vector FROM latent_snapshots "
        f"WHERE tag = ? AND {_ALIVE} AND round_number = ? AND epoch_number = ? AND step_number = ?",
        conn,
        params=(tag, key["round_number"], key["epoch_number"], key["step_number"]),
    )


def load_inference_results(conn: sqlite3.Connection, tag: str) -> pd.DataFrame:
    """Non-superseded inference_results for `tag` (prediction, confidence, target, band,
    frequency_mhz). Empty frame if the table is absent."""
    if not _table_exists(conn, "inference_results"):
        return pd.DataFrame(columns=["prediction", "confidence", "target", "band", "frequency_mhz"])
    return pd.read_sql_query(
        "SELECT prediction, confidence, target, band, frequency_mhz "
        f"FROM inference_results WHERE tag = ? AND {_ALIVE}",
        conn,
        params=(tag,),
    )


def load_inference_cadences(conn: sqlite3.Connection, tag: str) -> pd.DataFrame:
    """Non-superseded inference_cadences manifest rows for `tag` (status, duration_s, n_stamps,
    n_candidates). Empty frame if the table is absent."""
    if not _table_exists(conn, "inference_cadences"):
        return pd.DataFrame(columns=["status", "duration_s", "n_stamps", "n_candidates"])
    return pd.read_sql_query(
        "SELECT status, duration_s, n_stamps, n_candidates "
        f"FROM inference_cadences WHERE tag = ? AND {_ALIVE}",
        conn,
        params=(tag,),
    )


def load_stages(conn: sqlite3.Connection, tag: str) -> pd.DataFrame:
    """pipeline_stages spans for `tag` (stage, start_time, end_time, duration_s, metadata) plus a
    derived `depth` (dot-name component count), ordered by start. Empty if the table is absent."""
    if not _table_exists(conn, "pipeline_stages"):
        return pd.DataFrame(columns=["stage", "start_time", "end_time", "duration_s", "metadata"])
    df = pd.read_sql_query(
        "SELECT stage, start_time, end_time, duration_s, metadata "
        "FROM pipeline_stages WHERE tag = ? ORDER BY start_time",
        conn,
        params=(tag,),
    )
    if not df.empty:
        df["depth"] = df["stage"].str.split(".").str.len()
    return df


def parse_latent_matrix(snapshots: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Parse a latent_snapshots frame's JSON latent_vector column into an (n, d) float array +
    the matching signal_type labels. Rows whose vector is malformed, ragged, or non-finite
    (NaN/inf — json.loads accepts these and they blow up the downstream SVD) are dropped."""
    vecs, labels = [], []
    for _, row in snapshots.iterrows():
        try:
            v = json.loads(row["latent_vector"])
        except (TypeError, ValueError):
            continue
        if isinstance(v, list) and v:
            vecs.append(v)
            labels.append(row["signal_type"])
    if not vecs:
        return np.empty((0, 0)), []
    width = len(vecs[0])
    keep = [(v, s) for v, s in zip(vecs, labels, strict=False) if len(v) == width]
    mat = np.array([v for v, _ in keep], dtype=float)
    labels_keep = [s for _, s in keep]
    if mat.size:
        finite = np.isfinite(mat).all(axis=1)
        mat = mat[finite]
        labels_keep = [s for s, f in zip(labels_keep, finite, strict=False) if f]
    return mat, labels_keep


def pca_2d(matrix: np.ndarray) -> np.ndarray:
    """Project (n, d) -> (n, 2) via mean-centered SVD (numpy-only PCA; no sklearn). Returns a
    (n, 2) array; if d < 2 the missing component(s) are zero-filled."""
    if matrix.size == 0 or matrix.shape[0] == 0:
        return np.empty((0, 2))
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    # economy SVD; right singular vectors are the principal axes
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    comps = vt[:2] if vt.shape[0] >= 2 else vt
    proj = centered @ comps.T
    if proj.shape[1] == 1:  # degenerate d==1
        proj = np.hstack([proj, np.zeros((proj.shape[0], 1))])
    return proj


def list_png_artifacts(plots_dir: str, limit: int = 60) -> list[dict]:
    """Every *.png/*.gif under plots_dir (recursive), newest first: {path, name, rel, mtime}.

    Containment: plots_dir is realpath-resolved and every returned file's realpath is verified to
    stay within that root, so a symlink inside the tree can't make the gallery serve a file from
    outside the run's plots directory. (The dashboard is a single-operator local tool pointed at
    their own run via the launch args, but this keeps the file-serving surface tightly bounded.)
    """
    if not plots_dir or not os.path.isdir(plots_dir):
        return []
    base = os.path.realpath(plots_dir)
    found = []
    for root, _dirs, files in os.walk(base):
        for f in files:
            if not f.lower().endswith((".png", ".gif")):
                continue
            real = os.path.realpath(os.path.join(root, f))
            if real != base and not real.startswith(base + os.sep):
                continue  # symlink escaping the plots tree — never serve it
            try:
                mtime = os.path.getmtime(real)
            except OSError:
                continue
            found.append(
                {"path": real, "name": f, "rel": os.path.relpath(real, base), "mtime": mtime}
            )
    found.sort(key=lambda d: d["mtime"], reverse=True)
    return found[:limit]


def default_plots_dir(db_path: str) -> str:
    """Infer {output_path}/plots from the DB path (…/{output_path}/db/aetherscan.db)."""
    db_dir = os.path.dirname(os.path.abspath(db_path))  # …/db
    return os.path.join(os.path.dirname(db_dir), "plots")


def run_summary(resources: pd.DataFrame, stages: pd.DataFrame) -> dict:
    """Headline dict: wall-clock span, #stages, latest stage, peak system RAM %."""
    t_start: float | None = None
    t_end: float | None = None
    if not stages.empty:
        t_start, t_end = stages["start_time"].min(), stages["end_time"].max()
    elif not resources.empty:
        t_start, t_end = resources["timestamp"].min(), resources["timestamp"].max()
    wall = (t_end - t_start) if t_start is not None else 0.0

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
# Streamlit UI — thin shell over the data layer (imports plotly/streamlit lazily)
# --------------------------------------------------------------------------- #


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Live Aetherscan run dashboard")
    ap.add_argument("--db-path", required=True, help="Path to aetherscan.db")
    ap.add_argument("--tag", default=None, help="Run tag (else pick in the sidebar)")
    ap.add_argument(
        "--plots-dir", default=None, help="Plots dir (default: inferred from --db-path)"
    )
    ap.add_argument("--refresh", type=int, default=10, help="Auto-refresh seconds (0 = off)")
    args, _ = ap.parse_known_args()  # streamlit injects extra args after `--`
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
    import plotly.express as px  # noqa: PLC0415 (lazy so the data layer imports without plotly)
    import plotly.graph_objects as go  # noqa: PLC0415
    import streamlit as st  # noqa: PLC0415

    st.set_page_config(page_title="Aetherscan live", layout="wide")

    with st.sidebar:
        st.header("Aetherscan run")
        # DB + plots paths come from the launch args only (NOT browser-editable text inputs), so a
        # hosted instance can't be repointed at arbitrary filesystem paths from the browser.
        db_path = args.db_path
        st.caption(f"DB: `{db_path}`")
        try:
            conn = connect_ro(db_path)
        except sqlite3.OperationalError as e:
            st.error(f"Cannot open DB read-only: {e}")
            st.stop()
        tags = list_tags(conn)
        if not tags:
            conn.close()
            st.warning("No run tags found in this DB yet.")
            st.stop()
        default_idx = tags.index(args.tag) if args.tag in tags else len(tags) - 1
        tag = st.selectbox("Tag", tags, index=default_idx)
        refresh = st.number_input("Auto-refresh (s, 0=off)", 0, 600, value=args.refresh)
        plots_dir = args.plots_dir or default_plots_dir(db_path)
        st.caption(f"Plots: `{plots_dir}`")

    try:
        resources = load_resources(conn, tag)
        stages = load_stages(conn, tag)
        summary = run_summary(resources, stages)

        st.title(f"Aetherscan — {tag}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Wall clock", _fmt_duration(summary["wall_s"]))
        c2.metric("Stage spans", summary["n_stages"])
        c3.metric("Latest stage", summary["latest_stage"] or "—")
        c4.metric(
            "Peak sys RAM", f"{summary['peak_ram_pct']:.0f}%" if summary["peak_ram_pct"] else "—"
        )

        tabs = st.tabs(
            [
                "Training",
                "Injection",
                "Latent",
                "Resources",
                "Stages",
                "Inference",
                "All plots (PNG)",
            ]
        )

        # --- Training loss + stability ----------------------------------------
        with tabs[0]:
            loss = load_training_stats(conn, tag, _LOSS_STATS)
            stab = load_training_stats(conn, tag, _STABILITY_STATS)
            if loss.empty and stab.empty:
                st.caption("No beta_vae training_stats yet.")
            for title, df in (("Loss curves", loss), ("Training stability", stab)):
                if df.empty:
                    continue
                st.subheader(title)
                d = df.copy()
                d["kind"] = d["stat_name"].str.replace("^val_", "", regex=True)
                d["split"] = np.where(d["stat_name"].str.startswith("val_"), "val", "train")
                for kind in sorted(d["kind"].unique()):
                    sub = d[d["kind"] == kind].copy()
                    sub["step"] = range(len(sub))
                    fig = px.line(sub, x="step", y="value", color="split", title=kind)
                    fig.update_layout(height=240, margin={"l": 10, "r": 10, "t": 40, "b": 10})
                    st.plotly_chart(fig, use_container_width=True)

        # --- Injection stats ---------------------------------------------------
        with tabs[1]:
            inj = load_injection_stats(conn, tag)
            if inj.empty:
                st.caption("No injection_stats yet.")
            else:
                st.subheader("Injection stability (per round)")
                stability = (
                    inj.assign(nonfinite=1 - inj["is_finite"].fillna(1))
                    .groupby("round_number")[["nonfinite", "slope_clamped"]]
                    .mean()
                    .reset_index()
                )
                fig = px.line(
                    stability,
                    x="round_number",
                    y=["nonfinite", "slope_clamped"],
                    title="mean non-finite / slope-clamp rate",
                    markers=True,
                )
                fig.update_layout(height=260, margin={"l": 10, "r": 10, "t": 40, "b": 10})
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Injected-signal / intensity stat distributions")
                stat = st.selectbox("stat_name", sorted(inj["stat_name"].unique()))
                sub = inj[(inj["stat_name"] == stat) & inj["value"].notna()]
                color = "signal_type" if sub["signal_type"].notna().any() else None
                fig = px.histogram(
                    sub, x="value", color=color, barmode="overlay", nbins=60, title=stat
                )
                fig.update_layout(height=300, margin={"l": 10, "r": 10, "t": 40, "b": 10})
                st.plotly_chart(fig, use_container_width=True)

        # --- Latent scatter (live PCA of the latest snapshot) ------------------
        with tabs[2]:
            snaps = load_latent_snapshots_latest(conn, tag)
            mat, labels = parse_latent_matrix(snaps)
            if mat.shape[0] == 0:
                st.caption("No latent_snapshots yet.")
            else:
                proj = pca_2d(mat)
                df = pd.DataFrame({"pc1": proj[:, 0], "pc2": proj[:, 1], "signal_type": labels})
                st.subheader(
                    f"Latent space — latest snapshot (PCA of {mat.shape[0]}×{mat.shape[1]})"
                )
                fig = px.scatter(df, x="pc1", y="pc2", color="signal_type", opacity=0.7)
                fig.update_layout(height=520, margin={"l": 10, "r": 10, "t": 10, "b": 10})
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "Cheap PCA projection; the pipeline's saved UMAP animation is under All plots."
                )

        # --- Resource utilization ---------------------------------------------
        with tabs[3]:
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
        with tabs[4]:
            if stages.empty:
                st.caption("No pipeline_stages yet (needs PR #134's benchmarking layer).")
            else:
                s = stages.copy()
                t0 = s["start_time"].min()
                s["start_min"] = (s["start_time"] - t0) / 60.0
                s["dur_min"] = (s["end_time"] - s["start_time"]) / 60.0
                s["family"] = s["stage"].str.split(".").str[0]
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

        # --- Inference candidates ---------------------------------------------
        with tabs[5]:
            results = load_inference_results(conn, tag)
            cadences = load_inference_cadences(conn, tag)
            if results.empty and cadences.empty:
                st.caption("No inference_results / inference_cadences yet.")
            if not results.empty:
                cands = results[results["prediction"] == 1]
                st.subheader(f"Candidate confidence ({len(cands)} candidates)")
                if not cands.empty:
                    fig = px.histogram(cands, x="confidence", nbins=40)
                    fig.update_layout(height=240, margin={"l": 10, "r": 10, "t": 10, "b": 10})
                    st.plotly_chart(fig, use_container_width=True)
                    for dim in ("target", "band"):
                        if cands[dim].notna().any():
                            counts = cands[dim].value_counts().reset_index()
                            counts.columns = [dim, "candidates"]
                            fig = px.bar(
                                counts, x=dim, y="candidates", title=f"candidates per {dim}"
                            )
                            fig.update_layout(
                                height=240, margin={"l": 10, "r": 10, "t": 40, "b": 10}
                            )
                            st.plotly_chart(fig, use_container_width=True)
            if not cadences.empty:
                st.subheader("Per-cadence manifest")
                st.dataframe(cadences, use_container_width=True)

        # --- All plots (PNG gallery) ------------------------------------------
        with tabs[6]:
            pngs = list_png_artifacts(plots_dir)
            if not pngs:
                st.caption(f"No plot PNGs under {plots_dir} yet.")
            else:
                st.caption(f"{len(pngs)} figures under {plots_dir} (newest first)")
                cols = st.columns(2)
                for i, art in enumerate(pngs):
                    with cols[i % 2]:
                        st.image(art["path"], caption=art["rel"], use_container_width=True)

    finally:
        conn.close()

    if refresh and refresh > 0:
        # sleep()+rerun() is the version-robust live refresh (works on any Streamlit); for a
        # smoother non-blocking refresh, `pip install streamlit-autorefresh` and swap it in.
        import time  # noqa: PLC0415

        time.sleep(refresh)
        st.rerun()


def main() -> None:  # pragma: no cover
    render(_parse_args())


if __name__ == "__main__":
    main()
