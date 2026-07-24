"""
Process-pool wrapper for the RF SHAP passes.

shap's TreeSHAP C extension is single-threaded (no OpenMP, no ``n_jobs``), so explaining a
1000-tree forest at production sample counts runs for hours-to-days — dominated by the interaction
pass. SHAP values are per-sample independent, though, so we chunk the samples across processes and
call a **stock** ``shap.TreeExplainer`` in each worker (not a fork of the algorithm — just
parallelism). Measured ~40-45x on a 96-core node, byte-identical to the single-threaded result.

This module is deliberately kept small and off the TF/training import graph. ``train.py`` imports
TensorFlow at module level (and pulls in matplotlib/umap/…), and multiprocessing's ``spawn`` re-imports
the parent's ``__main__`` (``aetherscan.main`` → TF → the whole training stack) into every worker. So
the pool runs under the **forkserver** start method with an **empty preload**: the fork server is
spawned once, clean (no ``__main__`` re-import), and workers fork from it — importing only this module
+ shap, never TF. See ``_compute_or_load_shap_values`` in ``train.py`` for the caller and
``docs/TRAINING_PIPELINE.md`` for the CPU-vs-GPU comparison behind this choice.

Design notes:
  * **Rebuild the explainer inside each worker** from the picklable sklearn RF — never pickle a
    pre-built ``TreeExplainer`` into workers (segfaults large-model workers; shap #1204). Workers
    load the RF from its persisted joblib path so the model isn't re-serialised through the pool.
  * **forkserver + empty preload** keeps workers off the TF/training stack (see above); the fork
    server is spawned once and forks lightweight workers.
  * **Pin BLAS/OpenMP threads to 1 per worker** so N workers don't oversubscribe the CPU (best-effort:
    shap's TreeSHAP is single-threaded, so this only tames incidental numpy/BLAS threads).
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# Kinds of SHAP pass a worker can run.
_RAW_KINDS = ("summary", "interaction")
_LOGLOSS_KIND = "logloss"

# Per-worker globals, populated once by the pool initializer (or in-process for the n_workers==1
# fallback). Module-level so the forkserver workers can resolve them.
_RF = None
_BACKGROUND = None


def select_positive_class_shap(values, log_loss: bool = False) -> np.ndarray:
    """
    Normalize SHAP output across shap versions into a single positive-class ndarray.

    TreeExplainer returns results in several shapes depending on shap version & task:
    - classic binary classification: list ``[neg, pos]`` of ``(n, F)`` arrays
    - newer shap, last axis is class: ``(n, F, 2)`` for values, ``(n, F, F, 2)`` for interactions
    - log-loss (single scalar output): ``(n, F)`` or ``(n, F, F)``

    Picks the positive-class slice in all cases. For log-loss there is only one output, so it is
    returned as-is (modulo a preserved class axis). Kept here (not in train.py) so both the pipeline
    and the TF-free workers share one definition.
    """
    if log_loss:
        if isinstance(values, list):
            return np.asarray(values[0])
        values = np.asarray(values)
        # newer shap preserves the class axis even for model_output="log_loss":
        # (n, F, 2) -> (n, F); (n, F, F, 2) -> (n, F, F)
        if values.ndim >= 3 and values.shape[-1] == 2:
            return values[..., 1]
        return values

    if isinstance(values, list):
        return np.asarray(values[1])

    values = np.asarray(values)
    # (n, F, 2) -> (n, F); (n, F, F, 2) -> (n, F, F)
    if values.ndim >= 3 and values.shape[-1] == 2:
        return values[..., 1]
    return values


def _pin_worker_threads() -> None:
    # Must run before numpy/shap import their native thread pools. shap's TreeSHAP is single-threaded
    # anyway, so this only tames incidental numpy/BLAS threads — but with all cores as workers that
    # still matters. setdefault so an explicitly-set env is respected.
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")


def _worker_init(rf_path: str, background) -> None:
    """Pool initializer: pin threads, then load the RF once per worker (rebuild-per-worker source)."""
    _pin_worker_threads()
    import joblib  # noqa: PLC0415  (deferred so _pin_worker_threads() sets env before numpy/BLAS init)

    global _RF, _BACKGROUND
    _RF = joblib.load(rf_path)
    _BACKGROUND = background


def _worker(task):
    """Compute one SHAP pass on one chunk of samples using the module-global RF. Returns the
    positive-class array for the chunk; ProcessPoolExecutor.map preserves input order."""
    kind, x_chunk, y_chunk = task
    import shap  # noqa: PLC0415  (deferred so the spawn worker pins threads before shap/BLAS init)

    # Suppress shap's tqdm (it writes to stderr, which the pipeline logger forwards to Slack).
    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        if kind == _LOGLOSS_KIND:
            explainer = shap.TreeExplainer(_RF, data=_BACKGROUND, model_output="log_loss")
            raw = explainer.shap_values(x_chunk, y=y_chunk)
        else:
            explainer = shap.TreeExplainer(_RF)
            raw = (
                explainer.shap_interaction_values(x_chunk)
                if kind == "interaction"
                else explainer.shap_values(x_chunk)
            )
    return select_positive_class_shap(raw, log_loss=(kind == _LOGLOSS_KIND))


def parallel_shap(
    rf_path: str,
    x: np.ndarray,
    kind: str,
    n_workers: int,
    *,
    background: np.ndarray | None = None,
    y: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute the positive-class SHAP values for ``x`` under one pass ``kind`` in
    ``{"summary", "interaction", "logloss"}``, chunked across up to ``n_workers`` worker processes.

    ``rf_path`` is the persisted sklearn RF joblib (each worker loads it). ``background`` and ``y`` are
    required only for the ``"logloss"`` (interventional) pass. TreeSHAP is per-sample deterministic, so
    the concatenated result is bitwise-identical to the single-process computation.
    """
    if kind != _LOGLOSS_KIND and kind not in _RAW_KINDS:
        raise ValueError(f"unknown SHAP kind {kind!r}")

    n = len(x)
    if n == 0:
        raise ValueError("cannot explain zero samples")
    n_workers = max(1, min(n_workers, n))

    # n_workers == 1: run in-process (no pool) — used on single-core hosts or tiny inputs.
    if n_workers == 1:
        _worker_init(rf_path, background)
        return _worker((kind, x, y))

    chunks = np.array_split(np.arange(n), n_workers)
    tasks = [(kind, x[c], (None if y is None else y[c])) for c in chunks if len(c)]
    # forkserver with an EMPTY preload: the fork server is spawned clean and does NOT re-import the
    # parent's __main__ (aetherscan.main -> TF -> the whole training stack), so workers stay light.
    ctx = mp.get_context("forkserver")
    ctx.set_forkserver_preload([])
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(rf_path, background),
    ) as pool:
        # ProcessPoolExecutor.map yields results in input (chunk) order — no re-sort needed.
        results = list(pool.map(_worker, tasks))
    return np.concatenate(results, axis=0)
