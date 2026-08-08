"""Unit tests for aetherscan.seeding: root-seed stream derivation (issues #49, #279)."""

from __future__ import annotations

import logging

import numpy as np

from aetherscan import seeding
from aetherscan.seeding import (
    STREAM_DATA_GEN,
    STREAM_DATASET,
    STREAM_INFERENCE_MC,
    STREAM_INFERENCE_VIZ,
    STREAM_KMEANS,
    STREAM_PLOT,
    STREAM_REFERENCE_CLOUD,
    STREAM_RF,
    STREAM_RF_PLOTS,
    STREAM_SHAP_SAMPLES,
    STREAM_TF,
    STREAM_UMAP,
    STREAM_VIZ,
    derive_rng,
    derive_seed,
)

_ALL_STREAM_IDS = [
    STREAM_DATA_GEN,
    STREAM_DATASET,
    STREAM_VIZ,
    STREAM_PLOT,
    STREAM_RF,
    STREAM_UMAP,
    STREAM_KMEANS,
    STREAM_TF,
    STREAM_INFERENCE_VIZ,
    STREAM_SHAP_SAMPLES,
    STREAM_RF_PLOTS,
    STREAM_INFERENCE_MC,
    STREAM_REFERENCE_CLOUD,
]


def _draws(rng: np.random.Generator, n: int = 16) -> list[int]:
    return rng.integers(0, 2**63, size=n).tolist()


class TestDeriveRng:
    def test_same_key_reproduces_stream(self):
        assert _draws(derive_rng(42, STREAM_DATA_GEN, 1)) == _draws(
            derive_rng(42, STREAM_DATA_GEN, 1)
        )

    def test_round_index_changes_stream(self):
        assert _draws(derive_rng(42, STREAM_DATA_GEN, 1)) != _draws(
            derive_rng(42, STREAM_DATA_GEN, 2)
        )

    def test_root_seed_changes_stream(self):
        assert _draws(derive_rng(42, STREAM_DATA_GEN, 1)) != _draws(
            derive_rng(43, STREAM_DATA_GEN, 1)
        )

    def test_stream_ids_are_distinct_and_independent(self):
        assert len(set(_ALL_STREAM_IDS)) == len(_ALL_STREAM_IDS)
        streams = [tuple(_draws(derive_rng(42, stream_id, 1))) for stream_id in _ALL_STREAM_IDS]
        assert len(set(streams)) == len(streams)

    def test_none_root_seed_falls_back_to_entropy(self):
        # None = the historical OS-entropy behavior: still a Generator, but two derivations
        # of the same key are (overwhelmingly likely) different streams
        rng = derive_rng(None, STREAM_DATA_GEN, 1)
        assert isinstance(rng, np.random.Generator)
        assert _draws(rng) != _draws(derive_rng(None, STREAM_DATA_GEN, 1))


class TestDeriveSeed:
    """#279: the int-valued sibling for APIs that take an integer random_state."""

    def test_deterministic_and_in_range(self):
        first = derive_seed(42, STREAM_RF)
        assert first == derive_seed(42, STREAM_RF)
        assert 0 <= first < 2**32

    def test_key_and_root_change_value(self):
        base = derive_seed(42, STREAM_UMAP, 0, 15, 100)
        assert base != derive_seed(42, STREAM_UMAP, 1, 15, 100)
        assert base != derive_seed(42, STREAM_UMAP, 0, 30, 100)
        assert base != derive_seed(43, STREAM_UMAP, 0, 15, 100)

    def test_none_root_still_returns_concrete_int(self):
        # The #279 constraint: sites that set a random_state today keep setting one — an
        # unseeded root yields entropy, never None
        first = derive_seed(None, STREAM_RF)
        assert isinstance(first, int) and 0 <= first < 2**32
        assert first != derive_seed(None, STREAM_RF)  # overwhelmingly likely


class TestUnseededWarning:
    """#279: an unseeded run warns exactly once instead of failing silently."""

    def _reset_flag(self):
        seeding._UNSEEDED_WARNED = False

    def test_warns_once_for_none_root(self, caplog):
        self._reset_flag()
        with caplog.at_level(logging.WARNING, logger="aetherscan.seeding"):
            derive_rng(None, STREAM_DATA_GEN, 1)
            derive_seed(None, STREAM_RF)
        warnings = [r for r in caplog.records if "NOT reproducible" in r.message]
        assert len(warnings) == 1
        self._reset_flag()

    def test_no_warning_with_concrete_root(self, caplog):
        self._reset_flag()
        with caplog.at_level(logging.WARNING, logger="aetherscan.seeding"):
            derive_rng(11, STREAM_DATA_GEN, 1)
            derive_seed(11, STREAM_RF)
        assert not [r for r in caplog.records if "NOT reproducible" in r.message]


class TestReferenceCloudKeyStreamDisjointness:
    """#401 review note: SeedSequence treats a trailing 0 entropy word as an identity, so
    (root, S, 2) IS catalog cadence 0's (root, S, 2, 0) — which is exactly why the keyless
    reservoir path derives from (root, S, 3). Pin both facts so a refactor can't quietly
    reintroduce the collision."""

    def test_keyless_stream_disjoint_from_cadence_zero(self):
        keyless = derive_rng(11, STREAM_REFERENCE_CLOUD, 3).random(8)
        cadence_zero = derive_rng(11, STREAM_REFERENCE_CLOUD, 2, 0).random(8)
        assert not (keyless == cadence_zero).all()

    def test_trailing_zero_identity_is_real(self):
        # The trap the (S, 3) choice avoids: a trailing 0 entropy word is a SeedSequence
        # identity, so these two "different" derivations are one stream
        bare = derive_rng(11, STREAM_REFERENCE_CLOUD, 2).random(8)
        trailing_zero = derive_rng(11, STREAM_REFERENCE_CLOUD, 2, 0).random(8)
        assert (bare == trailing_zero).all()
