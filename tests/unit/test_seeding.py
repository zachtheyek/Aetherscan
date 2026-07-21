"""Unit tests for aetherscan.seeding: root-seed stream derivation (issue #49)."""

from __future__ import annotations

import numpy as np

from aetherscan.seeding import (
    STREAM_DATA_GEN,
    STREAM_DATASET,
    STREAM_PLOT,
    STREAM_VIZ,
    derive_rng,
)


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
        ids = [STREAM_DATA_GEN, STREAM_DATASET, STREAM_VIZ, STREAM_PLOT]
        assert len(set(ids)) == len(ids)
        streams = [tuple(_draws(derive_rng(42, stream_id, 1))) for stream_id in ids]
        assert len(set(streams)) == len(streams)

    def test_none_root_seed_falls_back_to_entropy(self):
        # None = the historical OS-entropy behavior: still a Generator, but two derivations
        # of the same key are (overwhelmingly likely) different streams
        rng = derive_rng(None, STREAM_DATA_GEN, 1)
        assert isinstance(rng, np.random.Generator)
        assert _draws(rng) != _draws(derive_rng(None, STREAM_DATA_GEN, 1))
