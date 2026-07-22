# NOTE: come back to this later

"""Unit tests for aetherscan.config: singleton semantics, to_dict() field coverage, and the
coupled-defaults guard (inference.stamp_width == data.width_bin)."""

from __future__ import annotations

import dataclasses
import json
import os

import pytest

import aetherscan.config as config_module
from aetherscan.config import Config, InferenceConfig, get_config, init_config

# Keys to_dict() emits that are not sub-dataclass sections (derived/grouped scalars).
_NON_SECTION_KEYS = frozenset({"paths"})


def _config_sections(config: Config) -> set[str]:
    """Attribute names on a Config whose values are dataclass instances (its serialized sections).

    Derived at runtime so a newly added section is auto-covered by the coverage test below.
    """
    return {name for name, value in vars(config).items() if dataclasses.is_dataclass(value)}


class TestSingletonSemantics:
    def test_constructor_returns_same_instance(self):
        assert Config() is Config()

    def test_get_config_returns_singleton(self):
        assert get_config() is Config()

    def test_init_config_returns_singleton(self):
        assert init_config() is get_config()

    def test_mutation_visible_across_accessors(self):
        Config().training.num_training_rounds = 3
        assert get_config().training.num_training_rounds == 3

    def test_reset_produces_fresh_instance(self):
        first = Config()
        first.training.num_training_rounds = 3
        Config._reset()
        second = init_config()
        assert second is not first
        assert second.training.num_training_rounds != 3

    def test_env_var_paths_respected(self, tmp_path):
        # The autouse fixture points AETHERSCAN_* at tmp_path before init_config().
        config = get_config()
        assert config.data_path == os.environ["AETHERSCAN_DATA_PATH"]
        assert config.model_path == os.environ["AETHERSCAN_MODEL_PATH"]
        assert config.output_path == os.environ["AETHERSCAN_OUTPUT_PATH"]
        assert config.data_path.startswith(str(tmp_path))

    def test_file_path_helpers(self):
        config = get_config()
        assert config.get_training_file_path("a.npy") == os.path.join(
            config.data_path, "training", "a.npy"
        )
        assert config.get_test_file_path("b.npy") == os.path.join(
            config.data_path, "testing", "b.npy"
        )
        assert config.get_inference_file_path("c.csv") == os.path.join(
            config.data_path, "inference", "c.csv"
        )
        # base_path override (used by validate_args before --data-path is applied)
        assert config.get_training_file_path("a.npy", base_path="/elsewhere") == os.path.join(
            "/elsewhere", "training", "a.npy"
        )


class TestToDict:
    def test_sections_cover_every_dataclass_field(self):
        """to_dict() must be updated by hand for every new config section/field — catch drift."""
        config = get_config()
        serialized = config.to_dict()
        sections = _config_sections(config)

        # A forgotten (or spurious extra) whole section: the section keys to_dict() emits, minus
        # the known non-section keys, must equal the dynamically-derived set of Config sections.
        serialized_sections = set(serialized) - _NON_SECTION_KEYS
        assert serialized_sections == sections, (
            "Config.to_dict() sections are out of sync with Config: "
            f"missing={sections - serialized_sections}, extra={serialized_sections - sections}"
        )

        # A forgotten field within a section: each section's keys must equal its dataclass fields.
        for section in sections:
            expected = {f.name for f in dataclasses.fields(type(getattr(config, section)))}
            actual = set(serialized[section].keys())
            assert actual == expected, (
                f"Config.to_dict()['{section}'] is out of sync with "
                f"{type(getattr(config, section)).__name__}: "
                f"missing={expected - actual}, extra={actual - expected}"
            )

    def test_values_round_trip(self):
        config = get_config()
        config.training.num_training_rounds = 5
        config.gpu.num_replicas = 4
        serialized = config.to_dict()
        assert serialized["training"]["num_training_rounds"] == 5
        assert serialized["gpu"]["num_replicas"] == 4
        assert serialized["paths"]["data_path"] == config.data_path
        for section in _config_sections(config):
            for field_name, value in serialized[section].items():
                stored = getattr(getattr(config, section), field_name)
                if isinstance(stored, tuple):
                    # json-level equality: tuples serialize as lists downstream
                    assert list(stored) == list(value)
                else:
                    assert stored == value

    def test_json_serializable(self):
        # Saved run configs are json.dump'ed — the dict must be JSON-native throughout.
        json.dumps(get_config().to_dict())


class TestCoupledDefaultsGuard:
    """Config init must fail fast when the coupled defaults inference.stamp_width and
    data.width_bin diverge (a source-level edit to one but not the other), instead of
    surfacing only at load time deep inside load_inference_data."""

    def test_equal_defaults_initialize_cleanly(self):
        # The autouse fixture already ran init_config() without raising; pin the invariant.
        config = get_config()
        assert config.inference.stamp_width == config.data.width_bin

    def test_diverged_defaults_raise_at_init(self, monkeypatch):
        # Simulate a source-level default edit: Config.__init__ resolves InferenceConfig
        # through the module global, so patch it to construct a diverged instance.
        monkeypatch.setattr(
            config_module, "InferenceConfig", lambda: InferenceConfig(stamp_width=2048)
        )
        Config._reset()
        with pytest.raises(ValueError, match=r"stamp_width \(2048\) must equal data.width_bin"):
            Config()
        # Don't leave the half-built diverged singleton behind for the fixture teardown.
        Config._reset()
