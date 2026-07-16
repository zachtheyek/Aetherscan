# NOTE: come back to this later

"""Unit tests for aetherscan.config: singleton semantics and to_dict() field coverage."""

from __future__ import annotations

import dataclasses
import json
import os

from aetherscan.config import Config, get_config, init_config

# Sub-dataclass sections serialized by Config.to_dict(), keyed by attribute name.
_CONFIG_SECTIONS = (
    "db",
    "manager",
    "monitor",
    "logger",
    "beta_vae",
    "rf",
    "gpu",
    "data",
    "training",
    "inference",
    "checkpoint",
)


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
        """to_dict() must be updated by hand for every new config field — catch drift."""
        config = get_config()
        serialized = config.to_dict()
        for section in _CONFIG_SECTIONS:
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
        for section in _CONFIG_SECTIONS:
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
