"""Unit tests for main.py glue that isn't reachable through the higher-level commands —
the terminal training-status / exit-code contract (_report_final_training_status) and
non-retryable streaming-inference failures."""

from __future__ import annotations

import types

import pytest

from aetherscan import main
from aetherscan.main import NonRetryableInferenceError, _run_streaming_csv_inference
from aetherscan.preprocessing import DataPreprocessor
from aetherscan.run_state import STAGE_RF_PLOTS, STAGE_VAE_PLOTS, TrainingRunState


def _pipeline_with(stages_failed):
    state = TrainingRunState(tag="t", run_start_time=1.0, stages_failed=list(stages_failed))
    return types.SimpleNamespace(run_state=state)


class TestReportFinalTrainingStatus:
    def test_success_when_no_failed_stages(self):
        # No permanently-failed stage -> returns normally, no SystemExit.
        main._report_final_training_status(_pipeline_with([]))

    @pytest.mark.parametrize(
        "failed", [[STAGE_VAE_PLOTS], [STAGE_RF_PLOTS], [STAGE_VAE_PLOTS, STAGE_RF_PLOTS]]
    )
    def test_exits_nonzero_on_failed_plot_stage(self, failed):
        with pytest.raises(SystemExit) as exc:
            main._report_final_training_status(_pipeline_with(failed))
        assert exc.value.code == 1

    def test_exits_nonzero_when_pipeline_is_none(self):
        # Degenerate no-pipeline path must never report a false success.
        with pytest.raises(SystemExit) as exc:
            main._report_final_training_status(None)
        assert exc.value.code == 1


@pytest.fixture
def initialized_runtime():
    """DataPreprocessor needs live db + manager singletons; conftest tears them down."""
    from aetherscan.db import init_db  # noqa: PLC0415
    from aetherscan.manager import init_manager  # noqa: PLC0415

    init_manager()
    init_db()


class TestStreamingInferenceNonRetryable:
    def test_empty_catalog_raises_non_retryable(self, initialized_runtime):
        # No inference_files configured -> plan_cadences yields no units. This is a
        # permanent (config) failure: the retry loop in inference_command re-raises
        # NonRetryableInferenceError immediately instead of burning retry attempts.
        # The raise happens before any model loading, so no strategy is needed.
        preprocessor = DataPreprocessor()
        with pytest.raises(NonRetryableInferenceError, match="No cadence work units"):
            _run_streaming_csv_inference(preprocessor, strategy=None)

    def test_non_retryable_error_is_an_exception_subclass(self):
        # Sanity: it must be catchable as a plain Exception (cleanup paths) while being
        # distinguishable from transient failures by the retry loop.
        assert issubclass(NonRetryableInferenceError, RuntimeError)
