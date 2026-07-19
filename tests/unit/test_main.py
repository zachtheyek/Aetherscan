"""Unit tests for main.py glue that isn't reachable through the higher-level commands —
currently the terminal training-status / exit-code contract (_report_final_training_status)."""

from __future__ import annotations

import types

import pytest

from aetherscan import main
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
