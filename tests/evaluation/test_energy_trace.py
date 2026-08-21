from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _config():
    from lights_audio_engine.config import AudioEngineConfig

    return AudioEngineConfig(
        expected_sample_rate_hz=1_000,
        sensitivity=0.5,
        min_bpm=60.0,
        max_bpm=200.0,
        analysis_window_seconds=0.02,
        energy_history_seconds=1.0,
    )


def test_trace_records_production_window_values_before_history_update() -> None:
    """Catches a trace that samples the baseline after adding the current RMS window."""

    from lights_audio_engine.evaluation.energy_trace import trace_energy_frames
    from lights_audio_engine.models import AudioFrame

    samples = np.concatenate(
        (
            np.full(20, 0.2, dtype=np.float64),
            np.full(20, 0.1, dtype=np.float64),
            np.full(20, 0.5, dtype=np.float64),
        )
    )
    report = trace_energy_frames(
        (AudioFrame(samples, sample_rate_hz=1_000, start_time_seconds=0.0),),
        config=_config(),
    )

    assert [record.window_index for record in report.windows] == [0, 1, 2]
    assert [record.start_time_seconds for record in report.windows] == pytest.approx(
        (0.0, 0.02, 0.04)
    )
    assert [record.end_time_seconds for record in report.windows] == pytest.approx(
        (0.02, 0.04, 0.06)
    )
    assert [record.rms for record in report.windows] == pytest.approx((0.2, 0.1, 0.5))
    assert [record.baseline for record in report.windows] == pytest.approx((0.0, 0.2, 0.15))
    assert [record.threshold for record in report.windows] == pytest.approx((0.175, 0.45, 0.3375))
    assert [record.is_active for record in report.windows] == [True, False, True]


def test_trace_replays_a_checksum_validated_artifact_in_detector_windows(tmp_path: Path) -> None:
    """Catches a trace that bypasses the authoritative artifact reader or replay source."""

    from lights_audio_engine.evaluation.artifact import write_artifact
    from lights_audio_engine.evaluation.energy_trace import trace_energy_artifact

    path = tmp_path / "capture.npy"
    samples = np.concatenate(
        (np.full(20, 0.2, dtype=np.float64), np.full(20, 0.1, dtype=np.float64))
    )
    write_artifact(
        path,
        samples,
        label="synthetic-capture",
        sample_rate_hz=1_000,
        frame_lengths=(7, 13, 20),
    )

    report = trace_energy_artifact(path, config=_config(), block_size_frames=7)

    assert report.artifact_label == "synthetic-capture"
    assert report.sample_rate_hz == 1_000
    assert report.segment_index == 0
    assert len(report.windows) == 2
    assert report.windows[1].baseline == pytest.approx(0.2)
    assert report.windows[1].threshold == pytest.approx(0.45)
    assert report.windows[1].is_active is False


def test_trace_report_writes_an_explicit_json_window_record(tmp_path: Path) -> None:
    """Catches a report writer that omits an observed window value from its JSON output."""

    from lights_audio_engine.evaluation.energy_trace import (
        EnergyTraceReport,
        EnergyWindowTrace,
        write_energy_trace_report,
    )

    destination = tmp_path / "trace.json"
    write_energy_trace_report(
        destination,
        EnergyTraceReport(
            artifact_label="synthetic",
            sample_rate_hz=1_000,
            segment_index=0,
            windows=(EnergyWindowTrace(0, 0.0, 0.02, 0.2, 0.0, 0.175, True),),
        ),
    )

    assert destination.read_text(encoding="utf-8") == (
        "{\n"
        '  "artifact_label": "synthetic",\n'
        '  "kind": "production_energy_window_trace",\n'
        '  "sample_rate_hz": 1000,\n'
        '  "schema_version": 1,\n'
        '  "segment_index": 0,\n'
        '  "windows": [\n'
        "    {\n"
        '      "baseline": 0.0,\n'
        '      "end_time_seconds": 0.02,\n'
        '      "is_active": true,\n'
        '      "rms": 0.2,\n'
        '      "start_time_seconds": 0.0,\n'
        '      "threshold": 0.175,\n'
        '      "window_index": 0\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )


def test_candidate_control_preserves_production_trace_decisions_and_window_values() -> None:
    """Catches a comparison harness that changes the production control semantics."""

    from lights_audio_engine.evaluation.energy_trace import (
        evaluate_energy_trace,
        trace_energy_frames,
    )
    from lights_audio_engine.models import AudioFrame

    samples = np.concatenate(
        (
            np.full(20, 0.2, dtype=np.float64),
            np.full(20, 0.1, dtype=np.float64),
            np.full(20, 0.5, dtype=np.float64),
        )
    )
    trace = trace_energy_frames(
        (AudioFrame(samples, sample_rate_hz=1_000, start_time_seconds=0.0),),
        config=_config(),
    )

    result = evaluate_energy_trace(trace, config=_config(), formulation="production-control")

    assert [window.rms for window in result.windows] == pytest.approx(
        [window.rms for window in trace.windows]
    )
    assert [window.baseline for window in result.windows] == pytest.approx(
        [window.baseline for window in trace.windows]
    )
    assert [window.decision_boundary for window in result.windows] == pytest.approx(
        [window.threshold for window in trace.windows]
    )
    assert [window.is_active for window in result.windows] == [
        window.is_active for window in trace.windows
    ]


def test_candidate_formulations_use_hand_checked_decision_statistics() -> None:
    """Catches a candidate formula that changes its stated evaluation hypothesis."""

    from lights_audio_engine.evaluation.energy_trace import (
        EnergyTraceReport,
        EnergyWindowTrace,
        evaluate_energy_trace,
    )

    trace = EnergyTraceReport(
        artifact_label=None,
        sample_rate_hz=1_000,
        segment_index=None,
        windows=(
            EnergyWindowTrace(0, 0.0, 0.02, 0.2, 0.0, 0.175, True),
            EnergyWindowTrace(1, 0.02, 0.04, 0.1, 0.2, 0.45, False),
            EnergyWindowTrace(2, 0.04, 0.06, 0.5, 0.15, 0.3375, True),
        ),
    )

    c1 = evaluate_energy_trace(trace, config=_config(), formulation="c1-additive-margin")
    c3 = evaluate_energy_trace(trace, config=_config(), formulation="c3-bounded-multiplicative")
    c2 = evaluate_energy_trace(trace, config=_config(), formulation="c2-normalized-excess")

    assert [window.decision_boundary for window in c1.windows] == pytest.approx(
        (0.175, 0.375, 0.325)
    )
    assert [window.decision_boundary for window in c3.windows] == pytest.approx(
        (0.175, 0.255, 0.19125)
    )
    assert [window.decision_statistic for window in c2.windows] == pytest.approx(
        (20.0, -0.5, 7.0 / 3.0)
    )
    assert [window.decision_boundary for window in c2.windows] == pytest.approx((0.35, 0.35, 0.35))


def test_candidate_metrics_separate_rising_edges_refractory_and_intervals() -> None:
    """Catches metrics that hide dense candidates behind refractory filtering."""

    from lights_audio_engine.evaluation.energy_trace import (
        EnergyTraceReport,
        EnergyWindowTrace,
        evaluate_energy_trace,
    )

    trace = EnergyTraceReport(
        artifact_label=None,
        sample_rate_hz=1_000,
        segment_index=None,
        windows=(
            EnergyWindowTrace(0, 0.0, 0.1, 0.2, 0.0, 0.0, False),
            EnergyWindowTrace(1, 0.1, 0.2, 0.1, 0.2, 0.0, False),
            EnergyWindowTrace(2, 0.2, 0.3, 0.2, 0.15, 0.0, False),
            EnergyWindowTrace(3, 0.3, 0.4, 0.1, 0.2, 0.0, False),
            EnergyWindowTrace(4, 0.4, 0.5, 0.2, 0.15, 0.0, False),
            EnergyWindowTrace(5, 0.5, 0.6, 0.1, 0.2, 0.0, False),
            EnergyWindowTrace(6, 0.6, 0.7, 0.2, 0.15, 0.0, False),
        ),
    )
    config = _config()
    config = config.__class__(
        expected_sample_rate_hz=1_000,
        sensitivity=0.5,
        min_bpm=60.0,
        max_bpm=240.0,
        analysis_window_seconds=0.1,
        energy_history_seconds=1.0,
    )

    result = evaluate_energy_trace(trace, config=config, formulation="c3-bounded-multiplicative")

    assert result.metrics.rising_edge_count == 4
    assert result.metrics.emitted_event_count == 2
    assert result.metrics.refractory_suppressed_candidate_count == 2
    assert result.metrics.emitted_interval_minimum_seconds == pytest.approx(0.4)
    assert result.metrics.emitted_interval_median_seconds == pytest.approx(0.4)
    assert result.metrics.emitted_interval_count_below_half_second == 1
    assert result.metrics.active_window_count == 4
    assert result.metrics.active_window_density == pytest.approx(4 / 7)


def test_candidate_report_serializes_and_replays_an_artifact(tmp_path: Path) -> None:
    """Catches a report that cannot be deterministically persisted after artifact replay."""

    from lights_audio_engine.evaluation.artifact import write_artifact
    from lights_audio_engine.evaluation.energy_trace import (
        evaluate_energy_artifact,
        write_energy_candidate_report,
    )

    path = tmp_path / "capture.npy"
    write_artifact(
        path,
        np.concatenate((np.full(20, 0.2), np.full(20, 0.1))).astype(np.float64),
        label="synthetic-capture",
        sample_rate_hz=1_000,
        frame_lengths=(20, 20),
    )

    report = evaluate_energy_artifact(path, config=_config(), block_size_frames=7)
    destination = tmp_path / "candidate-report.json"
    write_energy_candidate_report(destination, report)
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert report.artifact_label == "synthetic-capture"
    assert len(report.results) == 12
    assert payload["schema_version"] == 1
    assert payload["kind"] == "energy_threshold_candidate_evaluation"
    assert payload["hypothesis_notice"].startswith("Evaluation hypotheses only")


@pytest.mark.parametrize(
    "formulation",
    (
        "production-control",
        "c1-additive-margin",
        "c3-bounded-multiplicative",
        "c2-normalized-excess",
    ),
)
def test_candidate_formulations_preserve_the_deterministic_pulse_fixture(formulation: str) -> None:
    """Catches a candidate implementation that loses the established isolated pulse response."""

    from lights_audio_engine.evaluation.energy_trace import (
        evaluate_energy_trace,
        trace_energy_frames,
    )
    from lights_audio_engine.models import AudioFrame

    samples = np.zeros(4_800, dtype=np.float64)
    samples[960:1920] = 0.8
    config = _config()
    trace = trace_energy_frames(
        (AudioFrame(samples, sample_rate_hz=1_000, start_time_seconds=0.0),), config=config
    )

    result = evaluate_energy_trace(trace, config=config, formulation=formulation)  # type: ignore[arg-type]

    assert result.metrics.rising_edge_count == 1
    assert result.metrics.emitted_event_count == 1
