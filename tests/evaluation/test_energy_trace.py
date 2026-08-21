from __future__ import annotations

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
