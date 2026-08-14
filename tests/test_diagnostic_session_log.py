from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from lights_audio_engine.capture import Discontinuity, DiscontinuityReason
from lights_audio_engine.config import AudioEngineConfig
from lights_audio_engine.diagnostic.session_log import (
    JsonlSessionLogger,
    LoggedAudioEngine,
    SessionInfo,
)
from lights_audio_engine.models import AudioAnalysisResult, AudioFrame, BeatEvent


class SequenceClock:
    def __init__(self, *values: float) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


def _info() -> SessionInfo:
    return SessionInfo(
        label="steady-house-normal-1",
        device_index=12,
        device_name="Microphone (Realtek Audio)",
        sample_rate_hz=48_000,
        channels=1,
        block_size_frames=960,
        sensitivity=0.5,
        min_bpm=50.0,
        max_bpm=240.0,
        analysis_window_seconds=0.02,
        energy_history_seconds=2.0,
        bpm_history_size=8,
    )


def _result(timestamp: float, beat_index: int, *, level: float, bpm: float | None = None):
    return AudioAnalysisResult(
        bpm=bpm,
        beat_events=(BeatEvent(timestamp, 0.8, beat_index),),
        current_level=level,
    )


def _frame(start: float) -> AudioFrame:
    return AudioFrame(np.zeros(960, dtype=np.float64), 48_000, start)


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_logger_writes_versioned_records_and_resets_interval_across_streams(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.jsonl"
    clock = SequenceClock(10.0, 10.1, 10.6, 10.7, 10.8, 11.9)
    logger = JsonlSessionLogger(
        path,
        _info(),
        monotonic=clock,
        wall_clock=lambda: datetime(2026, 8, 13, 23, 15, tzinfo=UTC),
    )

    logger.record_result(_frame(0.0), _result(0.0, 0, level=0.2), 0.001)
    logger.record_result(_frame(0.5), _result(0.5, 1, level=0.4, bpm=120.0), 0.003)
    logger.record_discontinuity(Discontinuity(DiscontinuityReason.OVERFLOW, 2, "input overflow"))
    logger.record_result(_frame(0.0), _result(0.0, 0, level=0.3), 0.002)
    logger.close()

    records = _records(path)
    assert {record["schema"] for record in records} == {1}
    assert records[0] == {
        "type": "session",
        "schema": 1,
        "label": "steady-house-normal-1",
        "wall_clock_start": "2026-08-13T23:15:00+00:00",
        "device_index": 12,
        "device_name": "Microphone (Realtek Audio)",
        "sample_rate_hz": 48_000,
        "channels": 1,
        "block_size_frames": 960,
        "sensitivity": 0.5,
        "min_bpm": 50.0,
        "max_bpm": 240.0,
        "analysis_window_seconds": 0.02,
        "energy_history_seconds": 2.0,
        "bpm_history_size": 8,
    }
    beats = [record for record in records if record["type"] == "beat"]
    assert [(beat["stream_ordinal"], beat["beat_index"]) for beat in beats] == [
        (0, 0),
        (0, 1),
        (1, 0),
    ]
    assert [beat["interval_seconds"] for beat in beats] == [None, 0.5, None]
    assert beats[1]["bpm"] == 120.0
    boundary = next(record for record in records if record["type"] == "discontinuity")
    assert boundary["reason"] == "overflow"
    assert boundary["detail"] == "input overflow"
    assert boundary["stream_ordinal"] == 0
    assert boundary["frames_emitted_before"] == 2
    summaries = [record for record in records if record["type"] == "summary"]
    assert summaries[0]["frame_count"] == 2
    assert summaries[0]["mean_input_level"] == pytest.approx(0.3)
    assert summaries[0]["max_input_level"] == 0.4
    assert summaries[0]["mean_process_seconds"] == pytest.approx(0.002)
    assert summaries[0]["max_process_seconds"] == 0.003


def test_logged_engine_measures_only_engine_process_time(tmp_path: Path) -> None:
    path = tmp_path / "timed.jsonl"
    logger = JsonlSessionLogger(
        path,
        _info(),
        monotonic=SequenceClock(20.0, 20.1, 20.2),
        wall_clock=lambda: datetime(2026, 8, 13, tzinfo=UTC),
    )
    engine = LoggedAudioEngine(
        AudioEngineConfig(),
        logger,
        performance_clock=SequenceClock(4.0, 4.002),
    )

    engine.process(_frame(0.0))
    logger.close()

    summary = next(record for record in _records(path) if record["type"] == "summary")
    assert summary["mean_process_seconds"] == pytest.approx(0.002)
