"""Diagnostic-only JSONL characterization logging for live engine sessions."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

from lights_audio_engine.capture import AudioSource, CaptureItem, Discontinuity
from lights_audio_engine.config import AudioEngineConfig
from lights_audio_engine.engine import AudioEngine
from lights_audio_engine.models import AudioAnalysisResult, AudioFrame


@dataclass(frozen=True, slots=True)
class SessionInfo:
    """Reliable bookkeeping and configuration metadata for one live run."""

    label: str
    device_index: int
    device_name: str
    sample_rate_hz: int
    channels: int
    block_size_frames: int
    sensitivity: float
    min_bpm: float
    max_bpm: float
    analysis_window_seconds: float
    energy_history_seconds: float
    bpm_history_size: int


class JsonlSessionLogger:
    """Write a flushed, versioned record stream without retaining audio samples."""

    def __init__(
        self,
        path: Path,
        info: SessionInfo,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._output: TextIO = path.open("w", encoding="utf-8", newline="\n")
        self._monotonic = monotonic
        self._started_at = monotonic()
        self._stream_ordinal = 0
        self._previous_beat_timestamp: float | None = None
        self._summary_started_at = 0.0
        self._frame_count = 0
        self._stream_frame_count = 0
        self._level_sum = 0.0
        self._level_max = 0.0
        self._process_sum = 0.0
        self._process_max = 0.0
        self._stream_time_seconds: float | None = None
        self._closed = False
        session = asdict(info)
        session.update(
            {
                "type": "session",
                "schema": 1,
                "wall_clock_start": wall_clock().astimezone(UTC).isoformat(),
            }
        )
        self._write(session)

    def record_result(
        self,
        frame: AudioFrame,
        result: AudioAnalysisResult,
        process_duration_seconds: float,
    ) -> None:
        """Record one engine result and accumulate its periodic summary context."""

        host_time = self._elapsed()
        self._frame_count += 1
        self._stream_frame_count += 1
        self._level_sum += result.current_level
        self._level_max = max(self._level_max, result.current_level)
        self._process_sum += process_duration_seconds
        self._process_max = max(self._process_max, process_duration_seconds)
        self._stream_time_seconds = (
            frame.start_time_seconds + frame.samples.size / frame.sample_rate_hz
        )
        for event in result.beat_events:
            interval = (
                None
                if self._previous_beat_timestamp is None
                else event.timestamp_seconds - self._previous_beat_timestamp
            )
            self._write(
                {
                    "type": "beat",
                    "schema": 1,
                    "host_time_seconds": host_time,
                    "beat_timestamp_seconds": event.timestamp_seconds,
                    "stream_ordinal": self._stream_ordinal,
                    "beat_index": event.beat_index,
                    "interval_seconds": interval,
                    "strength": event.strength,
                    "bpm": result.bpm,
                }
            )
            self._previous_beat_timestamp = event.timestamp_seconds
        if host_time - self._summary_started_at >= 1.0:
            self._flush_summary(host_time)

    def record_discontinuity(self, boundary: Discontinuity) -> None:
        """Record a stream boundary, then reset stream-local interval state."""

        host_time = self._elapsed()
        self._flush_summary(host_time)
        self._write(
            {
                "type": "discontinuity",
                "schema": 1,
                "host_time_seconds": host_time,
                "reason": boundary.reason.value,
                "detail": boundary.detail,
                "stream_ordinal": self._stream_ordinal,
                "frames_emitted_before": boundary.frames_emitted_before,
            }
        )
        self._stream_ordinal += 1
        self._previous_beat_timestamp = None
        self._stream_frame_count = 0
        self._stream_time_seconds = None

    def close(self) -> None:
        """Flush any partial summary and close the JSONL file once."""

        if self._closed:
            return
        if self._frame_count:
            self._flush_summary(self._elapsed())
        self._output.close()
        self._closed = True

    def _elapsed(self) -> float:
        return self._monotonic() - self._started_at

    def _flush_summary(self, host_time: float) -> None:
        if not self._frame_count:
            self._summary_started_at = host_time
            return
        self._write(
            {
                "type": "summary",
                "schema": 1,
                "host_time_seconds": host_time,
                "stream_ordinal": self._stream_ordinal,
                "stream_time_seconds": self._stream_time_seconds,
                "frame_count": self._frame_count,
                "stream_frame_count": self._stream_frame_count,
                "mean_input_level": self._level_sum / self._frame_count,
                "max_input_level": self._level_max,
                "mean_process_seconds": self._process_sum / self._frame_count,
                "max_process_seconds": self._process_max,
            }
        )
        self._summary_started_at = host_time
        self._frame_count = 0
        self._level_sum = 0.0
        self._level_max = 0.0
        self._process_sum = 0.0
        self._process_max = 0.0

    def _write(self, record: dict[str, object]) -> None:
        self._output.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._output.flush()


class LoggedAudioSource:
    """Observe yielded discontinuities while preserving the source contract."""

    def __init__(self, source: AudioSource, logger: JsonlSessionLogger) -> None:
        self._source = source
        self._logger = logger

    @property
    def sample_rate_hz(self) -> int:
        return self._source.sample_rate_hz

    def stream(self) -> Iterator[CaptureItem]:
        for item in self._source.stream():
            if isinstance(item, Discontinuity):
                self._logger.record_discontinuity(item)
            yield item

    def close(self) -> None:
        self._source.close()


class LoggedAudioEngine(AudioEngine):
    """Time existing synchronous processing and forward results to the logger."""

    def __init__(
        self,
        config: AudioEngineConfig,
        logger: JsonlSessionLogger,
        *,
        performance_clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        super().__init__(config)
        self._logger = logger
        self._performance_clock = performance_clock

    def process(self, frame: AudioFrame) -> AudioAnalysisResult:
        started_at = self._performance_clock()
        result = super().process(frame)
        duration = self._performance_clock() - started_at
        self._logger.record_result(frame, result, duration)
        return result
