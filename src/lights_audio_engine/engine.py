"""Stateful orchestration for deterministic streaming audio analysis."""

from __future__ import annotations

from collections import deque
from statistics import median

from lights_audio_engine.config import AudioEngineConfig
from lights_audio_engine.detectors.drop import NoOpDropDetector
from lights_audio_engine.detectors.energy import EnergyBeatDetector
from lights_audio_engine.models import AudioAnalysisResult, AudioFrame, BeatEvent


def _require_config(config: object) -> AudioEngineConfig:
    if not isinstance(config, AudioEngineConfig):
        raise TypeError("config must be an AudioEngineConfig")
    return config


def _require_frame(frame: object) -> AudioFrame:
    if not isinstance(frame, AudioFrame):
        raise TypeError("frame must be an AudioFrame")
    return frame


class AudioEngine:
    """Process sequential audio frames into immutable analysis results."""

    def __init__(self, config: AudioEngineConfig | None = None) -> None:
        self._config = _require_config(config) if config is not None else AudioEngineConfig()
        self._energy_detector = EnergyBeatDetector(self._config)
        self._drop_detector = NoOpDropDetector()
        self._beat_timestamps: deque[float] = deque(maxlen=self._config.bpm_history_size)
        self._next_beat_index = 0

    def process(self, frame: AudioFrame) -> AudioAnalysisResult:
        """Analyze the next contiguous frame in the stream."""

        validated_frame = _require_frame(frame)
        energy_detection = self._energy_detector.process(validated_frame)
        beat_events: list[BeatEvent] = []
        for transient in energy_detection.transients:
            event = BeatEvent(
                timestamp_seconds=transient.timestamp_seconds,
                strength=transient.strength,
                beat_index=self._next_beat_index,
            )
            beat_events.append(event)
            self._beat_timestamps.append(event.timestamp_seconds)
            self._next_beat_index += 1

        return AudioAnalysisResult(
            bpm=self._estimate_bpm(),
            beat_events=tuple(beat_events),
            drop_events=self._drop_detector.process(validated_frame),
            current_level=energy_detection.current_level,
        )

    def reset(self) -> None:
        """Clear detector, timing, event-index, and BPM state."""

        self._energy_detector.reset()
        self._drop_detector.reset()
        self._beat_timestamps.clear()
        self._next_beat_index = 0

    def _estimate_bpm(self) -> float | None:
        if len(self._beat_timestamps) < 3:
            return None

        minimum_interval = 60.0 / self._config.max_bpm
        maximum_interval = 60.0 / self._config.min_bpm
        timestamps = tuple(self._beat_timestamps)
        valid_intervals = [
            current - previous
            for previous, current in zip(timestamps, timestamps[1:], strict=False)
            if minimum_interval <= current - previous <= maximum_interval
        ]
        if len(valid_intervals) < 2:
            return None

        estimate = 60.0 / median(valid_intervals)
        if not self._config.min_bpm <= estimate <= self._config.max_bpm:
            return None
        return estimate
