"""Deterministic short-term-energy transient detection."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import ceil
from statistics import median

import numpy as np

from lights_audio_engine.config import AudioEngineConfig
from lights_audio_engine.models import AudioFrame, Float64Samples


@dataclass(frozen=True, slots=True)
class EnergyTransient:
    """Internal transient observation before engine-level event indexing."""

    timestamp_seconds: float
    strength: float


@dataclass(frozen=True, slots=True)
class EnergyDetection:
    """Energy observations produced for one input frame."""

    transients: tuple[EnergyTransient, ...]
    current_level: float


class EnergyBeatDetector:
    """Detect onsets from fixed-duration RMS windows.

    This intentionally small M1 detector combines a sensitivity-dependent
    absolute gate with a median-history relative gate. It is an architectural
    proof, not a production beat tracker.
    """

    def __init__(self, config: AudioEngineConfig) -> None:
        self._config = config
        self._window_size = max(
            1,
            round(config.expected_sample_rate_hz * config.analysis_window_seconds),
        )
        history_windows = max(
            1,
            ceil(config.energy_history_seconds / config.analysis_window_seconds),
        )
        self._energy_history: deque[float] = deque(maxlen=history_windows)
        self._pending = np.empty(0, dtype=np.float64)
        self._stream_start_time_seconds: float | None = None
        self._samples_received = 0
        self._last_event_time_seconds: float | None = None
        self._is_active = False

    def process(self, frame: AudioFrame) -> EnergyDetection:
        """Process a sequential frame and return transient observations."""

        sample_rate_hz = self._config.expected_sample_rate_hz
        if frame.sample_rate_hz != sample_rate_hz:
            raise ValueError(
                f"frame sample rate {frame.sample_rate_hz} does not match configured "
                f"sample rate {sample_rate_hz}"
            )

        stream_start_time = self._validate_frame_start(frame)
        combined_start_sample = self._samples_received - self._pending.size
        self._samples_received += frame.samples.size
        if self._pending.size:
            combined = np.empty(self._pending.size + frame.samples.size, dtype=np.float64)
            combined[: self._pending.size] = self._pending
            combined[self._pending.size :] = frame.samples
        else:
            combined = np.array(frame.samples, copy=True)

        transients: list[EnergyTransient] = []
        processed_samples = 0
        while combined.size - processed_samples >= self._window_size:
            window = combined[processed_samples : processed_samples + self._window_size]
            energy = self._rms(window)
            threshold = self._threshold()
            above_threshold = energy >= threshold

            if above_threshold and not self._is_active:
                peak_offset = int(np.abs(window).argmax())
                timestamp = (
                    stream_start_time
                    + (combined_start_sample + processed_samples + peak_offset) / sample_rate_hz
                )
                minimum_interval = 60.0 / self._config.max_bpm
                if (
                    self._last_event_time_seconds is None
                    or timestamp - self._last_event_time_seconds >= minimum_interval
                ):
                    transients.append(
                        EnergyTransient(
                            timestamp_seconds=timestamp,
                            strength=min(1.0, energy),
                        )
                    )
                    self._last_event_time_seconds = timestamp

            self._is_active = above_threshold
            self._energy_history.append(energy)
            processed_samples += self._window_size

        remainder = combined[processed_samples:]
        self._pending = np.array(remainder, dtype=np.float64, copy=True)
        return EnergyDetection(
            transients=tuple(transients),
            current_level=self._rms(frame.samples),
        )

    def reset(self) -> None:
        """Clear all streaming detector state."""

        self._energy_history.clear()
        self._pending = np.empty(0, dtype=np.float64)
        self._stream_start_time_seconds = None
        self._samples_received = 0
        self._last_event_time_seconds = None
        self._is_active = False

    def _validate_frame_start(self, frame: AudioFrame) -> float:
        stream_start_time = self._stream_start_time_seconds
        if stream_start_time is None:
            self._stream_start_time_seconds = frame.start_time_seconds
            return frame.start_time_seconds
        expected = stream_start_time + self._samples_received / self._config.expected_sample_rate_hz
        tolerance = 0.5 / self._config.expected_sample_rate_hz
        if abs(frame.start_time_seconds - expected) > tolerance:
            raise ValueError("audio frames must have contiguous start timestamps")
        return stream_start_time

    def _threshold(self) -> float:
        sensitivity = self._config.sensitivity
        absolute_gate = 0.30 - 0.25 * sensitivity
        baseline = median(self._energy_history) if self._energy_history else 0.0
        relative_multiplier = 3.0 - 1.5 * sensitivity
        return max(absolute_gate, baseline * relative_multiplier)

    @staticmethod
    def _rms(samples: Float64Samples) -> float:
        return float(np.sqrt(np.mean(np.square(samples))))
