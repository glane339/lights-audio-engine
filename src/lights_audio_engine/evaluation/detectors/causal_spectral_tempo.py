"""Evaluation-only causal spectral-flux detector with bounded tempo state."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import median

import numpy as np
import numpy.typing as npt

from lights_audio_engine.evaluation.detectors.broadband_onset import OnsetObservation
from lights_audio_engine.models import AudioFrame


@dataclass(frozen=True, slots=True)
class CausalSpectralTempoConfig:
    """Frozen parameters for the evaluation-only causal spectral-tempo candidate."""

    sample_rate_hz: int = 48_000
    hop_size_frames: int = 240
    window_size_frames: int = 960
    fft_size_frames: int = 1_024
    band_count: int = 48
    minimum_frequency_hz: float = 40.0
    maximum_frequency_hz: float = 16_000.0
    history_hops: int = 400
    minimum_history_hops: int = 12
    novelty_floor: float = 1e-5
    mad_multiplier: float = 3.0
    min_bpm: float = 50.0
    max_bpm: float = 240.0
    phase_tolerance_fraction: float = 0.16
    phase_threshold_reduction: float = 0.70
    off_phase_threshold_multiplier: float = 1.60


class CausalSpectralTempoDetector:
    """Streaming spectral novelty with one-hop confirmation and causal phase gating."""

    def __init__(self, config: CausalSpectralTempoConfig | None = None) -> None:
        self._config = config or CausalSpectralTempoConfig()
        self._window = np.hanning(self._config.window_size_frames)
        frequencies = np.fft.rfftfreq(
            self._config.fft_size_frames, 1.0 / self._config.sample_rate_hz
        )
        edges = np.geomspace(
            self._config.minimum_frequency_hz,
            self._config.maximum_frequency_hz,
            self._config.band_count + 1,
        )
        self._band_masks = tuple(
            (frequencies >= low) & (frequencies < high)
            for low, high in zip(edges[:-1], edges[1:], strict=True)
        )
        self.reset()

    def process(self, frame: AudioFrame) -> tuple[OnsetObservation, ...]:
        if frame.sample_rate_hz != self._config.sample_rate_hz:
            raise ValueError(
                f"frame sample rate {frame.sample_rate_hz} does not match configured sample rate "
                f"{self._config.sample_rate_hz}"
            )
        stream_start = self._validate_start(frame)
        combined = np.concatenate((self._pending, frame.samples))
        combined_start = self._samples_received - self._pending.size
        self._samples_received += frame.samples.size
        emitted: list[OnsetObservation] = []
        consumed = 0
        while combined.size - consumed >= self._config.hop_size_frames:
            hop = combined[consumed : consumed + self._config.hop_size_frames]
            hop_start = combined_start + consumed
            self._trailing_samples.extend(float(sample) for sample in hop)
            if len(self._trailing_samples) == self._config.window_size_frames:
                novelty = self._spectral_flux()
                threshold = self._threshold()
                if (
                    self._previous_novelty is not None
                    and self._previous_timestamp is not None
                    and len(self._novelty_history) >= self._config.minimum_history_hops
                    and self._previous_novelty > novelty
                ):
                    phase_multiplier = self._phase_multiplier(self._previous_timestamp)
                    required = self._previous_threshold * phase_multiplier
                    off_phase = phase_multiplier == self._config.off_phase_threshold_multiplier
                    can_reacquire = off_phase and self._record_off_phase_candidate(
                        self._previous_timestamp
                    )
                    if (
                        self._previous_novelty >= required
                        and self._can_accept(self._previous_timestamp)
                        and (not off_phase or can_reacquire)
                    ):
                        emitted.append(
                            OnsetObservation(
                                self._previous_timestamp,
                                min(1.0, self._previous_novelty / max(required * 2.0, 1e-12)),
                            )
                        )
                        self._accept(self._previous_timestamp)
                        self._off_phase_candidates.clear()
                self._novelty_history.append(novelty)
                self._previous_novelty = novelty
                self._previous_threshold = threshold
                self._previous_timestamp = (
                    stream_start
                    + (hop_start + int(np.abs(hop).argmax())) / self._config.sample_rate_hz
                )
            consumed += self._config.hop_size_frames
        self._pending = np.array(combined[consumed:], dtype=np.float64, copy=True)
        return tuple(emitted)

    def reset(self) -> None:
        self._pending = np.empty(0, dtype=np.float64)
        self._trailing_samples: deque[float] = deque(maxlen=self._config.window_size_frames)
        self._novelty_history: deque[float] = deque(maxlen=self._config.history_hops)
        self._accepted_intervals: deque[float] = deque(maxlen=16)
        self._off_phase_candidates: deque[float] = deque(maxlen=2)
        self._tempo_hypotheses: tuple[float, ...] = ()
        self._samples_received = 0
        self._stream_start: float | None = None
        self._previous_bands: npt.NDArray[np.float64] | None = None
        self._previous_novelty: float | None = None
        self._previous_threshold = self._config.novelty_floor
        self._previous_timestamp: float | None = None
        self._last_event_time: float | None = None

    def _validate_start(self, frame: AudioFrame) -> float:
        if self._stream_start is None:
            self._stream_start = frame.start_time_seconds
        expected = self._stream_start + self._samples_received / self._config.sample_rate_hz
        if abs(frame.start_time_seconds - expected) > 0.5 / self._config.sample_rate_hz:
            raise ValueError("audio frames must have contiguous start timestamps")
        return self._stream_start

    def _spectral_flux(self) -> float:
        samples = np.asarray(self._trailing_samples, dtype=np.float64)
        spectrum = np.abs(np.fft.rfft(samples * self._window, n=self._config.fft_size_frames))
        bands = np.asarray(
            [
                float(np.log1p(np.sum(np.square(spectrum[mask])))) if bool(mask.any()) else 0.0
                for mask in self._band_masks
            ],
            dtype=np.float64,
        )
        if self._previous_bands is None:
            novelty = 0.0
        else:
            novelty = float(np.median(np.maximum(0.0, bands - self._previous_bands)))
        self._previous_bands = bands
        return novelty

    def _threshold(self) -> float:
        if not self._novelty_history:
            return self._config.novelty_floor
        center = median(self._novelty_history)
        deviations = tuple(abs(value - center) for value in self._novelty_history)
        return max(
            self._config.novelty_floor,
            center + self._config.mad_multiplier * median(deviations),
        )

    def _can_accept(self, timestamp: float) -> bool:
        return self._last_event_time is None or timestamp - self._last_event_time >= (
            60.0 / self._config.max_bpm - 1e-12
        )

    def _accept(self, timestamp: float) -> None:
        if self._last_event_time is not None:
            interval = timestamp - self._last_event_time
            if 60.0 / self._config.max_bpm <= interval <= 60.0 / self._config.min_bpm:
                self._accepted_intervals.append(interval)
        self._last_event_time = timestamp

    def _phase_multiplier(self, timestamp: float) -> float:
        if self._last_event_time is None or len(self._accepted_intervals) < 3:
            return 1.0
        primary_period = median(self._accepted_intervals)
        self._tempo_hypotheses = tuple(
            period
            for period in (primary_period / 2.0, primary_period, primary_period * 2.0)
            if 60.0 / self._config.max_bpm <= period <= 60.0 / self._config.min_bpm
        )
        offset = (timestamp - self._last_event_time) % primary_period
        residual = min(offset, primary_period - offset) / primary_period
        if residual <= self._config.phase_tolerance_fraction:
            return self._config.phase_threshold_reduction
        if residual >= 0.5 - self._config.phase_tolerance_fraction:
            return self._config.off_phase_threshold_multiplier
        return 1.0

    def _record_off_phase_candidate(self, timestamp: float) -> bool:
        """Permit a stable new cadence to reacquire, but reject isolated phase errors."""

        self._off_phase_candidates.append(timestamp)
        if len(self._off_phase_candidates) < 2:
            return False
        interval = self._off_phase_candidates[-1] - self._off_phase_candidates[-2]
        return 60.0 / self._config.max_bpm <= interval <= 60.0 / self._config.min_bpm
