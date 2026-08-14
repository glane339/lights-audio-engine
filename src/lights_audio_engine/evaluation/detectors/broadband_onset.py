"""Causal short-hop broadband novelty detector."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import median

import numpy as np
import numpy.typing as npt

from lights_audio_engine.models import AudioFrame


@dataclass(frozen=True, slots=True)
class BroadbandOnsetConfig:
    """Frozen M2C broadband parameters, independent of production configuration."""

    sample_rate_hz: int = 48_000
    hop_size_frames: int = 240
    novelty_floor: float = 0.0005
    adaptive_multiplier: float = 3.0
    history_hops: int = 400
    max_bpm: float = 240.0


@dataclass(frozen=True, slots=True)
class OnsetObservation:
    timestamp_seconds: float
    strength: float


class CausalNoveltyDetector:
    """Online novelty peak picker with exactly one hop of bounded lookahead."""

    def __init__(self, config: BroadbandOnsetConfig) -> None:
        self._config = config
        self._novelty_history: deque[float] = deque(maxlen=config.history_hops)
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
            feature = self._feature(hop)
            novelty = self._novelty(feature)
            threshold = max(
                self._config.novelty_floor,
                (median(self._novelty_history) if self._novelty_history else 0.0)
                * self._config.adaptive_multiplier,
            )
            if (
                self._previous_novelty is not None
                and self._previous_novelty >= self._previous_threshold
                and self._previous_novelty > novelty
                and self._previous_peak_sample is not None
            ):
                timestamp = stream_start + self._previous_peak_sample / self._config.sample_rate_hz
                minimum_interval = 60.0 / self._config.max_bpm
                if (
                    self._last_event_time is None
                    or timestamp - self._last_event_time >= minimum_interval - 1e-12
                ):
                    emitted.append(
                        OnsetObservation(
                            timestamp,
                            min(1.0, self._previous_novelty / self._strength_scale()),
                        )
                    )
                    self._last_event_time = timestamp
            self._novelty_history.append(novelty)
            self._previous_novelty = novelty
            self._previous_threshold = threshold
            self._previous_peak_sample = hop_start + int(np.abs(hop).argmax())
            self._previous_feature = feature
            self._after_hop(hop)
            consumed += self._config.hop_size_frames
        self._pending = np.array(combined[consumed:], dtype=np.float64, copy=True)
        return tuple(emitted)

    def reset(self) -> None:
        self._novelty_history.clear()
        self._pending = np.empty(0, dtype=np.float64)
        self._samples_received = 0
        self._stream_start: float | None = None
        self._previous_feature: float | npt.NDArray[np.float64] | None = None
        self._previous_novelty: float | None = None
        self._previous_threshold = 0.0
        self._previous_peak_sample: int | None = None
        self._last_event_time: float | None = None
        self._reset_feature_state()

    def _validate_start(self, frame: AudioFrame) -> float:
        if self._stream_start is None:
            self._stream_start = frame.start_time_seconds
        expected = self._stream_start + self._samples_received / self._config.sample_rate_hz
        if abs(frame.start_time_seconds - expected) > 0.5 / self._config.sample_rate_hz:
            raise ValueError("audio frames must have contiguous start timestamps")
        return self._stream_start

    def _feature(self, hop: npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        return float(np.sqrt(np.mean(np.square(hop))))

    def _novelty(self, feature: float | npt.NDArray[np.float64]) -> float:
        if self._previous_feature is None:
            return max(0.0, float(np.max(feature)))
        difference = np.asarray(feature) - np.asarray(self._previous_feature)
        return max(0.0, float(np.max(difference)))

    def _strength_scale(self) -> float:
        return 0.1

    def _after_hop(self, hop: npt.NDArray[np.float64]) -> None:
        del hop

    def _reset_feature_state(self) -> None:
        pass


class BroadbandOnsetDetector(CausalNoveltyDetector):
    """Broadband RMS-rise novelty at a 5 ms cadence."""

    def __init__(self, config: BroadbandOnsetConfig | None = None) -> None:
        super().__init__(config or BroadbandOnsetConfig())
