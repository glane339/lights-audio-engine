"""Immutable models that cross the audio engine boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import TypeAlias, cast

import numpy as np
import numpy.typing as npt

InputSamples: TypeAlias = (
    npt.NDArray[np.float16] | npt.NDArray[np.float32] | npt.NDArray[np.float64]
)
Float64Samples: TypeAlias = npt.NDArray[np.float64]


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _validate_normalized(name: str, value: object) -> None:
    normalized = _finite_number(name, value)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be finite and between 0.0 and 1.0")


def _validate_timestamp(name: str, value: object) -> float:
    timestamp_seconds = _finite_number(name, value)
    if timestamp_seconds < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return timestamp_seconds


def _validate_positive(name: str, value: object) -> None:
    if _finite_number(name, value) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _copy_validated_samples(samples: object) -> Float64Samples:
    if not isinstance(samples, np.ndarray):
        raise TypeError("samples must be a NumPy array")
    raw_samples = cast(npt.NDArray[np.generic], samples)
    if raw_samples.ndim != 1:
        raise ValueError("samples must be one-dimensional mono audio")
    if raw_samples.size == 0:
        raise ValueError("samples must not be empty")
    if not np.issubdtype(raw_samples.dtype, np.floating):
        raise TypeError("samples must use a floating-point dtype")

    normalized = raw_samples.astype(np.float64, copy=False)
    if not bool(np.isfinite(normalized).all()):
        raise ValueError("samples must contain only finite values")
    if bool((np.abs(normalized) > 1.0).any()):
        raise ValueError("samples must be normalized to the range -1.0 through 1.0")

    owned_samples = np.array(normalized, dtype=np.float64, copy=True, order="C")
    owned_samples.setflags(write=False)
    read_only_view = owned_samples.view()
    read_only_view.setflags(write=False)
    return read_only_view


def _validate_sample_rate(sample_rate_hz: object) -> int:
    if isinstance(sample_rate_hz, bool) or not isinstance(sample_rate_hz, int):
        raise TypeError("sample_rate_hz must be an integer")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    return sample_rate_hz


def _validate_beat_index(beat_index: object) -> None:
    if isinstance(beat_index, bool) or not isinstance(beat_index, int):
        raise TypeError("beat_index must be an integer")
    if beat_index < 0:
        raise ValueError("beat_index must be non-negative")


def _validate_beat_events(events: object) -> None:
    if not isinstance(events, tuple):
        raise TypeError("beat_events must be a tuple of BeatEvent values")
    event_values = cast(tuple[object, ...], events)
    if not all(isinstance(event, BeatEvent) for event in event_values):
        raise TypeError("beat_events must be a tuple of BeatEvent values")


def _validate_drop_events(events: object) -> None:
    if not isinstance(events, tuple):
        raise TypeError("drop_events must be a tuple of DropEvent values")
    event_values = cast(tuple[object, ...], events)
    if not all(isinstance(event, DropEvent) for event in event_values):
        raise TypeError("drop_events must be a tuple of DropEvent values")


@dataclass(frozen=True, slots=True, init=False, eq=False)
class AudioFrame:
    """One owned block of normalized mono PCM and its deterministic start time."""

    samples: Float64Samples
    sample_rate_hz: int
    start_time_seconds: float

    def __init__(
        self,
        samples: InputSamples,
        sample_rate_hz: int,
        start_time_seconds: float,
    ) -> None:
        validated_start_time = _validate_timestamp("start_time_seconds", start_time_seconds)

        object.__setattr__(self, "samples", _copy_validated_samples(samples))
        object.__setattr__(self, "sample_rate_hz", _validate_sample_rate(sample_rate_hz))
        object.__setattr__(self, "start_time_seconds", validated_start_time)


@dataclass(frozen=True, slots=True)
class BeatEvent:
    """A beat candidate with detector-relative transient energy as strength."""

    timestamp_seconds: float
    strength: float
    beat_index: int

    def __post_init__(self) -> None:
        _validate_timestamp("timestamp_seconds", self.timestamp_seconds)
        _validate_normalized("strength", self.strength)
        _validate_beat_index(self.beat_index)


@dataclass(frozen=True, slots=True)
class DropEvent:
    """A future-facing drop event model; M1 emits no drop events."""

    timestamp_seconds: float
    strength: float
    confidence: float | None = None

    def __post_init__(self) -> None:
        _validate_timestamp("timestamp_seconds", self.timestamp_seconds)
        _validate_normalized("strength", self.strength)
        if self.confidence is not None:
            _validate_normalized("confidence", self.confidence)


@dataclass(frozen=True, slots=True)
class AudioAnalysisResult:
    """Immutable result produced for one processed audio frame."""

    bpm: float | None = None
    beat_events: tuple[BeatEvent, ...] = field(default_factory=tuple)
    drop_events: tuple[DropEvent, ...] = field(default_factory=tuple)
    current_level: float = 0.0

    def __post_init__(self) -> None:
        if self.bpm is not None:
            _validate_positive("bpm", self.bpm)
        _validate_normalized("current_level", self.current_level)
        _validate_beat_events(self.beat_events)
        _validate_drop_events(self.drop_events)
