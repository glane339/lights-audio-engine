"""Validated configuration for the audio analysis engine."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


def _validate_positive_int(name: str, value: object, *, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


@dataclass(frozen=True, slots=True)
class AudioEngineConfig:
    """Configuration shared by deterministic M0/M1 analysis components.

    Sensitivity is normalized to ``0.0`` through ``1.0``. Higher values lower
    the energy detector's onset threshold, making quiet transients more likely
    to produce beat events.
    """

    expected_sample_rate_hz: int = 48_000
    sensitivity: float = 0.5
    min_bpm: float = 50.0
    max_bpm: float = 240.0
    analysis_window_seconds: float = 0.02
    energy_history_seconds: float = 2.0
    bpm_history_size: int = 8

    def __post_init__(self) -> None:
        _validate_positive_int("expected sample rate", self.expected_sample_rate_hz)
        sensitivity = _finite_number("sensitivity", self.sensitivity)
        if not 0.0 <= sensitivity <= 1.0:
            raise ValueError("sensitivity must be finite and between 0.0 and 1.0")
        min_bpm = _finite_number("minimum BPM", self.min_bpm)
        max_bpm = _finite_number("maximum BPM", self.max_bpm)
        if min_bpm <= 0.0 or max_bpm <= min_bpm:
            raise ValueError("BPM limits must be finite, positive, and ordered min < max")
        analysis_window_seconds = _finite_number(
            "analysis_window_seconds", self.analysis_window_seconds
        )
        if analysis_window_seconds <= 0.0:
            raise ValueError("analysis_window_seconds must be finite and positive")
        energy_history_seconds = _finite_number(
            "energy_history_seconds", self.energy_history_seconds
        )
        if energy_history_seconds < analysis_window_seconds:
            raise ValueError(
                "energy_history_seconds must be finite and at least one analysis window"
            )
        _validate_positive_int("bpm_history_size", self.bpm_history_size, minimum=3)
