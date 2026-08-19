"""The three hardcoded M2C detector candidates."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from lights_audio_engine.config import AudioEngineConfig
from lights_audio_engine.detectors.energy import EnergyBeatDetector
from lights_audio_engine.evaluation.detectors.broadband_onset import BroadbandOnsetDetector
from lights_audio_engine.evaluation.detectors.causal_spectral_tempo import (
    CausalSpectralTempoDetector,
)
from lights_audio_engine.evaluation.detectors.multiband_onset import MultibandOnsetDetector
from lights_audio_engine.models import AudioFrame


@dataclass(frozen=True, slots=True)
class DetectedOnset:
    """Minimal evaluation-only event assigned by a candidate."""

    timestamp_seconds: float
    strength: float

    def __post_init__(self) -> None:
        if not isfinite(self.timestamp_seconds):
            raise ValueError("onset timestamp must be finite")
        if self.timestamp_seconds < 0.0:
            raise ValueError("onset timestamp must be non-negative")
        if not isfinite(self.strength) or not 0.0 <= self.strength <= 1.0:
            raise ValueError("onset strength must be finite and between 0.0 and 1.0")


class Candidate(Protocol):
    def process(self, frame: AudioFrame) -> tuple[DetectedOnset, ...]: ...

    def reset(self) -> None: ...


class BaselineCandidate:
    """Candidate A: an unchanged production ``EnergyBeatDetector`` wrapper."""

    def __init__(self, config: AudioEngineConfig | None = None) -> None:
        self._detector = EnergyBeatDetector(config or AudioEngineConfig())

    def process(self, frame: AudioFrame) -> tuple[DetectedOnset, ...]:
        return tuple(
            DetectedOnset(event.timestamp_seconds, event.strength)
            for event in self._detector.process(frame).transients
        )

    def reset(self) -> None:
        self._detector.reset()


class _NoveltyCandidate:
    def __init__(
        self,
        detector: BroadbandOnsetDetector | MultibandOnsetDetector | CausalSpectralTempoDetector,
    ) -> None:
        self._detector = detector

    def process(self, frame: AudioFrame) -> tuple[DetectedOnset, ...]:
        return tuple(
            DetectedOnset(event.timestamp_seconds, event.strength)
            for event in self._detector.process(frame)
        )

    def reset(self) -> None:
        self._detector.reset()


def create_candidate(name: str) -> Candidate:
    """Construct one of the deliberately hardcoded M2C candidates."""

    if name == "baseline":
        return BaselineCandidate()
    if name == "broadband":
        return _NoveltyCandidate(BroadbandOnsetDetector())
    if name == "multiband":
        return _NoveltyCandidate(MultibandOnsetDetector())
    if name == "causal-spectral-tempo":
        return _NoveltyCandidate(CausalSpectralTempoDetector())
    raise ValueError(f"unknown M2C candidate: {name}")
