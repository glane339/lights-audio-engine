"""Candidate execution with explicit stream-audio emission times."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from time import perf_counter

from lights_audio_engine.evaluation.candidates import Candidate, DetectedOnset
from lights_audio_engine.models import AudioFrame


@dataclass(frozen=True, slots=True)
class EmittedDetection:
    event: DetectedOnset
    emitted_stream_time_seconds: float

    @property
    def decision_latency_seconds(self) -> float:
        return self.emitted_stream_time_seconds - self.event.timestamp_seconds


@dataclass(frozen=True, slots=True)
class CandidateRun:
    detections: tuple[EmittedDetection, ...]
    processing_seconds: float


def run_candidate(candidate: Candidate, frames: Iterable[AudioFrame]) -> CandidateRun:
    """Run a candidate and record the end of available audio for each returned event."""

    detections: list[EmittedDetection] = []
    processing_seconds = 0.0
    for frame in frames:
        started = perf_counter()
        events = candidate.process(frame)
        processing_seconds += perf_counter() - started
        emitted_stream_time = frame.start_time_seconds + frame.samples.size / frame.sample_rate_hz
        detections.extend(EmittedDetection(event, emitted_stream_time) for event in events)
    return CandidateRun(tuple(detections), processing_seconds)
