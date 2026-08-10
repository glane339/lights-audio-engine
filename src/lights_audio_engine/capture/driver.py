"""Minimal ownership boundary between an audio source and the analysis engine."""

from __future__ import annotations

from collections.abc import Iterator

from lights_audio_engine.capture.models import Discontinuity
from lights_audio_engine.capture.source import AudioSource
from lights_audio_engine.engine import AudioEngine
from lights_audio_engine.models import AudioAnalysisResult


def run_engine(source: AudioSource, engine: AudioEngine) -> Iterator[AudioAnalysisResult]:
    """Process frames and reset the engine at every explicit stream boundary."""

    for item in source.stream():
        if isinstance(item, Discontinuity):
            engine.reset()
            continue
        yield engine.process(item)
