"""Thin causal adapter around Aubio's native fixed-hop onset detector."""

from __future__ import annotations

from math import isfinite
from typing import Any, cast

import numpy as np

from lights_audio_engine.evaluation.candidates import DetectedOnset
from lights_audio_engine.models import AudioFrame

METHOD = "default"
WINDOW_SIZE_FRAMES = 1024
HOP_SIZE_FRAMES = 256


class AubioUnavailableError(ImportError):
    """Raised when the optional Aubio package is requested but unavailable."""


def _import_aubio() -> Any:
    try:
        import aubio  # pyright: ignore[reportMissingImports, reportMissingTypeStubs]
    except ImportError as error:
        message = (
            "Aubio is required for the causal advisory benchmark; "
            'install it with pip install -e ".[aubio]".'
        )
        raise AubioUnavailableError(message) from error
    return cast(Any, aubio)


class AubioOnsetCandidate:
    """Stream replay samples through an isolated Aubio ``onset('default')`` instance."""

    def __init__(self) -> None:
        self._aubio = _import_aubio()
        self._sample_rate_hz: int | None = None
        self._next_frame_start_seconds = 0.0
        self._pending = np.empty(0, dtype=np.float32)
        self._onset: Any | None = None

    def _start(self, sample_rate_hz: int) -> None:
        self._sample_rate_hz = sample_rate_hz
        self._onset = self._aubio.onset(METHOD, WINDOW_SIZE_FRAMES, HOP_SIZE_FRAMES, sample_rate_hz)

    def process(self, frame: AudioFrame) -> tuple[DetectedOnset, ...]:
        if self._sample_rate_hz is None:
            self._start(frame.sample_rate_hz)
        if frame.sample_rate_hz != self._sample_rate_hz:
            raise ValueError("Aubio benchmark stream sample rate changed")
        if not np.isclose(frame.start_time_seconds, self._next_frame_start_seconds):
            raise ValueError("Aubio benchmark frames must be contiguous and stream-relative")
        self._next_frame_start_seconds += frame.samples.size / frame.sample_rate_hz
        self._pending = np.concatenate(
            (self._pending, frame.samples.astype(np.float32, copy=False))
        )

        events: list[DetectedOnset] = []
        while self._pending.size >= HOP_SIZE_FRAMES:
            hop = self._pending[:HOP_SIZE_FRAMES]
            self._pending = self._pending[HOP_SIZE_FRAMES:]
            onset = self._onset
            if onset is None:
                raise RuntimeError("Aubio onset state was not initialized")
            result = onset(hop)
            if bool(np.asarray(result).reshape(-1)[0]):
                timestamp_seconds = float(onset.get_last_s())
                if not isfinite(timestamp_seconds) or timestamp_seconds < 0.0:
                    raise ValueError("Aubio returned an invalid onset timestamp")
                # Aubio's descriptor has no normalized cross-method scale. Preserve a binary event.
                events.append(DetectedOnset(timestamp_seconds, 1.0))
        return tuple(events)

    def reset(self) -> None:
        self._sample_rate_hz = None
        self._next_frame_start_seconds = 0.0
        self._pending = np.empty(0, dtype=np.float32)
        self._onset = None
