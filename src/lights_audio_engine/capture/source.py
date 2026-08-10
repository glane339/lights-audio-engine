"""Synchronous blocking-pull source contract for future capture backends."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from lights_audio_engine.capture.models import CaptureItem


@runtime_checkable
class AudioSource(Protocol):
    """Yield capture items synchronously until the source is exhausted or closed."""

    @property
    def sample_rate_hz(self) -> int:
        """Configured output sample rate for items yielded by this source."""
        ...

    def stream(self) -> Iterator[CaptureItem]:
        """Block as needed and yield frames or explicit discontinuities."""
        ...

    def close(self) -> None:
        """Release resources owned by the source implementation."""
        ...
