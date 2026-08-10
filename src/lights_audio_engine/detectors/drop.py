"""M1 drop-detection boundary."""

from __future__ import annotations

from lights_audio_engine.models import AudioFrame, DropEvent


class NoOpDropDetector:
    """Return no events until drop semantics are experimentally defined."""

    def process(self, frame: AudioFrame) -> tuple[DropEvent, ...]:
        """Accept a frame without inventing a drop event."""

        _ = frame
        return ()

    def reset(self) -> None:
        """Reset detector state; the M1 no-op detector has none."""
