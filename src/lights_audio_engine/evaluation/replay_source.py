"""Deterministic stream-relative replay of exact post-assembler samples."""

from __future__ import annotations

from collections.abc import Iterator

from lights_audio_engine.evaluation.artifact import PcmArtifact
from lights_audio_engine.models import AudioFrame


class ReplayAudioSource:
    """Slice one contiguous artifact segment directly into ``AudioFrame`` values."""

    def __init__(
        self,
        artifact: PcmArtifact,
        *,
        block_size_frames: int,
        segment_index: int | None = None,
    ) -> None:
        if block_size_frames <= 0:
            raise ValueError("block size must be positive")
        if segment_index is None:
            if len(artifact.segments) != 1:
                raise ValueError("a discontinuous artifact requires an explicit segment index")
            segment_index = 0
        if not 0 <= segment_index < len(artifact.segments):
            raise ValueError("segment index is out of range")
        self._artifact = artifact
        self._block_size = block_size_frames
        self._segment = artifact.segments[segment_index]

    @property
    def sample_rate_hz(self) -> int:
        return self._artifact.sample_rate_hz

    def stream(self) -> Iterator[AudioFrame]:
        start = self._segment.start_sample
        end = start + self._segment.sample_count
        segment_samples = self._artifact.samples[start:end]
        for offset in range(0, segment_samples.size, self._block_size):
            yield AudioFrame(
                segment_samples[offset : offset + self._block_size],
                self.sample_rate_hz,
                offset / self.sample_rate_hz,
            )

    def close(self) -> None:
        """Match the ``AudioSource`` boundary; replay owns no external resource."""
