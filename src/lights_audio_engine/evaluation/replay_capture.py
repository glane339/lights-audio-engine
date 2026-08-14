"""Diagnostic-only tap for exact post-assembler ``AudioFrame`` samples."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from lights_audio_engine.capture import AudioSource, CaptureItem
from lights_audio_engine.evaluation.artifact import SegmentInfo, write_artifact
from lights_audio_engine.models import AudioFrame


class CapturingAudioSource:
    """Observe and persist yielded capture items without changing them."""

    def __init__(self, source: AudioSource, path: Path, *, label: str) -> None:
        self._source = source
        self._path = Path(path)
        self._label = label or self._path.stem
        self._frames: list[np.ndarray] = []
        self._frame_lengths: list[int] = []
        self._segments: list[SegmentInfo] = []
        self._segment_start_sample: int | None = None
        self._segment_start_time = 0.0
        self._segment_sample_count = 0
        self._pending_discontinuity: str | None = None
        self._sample_count = 0
        self._closed = False

    @property
    def sample_rate_hz(self) -> int:
        return self._source.sample_rate_hz

    def stream(self) -> Iterator[CaptureItem]:
        for item in self._source.stream():
            if isinstance(item, AudioFrame):
                self._record_frame(item)
            else:
                self._finish_segment()
                self._pending_discontinuity = item.reason.value
            yield item

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._source.close()
        finally:
            self._finish_segment()
            samples = (
                np.concatenate(self._frames).astype(np.float64, copy=False)
                if self._frames
                else np.empty(0, dtype=np.float64)
            )
            write_artifact(
                self._path,
                samples,
                label=self._label,
                sample_rate_hz=self.sample_rate_hz,
                frame_lengths=tuple(self._frame_lengths),
                segments=tuple(self._segments) or None,
            )
            self._closed = True

    def _record_frame(self, frame: AudioFrame) -> None:
        if self._segment_start_sample is None:
            self._segment_start_sample = self._sample_count
            self._segment_start_time = frame.start_time_seconds
        copied = np.array(frame.samples, dtype=np.float64, copy=True)
        self._frames.append(copied)
        self._frame_lengths.append(int(copied.size))
        self._segment_sample_count += int(copied.size)
        self._sample_count += int(copied.size)

    def _finish_segment(self) -> None:
        if self._segment_start_sample is None:
            return
        self._segments.append(
            SegmentInfo(
                self._segment_start_sample,
                self._segment_sample_count,
                self._segment_start_time,
                self._pending_discontinuity,
            )
        )
        self._segment_start_sample = None
        self._segment_sample_count = 0
        self._pending_discontinuity = None
