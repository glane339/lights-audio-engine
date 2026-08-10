"""Explicit policies for converting capture blocks to mono."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, cast, runtime_checkable

import numpy as np
import numpy.typing as npt

from lights_audio_engine.capture.models import FrameAssemblyError

FloatBlock: TypeAlias = npt.NDArray[np.floating[Any]]


def _validate_index(index: object) -> int:
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("channel index must be an integer")
    if index < 0:
        raise ValueError("channel index must be non-negative")
    return index


def _validate_indices(indices: object) -> tuple[int, ...] | None:
    if indices is None:
        return None
    if not isinstance(indices, tuple):
        raise TypeError("channel indices must be a tuple of integers or None")
    if not indices:
        raise ValueError("explicit channel indices must not be empty")
    raw_indices = cast(tuple[object, ...], indices)
    if any(isinstance(index, bool) or not isinstance(index, int) for index in raw_indices):
        raise TypeError("channel indices must be integers")
    validated = cast(tuple[int, ...], raw_indices)
    if any(index < 0 for index in validated):
        raise ValueError("channel indices must be non-negative")
    if len(set(validated)) != len(validated):
        raise ValueError("channel indices must not contain duplicates")
    return validated


def _channel_count(block: FloatBlock) -> int:
    if block.ndim == 1:
        return 1
    if block.ndim != 2:
        raise FrameAssemblyError("audio blocks must be one- or two-dimensional")
    channel_count = block.shape[1]
    if channel_count == 0:
        raise FrameAssemblyError("two-dimensional audio blocks must contain a channel")
    return channel_count


@runtime_checkable
class ChannelPolicy(Protocol):
    """Convert a validated mono or multichannel capture block to mono."""

    def to_mono(self, block: FloatBlock) -> FloatBlock:
        """Return one-dimensional mono samples from ``block``."""
        ...


@dataclass(frozen=True, slots=True)
class MonoPassthrough:
    """Accept mono input without silently downmixing multiple channels."""

    def to_mono(self, block: FloatBlock) -> FloatBlock:
        if _channel_count(block) != 1:
            raise FrameAssemblyError("MonoPassthrough requires exactly one input channel")
        return block if block.ndim == 1 else block[:, 0]


@dataclass(frozen=True, slots=True)
class SelectChannel:
    """Select one deterministic zero-based channel index."""

    index: int

    def __post_init__(self) -> None:
        _validate_index(self.index)

    def to_mono(self, block: FloatBlock) -> FloatBlock:
        channel_count = _channel_count(block)
        if self.index >= channel_count:
            raise FrameAssemblyError(
                f"channel index {self.index} is outside observed channel count {channel_count}"
            )
        return block if block.ndim == 1 else block[:, self.index]


@dataclass(frozen=True, slots=True)
class AverageChannels:
    """Average all observed channels or one explicit channel subset."""

    indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        _validate_indices(self.indices)

    def to_mono(self, block: FloatBlock) -> FloatBlock:
        channel_count = _channel_count(block)
        indices = self.indices if self.indices is not None else tuple(range(channel_count))
        if any(index >= channel_count for index in indices):
            raise FrameAssemblyError(
                f"channel index is outside observed channel count {channel_count}"
            )
        if block.ndim == 1:
            return block
        return np.mean(block[:, indices], axis=1, dtype=np.float64)
