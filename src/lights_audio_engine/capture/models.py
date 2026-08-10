"""Hardware-independent capture configuration, values, and errors."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Literal, TypeAlias

from lights_audio_engine.models import AudioFrame

if TYPE_CHECKING:
    from lights_audio_engine.capture.channels import ChannelPolicy


def _validate_discontinuity_reason(reason: object) -> None:
    if not isinstance(reason, DiscontinuityReason):
        raise TypeError("reason must be a DiscontinuityReason")


def _validate_non_negative_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_optional_detail(detail: object) -> None:
    if detail is not None and not isinstance(detail, str):
        raise TypeError("detail must be a string or None")


def _validate_capture_sample_rate(sample_rate_hz: object) -> None:
    if (
        isinstance(sample_rate_hz, bool)
        or not isinstance(sample_rate_hz, int)
        or sample_rate_hz <= 0
    ):
        raise ValueError("sample_rate_hz must be a positive integer")


def _validate_channel_policy(channel_policy: object) -> None:
    from lights_audio_engine.capture.channels import ChannelPolicy

    if not isinstance(channel_policy, ChannelPolicy):
        raise TypeError("channel_policy must implement ChannelPolicy")


class CaptureError(Exception):
    """Base error raised at the production capture boundary."""


class FrameAssemblyError(CaptureError):
    """A raw capture block cannot be converted into an ``AudioFrame``."""


class DiscontinuityReason(StrEnum):
    """Hardware-independent reasons why one logical stream ended."""

    OVERFLOW = "overflow"
    DROPPED_BLOCK = "dropped_block"
    SAMPLE_RATE_CHANGE = "sample_rate_change"
    RESTART = "restart"


@dataclass(frozen=True, slots=True)
class Discontinuity:
    """A yielded stream boundary; the next frame begins a new logical stream."""

    reason: DiscontinuityReason
    frames_emitted_before: int
    detail: str | None = None

    def __post_init__(self) -> None:
        _validate_discontinuity_reason(self.reason)
        _validate_non_negative_int("frames_emitted_before", self.frames_emitted_before)
        _validate_optional_detail(self.detail)


def _default_channel_policy() -> ChannelPolicy:
    from lights_audio_engine.capture.channels import MonoPassthrough

    return MonoPassthrough()


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    """Hardware-independent configuration for raw-block assembly."""

    sample_rate_hz: int = 48_000
    channel_policy: ChannelPolicy = field(default_factory=_default_channel_policy)
    out_of_range: Literal["clip", "reject"] = "clip"

    def __post_init__(self) -> None:
        _validate_capture_sample_rate(self.sample_rate_hz)
        _validate_channel_policy(self.channel_policy)
        if self.out_of_range not in ("clip", "reject"):
            raise ValueError("out_of_range must be 'clip' or 'reject'")


CaptureItem: TypeAlias = AudioFrame | Discontinuity
