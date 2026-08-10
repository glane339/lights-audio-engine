"""Pure stateful conversion from backend-neutral blocks to ``AudioFrame`` values."""

from __future__ import annotations

from math import isfinite
from typing import cast

import numpy as np
import numpy.typing as npt

from lights_audio_engine.capture.channels import FloatBlock
from lights_audio_engine.capture.models import (
    CaptureConfig,
    CaptureItem,
    Discontinuity,
    DiscontinuityReason,
    FrameAssemblyError,
)
from lights_audio_engine.models import AudioFrame, InputSamples


def _validate_epoch(epoch_seconds: object) -> float:
    if isinstance(epoch_seconds, bool) or not isinstance(epoch_seconds, (int, float)):
        raise TypeError("epoch_seconds must be a real number")
    epoch = float(epoch_seconds)
    if not isfinite(epoch) or epoch < 0.0:
        raise ValueError("epoch_seconds must be finite and non-negative")
    return epoch


def _validate_observed_sample_rate(sample_rate_hz: object) -> int:
    if (
        isinstance(sample_rate_hz, bool)
        or not isinstance(sample_rate_hz, int)
        or sample_rate_hz <= 0
    ):
        raise FrameAssemblyError("observed sample_rate_hz must be a positive integer")
    return sample_rate_hz


def _validate_detail(detail: object) -> str | None:
    if detail is not None and not isinstance(detail, str):
        raise TypeError("detail must be a string or None")
    return detail


def _require_config(config: object) -> CaptureConfig:
    if not isinstance(config, CaptureConfig):
        raise TypeError("config must be a CaptureConfig")
    return config


def _validate_overflow(overflow: object) -> bool:
    if not isinstance(overflow, bool):
        raise TypeError("overflow must be a bool")
    return overflow


def _normalize_reason(reason: object) -> DiscontinuityReason:
    if not isinstance(reason, str):
        raise TypeError("reason must be a DiscontinuityReason or string value")
    try:
        return DiscontinuityReason(reason)
    except ValueError as error:
        raise ValueError(f"unknown discontinuity reason: {reason}") from error


class FrameAssembler:
    """Build contiguous frames and explicit boundaries from raw capture blocks."""

    def __init__(
        self,
        config: CaptureConfig | None = None,
        *,
        epoch_seconds: float = 0.0,
    ) -> None:
        self._config = _require_config(config) if config is not None else CaptureConfig()
        self._epoch_seconds = _validate_epoch(epoch_seconds)
        self._samples_emitted = 0
        self._frames_emitted = 0
        self._channel_count: int | None = None

    @property
    def samples_emitted(self) -> int:
        """Number of samples emitted in the current logical stream."""

        return self._samples_emitted

    @property
    def frames_emitted(self) -> int:
        """Number of ``AudioFrame`` values emitted in the current logical stream."""

        return self._frames_emitted

    def assemble(
        self,
        block: object,
        *,
        sample_rate_hz: int | None = None,
        overflow: bool = False,
        detail: str | None = None,
    ) -> CaptureItem | None:
        """Convert one raw block, discarding it when it marks a discontinuity."""

        validated_detail = _validate_detail(detail)
        if _validate_overflow(overflow):
            return self._mark(DiscontinuityReason.OVERFLOW, validated_detail)

        observed_sample_rate = (
            self._config.sample_rate_hz
            if sample_rate_hz is None
            else _validate_observed_sample_rate(sample_rate_hz)
        )
        if observed_sample_rate != self._config.sample_rate_hz:
            return self._mark(
                DiscontinuityReason.SAMPLE_RATE_CHANGE,
                f"expected {self._config.sample_rate_hz} Hz, observed {observed_sample_rate} Hz",
            )

        raw_block, frame_count, channel_count = self._validate_block(block)
        if frame_count == 0:
            return None
        if self._channel_count is not None and channel_count != self._channel_count:
            raise FrameAssemblyError(
                f"channel count changed from {self._channel_count} to {channel_count}"
            )

        try:
            mono = self._config.channel_policy.to_mono(raw_block)
        except FrameAssemblyError:
            raise
        except (TypeError, ValueError) as error:
            raise FrameAssemblyError(
                "channel policy could not convert the capture block"
            ) from error

        normalized = self._normalize_mono(mono)
        start_time_seconds = (
            self._epoch_seconds + self._samples_emitted / self._config.sample_rate_hz
        )
        try:
            frame = AudioFrame(
                samples=cast(InputSamples, normalized),
                sample_rate_hz=self._config.sample_rate_hz,
                start_time_seconds=start_time_seconds,
            )
        except (TypeError, ValueError) as error:
            raise FrameAssemblyError(
                "capture block could not be assembled into an AudioFrame"
            ) from error

        self._samples_emitted += frame.samples.size
        self._frames_emitted += 1
        if self._channel_count is None:
            self._channel_count = channel_count
        return frame

    def mark_discontinuity(
        self,
        reason: DiscontinuityReason | str,
        *,
        detail: str | None = None,
    ) -> Discontinuity:
        """End the current logical stream and return its explicit boundary value."""

        return self._mark(_normalize_reason(reason), _validate_detail(detail))

    @staticmethod
    def _validate_block(block: object) -> tuple[FloatBlock, int, int]:
        if not isinstance(block, np.ndarray):
            raise FrameAssemblyError("capture block must be a NumPy array")
        raw_block = cast(npt.NDArray[np.generic], block)
        if not np.issubdtype(raw_block.dtype, np.floating):
            raise FrameAssemblyError("capture block must use a floating-point dtype")
        if raw_block.ndim == 1:
            frame_count = raw_block.shape[0]
            channel_count = 1
        elif raw_block.ndim == 2:
            frame_count, channel_count = raw_block.shape
            if channel_count == 0:
                raise FrameAssemblyError("two-dimensional capture block must contain a channel")
        else:
            raise FrameAssemblyError("capture block must be one- or two-dimensional")
        if not bool(np.isfinite(raw_block).all()):
            raise FrameAssemblyError("capture block must contain only finite values")
        return cast(FloatBlock, raw_block), frame_count, channel_count

    def _normalize_mono(self, mono: object) -> npt.NDArray[np.float64]:
        if not isinstance(mono, np.ndarray) or mono.ndim != 1 or mono.size == 0:
            raise FrameAssemblyError("channel policy must return non-empty one-dimensional samples")
        converted = cast(npt.NDArray[np.generic], mono).astype(np.float64, copy=False)
        if not bool(np.isfinite(converted).all()):
            raise FrameAssemblyError("mono capture samples must contain only finite values")
        is_out_of_range = bool((np.abs(converted) > 1.0).any())
        if is_out_of_range and self._config.out_of_range == "reject":
            raise FrameAssemblyError("capture samples must be in the range -1.0 through 1.0")
        if is_out_of_range:
            clipped = np.array(converted, dtype=np.float64, copy=True)
            clipped[clipped < -1.0] = -1.0
            clipped[clipped > 1.0] = 1.0
            return clipped
        return converted

    def _mark(
        self,
        reason: DiscontinuityReason,
        detail: str | None,
    ) -> Discontinuity:
        discontinuity = Discontinuity(
            reason=reason,
            frames_emitted_before=self._frames_emitted,
            detail=detail,
        )
        self._samples_emitted = 0
        self._frames_emitted = 0
        self._channel_count = None
        return discontinuity
