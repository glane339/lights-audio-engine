"""Typed, immutable records for the experimental hardware probe."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal, TypeAlias

CaptureMode: TypeAlias = Literal["shared", "exclusive"]
CaptureTermination: TypeAlias = Literal["completed", "interrupted", "failed"]


class ProbeError(Exception):
    """Base class for expected probe failures."""


class BackendUnavailableError(ProbeError):
    """The optional audio backend could not be loaded or initialized."""


class DeviceSelectionError(ProbeError):
    """A requested input device could not be selected unambiguously."""


class UnsupportedConfigurationError(ProbeError):
    """The requested probe mode cannot apply to the selected device."""


def _require_int(name: str, value: object, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")


def _require_finite(name: str, value: object, *, minimum: float, inclusive: bool) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    invalid = normalized < minimum if inclusive else normalized <= minimum
    if not isfinite(normalized) or invalid:
        qualifier = "at least" if inclusive else "greater than"
        raise ValueError(f"{name} must be finite and {qualifier} {minimum}")


def _require_optional_index(name: str, value: int | None) -> None:
    if value is not None:
        _require_int(name, value)


def _require_mode(mode: object) -> None:
    if mode not in ("shared", "exclusive"):
        raise ValueError("mode must be 'shared' or 'exclusive'")


def _require_termination(termination: object) -> None:
    if termination not in ("completed", "interrupted", "failed"):
        raise ValueError("termination must be 'completed', 'interrupted', or 'failed'")


@dataclass(frozen=True, slots=True)
class HostApiInfo:
    """One PortAudio host API as reported during the current enumeration."""

    index: int
    name: str
    device_indexes: tuple[int, ...]
    default_input_device_index: int | None
    default_output_device_index: int | None

    def __post_init__(self) -> None:
        _require_int("index", self.index)
        if not self.name:
            raise ValueError("name must not be empty")
        for device_index in self.device_indexes:
            _require_int("device_indexes entry", device_index)
        _require_optional_index("default_input_device_index", self.default_input_device_index)
        _require_optional_index("default_output_device_index", self.default_output_device_index)


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """One run-local PortAudio device record."""

    index: int
    name: str
    host_api_index: int
    max_input_channels: int
    max_output_channels: int
    default_sample_rate_hz: float
    default_low_input_latency_seconds: float
    default_high_input_latency_seconds: float

    def __post_init__(self) -> None:
        _require_int("index", self.index)
        if not self.name:
            raise ValueError("name must not be empty")
        _require_int("host_api_index", self.host_api_index)
        _require_int("max_input_channels", self.max_input_channels)
        _require_int("max_output_channels", self.max_output_channels)
        _require_finite(
            "default_sample_rate_hz", self.default_sample_rate_hz, minimum=0.0, inclusive=False
        )
        _require_finite(
            "default_low_input_latency_seconds",
            self.default_low_input_latency_seconds,
            minimum=0.0,
            inclusive=True,
        )
        _require_finite(
            "default_high_input_latency_seconds",
            self.default_high_input_latency_seconds,
            minimum=0.0,
            inclusive=True,
        )


@dataclass(frozen=True, slots=True)
class ProbeSnapshot:
    """A typed snapshot of one PortAudio enumeration."""

    sounddevice_version: str
    portaudio_version_text: str
    host_apis: tuple[HostApiInfo, ...]
    devices: tuple[DeviceInfo, ...]
    default_input_device_index: int | None
    default_output_device_index: int | None

    def __post_init__(self) -> None:
        _require_optional_index("default_input_device_index", self.default_input_device_index)
        _require_optional_index("default_output_device_index", self.default_output_device_index)


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    """PortAudio's answer for one input configuration."""

    device_index: int
    sample_rate_hz: float
    channels: int
    mode: CaptureMode
    supported: bool
    portaudio_message: str | None

    def __post_init__(self) -> None:
        _require_int("device_index", self.device_index)
        _require_finite("sample_rate_hz", self.sample_rate_hz, minimum=0.0, inclusive=False)
        _require_int("channels", self.channels, minimum=1)
        _require_mode(self.mode)


@dataclass(frozen=True, slots=True)
class CaptureDiagnostics:
    """Incremental capture diagnostics with no retained PCM."""

    device_index: int
    mode: CaptureMode
    requested_duration_seconds: float
    requested_sample_rate_hz: float
    requested_channels: int
    requested_frame_count: int
    actual_sample_rate_hz: float | None
    reported_input_latency_seconds: float | None
    frames_received: int
    captured_duration_seconds: float
    per_channel_rms: tuple[float, ...]
    per_channel_peak: tuple[float, ...]
    per_channel_all_zero: tuple[bool, ...]
    per_channel_near_full_scale_count: tuple[int, ...]
    per_channel_non_finite_count: tuple[int, ...]
    overflowed_read_count: int
    termination: CaptureTermination
    error_message: str | None

    def __post_init__(self) -> None:
        _require_int("device_index", self.device_index)
        _require_mode(self.mode)
        _require_finite(
            "requested_duration_seconds",
            self.requested_duration_seconds,
            minimum=0.0,
            inclusive=False,
        )
        _require_finite(
            "requested_sample_rate_hz",
            self.requested_sample_rate_hz,
            minimum=0.0,
            inclusive=False,
        )
        _require_int("requested_channels", self.requested_channels, minimum=1)
        _require_int("requested_frame_count", self.requested_frame_count, minimum=1)
        if self.actual_sample_rate_hz is not None:
            _require_finite(
                "actual_sample_rate_hz", self.actual_sample_rate_hz, minimum=0.0, inclusive=False
            )
        if self.reported_input_latency_seconds is not None:
            _require_finite(
                "reported_input_latency_seconds",
                self.reported_input_latency_seconds,
                minimum=0.0,
                inclusive=True,
            )
        _require_int("frames_received", self.frames_received)
        _require_finite(
            "captured_duration_seconds",
            self.captured_duration_seconds,
            minimum=0.0,
            inclusive=True,
        )
        _require_int("overflowed_read_count", self.overflowed_read_count)
        _require_termination(self.termination)
        channel_fields = (
            self.per_channel_rms,
            self.per_channel_peak,
            self.per_channel_all_zero,
            self.per_channel_near_full_scale_count,
            self.per_channel_non_finite_count,
        )
        if any(len(values) != self.requested_channels for values in channel_fields):
            raise ValueError("per-channel fields must match requested_channels")
