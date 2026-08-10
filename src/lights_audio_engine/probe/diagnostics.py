"""Pure device-selection and incremental signal-diagnostic logic."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeAlias, cast

import numpy as np
import numpy.typing as npt

from lights_audio_engine.probe.models import (
    CaptureDiagnostics,
    CaptureMode,
    CaptureTermination,
    DeviceInfo,
    DeviceSelectionError,
    HostApiInfo,
    ProbeSnapshot,
)

FloatBlock: TypeAlias = npt.NDArray[np.float32] | npt.NDArray[np.float64]


def input_devices(snapshot: ProbeSnapshot) -> tuple[DeviceInfo, ...]:
    """Return devices that report one or more input channels."""

    return tuple(device for device in snapshot.devices if device.max_input_channels > 0)


def find_wasapi_host_api(snapshot: ProbeSnapshot) -> HostApiInfo | None:
    """Return the first host API whose reported name contains ``wasapi``."""

    return next(
        (host_api for host_api in snapshot.host_apis if "wasapi" in host_api.name.casefold()),
        None,
    )


def _candidate_text(snapshot: ProbeSnapshot) -> str:
    candidates = input_devices(snapshot)
    if not candidates:
        return "Available input devices: none"
    return "Available input devices:\n" + "\n".join(
        f"  [{device.index}] {device.name}" for device in candidates
    )


def resolve_input_device(snapshot: ProbeSnapshot, selector: str) -> DeviceInfo:
    """Resolve a run-local index or unique case-insensitive name substring."""

    normalized = selector.strip()
    try:
        selected_index = int(normalized)
    except ValueError:
        selected_index = None

    if selected_index is not None:
        device = next((item for item in snapshot.devices if item.index == selected_index), None)
        if device is None:
            raise DeviceSelectionError(
                f"device index {selected_index} is not an input device\n{_candidate_text(snapshot)}"
            )
        if device.max_input_channels <= 0:
            raise DeviceSelectionError(
                f"device index {selected_index} is output-only\n{_candidate_text(snapshot)}"
            )
        return device

    matches = tuple(
        device
        for device in input_devices(snapshot)
        if normalized.casefold() in device.name.casefold()
    )
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise DeviceSelectionError(
            f"no input device matches {selector!r}\n{_candidate_text(snapshot)}"
        )
    match_list = ", ".join(f"[{device.index}] {device.name}" for device in matches)
    raise DeviceSelectionError(
        f"input device selector {selector!r} is ambiguous: {match_list}\n"
        f"{_candidate_text(snapshot)}"
    )


def reported_device_label(device: DeviceInfo, snapshot: ProbeSnapshot) -> str:
    """Build a display label with no stability guarantee across enumerations."""

    host_api = next(
        (item for item in snapshot.host_apis if item.index == device.host_api_index), None
    )
    host_name = host_api.name if host_api is not None else f"host API {device.host_api_index}"
    qualifiers: list[str] = []
    if device.index == snapshot.default_input_device_index or (
        host_api is not None and device.index == host_api.default_input_device_index
    ):
        qualifiers.append("default input")
    if device.index == snapshot.default_output_device_index or (
        host_api is not None and device.index == host_api.default_output_device_index
    ):
        qualifiers.append("default output")
    suffix = f", {', '.join(qualifiers)}" if qualifiers else ""
    return (
        f"[{device.index}] {device.name} — {host_name}, {device.max_input_channels} in / "
        f"{device.max_output_channels} out{suffix}"
    )


def _ordered_unique(values: Iterable[float | int]) -> tuple[float | int, ...]:
    return tuple(dict.fromkeys(values))


def compatibility_requests(
    device: DeviceInfo,
    sample_rate_hz: float | None = None,
    channels: int | None = None,
) -> tuple[tuple[float, int], ...]:
    """Create an order-stable compatibility matrix for one input device."""

    rates = (
        (float(sample_rate_hz),)
        if sample_rate_hz is not None
        else cast(
            tuple[float, ...], _ordered_unique((44_100.0, 48_000.0, device.default_sample_rate_hz))
        )
    )
    channel_values = (
        (channels,)
        if channels is not None
        else cast(
            tuple[int, ...],
            _ordered_unique(
                (1, *((2,) if device.max_input_channels >= 2 else ()), device.max_input_channels)
            ),
        )
    )
    return tuple((rate, channel_count) for rate in rates for channel_count in channel_values)


class CaptureAccumulator:
    """Aggregate per-channel signal diagnostics in constant capture-duration memory."""

    def __init__(self, channels: int) -> None:
        if isinstance(channels, bool) or channels <= 0:
            raise ValueError("channels must be a positive integer")
        self._channels = channels
        self._finite_count = np.zeros(channels, dtype=np.int64)
        self._sum_squares = np.zeros(channels, dtype=np.float64)
        self._peak = np.zeros(channels, dtype=np.float64)
        self._all_zero = np.ones(channels, dtype=np.bool_)
        self._near_full_scale_count = np.zeros(channels, dtype=np.int64)
        self._non_finite_count = np.zeros(channels, dtype=np.int64)
        self._frames_received = 0
        self._overflowed_read_count = 0

    def add_block(self, block_2d: FloatBlock, *, overflowed: bool) -> None:
        """Add one ``frames × channels`` block without retaining it."""

        if block_2d.ndim != 2:
            raise ValueError("capture block must be a two-dimensional NumPy array")
        if block_2d.shape[1] != self._channels:
            raise ValueError("capture block channel count does not match accumulator")
        self._frames_received += int(block_2d.shape[0])
        self._overflowed_read_count += int(overflowed)
        for channel_index in range(self._channels):
            column = np.asarray(block_2d[:, channel_index], dtype=np.float64)
            finite_mask = cast(npt.NDArray[np.bool_], np.isfinite(column))
            finite_values = column[finite_mask]
            self._finite_count[channel_index] += int(finite_values.size)
            self._non_finite_count[channel_index] += int(column.size - finite_values.size)
            column_all_zero = finite_values.size == column.size and (
                finite_values.size == 0 or float(np.max(np.abs(finite_values))) == 0.0
            )
            self._all_zero[channel_index] = bool(self._all_zero[channel_index] and column_all_zero)
            if finite_values.size:
                self._sum_squares[channel_index] += float(
                    np.sum(np.square(finite_values), dtype=np.float64)
                )
                self._peak[channel_index] = max(
                    self._peak[channel_index], float(np.max(np.abs(finite_values)))
                )
                self._near_full_scale_count[channel_index] += int(
                    np.sum(np.abs(finite_values) >= 0.999)
                )

    def finalize(
        self,
        *,
        device_index: int,
        mode: CaptureMode,
        requested_duration_seconds: float,
        requested_sample_rate_hz: float,
        requested_frame_count: int,
        actual_sample_rate_hz: float | None,
        reported_input_latency_seconds: float | None,
        termination: CaptureTermination,
        error_message: str | None,
    ) -> CaptureDiagnostics:
        """Produce an immutable diagnostic summary."""

        rms = np.zeros(self._channels, dtype=np.float64)
        nonempty = self._finite_count > 0
        rms[nonempty] = np.sqrt(self._sum_squares[nonempty] / self._finite_count[nonempty])
        duration_rate = actual_sample_rate_hz or requested_sample_rate_hz
        return CaptureDiagnostics(
            device_index=device_index,
            mode=mode,
            requested_duration_seconds=requested_duration_seconds,
            requested_sample_rate_hz=requested_sample_rate_hz,
            requested_channels=self._channels,
            requested_frame_count=requested_frame_count,
            actual_sample_rate_hz=actual_sample_rate_hz,
            reported_input_latency_seconds=reported_input_latency_seconds,
            frames_received=self._frames_received,
            captured_duration_seconds=self._frames_received / duration_rate,
            per_channel_rms=tuple(float(value) for value in rms),
            per_channel_peak=tuple(float(value) for value in self._peak),
            per_channel_all_zero=tuple(
                bool(value and self._frames_received > 0) for value in self._all_zero
            ),
            per_channel_near_full_scale_count=tuple(
                int(value) for value in self._near_full_scale_count
            ),
            per_channel_non_finite_count=tuple(int(value) for value in self._non_finite_count),
            overflowed_read_count=self._overflowed_read_count,
            termination=termination,
            error_message=error_message,
        )
