"""Developer-oriented command line for experimental audio-hardware diagnostics."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from typing import cast

from lights_audio_engine.probe.backend import AudioProbeBackend, SoundDeviceBackend
from lights_audio_engine.probe.diagnostics import (
    compatibility_requests,
    find_wasapi_host_api,
    input_devices,
    reported_device_label,
    resolve_input_device,
)
from lights_audio_engine.probe.models import (
    BackendUnavailableError,
    CaptureDiagnostics,
    CaptureMode,
    CompatibilityResult,
    DeviceInfo,
    DeviceSelectionError,
    ProbeSnapshot,
    UnsupportedConfigurationError,
)

BackendFactory = Callable[[], AudioProbeBackend]


@dataclass(frozen=True, slots=True)
class _Options:
    command: str
    device: str | None
    duration: float
    sample_rate_hz: float | None
    channels: int | None
    exclusive: bool


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _add_format_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sample-rate", dest="sample_rate_hz", type=_positive_float)
    parser.add_argument("--channels", type=_positive_int)
    parser.add_argument(
        "--exclusive",
        action="store_true",
        help="diagnose explicit-format WASAPI exclusive mode",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lights_audio_engine.probe",
        description="Experimental developer diagnostics for Windows audio hardware.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("devices", help="list PortAudio host APIs and devices")

    check = subparsers.add_parser("check", help="check input-format compatibility")
    check.add_argument("--device", required=True)
    _add_format_arguments(check)

    capture = subparsers.add_parser("capture", help="capture incremental signal diagnostics")
    capture.add_argument("--device", required=True)
    capture.add_argument("--duration", type=_positive_float, default=3.0)
    _add_format_arguments(capture)

    report = subparsers.add_parser("report", help="run inventory, checks, and one capture")
    report.add_argument("--device")
    report.add_argument("--duration", type=_positive_float, default=3.0)
    _add_format_arguments(report)
    return parser


def _options(namespace: argparse.Namespace) -> _Options:
    values = cast(dict[str, object], vars(namespace))
    return _Options(
        command=cast(str, values["command"]),
        device=cast(str | None, values.get("device")),
        duration=cast(float, values.get("duration", 3.0)),
        sample_rate_hz=cast(float | None, values.get("sample_rate_hz")),
        channels=cast(int | None, values.get("channels")),
        exclusive=cast(bool, values.get("exclusive", False)),
    )


def _mode(options: _Options) -> CaptureMode:
    return "exclusive" if options.exclusive else "shared"


def _print_inventory(snapshot: ProbeSnapshot) -> None:
    print(f"sounddevice: {snapshot.sounddevice_version}")
    print(f"PortAudio: {snapshot.portaudio_version_text}")
    print(
        "Global defaults: "
        f"input={snapshot.default_input_device_index}, "
        f"output={snapshot.default_output_device_index}"
    )
    if snapshot.host_apis:
        print("Host APIs:")
        for host_api in snapshot.host_apis:
            print(
                f"  [{host_api.index}] {host_api.name}; devices={host_api.device_indexes}; "
                f"default input={host_api.default_input_device_index}; "
                f"default output={host_api.default_output_device_index}"
            )
    else:
        print("Host APIs: none")
    if snapshot.devices:
        print("Devices:")
        for device in snapshot.devices:
            print(f"  {reported_device_label(device, snapshot)}")
    else:
        print("Devices: none")


def _run_checks(
    backend: AudioProbeBackend,
    device: DeviceInfo,
    sample_rate_hz: float | None,
    channels: int | None,
    mode: CaptureMode,
) -> tuple[CompatibilityResult, ...]:
    results = tuple(
        backend.check_input_settings(device.index, rate, channel_count, mode)
        for rate, channel_count in compatibility_requests(device, sample_rate_hz, channels)
    )
    print(f"Compatibility checks for [{device.index}] {device.name} ({mode}):")
    for result in results:
        state = "supported" if result.supported else "unsupported"
        detail = f" — {result.portaudio_message}" if result.portaudio_message else ""
        print(f"  {result.sample_rate_hz:g} Hz × {result.channels} ch: {state}{detail}")
    if not any(result.supported for result in results):
        print("No tested combinations are supported.")
    return results


def _print_capture(diagnostics: CaptureDiagnostics) -> None:
    print("Capture diagnostics:")
    print(f"  device index: {diagnostics.device_index}")
    print(f"  mode: {diagnostics.mode}")
    print(f"  termination: {diagnostics.termination}")
    print(
        f"  requested: {diagnostics.requested_frame_count} frames, "
        f"{diagnostics.requested_duration_seconds:g} s, "
        f"{diagnostics.requested_sample_rate_hz:g} Hz, "
        f"{diagnostics.requested_channels} ch"
    )
    print(
        f"  received: {diagnostics.frames_received} frames, "
        f"{diagnostics.captured_duration_seconds:.6f} s"
    )
    print(f"  actual sample rate: {diagnostics.actual_sample_rate_hz}")
    print(f"  reported input latency: {diagnostics.reported_input_latency_seconds}")
    for index in range(diagnostics.requested_channels):
        print(
            f"  channel {index + 1}: RMS={diagnostics.per_channel_rms[index]:.8f}, "
            f"peak={diagnostics.per_channel_peak[index]:.8f}, "
            f"all_zero={diagnostics.per_channel_all_zero[index]}, "
            "near_full_scale="
            f"{diagnostics.per_channel_near_full_scale_count[index]}, "
            f"non_finite={diagnostics.per_channel_non_finite_count[index]}"
        )
        if diagnostics.per_channel_all_zero[index]:
            print(f"  WARNING channel {index + 1}: all_samples_zero")
        if diagnostics.per_channel_near_full_scale_count[index]:
            print(
                f"  WARNING channel {index + 1}: samples reached abs(sample) >= 0.999; "
                "this is not proof of clipping"
            )
        if diagnostics.per_channel_non_finite_count[index]:
            print(
                f"  WARNING channel {index + 1}: non-finite samples make this capture untrustworthy"
            )
    print(
        f"  overflowed reads: {diagnostics.overflowed_read_count} "
        "(reads reporting overflow, not estimated dropped frames)"
    )
    if diagnostics.overflowed_read_count:
        print("  WARNING: one or more reads reporting overflow")
    if diagnostics.error_message:
        print(f"  error: {diagnostics.error_message}")


def _capture_exit_code(diagnostics: CaptureDiagnostics) -> int:
    if diagnostics.termination == "interrupted":
        return 130
    if diagnostics.termination == "failed":
        return 6
    if any(diagnostics.per_channel_non_finite_count):
        return 6
    return 0


def _capture(
    backend: AudioProbeBackend,
    device: DeviceInfo,
    options: _Options,
    *,
    sample_rate_override: float | None = None,
    channels_override: int | None = None,
) -> int:
    sample_rate_hz = sample_rate_override or options.sample_rate_hz or device.default_sample_rate_hz
    channels = channels_override or options.channels or 1
    mode = _mode(options)
    compatibility = backend.check_input_settings(device.index, sample_rate_hz, channels, mode)
    print(
        f"Preflight {sample_rate_hz:g} Hz × {channels} ch: "
        f"{'supported' if compatibility.supported else 'unsupported'}"
    )
    if not compatibility.supported:
        if compatibility.portaudio_message:
            print(compatibility.portaudio_message)
        return 5
    diagnostics = backend.capture(
        device.index,
        options.duration,
        sample_rate_hz,
        channels,
        mode,
    )
    _print_capture(diagnostics)
    return _capture_exit_code(diagnostics)


def _default_report_device(snapshot: ProbeSnapshot) -> DeviceInfo:
    wasapi = find_wasapi_host_api(snapshot)
    if wasapi is not None and wasapi.default_input_device_index is not None:
        candidate = next(
            (
                device
                for device in input_devices(snapshot)
                if device.index == wasapi.default_input_device_index
            ),
            None,
        )
        if candidate is not None:
            print(f"Selected WASAPI default input device [{candidate.index}] {candidate.name}.")
            return candidate
    if snapshot.default_input_device_index is not None:
        candidate = next(
            (
                device
                for device in input_devices(snapshot)
                if device.index == snapshot.default_input_device_index
            ),
            None,
        )
        if candidate is not None:
            print(
                f"Falling back to global default input device [{candidate.index}] {candidate.name}."
            )
            return candidate
    candidates = input_devices(snapshot)
    candidate_text = "\n".join(f"  [{item.index}] {item.name}" for item in candidates)
    raise DeviceSelectionError(
        "no usable default input device was reported\nAvailable input devices:\n"
        + (candidate_text or "  none")
    )


def _dispatch(backend: AudioProbeBackend, options: _Options) -> int:
    snapshot = backend.snapshot()
    if options.command == "devices":
        _print_inventory(snapshot)
        return 0
    if options.command == "report":
        _print_inventory(snapshot)
        device = (
            resolve_input_device(snapshot, options.device)
            if options.device is not None
            else _default_report_device(snapshot)
        )
        results = _run_checks(
            backend,
            device,
            options.sample_rate_hz,
            options.channels,
            _mode(options),
        )
        if not any(result.supported for result in results):
            return 5
        preferred_rate = options.sample_rate_hz or device.default_sample_rate_hz
        preferred_channels = options.channels or 1
        supported = tuple(result for result in results if result.supported)
        selected = next(
            (
                result
                for result in supported
                if result.sample_rate_hz == preferred_rate and result.channels == preferred_channels
            ),
            supported[0],
        )
        if (selected.sample_rate_hz, selected.channels) != (
            preferred_rate,
            preferred_channels,
        ):
            print(
                f"Capture fallback: {preferred_rate:g} Hz × {preferred_channels} ch is "
                f"unsupported; using {selected.sample_rate_hz:g} Hz × "
                f"{selected.channels} ch."
            )
        return _capture(
            backend,
            device,
            options,
            sample_rate_override=selected.sample_rate_hz,
            channels_override=selected.channels,
        )
    if options.device is None:
        raise DeviceSelectionError("this command requires an input device selector")
    device = resolve_input_device(snapshot, options.device)
    if options.command == "check":
        results = _run_checks(
            backend,
            device,
            options.sample_rate_hz,
            options.channels,
            _mode(options),
        )
        return 0 if any(result.supported for result in results) else 5
    if options.command == "capture":
        return _capture(backend, device, options)
    raise RuntimeError(f"unhandled command {options.command}")


def main(
    argv: list[str] | None = None,
    backend_factory: BackendFactory = SoundDeviceBackend,
) -> int:
    """Run the experimental developer CLI and return its documented exit code."""

    try:
        namespace = _parser().parse_args(argv)
    except SystemExit as error:
        return int(error.code) if isinstance(error.code, int) else 2
    try:
        return _dispatch(backend_factory(), _options(namespace))
    except BackendUnavailableError as error:
        print(f"Backend unavailable: {error}", file=sys.stderr)
        return 3
    except DeviceSelectionError as error:
        print(f"Device selection error: {error}", file=sys.stderr)
        return 4
    except UnsupportedConfigurationError as error:
        print(f"Unsupported configuration: {error}", file=sys.stderr)
        return 5
    except RuntimeError as error:
        print(f"Probe failure: {error}", file=sys.stderr)
        return 6
    except KeyboardInterrupt:
        print("Probe interrupted.", file=sys.stderr)
        return 130
