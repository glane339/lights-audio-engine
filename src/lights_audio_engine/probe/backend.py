"""Lazy sounddevice adapter for the experimental hardware probe."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, cast

import numpy as np

from lights_audio_engine.probe.diagnostics import CaptureAccumulator, find_wasapi_host_api
from lights_audio_engine.probe.models import (
    BackendUnavailableError,
    CaptureDiagnostics,
    CaptureMode,
    CaptureTermination,
    CompatibilityResult,
    DeviceInfo,
    HostApiInfo,
    ProbeSnapshot,
    UnsupportedConfigurationError,
)

ModuleLoader = Callable[[], object]


class AudioProbeBackend(Protocol):
    """Narrow hardware boundary used by the developer CLI and local fakes."""

    def snapshot(self) -> ProbeSnapshot: ...

    def check_input_settings(
        self,
        device_index: int,
        sample_rate_hz: float,
        channels: int,
        mode: CaptureMode,
    ) -> CompatibilityResult: ...

    def capture(
        self,
        device_index: int,
        duration_seconds: float,
        sample_rate_hz: float,
        channels: int,
        mode: CaptureMode,
    ) -> CaptureDiagnostics: ...


def _load_sounddevice() -> object:
    return importlib.import_module("sounddevice")


def _call(target: object, name: str, *args: object, **kwargs: object) -> object:
    function = getattr(target, name, None)
    if not callable(function):
        raise RuntimeError(f"audio backend has no callable {name}")
    return function(*args, **kwargs)


def _attribute(target: object, name: str) -> object:
    return getattr(target, name)


def _exception_type(module: object) -> type[BaseException] | None:
    candidate = getattr(module, "PortAudioError", None)
    if isinstance(candidate, type) and issubclass(candidate, BaseException):
        return candidate
    return None


def _is_portaudio_error(module: object, error: BaseException) -> bool:
    error_type = _exception_type(module)
    return error_type is not None and isinstance(error, error_type)


def _is_capture_error(module: object, error: BaseException) -> bool:
    return _is_portaudio_error(module, error) or isinstance(
        error, (OSError, RuntimeError, ValueError)
    )


def _number(record: Mapping[str, object], key: str) -> float:
    value = record[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"audio backend field {key} is not numeric")
    return float(value)


def _integer(record: Mapping[str, object], key: str) -> int:
    value = record[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"audio backend field {key} is not an integer")
    return value


def _optional_index(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("audio backend device index is not an integer")
    return None if value < 0 else value


def _required_index(value: object) -> int:
    index = _optional_index(value)
    if index is None:
        raise ValueError("audio backend device list contains a negative index")
    return index


class SoundDeviceBackend:
    """Blocking sounddevice/PortAudio diagnostics behind a lazy adapter."""

    def __init__(self, module_loader: ModuleLoader = _load_sounddevice) -> None:
        self._module_loader = module_loader
        self._module: object | None = None

    def _sounddevice(self) -> object:
        if self._module is not None:
            return self._module
        try:
            module = self._module_loader()
        except Exception as error:
            raise BackendUnavailableError(
                "sounddevice/PortAudio is unavailable. Install the optional backend with "
                '`uv sync --extra probe` or `pip install -e ".[probe]"`. '
                f"Backend detail: {error}"
            ) from error
        self._module = module
        return module

    def snapshot(self) -> ProbeSnapshot:
        """Enumerate host APIs and devices into backend-independent models."""

        module = self._sounddevice()
        try:
            raw_host_apis = cast(Sequence[Mapping[str, object]], _call(module, "query_hostapis"))
            raw_devices = cast(Sequence[Mapping[str, object]], _call(module, "query_devices"))
            version_result = cast(Sequence[object], _call(module, "get_portaudio_version"))
            default_object = _attribute(module, "default")
            default_devices = cast(Sequence[object], _attribute(default_object, "device"))

            host_apis = tuple(
                HostApiInfo(
                    index=index,
                    name=str(record["name"]),
                    device_indexes=tuple(
                        _required_index(value)
                        for value in cast(Sequence[object], record["devices"])
                    ),
                    default_input_device_index=_optional_index(record["default_input_device"]),
                    default_output_device_index=_optional_index(record["default_output_device"]),
                )
                for index, record in enumerate(raw_host_apis)
            )
            devices = tuple(
                DeviceInfo(
                    index=index,
                    name=str(record["name"]),
                    host_api_index=_integer(record, "hostapi"),
                    max_input_channels=_integer(record, "max_input_channels"),
                    max_output_channels=_integer(record, "max_output_channels"),
                    default_sample_rate_hz=_number(record, "default_samplerate"),
                    default_low_input_latency_seconds=_number(record, "default_low_input_latency"),
                    default_high_input_latency_seconds=_number(
                        record, "default_high_input_latency"
                    ),
                )
                for index, record in enumerate(raw_devices)
            )
            device_indexes = tuple(_optional_index(value) for value in default_devices)
            if len(device_indexes) != 2 or len(version_result) < 2:
                raise RuntimeError("audio backend returned malformed defaults or version data")
            return ProbeSnapshot(
                sounddevice_version=str(getattr(module, "__version__", "unknown")),
                portaudio_version_text=str(version_result[1]),
                host_apis=host_apis,
                devices=devices,
                default_input_device_index=device_indexes[0],
                default_output_device_index=device_indexes[1],
            )
        except Exception as error:
            if _is_portaudio_error(module, error) or isinstance(error, OSError):
                raise BackendUnavailableError(
                    f"sounddevice loaded, but PortAudio initialization failed: {error}"
                ) from error
            raise

    def _extra_settings(
        self, module: object, device_index: int, mode: CaptureMode
    ) -> object | None:
        if mode == "shared":
            return None
        snapshot = self.snapshot()
        device = next((item for item in snapshot.devices if item.index == device_index), None)
        wasapi = find_wasapi_host_api(snapshot)
        if device is None or wasapi is None or device.host_api_index != wasapi.index:
            raise UnsupportedConfigurationError(
                "exclusive mode is available only for a Windows WASAPI device"
            )
        return _call(
            module,
            "WasapiSettings",
            exclusive=True,
            explicit_sample_format=True,
        )

    def check_input_settings(
        self,
        device_index: int,
        sample_rate_hz: float,
        channels: int,
        mode: CaptureMode,
    ) -> CompatibilityResult:
        """Ask PortAudio whether one configuration is supported without opening a stream."""

        module = self._sounddevice()
        extra_settings = self._extra_settings(module, device_index, mode)
        try:
            _call(
                module,
                "check_input_settings",
                device=device_index,
                samplerate=sample_rate_hz,
                channels=channels,
                dtype="float32",
                extra_settings=extra_settings,
            )
        except Exception as error:
            if _is_portaudio_error(module, error):
                return CompatibilityResult(
                    device_index,
                    sample_rate_hz,
                    channels,
                    mode,
                    False,
                    str(error),
                )
            if isinstance(error, OSError):
                raise BackendUnavailableError(
                    f"PortAudio settings check failed: {error}"
                ) from error
            raise
        return CompatibilityResult(device_index, sample_rate_hz, channels, mode, True, None)

    def capture(
        self,
        device_index: int,
        duration_seconds: float,
        sample_rate_hz: float,
        channels: int,
        mode: CaptureMode,
    ) -> CaptureDiagnostics:
        """Capture through bounded blocking reads and return only incremental diagnostics."""

        module = self._sounddevice()
        extra_settings = self._extra_settings(module, device_index, mode)
        requested_frame_count = max(1, round(duration_seconds * sample_rate_hz))
        accumulator = CaptureAccumulator(channels)
        stream: object | None = None
        started = False
        frames_received = 0
        actual_sample_rate_hz: float | None = None
        input_latency_seconds: float | None = None
        termination: CaptureTermination = "completed"
        error_message: str | None = None

        try:
            stream = _call(
                module,
                "InputStream",
                device=device_index,
                samplerate=sample_rate_hz,
                channels=channels,
                dtype="float32",
                callback=None,
                extra_settings=extra_settings,
            )
            actual_sample_rate_hz = float(cast(float, _attribute(stream, "samplerate")))
            input_latency_seconds = float(cast(float, _attribute(stream, "latency")))
            maximum_read_frames = max(1, int(0.25 * actual_sample_rate_hz))
            _call(stream, "start")
            started = True
            while frames_received < requested_frame_count:
                requested_read = min(requested_frame_count - frames_received, maximum_read_frames)
                raw_result = cast(Sequence[object], _call(stream, "read", requested_read))
                if len(raw_result) != 2:
                    raise RuntimeError("audio backend returned a malformed read result")
                block = np.asarray(raw_result[0], dtype=np.float32)
                if block.ndim != 2:
                    raise RuntimeError("audio backend returned a non-matrix capture block")
                overflowed = bool(raw_result[1])
                accumulator.add_block(block, overflowed=overflowed)
                received_now = int(block.shape[0])
                frames_received += received_now
                if received_now != requested_read:
                    termination = "failed"
                    comparison = "fewer" if received_now < requested_read else "more"
                    error_message = (
                        f"stream returned {comparison} frames than requested "
                        f"({received_now} of {requested_read})"
                    )
                    break
        except KeyboardInterrupt:
            termination = "interrupted"
        except Exception as error:
            if _is_capture_error(module, error):
                termination = "failed"
                error_message = str(error)
            else:
                raise
        finally:
            if stream is not None:
                if started:
                    try:
                        _call(stream, "stop")
                    except Exception as error:
                        if not _is_capture_error(module, error):
                            raise
                        if termination != "interrupted":
                            termination = "failed"
                        error_message = _append_error(error_message, str(error))
                try:
                    _call(stream, "close")
                except Exception as error:
                    if not _is_capture_error(module, error):
                        raise
                    if termination != "interrupted":
                        termination = "failed"
                    error_message = _append_error(error_message, str(error))

        if termination == "completed" and frames_received != requested_frame_count:
            termination = "failed"
            error_message = (
                f"stream returned fewer frames than requested "
                f"({frames_received} of {requested_frame_count})"
            )
        return accumulator.finalize(
            device_index=device_index,
            mode=mode,
            requested_duration_seconds=duration_seconds,
            requested_sample_rate_hz=sample_rate_hz,
            requested_frame_count=requested_frame_count,
            actual_sample_rate_hz=actual_sample_rate_hz,
            reported_input_latency_seconds=input_latency_seconds,
            termination=termination,
            error_message=error_message,
        )


def _append_error(existing: str | None, new: str) -> str:
    return new if existing is None else f"{existing}; cleanup failure: {new}"
