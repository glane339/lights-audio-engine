"""Blocking live-audio source backed by the optional sounddevice runtime."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator, Sequence
from typing import cast

from lights_audio_engine.capture.assembler import FrameAssembler
from lights_audio_engine.capture.channels import AverageChannels, MonoPassthrough
from lights_audio_engine.capture.models import (
    CaptureConfig,
    CaptureItem,
    Discontinuity,
    DiscontinuityReason,
)

ModuleLoader = Callable[[], object]
DiscontinuityObserver = Callable[[Discontinuity], object]


def _load_sounddevice() -> object:
    return importlib.import_module("sounddevice")


def _call(target: object, name: str, *args: object, **kwargs: object) -> object:
    function = getattr(target, name, None)
    if not callable(function):
        raise RuntimeError(f"audio backend has no callable {name}")
    return function(*args, **kwargs)


def _exception_type(module: object) -> type[BaseException] | None:
    candidate = getattr(module, "PortAudioError", None)
    if isinstance(candidate, type) and issubclass(candidate, BaseException):
        return candidate
    return None


def _is_capture_error(module: object, error: BaseException) -> bool:
    portaudio_error = _exception_type(module)
    return (portaudio_error is not None and isinstance(error, portaudio_error)) or isinstance(
        error, (OSError, RuntimeError, ValueError)
    )


def _positive_int(name: str, value: object, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _device_selector(value: object) -> int | str:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("device_index must be a non-negative integer or non-empty string")
        return value
    return _positive_int("device_index", value, allow_zero=True)


class SoundDeviceAudioSource:
    """Synchronous ``AudioSource`` backed by blocking sounddevice reads.

    The default 960-frame read is 20 ms at 48 kHz and supplies one complete
    analysis window to the current detector. It is an alignment choice, not a
    measured physical-event-to-Python or end-to-end latency claim.
    """

    def __init__(
        self,
        device_index: int | str,
        *,
        sample_rate_hz: int = 48_000,
        channels: int = 1,
        block_size_frames: int = 960,
        module_loader: ModuleLoader = _load_sounddevice,
        assembler: FrameAssembler | None = None,
        on_discontinuity: DiscontinuityObserver | None = None,
    ) -> None:
        self._device_index = _device_selector(device_index)
        self._sample_rate_hz = _positive_int("sample_rate_hz", sample_rate_hz)
        self._channels = _positive_int("channels", channels)
        self._block_size_frames = _positive_int("block_size_frames", block_size_frames)
        if not callable(module_loader):
            raise TypeError("module_loader must be callable")
        if on_discontinuity is not None and not callable(on_discontinuity):
            raise TypeError("on_discontinuity must be callable or None")
        channel_policy = MonoPassthrough() if channels == 1 else AverageChannels()
        self._assembler = assembler or FrameAssembler(
            CaptureConfig(sample_rate_hz=sample_rate_hz, channel_policy=channel_policy)
        )
        self._module_loader = module_loader
        self._on_discontinuity = on_discontinuity
        self._module: object | None = None
        self._stream: object | None = None
        self._started = False
        self._closed = False

    @property
    def sample_rate_hz(self) -> int:
        """Configured sample rate of yielded frames."""

        return self._sample_rate_hz

    def stream(self) -> Iterator[CaptureItem]:
        """Yield assembled frames and explicit capture discontinuities."""

        if self._closed:
            return
        module = self._module_loader()
        self._module = module
        try:
            stream = _call(
                module,
                "InputStream",
                device=self._device_index,
                samplerate=self._sample_rate_hz,
                channels=self._channels,
                dtype="float32",
                callback=None,
            )
            self._stream = stream
            _call(stream, "start")
            self._started = True
            while not self._closed:
                raw_result = cast(
                    Sequence[object],
                    _call(stream, "read", self._block_size_frames),
                )
                if len(raw_result) != 2:
                    raise RuntimeError("audio backend returned a malformed read result")
                item = self._assembler.assemble(
                    raw_result[0],
                    sample_rate_hz=self._sample_rate_hz,
                    overflow=bool(raw_result[1]),
                )
                if item is None:
                    continue
                if isinstance(item, Discontinuity):
                    self._notify(item)
                yield item
        except KeyboardInterrupt:
            raise
        except Exception as error:
            if not _is_capture_error(module, error):
                raise
            if self._closed:
                return
            boundary = self._assembler.mark_discontinuity(
                DiscontinuityReason.DROPPED_BLOCK,
                detail=str(error),
            )
            self._notify(boundary)
            yield boundary

    def close(self) -> None:
        """Stop and close the stream once, if it was opened."""

        if self._closed:
            return
        self._closed = True
        stream = self._stream
        module = self._module
        self._stream = None
        if stream is None or module is None:
            return

        unexpected: BaseException | None = None
        if self._started:
            try:
                _call(stream, "stop")
            except Exception as error:
                if not _is_capture_error(module, error):
                    unexpected = error
        self._started = False
        try:
            _call(stream, "close")
        except Exception as error:
            if not _is_capture_error(module, error) and unexpected is None:
                unexpected = error
        if unexpected is not None:
            raise unexpected

    def _notify(self, boundary: Discontinuity) -> None:
        observer = self._on_discontinuity
        if observer is None:
            return
        try:
            observer(boundary)
        except Exception:
            # Observation must never replace delivery or run_engine reset semantics.
            return
