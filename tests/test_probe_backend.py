from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pytest


class FakePortAudioError(Exception):
    pass


@dataclass(slots=True)
class FakeDefault:
    device: tuple[int, int] = (3, 2)


class FakeStream:
    def __init__(
        self,
        *,
        channels: int = 1,
        samplerate: float = 48_000.0,
        short_on_call: int | None = None,
        fail_on_call: int | None = None,
        interrupt_on_call: int | None = None,
        start_failure: bool = False,
        close_failure: bool = False,
        overflow_on_call: int | None = None,
        actual_samplerate: float | None = None,
        malformed_on_call: int | None = None,
    ) -> None:
        self.channels = channels
        self.samplerate = samplerate
        self.latency = 0.012
        self.short_on_call = short_on_call
        self.fail_on_call = fail_on_call
        self.interrupt_on_call = interrupt_on_call
        self.start_failure = start_failure
        self.close_failure = close_failure
        self.overflow_on_call = overflow_on_call
        self.actual_samplerate = actual_samplerate
        self.malformed_on_call = malformed_on_call
        self.read_sizes: list[int] = []
        self.started = False
        self.closed = False

    def start(self) -> None:
        if self.start_failure:
            raise FakePortAudioError("start failed")
        self.started = True

    def read(self, frames: int) -> tuple[np.ndarray, bool]:
        self.read_sizes.append(frames)
        call = len(self.read_sizes)
        if call == self.interrupt_on_call:
            raise KeyboardInterrupt
        if call == self.fail_on_call:
            raise FakePortAudioError("device disconnected")
        if call == self.malformed_on_call:
            return cast(tuple[np.ndarray, bool], (np.zeros((frames, self.channels)),))
        returned_frames = frames - 1 if call == self.short_on_call else frames
        block = np.full((returned_frames, self.channels), 0.25, dtype=np.float32)
        return block, call == self.overflow_on_call

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.closed = True
        if self.close_failure:
            raise FakePortAudioError("close failed")


class FakeSoundDevice:
    PortAudioError = FakePortAudioError
    __version__ = "0.5.5"

    def __init__(self, stream: FakeStream | None = None) -> None:
        self.default = FakeDefault()
        self.stream = stream or FakeStream()
        self.input_stream_failure = False
        self.check_failure = False
        self.snapshot_failure = False
        self.input_stream_kwargs: dict[str, object] | None = None
        self.check_kwargs: dict[str, object] | None = None
        self.wasapi_kwargs: dict[str, object] | None = None

    def get_portaudio_version(self) -> tuple[int, str]:
        return (1_246_720, "PortAudio V19.7.0")

    def query_hostapis(self) -> tuple[dict[str, object], ...]:
        if self.snapshot_failure:
            raise FakePortAudioError("PortAudio not initialized")
        return (
            {
                "name": "MME",
                "devices": (0, 1),
                "default_input_device": 0,
                "default_output_device": 1,
            },
            {
                "name": "Windows WASAPI",
                "devices": (2, 3),
                "default_input_device": 3,
                "default_output_device": 2,
            },
        )

    def query_devices(self) -> tuple[dict[str, object], ...]:
        return (
            {
                "name": "MME Mic",
                "hostapi": 0,
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 44_100.0,
                "default_low_input_latency": 0.01,
                "default_high_input_latency": 0.1,
            },
            {
                "name": "MME Speakers",
                "hostapi": 0,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48_000.0,
                "default_low_input_latency": 0.0,
                "default_high_input_latency": 0.0,
            },
            {
                "name": "WASAPI Speakers",
                "hostapi": 1,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48_000.0,
                "default_low_input_latency": 0.0,
                "default_high_input_latency": 0.0,
            },
            {
                "name": "WASAPI Mic",
                "hostapi": 1,
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48_000.0,
                "default_low_input_latency": 0.01,
                "default_high_input_latency": 0.1,
            },
        )

    def check_input_settings(self, **kwargs: object) -> None:
        self.check_kwargs = kwargs
        if self.check_failure:
            raise FakePortAudioError("Invalid sample rate [PaErrorCode -9997]")

    def WasapiSettings(self, **kwargs: object) -> object:
        self.wasapi_kwargs = kwargs
        return ("wasapi", kwargs)

    def InputStream(self, **kwargs: object) -> FakeStream:
        self.input_stream_kwargs = kwargs
        if self.input_stream_failure:
            raise FakePortAudioError("open failed")
        self.stream.channels = cast(int, kwargs["channels"])
        self.stream.samplerate = self.stream.actual_samplerate or cast(float, kwargs["samplerate"])
        return self.stream


def test_backend_import_and_construction_are_lazy() -> None:
    from lights_audio_engine.probe.backend import SoundDeviceBackend

    loader_calls = 0

    def loader() -> object:
        nonlocal loader_calls
        loader_calls += 1
        return FakeSoundDevice()

    backend = SoundDeviceBackend(loader)

    assert backend is not None
    assert loader_calls == 0


@pytest.mark.parametrize("failure", [ImportError("missing"), OSError("DLL missing")])
def test_backend_wraps_optional_dependency_load_failures(failure: Exception) -> None:
    from lights_audio_engine.probe.backend import SoundDeviceBackend
    from lights_audio_engine.probe.models import BackendUnavailableError

    def loader() -> object:
        raise failure

    with pytest.raises(BackendUnavailableError, match=r"\[probe\]"):
        SoundDeviceBackend(loader).snapshot()


def test_snapshot_wraps_portaudio_initialization_failure() -> None:
    from lights_audio_engine.probe.backend import SoundDeviceBackend
    from lights_audio_engine.probe.models import BackendUnavailableError

    fake = FakeSoundDevice()
    fake.snapshot_failure = True

    with pytest.raises(BackendUnavailableError, match="PortAudio not initialized"):
        SoundDeviceBackend(lambda: fake).snapshot()


def test_snapshot_converts_backend_records_and_minus_one_defaults() -> None:
    from lights_audio_engine.probe.backend import SoundDeviceBackend

    fake = FakeSoundDevice()
    fake.default.device = (-1, -1)

    snapshot = SoundDeviceBackend(lambda: fake).snapshot()

    assert snapshot.sounddevice_version == "0.5.5"
    assert snapshot.portaudio_version_text == "PortAudio V19.7.0"
    assert snapshot.default_input_device_index is None
    assert snapshot.default_output_device_index is None
    assert snapshot.host_apis[1].default_input_device_index == 3
    assert snapshot.devices[3].name == "WASAPI Mic"
    assert snapshot.devices[3].host_api_index == 1


def test_check_input_settings_maps_success_and_portaudio_failure() -> None:
    from lights_audio_engine.probe.backend import SoundDeviceBackend

    fake = FakeSoundDevice()
    backend = SoundDeviceBackend(lambda: fake)

    supported = backend.check_input_settings(3, 48_000.0, 2, "shared")
    fake.check_failure = True
    unsupported = backend.check_input_settings(3, 12_345.0, 2, "shared")

    assert supported.supported
    assert supported.portaudio_message is None
    assert unsupported.supported is False
    assert "Invalid sample rate" in (unsupported.portaudio_message or "")
    assert fake.check_kwargs == {
        "device": 3,
        "samplerate": 12_345.0,
        "channels": 2,
        "dtype": "float32",
        "extra_settings": None,
    }


def test_exclusive_settings_are_wasapi_only_and_never_auto_convert() -> None:
    from lights_audio_engine.probe.backend import SoundDeviceBackend
    from lights_audio_engine.probe.models import UnsupportedConfigurationError

    fake = FakeSoundDevice()
    backend = SoundDeviceBackend(lambda: fake)

    result = backend.check_input_settings(3, 48_000.0, 2, "exclusive")

    assert result.supported
    assert fake.wasapi_kwargs == {"exclusive": True, "explicit_sample_format": True}
    assert fake.check_kwargs is not None
    assert fake.check_kwargs["extra_settings"] == ("wasapi", fake.wasapi_kwargs)
    with pytest.raises(UnsupportedConfigurationError, match="WASAPI"):
        backend.check_input_settings(0, 44_100.0, 1, "exclusive")


def test_capture_uses_bounded_blocking_reads_and_aggregates_overflows() -> None:
    from lights_audio_engine.probe.backend import SoundDeviceBackend

    stream = FakeStream(overflow_on_call=2)
    fake = FakeSoundDevice(stream)

    result = SoundDeviceBackend(lambda: fake).capture(3, 0.6, 48_000.0, 2, "shared")

    assert result.termination == "completed"
    assert result.frames_received == 28_800
    assert result.actual_sample_rate_hz == 48_000.0
    assert result.reported_input_latency_seconds == 0.012
    assert result.overflowed_read_count == 1
    assert stream.read_sizes == [12_000, 12_000, 4_800]
    assert max(stream.read_sizes) <= 0.25 * 48_000
    assert fake.input_stream_kwargs == {
        "device": 3,
        "samplerate": 48_000.0,
        "channels": 2,
        "dtype": "float32",
        "callback": None,
        "extra_settings": None,
    }
    assert stream.closed


def test_capture_sizes_reads_from_the_opened_stream_sample_rate() -> None:
    from lights_audio_engine.probe.backend import SoundDeviceBackend

    stream = FakeStream(actual_samplerate=24_000.0)

    result = SoundDeviceBackend(lambda: FakeSoundDevice(stream)).capture(
        3, 0.5, 48_000.0, 1, "shared"
    )

    assert result.actual_sample_rate_hz == 24_000.0
    assert max(stream.read_sizes) <= 0.25 * 24_000


def test_malformed_stream_read_returns_failed_diagnostics() -> None:
    from lights_audio_engine.probe.backend import SoundDeviceBackend

    result = SoundDeviceBackend(lambda: FakeSoundDevice(FakeStream(malformed_on_call=1))).capture(
        3, 0.5, 48_000.0, 1, "shared"
    )

    assert result.termination == "failed"
    assert "malformed read result" in (result.error_message or "")


@pytest.mark.parametrize(
    ("stream", "expected_termination", "message"),
    [
        (FakeStream(short_on_call=1), "failed", "fewer frames"),
        (FakeStream(fail_on_call=2), "failed", "device disconnected"),
        (FakeStream(interrupt_on_call=2), "interrupted", None),
        (FakeStream(start_failure=True), "failed", "start failed"),
        (FakeStream(close_failure=True), "failed", "close failed"),
    ],
)
def test_capture_preserves_partial_diagnostics_for_stream_endings(
    stream: FakeStream, expected_termination: str, message: str | None
) -> None:
    from lights_audio_engine.probe.backend import SoundDeviceBackend

    result = SoundDeviceBackend(lambda: FakeSoundDevice(stream)).capture(
        3, 0.5, 48_000.0, 1, "shared"
    )

    assert result.termination == expected_termination
    assert result.frames_received >= 0
    if message is not None:
        assert message in (result.error_message or "")


def test_capture_open_failure_returns_zero_frame_failed_diagnostics() -> None:
    from lights_audio_engine.probe.backend import SoundDeviceBackend

    fake = FakeSoundDevice()
    fake.input_stream_failure = True

    result = SoundDeviceBackend(lambda: fake).capture(3, 1.0, 48_000.0, 1, "shared")

    assert result.termination == "failed"
    assert result.frames_received == 0
    assert result.actual_sample_rate_hz is None
    assert result.per_channel_all_zero == (False,)
    assert "open failed" in (result.error_message or "")
