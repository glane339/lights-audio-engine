from dataclasses import FrozenInstanceError, fields
from typing import get_args

import pytest


def test_probe_models_are_frozen_slotted_and_keep_expected_fields() -> None:
    from lights_audio_engine.probe.models import DeviceInfo, HostApiInfo, ProbeSnapshot

    host_api = HostApiInfo(
        index=1,
        name="Windows WASAPI",
        device_indexes=(3,),
        default_input_device_index=3,
        default_output_device_index=None,
    )
    device = DeviceInfo(
        index=3,
        name="Microphone Array",
        host_api_index=1,
        max_input_channels=2,
        max_output_channels=0,
        default_sample_rate_hz=48_000.0,
        default_low_input_latency_seconds=0.01,
        default_high_input_latency_seconds=0.1,
    )
    snapshot = ProbeSnapshot(
        sounddevice_version="0.5.5",
        portaudio_version_text="PortAudio V19",
        host_apis=(host_api,),
        devices=(device,),
        default_input_device_index=3,
        default_output_device_index=None,
    )

    assert snapshot.devices == (device,)
    assert not hasattr(device, "__dict__")
    assert [field.name for field in fields(DeviceInfo)] == [
        "index",
        "name",
        "host_api_index",
        "max_input_channels",
        "max_output_channels",
        "default_sample_rate_hz",
        "default_low_input_latency_seconds",
        "default_high_input_latency_seconds",
    ]
    with pytest.raises(FrozenInstanceError):
        device.name = "changed"  # type: ignore[misc]


def test_probe_model_literal_contracts_and_error_hierarchy() -> None:
    from lights_audio_engine.probe.models import (
        BackendUnavailableError,
        CaptureMode,
        CaptureTermination,
        DeviceSelectionError,
        ProbeError,
        UnsupportedConfigurationError,
    )

    assert get_args(CaptureMode) == ("shared", "exclusive")
    assert get_args(CaptureTermination) == ("completed", "interrupted", "failed")
    assert issubclass(BackendUnavailableError, ProbeError)
    assert issubclass(DeviceSelectionError, ProbeError)
    assert issubclass(UnsupportedConfigurationError, ProbeError)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("index", -1),
        ("max_input_channels", -1),
        ("default_sample_rate_hz", 0.0),
        ("default_low_input_latency_seconds", -0.01),
    ],
)
def test_device_info_rejects_invalid_hardware_metadata(field_name: str, value: object) -> None:
    from lights_audio_engine.probe.models import DeviceInfo

    values: dict[str, object] = {
        "index": 0,
        "name": "Input",
        "host_api_index": 0,
        "max_input_channels": 1,
        "max_output_channels": 0,
        "default_sample_rate_hz": 48_000.0,
        "default_low_input_latency_seconds": 0.01,
        "default_high_input_latency_seconds": 0.1,
    }
    values[field_name] = value

    with pytest.raises((TypeError, ValueError), match=field_name):
        DeviceInfo(**values)  # type: ignore[arg-type]


def test_capture_diagnostics_accepts_completed_capture_with_quality_findings() -> None:
    from lights_audio_engine.probe.models import CaptureDiagnostics

    diagnostics = CaptureDiagnostics(
        device_index=3,
        mode="shared",
        requested_duration_seconds=1.0,
        requested_sample_rate_hz=48_000.0,
        requested_channels=2,
        requested_frame_count=48_000,
        actual_sample_rate_hz=48_000.0,
        reported_input_latency_seconds=0.01,
        frames_received=48_000,
        captured_duration_seconds=1.0,
        per_channel_rms=(0.0, 0.2),
        per_channel_peak=(0.0, 0.9),
        per_channel_all_zero=(True, False),
        per_channel_near_full_scale_count=(0, 1),
        per_channel_non_finite_count=(1, 0),
        overflowed_read_count=0,
        termination="completed",
        error_message=None,
    )

    assert diagnostics.termination == "completed"
    assert diagnostics.per_channel_non_finite_count == (1, 0)
