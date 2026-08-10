import numpy as np
import pytest


def _snapshot():
    from lights_audio_engine.probe.models import DeviceInfo, HostApiInfo, ProbeSnapshot

    wasapi = HostApiInfo(
        index=1,
        name="Windows WASAPI",
        device_indexes=(2, 3, 4),
        default_input_device_index=3,
        default_output_device_index=2,
    )
    mme = HostApiInfo(
        index=0,
        name="MME",
        device_indexes=(0, 1),
        default_input_device_index=0,
        default_output_device_index=1,
    )
    devices = (
        DeviceInfo(0, "USB Microphone", 0, 1, 0, 44_100.0, 0.01, 0.1),
        DeviceInfo(1, "Speakers", 0, 0, 2, 48_000.0, 0.0, 0.0),
        DeviceInfo(2, "Speakers", 1, 0, 2, 48_000.0, 0.0, 0.0),
        DeviceInfo(3, "Microphone Array", 1, 2, 0, 48_000.0, 0.01, 0.1),
        DeviceInfo(4, "USB Microphone", 1, 6, 0, 96_000.0, 0.01, 0.1),
    )
    return ProbeSnapshot("0.5.5", "PortAudio V19", (mme, wasapi), devices, 3, 2)


def test_input_device_filter_and_wasapi_detection() -> None:
    from lights_audio_engine.probe.diagnostics import find_wasapi_host_api, input_devices

    snapshot = _snapshot()

    assert tuple(device.index for device in input_devices(snapshot)) == (0, 3, 4)
    assert find_wasapi_host_api(snapshot) == snapshot.host_apis[1]


def test_wasapi_detection_is_case_insensitive_and_can_be_absent() -> None:
    from lights_audio_engine.probe.diagnostics import find_wasapi_host_api
    from lights_audio_engine.probe.models import HostApiInfo, ProbeSnapshot

    lowercase = HostApiInfo(0, "windows wasapi", (), None, None)
    assert find_wasapi_host_api(ProbeSnapshot("v", "pa", (lowercase,), (), None, None)) == lowercase
    assert find_wasapi_host_api(ProbeSnapshot("v", "pa", (), (), None, None)) is None


@pytest.mark.parametrize(("selector", "expected_index"), [("3", 3), ("array", 3), ("usb", None)])
def test_resolve_input_device_by_index_or_unique_substring(
    selector: str, expected_index: int | None
) -> None:
    from lights_audio_engine.probe.diagnostics import resolve_input_device
    from lights_audio_engine.probe.models import DeviceSelectionError

    if expected_index is None:
        with pytest.raises(DeviceSelectionError, match="ambiguous"):
            resolve_input_device(_snapshot(), selector)
    else:
        assert resolve_input_device(_snapshot(), selector).index == expected_index


@pytest.mark.parametrize(
    ("selector", "message"),
    [("99", "not an input device"), ("2", "output-only"), ("missing", "no input device")],
)
def test_resolve_input_device_reports_stale_output_only_and_missing_selectors(
    selector: str, message: str
) -> None:
    from lights_audio_engine.probe.diagnostics import resolve_input_device
    from lights_audio_engine.probe.models import DeviceSelectionError

    with pytest.raises(DeviceSelectionError, match=message) as exc_info:
        resolve_input_device(_snapshot(), selector)

    assert "[3] Microphone Array" in str(exc_info.value)


def test_reported_device_label_is_descriptive_without_claiming_stability() -> None:
    from lights_audio_engine.probe.diagnostics import reported_device_label

    snapshot = _snapshot()

    assert reported_device_label(snapshot.devices[3], snapshot) == (
        "[3] Microphone Array — Windows WASAPI, 2 in / 0 out, default input"
    )


@pytest.mark.parametrize(
    ("device_index", "sample_rate_hz", "channels", "expected"),
    [
        (0, None, None, ((44_100.0, 1), (48_000.0, 1))),
        (
            3,
            None,
            None,
            ((44_100.0, 1), (44_100.0, 2), (48_000.0, 1), (48_000.0, 2)),
        ),
        (
            4,
            None,
            None,
            (
                (44_100.0, 1),
                (44_100.0, 2),
                (44_100.0, 6),
                (48_000.0, 1),
                (48_000.0, 2),
                (48_000.0, 6),
                (96_000.0, 1),
                (96_000.0, 2),
                (96_000.0, 6),
            ),
        ),
        (4, 32_000.0, None, ((32_000.0, 1), (32_000.0, 2), (32_000.0, 6))),
        (4, None, 4, ((44_100.0, 4), (48_000.0, 4), (96_000.0, 4))),
        (4, 32_000.0, 4, ((32_000.0, 4),)),
    ],
)
def test_compatibility_requests_are_ordered_deduplicated_and_pinnable(
    device_index: int,
    sample_rate_hz: float | None,
    channels: int | None,
    expected: tuple[tuple[float, int], ...],
) -> None:
    from lights_audio_engine.probe.diagnostics import compatibility_requests

    assert (
        compatibility_requests(_snapshot().devices[device_index], sample_rate_hz, channels)
        == expected
    )


def test_capture_accumulator_tracks_multichannel_quality_without_retaining_blocks() -> None:
    from lights_audio_engine.probe.diagnostics import CaptureAccumulator

    accumulator = CaptureAccumulator(channels=2)
    accumulator.add_block(
        np.array([[0.0, 0.5], [0.0, -0.5], [np.nan, 0.999]], dtype=np.float32),
        overflowed=True,
    )
    accumulator.add_block(np.array([[0.0, np.inf]], dtype=np.float32), overflowed=False)
    diagnostics = accumulator.finalize(
        device_index=3,
        mode="shared",
        requested_duration_seconds=4 / 48_000,
        requested_sample_rate_hz=48_000.0,
        requested_frame_count=4,
        actual_sample_rate_hz=48_000.0,
        reported_input_latency_seconds=0.01,
        termination="completed",
        error_message=None,
    )

    assert diagnostics.frames_received == 4
    assert diagnostics.per_channel_rms[0] == 0.0
    assert diagnostics.per_channel_rms[1] == pytest.approx(
        np.sqrt((0.5**2 + 0.5**2 + np.float32(0.999) ** 2) / 3)
    )
    assert diagnostics.per_channel_peak == pytest.approx((0.0, np.float32(0.999)))
    assert diagnostics.per_channel_all_zero == (False, False)
    assert diagnostics.per_channel_near_full_scale_count == (0, 1)
    assert diagnostics.per_channel_non_finite_count == (1, 1)
    assert diagnostics.overflowed_read_count == 1


def test_capture_accumulator_incremental_result_matches_one_shot_result() -> None:
    from lights_audio_engine.probe.diagnostics import CaptureAccumulator

    block = np.array([[0.0, 0.2], [0.4, -0.6], [0.8, 1.0]], dtype=np.float32)
    one_shot = CaptureAccumulator(2)
    one_shot.add_block(block, overflowed=False)
    incremental = CaptureAccumulator(2)
    incremental.add_block(block[:1], overflowed=False)
    incremental.add_block(block[1:], overflowed=False)

    kwargs = {
        "device_index": 3,
        "mode": "shared",
        "requested_duration_seconds": 3 / 48_000,
        "requested_sample_rate_hz": 48_000.0,
        "requested_frame_count": 3,
        "actual_sample_rate_hz": 48_000.0,
        "reported_input_latency_seconds": 0.01,
        "termination": "completed",
        "error_message": None,
    }
    assert one_shot.finalize(**kwargs) == incremental.finalize(**kwargs)  # type: ignore[arg-type]


def test_capture_accumulator_marks_exact_finite_zero_signal() -> None:
    from lights_audio_engine.probe.diagnostics import CaptureAccumulator

    accumulator = CaptureAccumulator(1)
    accumulator.add_block(np.zeros((4, 1), dtype=np.float32), overflowed=False)
    diagnostics = accumulator.finalize(
        device_index=0,
        mode="shared",
        requested_duration_seconds=4 / 48_000,
        requested_sample_rate_hz=48_000.0,
        requested_frame_count=4,
        actual_sample_rate_hz=None,
        reported_input_latency_seconds=None,
        termination="failed",
        error_message="test",
    )

    assert diagnostics.per_channel_all_zero == (True,)
    assert diagnostics.captured_duration_seconds == pytest.approx(4 / 48_000)
