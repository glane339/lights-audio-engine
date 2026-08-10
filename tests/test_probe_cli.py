from __future__ import annotations

import sys

import pytest

from lights_audio_engine.probe.models import (
    BackendUnavailableError,
    CaptureDiagnostics,
    CaptureMode,
    CompatibilityResult,
    DeviceInfo,
    HostApiInfo,
    ProbeSnapshot,
    UnsupportedConfigurationError,
)


def _snapshot(*, empty: bool = False) -> ProbeSnapshot:
    if empty:
        return ProbeSnapshot("0.5.5", "PortAudio V19", (), (), None, None)
    host_apis = (
        HostApiInfo(0, "MME", (0,), 0, None),
        HostApiInfo(1, "Windows WASAPI", (2, 3), 3, 2),
    )
    devices = (
        DeviceInfo(0, "USB Microphone", 0, 1, 0, 44_100.0, 0.01, 0.1),
        DeviceInfo(2, "Speakers", 1, 0, 2, 48_000.0, 0.0, 0.0),
        DeviceInfo(3, "Microphone Array", 1, 2, 0, 48_000.0, 0.01, 0.1),
    )
    return ProbeSnapshot("0.5.5", "PortAudio V19", host_apis, devices, 0, 2)


def _diagnostics(
    *,
    termination: str = "completed",
    all_zero: bool = False,
    near_full_scale_count: int = 0,
    non_finite_count: int = 0,
    overflowed_read_count: int = 0,
) -> CaptureDiagnostics:
    return CaptureDiagnostics(
        device_index=3,
        mode="shared",
        requested_duration_seconds=1.0,
        requested_sample_rate_hz=48_000.0,
        requested_channels=1,
        requested_frame_count=48_000,
        actual_sample_rate_hz=48_000.0,
        reported_input_latency_seconds=0.01,
        frames_received=24_000 if termination != "completed" else 48_000,
        captured_duration_seconds=0.5 if termination != "completed" else 1.0,
        per_channel_rms=(0.0 if all_zero else 0.2,),
        per_channel_peak=(0.0 if all_zero else 0.9,),
        per_channel_all_zero=(all_zero,),
        per_channel_near_full_scale_count=(near_full_scale_count,),
        per_channel_non_finite_count=(non_finite_count,),
        overflowed_read_count=overflowed_read_count,
        termination=termination,  # type: ignore[arg-type]
        error_message="stream failed" if termination == "failed" else None,
    )


class FakeBackend:
    def __init__(
        self,
        *,
        snapshot: ProbeSnapshot | None = None,
        capture_result: CaptureDiagnostics | None = None,
    ) -> None:
        self.snapshot_result = snapshot or _snapshot()
        self.capture_result = capture_result or _diagnostics()
        self.supported_pairs: set[tuple[float, int]] | None = None
        self.snapshot_error: Exception | None = None
        self.exclusive_error = False
        self.capture_error: RuntimeError | None = None
        self.check_calls: list[tuple[int, float, int, CaptureMode]] = []
        self.capture_calls: list[tuple[int, float, float, int, CaptureMode]] = []

    def snapshot(self) -> ProbeSnapshot:
        if self.snapshot_error is not None:
            raise self.snapshot_error
        return self.snapshot_result

    def check_input_settings(
        self,
        device_index: int,
        sample_rate_hz: float,
        channels: int,
        mode: CaptureMode,
    ) -> CompatibilityResult:
        self.check_calls.append((device_index, sample_rate_hz, channels, mode))
        if self.exclusive_error and mode == "exclusive":
            raise UnsupportedConfigurationError("exclusive mode requires WASAPI")
        supported = (
            self.supported_pairs is None or (sample_rate_hz, channels) in self.supported_pairs
        )
        return CompatibilityResult(
            device_index,
            sample_rate_hz,
            channels,
            mode,
            supported,
            None if supported else "Invalid sample rate [PaErrorCode -9997]",
        )

    def capture(
        self,
        device_index: int,
        duration_seconds: float,
        sample_rate_hz: float,
        channels: int,
        mode: CaptureMode,
    ) -> CaptureDiagnostics:
        self.capture_calls.append((device_index, duration_seconds, sample_rate_hz, channels, mode))
        if self.capture_error is not None:
            raise self.capture_error
        return self.capture_result


def _main(arguments: list[str], backend: FakeBackend) -> int:
    from lights_audio_engine.probe.cli import main

    return main(arguments, backend_factory=lambda: backend)


def test_importing_probe_cli_does_not_import_sounddevice() -> None:
    assert "sounddevice" not in sys.modules

    import lights_audio_engine.probe.cli as probe_cli

    assert probe_cli.main is not None
    assert "sounddevice" not in sys.modules


def test_devices_prints_full_inventory_and_empty_inventory_is_success(
    capsys: pytest.CaptureFixture[str],
) -> None:
    populated = FakeBackend()
    empty = FakeBackend(snapshot=_snapshot(empty=True))

    assert _main(["devices"], populated) == 0
    populated_output = capsys.readouterr().out
    assert "sounddevice: 0.5.5" in populated_output
    assert "Windows WASAPI" in populated_output
    assert "[3] Microphone Array" in populated_output
    assert _main(["devices"], empty) == 0
    assert "Devices: none" in capsys.readouterr().out


@pytest.mark.parametrize(
    "arguments",
    [
        ["capture", "--device", "3", "--duration", "-1"],
        ["capture", "--device", "3", "--sample-rate", "0"],
        ["capture", "--device", "3", "--channels", "0"],
        ["unknown"],
    ],
)
def test_argparse_errors_return_exit_2(arguments: list[str]) -> None:
    assert _main(arguments, FakeBackend()) == 2


def test_backend_unavailable_returns_exit_3(capsys: pytest.CaptureFixture[str]) -> None:
    backend = FakeBackend()
    backend.snapshot_error = BackendUnavailableError("install [probe]")

    assert _main(["devices"], backend) == 3
    assert "install [probe]" in capsys.readouterr().err


@pytest.mark.parametrize("selector", ["99", "2", "missing", "microphone"])
def test_bad_or_ambiguous_device_selection_returns_exit_4(
    selector: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _main(["check", "--device", selector], FakeBackend()) == 4
    assert "Available input devices" in capsys.readouterr().err


def test_explicit_unsupported_capture_returns_exit_5_without_opening_stream(
    capsys: pytest.CaptureFixture[str],
) -> None:
    backend = FakeBackend()
    backend.supported_pairs = set()

    result = _main(
        ["capture", "--device", "3", "--sample-rate", "12345", "--channels", "2"],
        backend,
    )

    assert result == 5
    assert backend.capture_calls == []
    assert "Invalid sample rate" in capsys.readouterr().out


def test_check_matrix_returns_zero_if_any_combination_is_supported_and_5_if_none_are(
    capsys: pytest.CaptureFixture[str],
) -> None:
    partial = FakeBackend()
    partial.supported_pairs = {(48_000.0, 1)}
    none = FakeBackend()
    none.supported_pairs = set()

    assert _main(["check", "--device", "3"], partial) == 0
    assert "supported" in capsys.readouterr().out
    assert _main(["check", "--device", "3"], none) == 5
    assert "No tested combinations are supported" in capsys.readouterr().out


def test_exclusive_non_wasapi_error_returns_exit_5(capsys: pytest.CaptureFixture[str]) -> None:
    backend = FakeBackend()
    backend.exclusive_error = True

    assert _main(["check", "--device", "0", "--exclusive"], backend) == 5
    assert "requires WASAPI" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("diagnostics", "expected_exit"),
    [
        (_diagnostics(termination="failed"), 6),
        (_diagnostics(termination="interrupted"), 130),
        (_diagnostics(non_finite_count=1), 6),
        (_diagnostics(all_zero=True), 0),
        (_diagnostics(near_full_scale_count=2, overflowed_read_count=1), 0),
    ],
)
def test_capture_maps_diagnostics_to_exit_contract_and_warnings(
    diagnostics: CaptureDiagnostics,
    expected_exit: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    backend = FakeBackend(capture_result=diagnostics)

    assert _main(["capture", "--device", "3", "--duration", "1"], backend) == expected_exit
    output = capsys.readouterr().out
    assert f"termination: {diagnostics.termination}" in output
    if diagnostics.per_channel_all_zero[0]:
        assert "all_samples_zero" in output
    if diagnostics.per_channel_near_full_scale_count[0]:
        assert "not proof of clipping" in output
    if diagnostics.per_channel_non_finite_count[0]:
        assert "untrustworthy" in output
    if diagnostics.overflowed_read_count:
        assert "reads reporting overflow" in output


def test_unexpected_adapter_runtime_failure_returns_exit_6(
    capsys: pytest.CaptureFixture[str],
) -> None:
    backend = FakeBackend()
    backend.capture_error = RuntimeError("malformed backend result")

    assert _main(["capture", "--device", "3"], backend) == 6
    assert "malformed backend result" in capsys.readouterr().err


def test_report_prefers_wasapi_default_and_prints_fallback_reason(
    capsys: pytest.CaptureFixture[str],
) -> None:
    backend = FakeBackend()

    assert _main(["report", "--duration", "1"], backend) == 0
    output = capsys.readouterr().out
    assert "Selected WASAPI default input device [3]" in output
    assert backend.capture_calls == [(3, 1.0, 48_000.0, 1, "shared")]


def test_report_falls_back_to_global_default_when_wasapi_default_is_unavailable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    snapshot = _snapshot()
    host_apis = (
        snapshot.host_apis[0],
        HostApiInfo(1, "Windows WASAPI", (2, 3), None, 2),
    )
    backend = FakeBackend(
        snapshot=ProbeSnapshot(
            snapshot.sounddevice_version,
            snapshot.portaudio_version_text,
            host_apis,
            snapshot.devices,
            0,
            2,
        )
    )

    assert _main(["report", "--duration", "1"], backend) == 0
    assert "Falling back to global default input device [0]" in capsys.readouterr().out


def test_report_captures_a_supported_matrix_fallback_when_default_mono_is_unsupported(
    capsys: pytest.CaptureFixture[str],
) -> None:
    backend = FakeBackend()
    backend.supported_pairs = {(48_000.0, 2)}

    assert _main(["report", "--duration", "1"], backend) == 0
    assert backend.capture_calls == [(3, 1.0, 48_000.0, 2, "shared")]
    assert "Capture fallback" in capsys.readouterr().out
