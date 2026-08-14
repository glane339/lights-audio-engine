from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest

from lights_audio_engine.capture import CaptureItem, Discontinuity, DiscontinuityReason
from lights_audio_engine.diagnostic.cli import main
from lights_audio_engine.models import AudioFrame
from lights_audio_engine.probe.models import DeviceInfo, HostApiInfo, ProbeSnapshot


def _snapshot() -> ProbeSnapshot:
    return ProbeSnapshot(
        "0.5.5",
        "PortAudio V19",
        (HostApiInfo(1, "Windows WASAPI", (3, 7), 3, None),),
        (
            DeviceInfo(3, "Microphone Array", 1, 2, 0, 48_000.0, 0.01, 0.1),
            DeviceInfo(7, "USB Microphone", 1, 1, 0, 48_000.0, 0.01, 0.1),
        ),
        3,
        None,
    )


class FakeInventoryBackend:
    def snapshot(self) -> ProbeSnapshot:
        return _snapshot()


class FakeSource:
    def __init__(
        self,
        items: tuple[CaptureItem, ...],
        *,
        on_discontinuity: Callable[[Discontinuity], object] | None,
        interrupt_after_items: bool = False,
        close_error: bool = False,
    ) -> None:
        self.items = items
        self.on_discontinuity = on_discontinuity
        self.interrupt_after_items = interrupt_after_items
        self.close_error = close_error
        self.close_calls = 0

    @property
    def sample_rate_hz(self) -> int:
        return 48_000

    def stream(self) -> Iterator[CaptureItem]:
        for item in self.items:
            if isinstance(item, Discontinuity) and self.on_discontinuity is not None:
                self.on_discontinuity(item)
            yield item
        if self.interrupt_after_items:
            raise KeyboardInterrupt

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error:
            raise RuntimeError("source close failed")


class RecordingSourceFactory:
    def __init__(
        self,
        items: tuple[CaptureItem, ...] = (),
        *,
        interrupt_after_items: bool = False,
        close_error: bool = False,
    ) -> None:
        self.items = items
        self.interrupt_after_items = interrupt_after_items
        self.close_error = close_error
        self.calls: list[dict[str, object]] = []
        self.source: FakeSource | None = None

    def __call__(
        self,
        device_index: int,
        *,
        sample_rate_hz: int,
        channels: int,
        block_size_frames: int,
        on_discontinuity: Callable[[Discontinuity], object] | None,
    ) -> FakeSource:
        self.calls.append(
            {
                "device_index": device_index,
                "sample_rate_hz": sample_rate_hz,
                "channels": channels,
                "block_size_frames": block_size_frames,
            }
        )
        self.source = FakeSource(
            self.items,
            on_discontinuity=on_discontinuity,
            interrupt_after_items=self.interrupt_after_items,
            close_error=self.close_error,
        )
        return self.source


@pytest.mark.parametrize(("selector", "expected_index"), [("3", 3), ("USB", 7)])
def test_cli_resolves_current_device_without_hardcoded_index(
    selector: str,
    expected_index: int,
) -> None:
    factory = RecordingSourceFactory()

    exit_code = main(
        ["--device", selector, "--sample-rate", "48000", "--channels", "1"],
        source_factory=factory,
        backend_factory=FakeInventoryBackend,
    )

    assert exit_code == 0
    assert factory.calls == [
        {
            "device_index": expected_index,
            "sample_rate_hz": 48_000,
            "channels": 1,
            "block_size_frames": 960,
        }
    ]
    assert factory.source is not None
    assert factory.source.close_calls == 1


def test_cli_reports_invalid_or_ambiguous_device_selector(
    capsys: pytest.CaptureFixture[str],
) -> None:
    factory = RecordingSourceFactory()

    exit_code = main(
        ["--device", "Microphone"],
        source_factory=factory,
        backend_factory=FakeInventoryBackend,
    )

    captured = capsys.readouterr()
    assert exit_code == 4
    assert "Device selection error:" in captured.err
    assert "ambiguous" in captured.err
    assert factory.calls == []


def _beat_frames() -> tuple[AudioFrame, ...]:
    frames: list[AudioFrame] = []
    for index in range(3):
        samples = np.zeros(24_000, dtype=np.float64)
        samples[:960] = 0.8
        frames.append(AudioFrame(samples, 48_000, index * 0.5))
    return tuple(frames)


def test_cli_prints_beats_changed_bpm_and_discontinuities(
    capsys: pytest.CaptureFixture[str],
) -> None:
    boundary = Discontinuity(DiscontinuityReason.OVERFLOW, 3, "input overflow")
    factory = RecordingSourceFactory((*_beat_frames(), boundary))

    exit_code = main(
        ["--device", "Microphone Array"],
        source_factory=factory,
        backend_factory=FakeInventoryBackend,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert (
        "Listening: [3] Microphone Array, 48000 Hz, 1 channel(s), 960-frame reads" in captured.out
    )
    assert captured.out.count("Beat ") == 3
    assert "Beat 0: t=0.000000s strength=0.800000" in captured.out
    assert captured.out.count("BPM: 120.000") == 1
    assert "Discontinuity: overflow after 3 frame(s) (input overflow)" in captured.err
    assert factory.source is not None
    assert factory.source.close_calls == 1


def test_cli_jsonl_logging_preserves_console_output_and_records_discontinuity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    boundary = Discontinuity(DiscontinuityReason.OVERFLOW, 3, "input overflow")
    factory = RecordingSourceFactory((*_beat_frames(), boundary, _beat_frames()[0]))
    path = tmp_path / "live.jsonl"

    exit_code = main(
        [
            "--device",
            "Microphone Array",
            "--log-jsonl",
            str(path),
            "--label",
            "steady-house-normal-1",
        ],
        source_factory=factory,
        backend_factory=FakeInventoryBackend,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.count("Beat ") == 4
    assert "BPM: 120.000" in captured.out
    assert "Discontinuity: overflow after 3 frame(s) (input overflow)" in captured.err
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["label"] == "steady-house-normal-1"
    assert [record["stream_ordinal"] for record in records if record["type"] == "beat"] == [
        0,
        0,
        0,
        1,
    ]


def test_cli_without_log_path_does_not_construct_a_log_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    factory = RecordingSourceFactory(_beat_frames()[:1])
    untouched = tmp_path / "not-requested.jsonl"

    exit_code = main(
        ["--device", "3", "--label", "ignored-without-log"],
        source_factory=factory,
        backend_factory=FakeInventoryBackend,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Beat 0: t=0.000000s strength=0.800000" in captured.out
    assert not untouched.exists()


def test_cli_closes_logger_even_when_source_close_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    factory = RecordingSourceFactory(_beat_frames()[:1], close_error=True)
    path = tmp_path / "close-error.jsonl"

    exit_code = main(
        ["--device", "3", "--log-jsonl", str(path)],
        source_factory=factory,
        backend_factory=FakeInventoryBackend,
    )

    captured = capsys.readouterr()
    assert exit_code == 6
    assert "Live diagnostic failure: source close failed" in captured.err
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert any(record["type"] == "summary" for record in records)


def test_cli_ctrl_c_closes_once_without_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    factory = RecordingSourceFactory(_beat_frames()[:1], interrupt_after_items=True)

    exit_code = main(
        ["--device", "3"],
        source_factory=factory,
        backend_factory=FakeInventoryBackend,
    )

    captured = capsys.readouterr()
    assert exit_code == 130
    assert "Live diagnostic interrupted." in captured.err
    assert "Traceback" not in captured.err
    assert factory.source is not None
    assert factory.source.close_calls == 1


@pytest.mark.parametrize(
    "arguments",
    [
        ["--device", "3", "--sample-rate", "0"],
        ["--device", "3", "--channels", "0"],
        ["--device", "3", "--sensitivity", "1.1"],
        ["--device", "3", "--min-bpm", "120", "--max-bpm", "100"],
    ],
)
def test_cli_rejects_invalid_analysis_or_capture_arguments(arguments: list[str]) -> None:
    factory = RecordingSourceFactory()

    exit_code = main(
        arguments,
        source_factory=factory,
        backend_factory=FakeInventoryBackend,
    )

    assert exit_code == 2
    assert factory.calls == []
