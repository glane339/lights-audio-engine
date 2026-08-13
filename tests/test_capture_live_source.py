from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from lights_audio_engine import AudioEngine
from lights_audio_engine.capture import Discontinuity, DiscontinuityReason, run_engine
from lights_audio_engine.capture.live_source import SoundDeviceAudioSource
from lights_audio_engine.models import AudioFrame


class FakePortAudioError(Exception):
    pass


class FakeInputStream:
    def __init__(
        self,
        blocks: tuple[np.ndarray, ...],
        *,
        overflow_on_calls: frozenset[int] = frozenset(),
        fail_on_call: int | None = None,
    ) -> None:
        self.blocks = blocks
        self.overflow_on_calls = overflow_on_calls
        self.fail_on_call = fail_on_call
        self.read_sizes: list[int] = []
        self.started = False
        self.stop_calls = 0
        self.close_calls = 0

    def start(self) -> None:
        self.started = True

    def read(self, frames: int) -> tuple[np.ndarray, bool]:
        self.read_sizes.append(frames)
        call = len(self.read_sizes)
        if call == self.fail_on_call:
            raise FakePortAudioError("device disconnected")
        if call > len(self.blocks):
            raise FakePortAudioError("capture complete")
        return self.blocks[call - 1], call in self.overflow_on_calls

    def stop(self) -> None:
        self.stop_calls += 1
        self.started = False

    def close(self) -> None:
        self.close_calls += 1


class FakeSoundDevice:
    PortAudioError = FakePortAudioError

    def __init__(self, stream: FakeInputStream) -> None:
        self.stream = stream
        self.input_stream_kwargs: dict[str, object] | None = None

    def InputStream(self, **kwargs: object) -> FakeInputStream:  # noqa: N802
        self.input_stream_kwargs = kwargs
        return self.stream


def _source(
    stream: FakeInputStream,
    *,
    channels: int = 1,
    on_discontinuity: Callable[[Discontinuity], object] | None = None,
) -> SoundDeviceAudioSource:
    module = FakeSoundDevice(stream)
    return SoundDeviceAudioSource(
        17,
        channels=channels,
        module_loader=lambda: module,
        on_discontinuity=on_discontinuity,
    )


def test_live_source_opens_requested_device_and_preserves_sample_timing() -> None:
    stream = FakeInputStream(
        (
            np.full((960, 1), 0.25, dtype=np.float32),
            np.full((960, 1), 0.5, dtype=np.float32),
        )
    )
    module = FakeSoundDevice(stream)
    source = SoundDeviceAudioSource(17, module_loader=lambda: module)
    iterator = source.stream()

    first = next(iterator)
    second = next(iterator)
    source.close()

    assert isinstance(first, AudioFrame)
    assert isinstance(second, AudioFrame)
    assert source.sample_rate_hz == 48_000
    assert first.start_time_seconds == 0.0
    assert second.start_time_seconds == 0.02
    np.testing.assert_array_equal(first.samples, np.full(960, 0.25))
    np.testing.assert_array_equal(second.samples, np.full(960, 0.5))
    assert stream.read_sizes == [960, 960]
    assert module.input_stream_kwargs == {
        "device": 17,
        "samplerate": 48_000,
        "channels": 1,
        "dtype": "float32",
        "callback": None,
    }


def test_live_source_averages_requested_multichannel_capture() -> None:
    block = np.column_stack(
        (np.full(960, 0.2, dtype=np.float32), np.full(960, 0.6, dtype=np.float32))
    )
    source = _source(FakeInputStream((block,)), channels=2)
    iterator = source.stream()

    item = next(iterator)
    source.close()

    assert isinstance(item, AudioFrame)
    np.testing.assert_allclose(item.samples, np.full(960, 0.4), rtol=0.0, atol=1e-7)


def test_overflow_is_observed_before_yield_and_streaming_continues_rebased() -> None:
    stream = FakeInputStream(
        (
            np.full((960, 1), 0.1, dtype=np.float32),
            np.full((960, 1), 0.2, dtype=np.float32),
            np.full((960, 1), 0.3, dtype=np.float32),
        ),
        overflow_on_calls=frozenset({2}),
    )
    observed: list[Discontinuity] = []
    source = _source(stream, on_discontinuity=observed.append)
    iterator = source.stream()

    first = next(iterator)
    boundary = next(iterator)
    assert observed == [boundary]
    rebased = next(iterator)
    source.close()

    assert isinstance(first, AudioFrame)
    assert boundary == Discontinuity(DiscontinuityReason.OVERFLOW, 1)
    assert isinstance(rebased, AudioFrame)
    assert rebased.start_time_seconds == 0.0


def test_capture_error_yields_one_observed_final_discontinuity() -> None:
    stream = FakeInputStream(
        (np.full((960, 1), 0.25, dtype=np.float32),),
        fail_on_call=2,
    )
    observed: list[Discontinuity] = []
    source = _source(stream, on_discontinuity=observed.append)
    iterator = source.stream()

    first = next(iterator)
    boundary = next(iterator)

    assert isinstance(first, AudioFrame)
    assert boundary == Discontinuity(
        DiscontinuityReason.DROPPED_BLOCK,
        1,
        "device disconnected",
    )
    assert observed == [boundary]
    with pytest.raises(StopIteration):
        next(iterator)
    assert stream.read_sizes == [960, 960]
    source.close()


def test_observer_failure_does_not_change_discontinuity_delivery() -> None:
    stream = FakeInputStream(
        (np.zeros((960, 1), dtype=np.float32),),
        overflow_on_calls=frozenset({1}),
    )
    calls = 0

    def failing_observer(_boundary: Discontinuity) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("observer failed")

    source = _source(stream, on_discontinuity=failing_observer)

    boundary = next(source.stream())
    source.close()

    assert calls == 1
    assert boundary == Discontinuity(DiscontinuityReason.OVERFLOW, 0)


def test_close_is_safe_before_streaming_and_idempotent_after_start() -> None:
    unopened_stream = FakeInputStream(())
    unopened_module = FakeSoundDevice(unopened_stream)
    unopened = SoundDeviceAudioSource(17, module_loader=lambda: unopened_module)

    unopened.close()

    assert tuple(unopened.stream()) == ()
    assert unopened_module.input_stream_kwargs is None

    stream = FakeInputStream((np.zeros((960, 1), dtype=np.float32),))
    source = _source(stream)
    iterator = source.stream()
    assert isinstance(next(iterator), AudioFrame)

    source.close()
    source.close()

    assert tuple(iterator) == ()
    assert stream.stop_calls == 1
    assert stream.close_calls == 1


def _pulse_blocks() -> tuple[np.ndarray, ...]:
    samples = np.zeros(72_000, dtype=np.float32)
    for start in (0, 24_000, 48_000):
        samples[start : start + 960] = 0.8
    return tuple(samples[start : start + 960, None] for start in range(0, samples.size, 960))


def test_live_source_runs_through_existing_engine_without_observer() -> None:
    source = _source(FakeInputStream(_pulse_blocks()))

    results = tuple(run_engine(source, AudioEngine()))
    source.close()

    events = tuple(event for result in results for event in result.beat_events)
    assert [event.beat_index for event in events] == [0, 1, 2]
    assert results[-1].bpm == pytest.approx(120.0)


def test_live_source_rejects_invalid_capture_configuration() -> None:
    def module_loader() -> FakeSoundDevice:
        return FakeSoundDevice(FakeInputStream(()))

    with pytest.raises(ValueError, match="device_index"):
        SoundDeviceAudioSource(-1, module_loader=module_loader)
    with pytest.raises(ValueError, match="sample_rate_hz"):
        SoundDeviceAudioSource(0, sample_rate_hz=0, module_loader=module_loader)
    with pytest.raises(ValueError, match="channels"):
        SoundDeviceAudioSource(0, channels=0, module_loader=module_loader)
    with pytest.raises(ValueError, match="block_size_frames"):
        SoundDeviceAudioSource(0, block_size_frames=0, module_loader=module_loader)
