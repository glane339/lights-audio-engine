from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from lights_audio_engine import AudioEngine, AudioEngineConfig, AudioFrame

if TYPE_CHECKING:
    from lights_audio_engine.capture import CaptureItem


@dataclass
class FakeAudioSource:
    _sample_rate_hz: int
    items: tuple[CaptureItem, ...]
    closed: bool = False

    @property
    def sample_rate_hz(self) -> int:
        return self._sample_rate_hz

    def stream(self) -> Iterator[CaptureItem]:
        yield from self.items

    def close(self) -> None:
        self.closed = True


class TrackingAudioEngine(AudioEngine):
    def __init__(self, config: AudioEngineConfig) -> None:
        super().__init__(config)
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1
        super().reset()


def _config() -> AudioEngineConfig:
    return AudioEngineConfig(
        expected_sample_rate_hz=1_000,
        sensitivity=0.5,
        min_bpm=60.0,
        max_bpm=200.0,
        analysis_window_seconds=0.02,
        energy_history_seconds=1.0,
    )


def _signal() -> np.ndarray:
    samples = np.zeros(3_200, dtype=np.float64)
    for start in range(0, samples.size, 500):
        samples[start : start + 20] = 0.8
    return samples


def _varying_blocks(samples: np.ndarray) -> tuple[np.ndarray, ...]:
    sizes = (73, 211, 19, 512, 127)
    blocks: list[np.ndarray] = []
    start = 0
    size_index = 0
    while start < samples.size:
        end = min(samples.size, start + sizes[size_index % len(sizes)])
        blocks.append(samples[start:end])
        start = end
        size_index += 1
    return tuple(blocks)


def test_fake_audio_source_satisfies_runtime_protocol() -> None:
    from lights_audio_engine.capture import AudioSource

    source = FakeAudioSource(1_000, ())

    assert isinstance(source, AudioSource)
    assert source.sample_rate_hz == 1_000
    source.close()
    assert source.closed


def test_capture_driver_matches_existing_engine_contract_for_arbitrary_blocks() -> None:
    from lights_audio_engine.capture import CaptureConfig, FrameAssembler, run_engine

    blocks = _varying_blocks(_signal())
    assembler = FrameAssembler(CaptureConfig(sample_rate_hz=1_000))
    capture_items = tuple(
        frame
        for block in blocks
        if (frame := assembler.assemble(block, sample_rate_hz=1_000)) is not None
    )
    source = FakeAudioSource(1_000, capture_items)

    captured_results = tuple(run_engine(source, AudioEngine(_config())))

    direct_engine = AudioEngine(_config())
    direct_results = tuple(
        direct_engine.process(
            AudioFrame(
                samples=block,
                sample_rate_hz=1_000,
                start_time_seconds=sum(previous.size for previous in blocks[:index]) / 1_000,
            )
        )
        for index, block in enumerate(blocks)
    )
    assert captured_results == direct_results


def test_discontinuity_resets_engine_and_yields_no_result_for_boundary() -> None:
    from lights_audio_engine.capture import (
        CaptureConfig,
        DiscontinuityReason,
        FrameAssembler,
        run_engine,
    )

    assembler = FrameAssembler(CaptureConfig(sample_rate_hz=1_000))
    first = assembler.assemble(np.r_[np.ones(20) * 0.8, np.zeros(480)])
    boundary = assembler.mark_discontinuity(DiscontinuityReason.RESTART)
    second = assembler.assemble(np.r_[np.ones(20) * 0.8, np.zeros(480)])
    assert first is not None
    assert second is not None
    source = FakeAudioSource(1_000, (first, boundary, second))
    engine = TrackingAudioEngine(_config())

    results = tuple(run_engine(source, engine))

    assert engine.reset_calls == 1
    assert len(results) == 2
    assert [event.beat_index for result in results for event in result.beat_events] == [0, 0]
    assert [event.timestamp_seconds for result in results for event in result.beat_events] == [
        0.0,
        0.0,
    ]
