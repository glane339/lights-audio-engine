from dataclasses import FrozenInstanceError
from math import inf, nan

import numpy as np
import pytest

from lights_audio_engine import AudioFrame
from lights_audio_engine.capture import FrameAssembler


def _audio_frame(item: object) -> AudioFrame:
    assert isinstance(item, AudioFrame)
    return item


def _assembler(
    *,
    sample_rate_hz: int = 48_000,
    epoch_seconds: float = 0.0,
    out_of_range: str = "clip",
    channel_policy: object | None = None,
) -> FrameAssembler:
    from lights_audio_engine.capture import CaptureConfig

    config_kwargs: dict[str, object] = {
        "sample_rate_hz": sample_rate_hz,
        "out_of_range": out_of_range,
    }
    if channel_policy is not None:
        config_kwargs["channel_policy"] = channel_policy
    return FrameAssembler(CaptureConfig(**config_kwargs), epoch_seconds=epoch_seconds)  # type: ignore[arg-type]


def test_capture_config_uses_hardware_independent_defaults() -> None:
    from lights_audio_engine.capture import CaptureConfig, MonoPassthrough

    config = CaptureConfig()

    assert config.sample_rate_hz == 48_000
    assert isinstance(config.channel_policy, MonoPassthrough)
    assert config.out_of_range == "clip"


@pytest.mark.parametrize("sample_rate_hz", [0, -1, 48_000.5, True])
def test_capture_config_rejects_invalid_sample_rate(sample_rate_hz: object) -> None:
    from lights_audio_engine.capture import CaptureConfig

    with pytest.raises(ValueError, match="sample_rate_hz"):
        CaptureConfig(sample_rate_hz=sample_rate_hz)  # type: ignore[arg-type]


def test_capture_config_rejects_unknown_out_of_range_policy() -> None:
    from lights_audio_engine.capture import CaptureConfig

    with pytest.raises(ValueError, match="out_of_range"):
        CaptureConfig(out_of_range="normalize")  # type: ignore[arg-type]


def test_capture_config_rejects_non_channel_policy() -> None:
    from lights_audio_engine.capture import CaptureConfig

    with pytest.raises(TypeError, match="channel_policy"):
        CaptureConfig(channel_policy=object())  # type: ignore[arg-type]


def test_one_dimensional_mono_block_becomes_audio_frame() -> None:
    assembler = _assembler()
    block = np.array([0.25, -0.5, 0.75], dtype=np.float32)

    frame = assembler.assemble(block)

    assert isinstance(frame, AudioFrame)
    assert frame.samples.tolist() == [0.25, -0.5, 0.75]
    assert frame.samples.dtype == np.float64
    assert not frame.samples.flags.writeable
    assert frame.sample_rate_hz == 48_000
    assert frame.start_time_seconds == 0.0


def test_single_channel_matrix_becomes_audio_frame() -> None:
    frame = _assembler().assemble(np.array([[0.25], [-0.5]], dtype=np.float64))

    assert isinstance(frame, AudioFrame)
    assert frame.samples.tolist() == [0.25, -0.5]


def test_arbitrary_odd_blocks_have_exact_cumulative_sample_timing() -> None:
    assembler = _assembler(sample_rate_hz=1_000, epoch_seconds=2.5)
    sizes = [73, 511, 3, 2_049]

    frames = [_audio_frame(assembler.assemble(np.zeros(size, dtype=np.float64))) for size in sizes]

    assert [frame.start_time_seconds for frame in frames] == [
        2.5,
        2.5 + 73 / 1_000,
        2.5 + 584 / 1_000,
        2.5 + 587 / 1_000,
    ]
    assert assembler.samples_emitted == sum(sizes)
    assert assembler.frames_emitted == len(sizes)


def test_many_blocks_do_not_accumulate_timestamp_drift() -> None:
    assembler = _assembler(sample_rate_hz=48_000, epoch_seconds=1.25)

    for index in range(10_000):
        frame = _audio_frame(assembler.assemble(np.zeros(73, dtype=np.float64)))
        assert frame.start_time_seconds == 1.25 + (index * 73) / 48_000


@pytest.mark.parametrize("epoch_seconds", [0.0, 1.25, 10])
def test_frame_assembler_accepts_finite_non_negative_epoch(epoch_seconds: float) -> None:
    frame = _audio_frame(
        _assembler(epoch_seconds=epoch_seconds).assemble(np.zeros(1, dtype=np.float64))
    )

    assert frame.start_time_seconds == float(epoch_seconds)


@pytest.mark.parametrize("epoch_seconds", [-0.01, nan, inf])
def test_frame_assembler_rejects_invalid_epoch_value(epoch_seconds: float) -> None:
    with pytest.raises(ValueError, match="epoch_seconds"):
        _assembler(epoch_seconds=epoch_seconds)


def test_frame_assembler_rejects_boolean_epoch() -> None:
    with pytest.raises(TypeError, match="epoch_seconds"):
        _assembler(epoch_seconds=True)  # type: ignore[arg-type]


def test_sample_rate_mismatch_discards_block_and_rebases_next_stream() -> None:
    from lights_audio_engine.capture import Discontinuity, DiscontinuityReason

    assembler = _assembler(sample_rate_hz=1_000, epoch_seconds=3.0)
    first = assembler.assemble(np.zeros(7, dtype=np.float64))

    discontinuity = assembler.assemble(np.ones(11, dtype=np.float64), sample_rate_hz=2_000)
    next_frame = _audio_frame(assembler.assemble(np.zeros(5, dtype=np.float64)))

    assert first is not None
    assert isinstance(discontinuity, Discontinuity)
    assert discontinuity.reason is DiscontinuityReason.SAMPLE_RATE_CHANGE
    assert discontinuity.frames_emitted_before == 1
    assert discontinuity.detail == "expected 1000 Hz, observed 2000 Hz"
    assert next_frame.start_time_seconds == 3.0
    assert assembler.samples_emitted == 5


def test_empty_blocks_return_none_without_advancing_timing() -> None:
    assembler = _assembler(sample_rate_hz=1_000)

    assert assembler.assemble(np.array([], dtype=np.float64)) is None
    assert assembler.assemble(np.empty((0, 2), dtype=np.float32)) is None
    frame = _audio_frame(assembler.assemble(np.ones(3, dtype=np.float64)))

    assert frame.start_time_seconds == 0.0
    assert assembler.samples_emitted == 3


@pytest.mark.parametrize(
    "block",
    [
        np.array([0.0, nan], dtype=np.float64),
        np.array([0.0, inf], dtype=np.float32),
    ],
)
def test_non_finite_block_is_frame_assembly_error_without_state_change(
    block: np.ndarray,
) -> None:
    from lights_audio_engine.capture import FrameAssemblyError

    assembler = _assembler()

    with pytest.raises(FrameAssemblyError, match="finite"):
        assembler.assemble(block)

    assert assembler.samples_emitted == 0
    assert assembler.frames_emitted == 0


def test_integer_block_is_frame_assembly_error() -> None:
    from lights_audio_engine.capture import FrameAssemblyError

    with pytest.raises(FrameAssemblyError, match="floating-point"):
        _assembler().assemble(np.array([0, 1], dtype=np.int16))


@pytest.mark.parametrize(
    "block",
    [np.zeros((2, 2, 1), dtype=np.float64), np.zeros((2, 0), dtype=np.float64)],
)
def test_incompatible_block_shape_is_frame_assembly_error(block: np.ndarray) -> None:
    from lights_audio_engine.capture import FrameAssemblyError

    with pytest.raises(FrameAssemblyError, match="block"):
        _assembler().assemble(block)


def test_non_array_block_is_frame_assembly_error() -> None:
    from lights_audio_engine.capture import FrameAssemblyError

    with pytest.raises(FrameAssemblyError, match="NumPy array"):
        _assembler().assemble([0.0, 0.1])  # type: ignore[arg-type]


def test_reject_out_of_range_policy_raises_boundary_error_without_advancing() -> None:
    from lights_audio_engine.capture import FrameAssemblyError

    assembler = _assembler(out_of_range="reject")

    with pytest.raises(FrameAssemblyError, match="range"):
        assembler.assemble(np.array([-1.01, 0.0, 1.01], dtype=np.float64))

    assert assembler.samples_emitted == 0


def test_clip_out_of_range_policy_builds_clipped_audio_frame() -> None:
    frame = _audio_frame(
        _assembler(out_of_range="clip").assemble(np.array([-1.5, -0.25, 1.5], dtype=np.float32))
    )

    assert frame.samples.tolist() == [-1.0, -0.25, 1.0]


def test_overflow_returns_discontinuity_discards_block_and_rebases() -> None:
    from lights_audio_engine.capture import Discontinuity, DiscontinuityReason

    assembler = _assembler(sample_rate_hz=1_000, epoch_seconds=4.0)
    assert assembler.assemble(np.zeros(8, dtype=np.float64)) is not None

    discontinuity = assembler.assemble(
        np.ones(100, dtype=np.float64), overflow=True, detail="backend overflow flag"
    )
    next_frame = _audio_frame(assembler.assemble(np.zeros(3, dtype=np.float64)))

    assert isinstance(discontinuity, Discontinuity)
    assert discontinuity.reason is DiscontinuityReason.OVERFLOW
    assert discontinuity.frames_emitted_before == 1
    assert discontinuity.detail == "backend overflow flag"
    assert next_frame.start_time_seconds == 4.0
    assert assembler.samples_emitted == 3


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("dropped_block", "dropped_block"),
        ("restart", "restart"),
    ],
)
def test_mark_discontinuity_returns_value_and_rebases(reason: str, expected: str) -> None:
    from lights_audio_engine.capture import DiscontinuityReason

    assembler = _assembler(sample_rate_hz=1_000, epoch_seconds=2.0)
    assert assembler.assemble(np.zeros(9, dtype=np.float64)) is not None

    discontinuity = assembler.mark_discontinuity(reason, detail="source status")
    next_frame = _audio_frame(assembler.assemble(np.zeros(1, dtype=np.float64)))

    assert discontinuity.reason is DiscontinuityReason(expected)
    assert discontinuity.frames_emitted_before == 1
    assert discontinuity.detail == "source status"
    assert next_frame.start_time_seconds == 2.0


def test_discontinuity_model_is_frozen() -> None:
    from lights_audio_engine.capture import Discontinuity, DiscontinuityReason

    discontinuity = Discontinuity(
        reason=DiscontinuityReason.RESTART,
        frames_emitted_before=2,
    )

    with pytest.raises(FrozenInstanceError):
        discontinuity.frames_emitted_before = 3  # type: ignore[misc]


def test_channel_count_change_in_uninterrupted_stream_is_error() -> None:
    from lights_audio_engine.capture import FrameAssemblyError, SelectChannel

    assembler = _assembler(channel_policy=SelectChannel(0))
    assert assembler.assemble(np.zeros((5, 2), dtype=np.float64)) is not None

    with pytest.raises(FrameAssemblyError, match="channel count changed"):
        assembler.assemble(np.zeros((5, 1), dtype=np.float64))

    assert assembler.samples_emitted == 5
    assert assembler.frames_emitted == 1


def test_channel_count_may_change_after_restart_discontinuity() -> None:
    from lights_audio_engine.capture import SelectChannel

    assembler = _assembler(channel_policy=SelectChannel(0))
    assert assembler.assemble(np.zeros((5, 2), dtype=np.float64)) is not None

    assembler.mark_discontinuity("restart")
    frame = _audio_frame(assembler.assemble(np.zeros((3, 1), dtype=np.float64)))

    assert frame.samples.size == 3
    assert frame.start_time_seconds == 0.0
