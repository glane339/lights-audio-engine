import numpy as np
import pytest


def test_mono_passthrough_accepts_one_dimensional_mono() -> None:
    from lights_audio_engine.capture import MonoPassthrough

    samples = np.array([0.25, -0.5], dtype=np.float32)

    mono = MonoPassthrough().to_mono(samples)

    assert mono is samples


def test_mono_passthrough_accepts_single_channel_matrix() -> None:
    from lights_audio_engine.capture import MonoPassthrough

    samples = np.array([[0.25], [-0.5]], dtype=np.float64)

    mono = MonoPassthrough().to_mono(samples)

    assert mono.tolist() == [0.25, -0.5]


def test_mono_passthrough_rejects_multichannel_input() -> None:
    from lights_audio_engine.capture import FrameAssemblyError, MonoPassthrough

    with pytest.raises(FrameAssemblyError, match="exactly one input channel"):
        MonoPassthrough().to_mono(np.zeros((4, 2), dtype=np.float64))


def test_select_channel_returns_the_configured_channel() -> None:
    from lights_audio_engine.capture import SelectChannel

    samples = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)

    mono = SelectChannel(1).to_mono(samples)

    assert mono.tolist() == [0.2, 0.4]


@pytest.mark.parametrize("index", [-1, -20])
def test_select_channel_rejects_negative_index_at_construction(index: int) -> None:
    from lights_audio_engine.capture import SelectChannel

    with pytest.raises(ValueError, match="non-negative"):
        SelectChannel(index)


@pytest.mark.parametrize("index", [True, 1.5, "1"])
def test_select_channel_rejects_non_integer_index_at_construction(index: object) -> None:
    from lights_audio_engine.capture import SelectChannel

    with pytest.raises(TypeError, match="integer"):
        SelectChannel(index)  # type: ignore[arg-type]


def test_select_channel_rejects_index_outside_observed_channel_count() -> None:
    from lights_audio_engine.capture import FrameAssemblyError, SelectChannel

    with pytest.raises(FrameAssemblyError, match="outside observed channel count 2"):
        SelectChannel(2).to_mono(np.zeros((4, 2), dtype=np.float64))


def test_average_channels_with_no_indices_averages_all_channels() -> None:
    from lights_audio_engine.capture import AverageChannels

    samples = np.array([[0.0, 0.6, 0.9], [0.3, 0.6, 0.0]], dtype=np.float64)

    mono = AverageChannels().to_mono(samples)

    assert mono.tolist() == pytest.approx([0.5, 0.3])


def test_average_channels_with_indices_averages_exact_subset() -> None:
    from lights_audio_engine.capture import AverageChannels

    samples = np.array([[0.0, 0.4, 1.0], [0.2, 0.6, 0.8]], dtype=np.float64)

    mono = AverageChannels((0, 2)).to_mono(samples)

    assert mono.tolist() == pytest.approx([0.5, 0.5])


@pytest.mark.parametrize("indices", [(-1,), (0, -2)])
def test_average_channels_rejects_negative_indices(indices: tuple[int, ...]) -> None:
    from lights_audio_engine.capture import AverageChannels

    with pytest.raises(ValueError, match="non-negative"):
        AverageChannels(indices)


def test_average_channels_rejects_duplicate_indices() -> None:
    from lights_audio_engine.capture import AverageChannels

    with pytest.raises(ValueError, match="duplicate"):
        AverageChannels((0, 1, 0))


def test_average_channels_rejects_empty_explicit_indices() -> None:
    from lights_audio_engine.capture import AverageChannels

    with pytest.raises(ValueError, match="must not be empty"):
        AverageChannels(())


@pytest.mark.parametrize("indices", [(0, True), (0, 1.5)])
def test_average_channels_rejects_non_integer_indices(indices: tuple[object, ...]) -> None:
    from lights_audio_engine.capture import AverageChannels

    with pytest.raises(TypeError, match="integers"):
        AverageChannels(indices)  # type: ignore[arg-type]


def test_average_channels_rejects_index_outside_observed_channel_count() -> None:
    from lights_audio_engine.capture import AverageChannels, FrameAssemblyError

    with pytest.raises(FrameAssemblyError, match="outside observed channel count 2"):
        AverageChannels((0, 2)).to_mono(np.zeros((4, 2), dtype=np.float64))
