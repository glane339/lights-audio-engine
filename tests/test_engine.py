import numpy as np
import pytest


def _config(
    *,
    sensitivity: float = 0.5,
    min_bpm: float = 60.0,
    max_bpm: float = 200.0,
):
    from lights_audio_engine import AudioEngineConfig

    return AudioEngineConfig(
        expected_sample_rate_hz=1_000,
        sensitivity=sensitivity,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        analysis_window_seconds=0.02,
        energy_history_seconds=1.0,
    )


def _frame(
    samples: np.ndarray,
    *,
    start_time_seconds: float = 0.0,
    sample_rate_hz: int = 1_000,
):
    from lights_audio_engine import AudioFrame

    return AudioFrame(
        samples=samples.astype(np.float64),
        sample_rate_hz=sample_rate_hz,
        start_time_seconds=start_time_seconds,
    )


def _beat_block(amplitude: float = 0.8) -> np.ndarray:
    samples = np.zeros(500)
    samples[:20] = amplitude
    return samples


def test_silence_produces_no_events_or_bpm() -> None:
    from lights_audio_engine import AudioEngine

    result = AudioEngine(_config()).process(_frame(np.zeros(500)))

    assert result.beat_events == ()
    assert result.drop_events == ()
    assert result.bpm is None
    assert result.current_level == 0.0


def test_regular_beats_require_three_events_before_bpm_is_available() -> None:
    from lights_audio_engine import AudioEngine

    engine = AudioEngine(_config())

    first = engine.process(_frame(_beat_block(), start_time_seconds=0.0))
    second = engine.process(_frame(_beat_block(), start_time_seconds=0.5))
    third = engine.process(_frame(_beat_block(), start_time_seconds=1.0))

    assert [event.beat_index for event in first.beat_events] == [0]
    assert [event.beat_index for event in second.beat_events] == [1]
    assert [event.beat_index for event in third.beat_events] == [2]
    assert first.bpm is None
    assert second.bpm is None
    assert third.bpm == pytest.approx(120.0)


def test_bpm_is_not_reported_for_intervals_outside_configured_range() -> None:
    from lights_audio_engine import AudioEngine

    engine = AudioEngine(_config(min_bpm=90.0, max_bpm=200.0))

    engine.process(_frame(_beat_block(), start_time_seconds=0.0))
    engine.process(_frame(np.zeros(500), start_time_seconds=0.5))
    engine.process(_frame(_beat_block(), start_time_seconds=1.0))
    engine.process(_frame(np.zeros(500), start_time_seconds=1.5))
    result = engine.process(_frame(_beat_block(), start_time_seconds=2.0))

    assert len(result.beat_events) == 1
    assert result.bpm is None


def test_event_timestamp_is_derived_from_frame_time_and_sample_position() -> None:
    from lights_audio_engine import AudioEngine

    samples = np.zeros(100)
    samples[40:60] = 0.8

    result = AudioEngine(_config()).process(_frame(samples, start_time_seconds=7.25))

    assert result.beat_events[0].timestamp_seconds == pytest.approx(7.29)


def test_reset_clears_timing_beat_index_and_bpm_history() -> None:
    from lights_audio_engine import AudioEngine

    engine = AudioEngine(_config())
    engine.process(_frame(_beat_block(), start_time_seconds=0.0))
    engine.process(_frame(_beat_block(), start_time_seconds=0.5))
    assert engine.process(_frame(_beat_block(), start_time_seconds=1.0)).bpm == pytest.approx(120.0)

    engine.reset()
    replay = engine.process(_frame(_beat_block(), start_time_seconds=0.0))

    assert replay.bpm is None
    assert [event.beat_index for event in replay.beat_events] == [0]


def test_same_input_sequence_is_deterministic() -> None:
    from lights_audio_engine import AudioEngine

    frames = [
        _frame(_beat_block(), start_time_seconds=0.0),
        _frame(_beat_block(), start_time_seconds=0.5),
        _frame(_beat_block(), start_time_seconds=1.0),
    ]

    first_engine = AudioEngine(_config())
    first = [first_engine.process(frame) for frame in frames]
    second_engine = AudioEngine(_config())
    second = [second_engine.process(frame) for frame in frames]

    assert first == second


def test_sensitivity_propagates_to_energy_detection() -> None:
    from lights_audio_engine import AudioEngine

    quiet_beat = _beat_block(amplitude=0.15)

    low = AudioEngine(_config(sensitivity=0.0)).process(_frame(quiet_beat))
    high = AudioEngine(_config(sensitivity=1.0)).process(_frame(quiet_beat))

    assert low.beat_events == ()
    assert len(high.beat_events) == 1


def test_engine_rejects_noncontiguous_frames() -> None:
    from lights_audio_engine import AudioEngine

    engine = AudioEngine(_config())
    engine.process(_frame(np.zeros(100), start_time_seconds=0.0))

    with pytest.raises(ValueError, match="contiguous"):
        engine.process(_frame(np.zeros(100), start_time_seconds=0.2))


def test_engine_rejects_mismatched_sample_rate() -> None:
    from lights_audio_engine import AudioEngine

    with pytest.raises(ValueError, match="sample rate"):
        AudioEngine(_config()).process(_frame(np.zeros(100), sample_rate_hz=2_000))
