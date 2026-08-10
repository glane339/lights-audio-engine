from dataclasses import FrozenInstanceError, fields
from math import inf, nan

import numpy as np
import pytest


def test_audio_frame_owns_a_read_only_float64_copy() -> None:
    from lights_audio_engine.models import AudioFrame

    source = np.array([0.0, 0.25, -0.5], dtype=np.float32)
    frame = AudioFrame(samples=source, sample_rate_hz=48_000, start_time_seconds=1.25)
    source[1] = 0.75

    assert frame.samples.dtype == np.float64
    assert frame.samples.tolist() == [0.0, 0.25, -0.5]
    assert not frame.samples.flags.writeable
    assert frame.sample_rate_hz == 48_000
    assert frame.start_time_seconds == 1.25


def test_audio_frame_cannot_reenable_sample_writes() -> None:
    from lights_audio_engine.models import AudioFrame

    frame = AudioFrame(
        samples=np.zeros(4, dtype=np.float64),
        sample_rate_hz=48_000,
        start_time_seconds=0.0,
    )

    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        frame.samples.setflags(write=True)


@pytest.mark.parametrize(
    "samples",
    [
        np.array([], dtype=np.float64),
        np.zeros((2, 2), dtype=np.float64),
        np.array([0.0, nan], dtype=np.float64),
        np.array([0.0, inf], dtype=np.float64),
        np.array([0, 1], dtype=np.int16),
        np.array([0.0, 1.01], dtype=np.float64),
    ],
)
def test_audio_frame_rejects_malformed_samples(samples: np.ndarray) -> None:
    from lights_audio_engine.models import AudioFrame

    with pytest.raises((TypeError, ValueError)):
        AudioFrame(samples=samples, sample_rate_hz=48_000, start_time_seconds=0.0)


@pytest.mark.parametrize(
    ("sample_rate_hz", "start_time_seconds"),
    [(0, 0.0), (-1, 0.0), (48_000, -0.01), (48_000, nan), (48_000, inf)],
)
def test_audio_frame_rejects_invalid_timing(sample_rate_hz: int, start_time_seconds: float) -> None:
    from lights_audio_engine.models import AudioFrame

    with pytest.raises(ValueError):
        AudioFrame(
            samples=np.zeros(8, dtype=np.float64),
            sample_rate_hz=sample_rate_hz,
            start_time_seconds=start_time_seconds,
        )


def test_event_models_are_frozen_and_validate_normalized_fields() -> None:
    from lights_audio_engine.models import BeatEvent, DropEvent

    beat = BeatEvent(timestamp_seconds=1.5, strength=0.75, beat_index=2)
    drop = DropEvent(timestamp_seconds=2.0, strength=1.0, confidence=0.4)

    assert beat.beat_index == 2
    assert drop.confidence == 0.4
    with pytest.raises(FrozenInstanceError):
        beat.strength = 0.2  # type: ignore[misc]
    with pytest.raises(ValueError, match="strength"):
        BeatEvent(timestamp_seconds=0.0, strength=1.01, beat_index=0)
    with pytest.raises(ValueError, match="confidence"):
        DropEvent(timestamp_seconds=0.0, strength=0.5, confidence=-0.01)
    with pytest.raises(ValueError, match="beat_index"):
        BeatEvent(timestamp_seconds=0.0, strength=0.5, beat_index=-1)


def test_beat_event_exposes_observation_strength_without_speculative_confidence() -> None:
    from lights_audio_engine.models import BeatEvent

    assert [field.name for field in fields(BeatEvent)] == [
        "timestamp_seconds",
        "strength",
        "beat_index",
    ]


def test_analysis_result_uses_immutable_event_collections() -> None:
    from lights_audio_engine.models import AudioAnalysisResult, BeatEvent, DropEvent

    beat = BeatEvent(timestamp_seconds=1.0, strength=0.8, beat_index=0)
    drop = DropEvent(timestamp_seconds=1.0, strength=0.6)
    result = AudioAnalysisResult(
        bpm=120.0,
        beat_events=(beat,),
        drop_events=(drop,),
        current_level=0.5,
    )

    assert result.beat_events == (beat,)
    assert result.drop_events == (drop,)
    with pytest.raises(FrozenInstanceError):
        result.bpm = 90.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("bpm", "current_level"),
    [(0.0, 0.5), (nan, 0.5), (120.0, -0.01), (120.0, 1.01)],
)
def test_analysis_result_rejects_invalid_summary_values(bpm: float, current_level: float) -> None:
    from lights_audio_engine.models import AudioAnalysisResult

    with pytest.raises(ValueError):
        AudioAnalysisResult(bpm=bpm, current_level=current_level)


def test_boolean_values_are_rejected_for_numeric_domain_fields() -> None:
    from lights_audio_engine.config import AudioEngineConfig
    from lights_audio_engine.models import AudioAnalysisResult, AudioFrame, BeatEvent

    with pytest.raises(TypeError, match="sensitivity"):
        AudioEngineConfig(sensitivity=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="start_time_seconds"):
        AudioFrame(
            samples=np.zeros(1, dtype=np.float64),
            sample_rate_hz=48_000,
            start_time_seconds=True,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="strength"):
        BeatEvent(timestamp_seconds=0.0, strength=True, beat_index=0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bpm"):
        AudioAnalysisResult(bpm=True)  # type: ignore[arg-type]
