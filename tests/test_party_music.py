from dataclasses import dataclass

import numpy as np
import pytest

from lights_audio_engine import AudioEngine, AudioEngineConfig, AudioFrame, BeatEvent

SAMPLE_RATE_HZ = 6_000
SIGNAL_DURATION_SECONDS = 4.5
PARTY_TEMPOS_BPM = (80.0, 100.0, 120.0, 128.0, 140.0, 150.0, 174.0, 200.0, 240.0)


@dataclass(frozen=True, slots=True)
class AnalysisRun:
    events: tuple[BeatEvent, ...]
    evidence: tuple[tuple[int, float | None], ...]
    final_bpm: float | None


def _config(*, sensitivity: float = 0.5) -> AudioEngineConfig:
    return AudioEngineConfig(
        expected_sample_rate_hz=SAMPLE_RATE_HZ,
        sensitivity=sensitivity,
        min_bpm=50.0,
        max_bpm=240.0,
    )


def _pulse_train(
    bpm: float,
    *,
    amplitude: float = 0.8,
    alternate_amplitude: float | None = None,
) -> np.ndarray:
    samples = np.zeros(round(SIGNAL_DURATION_SECONDS * SAMPLE_RATE_HZ), dtype=np.float64)
    pulse_width = round(0.020 * SAMPLE_RATE_HZ)
    beat_time_seconds = 0.25
    beat_index = 0
    while beat_time_seconds < SIGNAL_DURATION_SECONDS - 0.05:
        start = round(beat_time_seconds * SAMPLE_RATE_HZ)
        beat_amplitude = (
            alternate_amplitude if alternate_amplitude is not None and beat_index % 2 else amplitude
        )
        samples[start : start + pulse_width] = beat_amplitude
        beat_time_seconds += 60.0 / bpm
        beat_index += 1
    return samples


def _analyze(
    samples: np.ndarray,
    *,
    chunk_size: int,
    config: AudioEngineConfig | None = None,
) -> AnalysisRun:
    engine = AudioEngine(config or _config())
    events: list[BeatEvent] = []
    evidence: list[tuple[int, float | None]] = []
    final_bpm: float | None = None

    for start in range(0, samples.size, chunk_size):
        result = engine.process(
            AudioFrame(
                samples=samples[start : start + chunk_size],
                sample_rate_hz=SAMPLE_RATE_HZ,
                start_time_seconds=start / SAMPLE_RATE_HZ,
            )
        )
        events.extend(result.beat_events)
        evidence.append((len(events), result.bpm))
        final_bpm = result.bpm

    return AnalysisRun(events=tuple(events), evidence=tuple(evidence), final_bpm=final_bpm)


@pytest.mark.parametrize("expected_bpm", PARTY_TEMPOS_BPM)
def test_party_tempo_matrix_is_direct_deterministic_and_chunk_invariant(
    expected_bpm: float,
) -> None:
    samples = _pulse_train(expected_bpm)

    small = _analyze(samples, chunk_size=73)
    medium = _analyze(samples, chunk_size=512)
    large = _analyze(samples, chunk_size=2_048)
    replay = _analyze(samples, chunk_size=512)

    assert len(medium.events) >= 5
    assert all(
        b.timestamp_seconds > a.timestamp_seconds
        for a, b in zip(medium.events, medium.events[1:], strict=False)
    )
    assert all(bpm is None for event_count, bpm in medium.evidence if event_count < 3)
    first_estimated_event_count = next(
        event_count for event_count, bpm in medium.evidence if bpm is not None
    )
    assert first_estimated_event_count == 3
    assert medium.final_bpm == pytest.approx(expected_bpm, abs=0.1)
    assert replay == medium
    assert small.events == medium.events == large.events
    assert small.final_bpm == medium.final_bpm == large.final_bpm


def test_higher_sensitivity_detects_more_borderline_beats_without_moving_shared_events() -> None:
    samples = _pulse_train(120.0, amplitude=0.8, alternate_amplitude=0.15)

    low = _analyze(samples, chunk_size=512, config=_config(sensitivity=0.0))
    high = _analyze(samples, chunk_size=512, config=_config(sensitivity=1.0))

    low_timestamps = tuple(event.timestamp_seconds for event in low.events)
    high_timestamps = tuple(event.timestamp_seconds for event in high.events)
    assert 0 < len(low_timestamps) < len(high_timestamps)
    assert low_timestamps == high_timestamps[::2]


def test_reasonable_amplitude_scaling_preserves_rhythm_at_reactive_sensitivity() -> None:
    reactive_config = _config(sensitivity=0.8)
    quiet = _analyze(
        _pulse_train(128.0, amplitude=0.35),
        chunk_size=512,
        config=reactive_config,
    )
    loud = _analyze(
        _pulse_train(128.0, amplitude=0.8),
        chunk_size=512,
        config=reactive_config,
    )

    assert len(quiet.events) == len(loud.events)
    assert all(
        abs(quiet_event.timestamp_seconds - loud_event.timestamp_seconds)
        <= reactive_config.analysis_window_seconds
        for quiet_event, loud_event in zip(quiet.events, loud.events, strict=True)
    )
    assert quiet.final_bpm == pytest.approx(128.0, abs=0.1)
    assert loud.final_bpm == pytest.approx(128.0, abs=0.1)
