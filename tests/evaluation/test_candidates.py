from __future__ import annotations

import numpy as np
import pytest

from lights_audio_engine.evaluation.candidates import Candidate, DetectedOnset
from lights_audio_engine.models import AudioFrame

SAMPLE_RATE = 48_000


@pytest.mark.parametrize(
    ("timestamp", "strength", "message"),
    [(-0.1, 0.5, "timestamp"), (float("nan"), 0.5, "finite"), (0.1, 1.1, "strength")],
)
def test_detected_onset_rejects_invalid_values(
    timestamp: float, strength: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        DetectedOnset(timestamp, strength)


def _frames(samples: np.ndarray, block_size: int) -> tuple[AudioFrame, ...]:
    return tuple(
        AudioFrame(samples[start : start + block_size], SAMPLE_RATE, start / SAMPLE_RATE)
        for start in range(0, samples.size, block_size)
    )


def _pulse_signal(
    *, amplitude: float = 0.8, starts: tuple[int, ...] = (4_800,), size: int = 24_000
) -> np.ndarray:
    samples = np.zeros(size, dtype=np.float64)
    shape = np.hanning(97) * amplitude
    for start in starts:
        samples[start : start + shape.size] += shape
    return samples


def _events(
    candidate: Candidate, samples: np.ndarray, block_size: int
) -> tuple[DetectedOnset, ...]:
    return tuple(
        event for frame in _frames(samples, block_size) for event in candidate.process(frame)
    )


def test_candidate_a_matches_direct_energy_detector() -> None:
    from lights_audio_engine.config import AudioEngineConfig
    from lights_audio_engine.detectors.energy import EnergyBeatDetector
    from lights_audio_engine.evaluation.candidates import BaselineCandidate

    config = AudioEngineConfig()
    samples = np.zeros(4_800, dtype=np.float64)
    samples[960:1920] = 0.8
    frames = _frames(samples, 480)
    direct = EnergyBeatDetector(config)
    candidate = BaselineCandidate(config)

    expected = tuple(
        (event.timestamp_seconds, event.strength)
        for frame in frames
        for event in direct.process(frame).transients
    )
    actual = tuple(
        (event.timestamp_seconds, event.strength)
        for frame in frames
        for event in candidate.process(frame)
    )
    assert actual == expected


@pytest.mark.parametrize("candidate_factory", ["broadband", "multiband"])
def test_low_latency_candidates_handle_silence_reset_sample_rate_and_causality(
    candidate_factory: str,
) -> None:
    from lights_audio_engine.evaluation.candidates import create_candidate

    candidate = create_candidate(candidate_factory)
    assert _events(candidate, np.zeros(2_400, dtype=np.float64), 240) == ()
    with pytest.raises(ValueError, match="sample rate"):
        candidate.process(AudioFrame(np.zeros(240), 44_100, 0.05))

    candidate.reset()
    signal = _pulse_signal(size=5_520)
    before_confirmation = signal[:5_040]
    confirmation_hop = signal[5_040:5_280]
    assert _events(candidate, before_confirmation, 240) == ()
    emitted = candidate.process(AudioFrame(confirmation_hop, SAMPLE_RATE, 5_040 / SAMPLE_RATE))
    assert len(emitted) == 1
    assert emitted[0].timestamp_seconds == pytest.approx(4_848 / SAMPLE_RATE, abs=1 / SAMPLE_RATE)

    candidate.reset()
    first = _events(candidate, signal, 480)
    candidate.reset()
    assert _events(candidate, signal, 480) == first


@pytest.mark.parametrize("candidate_factory", ["broadband", "multiband"])
def test_low_latency_candidates_are_chunk_invariant_and_sensitive_to_quiet_pulses(
    candidate_factory: str,
) -> None:
    from lights_audio_engine.evaluation.candidates import create_candidate

    samples = _pulse_signal(amplitude=0.04, starts=(4_800, 16_800), size=24_000)
    outputs: list[tuple[DetectedOnset, ...]] = []
    for block_size in (240, 480, 960):
        candidate = create_candidate(candidate_factory)
        outputs.append(_events(candidate, samples, block_size))
    assert outputs[0] == outputs[1] == outputs[2]
    assert len(outputs[0]) == 2


@pytest.mark.parametrize("candidate_factory", ["broadband", "multiband"])
def test_low_latency_candidates_suppress_refractory_double(candidate_factory: str) -> None:
    from lights_audio_engine.evaluation.candidates import create_candidate

    samples = _pulse_signal(starts=(4_800, 7_200), size=12_000)
    assert len(_events(create_candidate(candidate_factory), samples, 240)) == 1


def test_multiband_detects_low_frequency_burst_and_fuses_broadband_evidence() -> None:
    from lights_audio_engine.evaluation.candidates import create_candidate

    samples = np.zeros(24_000, dtype=np.float64)
    time = np.arange(480, dtype=np.float64) / SAMPLE_RATE
    samples[4_800:5_280] = 0.15 * np.sin(2.0 * np.pi * 100.0 * time) * np.hanning(480)
    samples[18_000:18_097] = np.hanning(97) * 0.3

    events = _events(create_candidate("multiband"), samples, 240)
    assert len(events) == 2
    assert events[0].timestamp_seconds == pytest.approx(0.105, abs=0.006)
    assert events[1].timestamp_seconds == pytest.approx(18_048 / SAMPLE_RATE, abs=1 / SAMPLE_RATE)


def test_candidate_runner_records_end_of_delivered_audio_as_emission_time() -> None:
    from lights_audio_engine.evaluation.candidates import DetectedOnset
    from lights_audio_engine.evaluation.runner import run_candidate

    class BackdatingCandidate:
        def process(self, frame: AudioFrame) -> tuple[DetectedOnset, ...]:
            if frame.start_time_seconds == 0.005:
                return (DetectedOnset(0.001, 0.5),)
            return ()

        def reset(self) -> None:
            pass

    result = run_candidate(BackdatingCandidate(), _frames(np.zeros(480), 240))
    assert result.detections[0].event.timestamp_seconds == 0.001
    assert result.detections[0].emitted_stream_time_seconds == 0.01
    assert result.detections[0].decision_latency_seconds == pytest.approx(0.009)
