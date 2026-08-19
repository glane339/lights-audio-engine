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


def test_causal_spectral_tempo_candidate_is_silent_for_silence_at_each_delivery_size() -> None:
    """A spectral-tempo candidate must never turn absent onset evidence into beats."""

    from lights_audio_engine.evaluation.candidates import create_candidate

    samples = np.zeros(4_800, dtype=np.float64)
    for block_size in (240, 480, 960):
        assert _events(create_candidate("causal-spectral-tempo"), samples, block_size) == ()


def test_causal_spectral_tempo_candidate_is_delivery_and_amplitude_invariant() -> None:
    """Changing packetization or uniform gain must not change causal event timing."""

    from lights_audio_engine.evaluation.candidates import create_candidate

    samples = _pulse_signal(starts=(4_800, 28_800, 52_800), size=57_600)
    signatures: list[tuple[tuple[float, float], ...]] = []
    for block_size in (240, 480, 960):
        events = _events(create_candidate("causal-spectral-tempo"), samples, block_size)
        signatures.append(tuple((event.timestamp_seconds, event.strength) for event in events))
    quiet_events = _events(create_candidate("causal-spectral-tempo"), samples * 0.05, 240)

    assert signatures[0] == signatures[1] == signatures[2]
    assert tuple(event.timestamp_seconds for event in quiet_events) == tuple(
        timestamp for timestamp, _ in signatures[0]
    )
    assert len(signatures[0]) == 3


def test_causal_spectral_tempo_candidate_never_synthesizes_beats_without_onset_evidence() -> None:
    """Tempo state must not turn a steady tone or later silence into synthetic events."""

    from lights_audio_engine.evaluation.candidates import create_candidate

    time = np.arange(96_000, dtype=np.float64) / SAMPLE_RATE
    assert _events(create_candidate("causal-spectral-tempo"), 0.2 * np.sin(440.0 * time), 240) == ()

    pulses = _pulse_signal(starts=(4_800, 28_800, 52_800), size=192_000)
    events = _events(create_candidate("causal-spectral-tempo"), pulses, 240)
    assert len(events) == 3
    assert events[-1].timestamp_seconds < 1.2


def test_causal_spectral_tempo_candidate_confirms_events_within_fifteen_milliseconds() -> None:
    """The real trailing window may warm up, but confirmed events must have bounded latency."""

    from lights_audio_engine.evaluation.candidates import create_candidate
    from lights_audio_engine.evaluation.runner import run_candidate

    samples = _pulse_signal(starts=(4_800, 28_800, 52_800), size=57_600)
    result = run_candidate(create_candidate("causal-spectral-tempo"), _frames(samples, 240))

    assert len(result.detections) == 3
    assert max(item.decision_latency_seconds for item in result.detections) <= 0.015


def test_causal_spectral_tempo_candidate_localizes_a_confirmed_event_to_its_hop() -> None:
    """A larger transient in the trailing window cannot backdate the next-hop candidate."""

    from lights_audio_engine.evaluation.detectors.causal_spectral_tempo import (
        CausalSpectralTempoConfig,
        CausalSpectralTempoDetector,
    )

    class ForcedPeakDetector(CausalSpectralTempoDetector):
        def force_previous_peak(self) -> None:
            self._previous_novelty = 1.0
            self._previous_threshold = 0.0

    detector = ForcedPeakDetector(CausalSpectralTempoConfig(minimum_history_hops=0))
    samples = np.zeros(1_200, dtype=np.float64)
    samples[100] = 1.0  # Larger, but in an earlier overlapping analysis-window hop.
    samples[840] = 0.5  # Peak of the newly completed candidate hop [720, 960).
    for frame in _frames(samples[:960], 240):
        detector.process(frame)

    detector.force_previous_peak()
    events = detector.process(AudioFrame(samples[960:], SAMPLE_RATE, 960 / SAMPLE_RATE))

    assert len(events) == 1
    assert events[0].timestamp_seconds == pytest.approx(840 / SAMPLE_RATE)
    assert events[0].timestamp_seconds >= 720 / SAMPLE_RATE


def test_causal_spectral_tempo_candidate_suppresses_an_isolated_strong_offbeat() -> None:
    """A stable phase estimate must reject one strong half-cycle distractor."""

    from lights_audio_engine.evaluation.candidates import create_candidate

    samples = _pulse_signal(
        amplitude=0.2,
        starts=(4_800, 28_800, 52_800, 76_800),
        size=120_000,
    )
    samples[88_800:88_897] += np.hanning(97)

    causal = _events(create_candidate("causal-spectral-tempo"), samples, 240)
    broadband = _events(create_candidate("broadband"), samples, 240)

    assert len(causal) == 4
    assert len(broadband) > len(causal)


def test_causal_spectral_tempo_candidate_reacquires_a_persistent_phase_change() -> None:
    """Two consistent off-phase peaks must deterministically establish a new phase."""

    from lights_audio_engine.evaluation.candidates import create_candidate

    events = _events(
        create_candidate("causal-spectral-tempo"),
        _pulse_signal(
            starts=(4_800, 28_800, 52_800, 76_800, 88_800, 112_800, 136_800),
            size=144_000,
        ),
        240,
    )

    assert tuple(round(event.timestamp_seconds, 3) for event in events) == (
        0.105,
        0.605,
        1.105,
        1.605,
        2.355,
        2.855,
    )


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
