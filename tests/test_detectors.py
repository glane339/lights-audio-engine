import numpy as np
import pytest


def _frame(
    samples: np.ndarray,
    *,
    start_time_seconds: float = 0.0,
    sample_rate_hz: int = 1_000,
):
    from lights_audio_engine.models import AudioFrame

    return AudioFrame(
        samples=samples.astype(np.float64),
        sample_rate_hz=sample_rate_hz,
        start_time_seconds=start_time_seconds,
    )


def _config(*, sensitivity: float = 0.5):
    from lights_audio_engine.config import AudioEngineConfig

    return AudioEngineConfig(
        expected_sample_rate_hz=1_000,
        sensitivity=sensitivity,
        min_bpm=60.0,
        max_bpm=200.0,
        analysis_window_seconds=0.02,
        energy_history_seconds=1.0,
    )


def test_energy_detector_emits_nothing_for_silence() -> None:
    from lights_audio_engine.detectors.energy import EnergyBeatDetector

    detection = EnergyBeatDetector(_config()).process(_frame(np.zeros(100)))

    assert detection.transients == ()
    assert detection.current_level == 0.0


def test_energy_detector_uses_peak_sample_for_event_timestamp() -> None:
    from lights_audio_engine.detectors.energy import EnergyBeatDetector

    samples = np.zeros(100)
    samples[40:60] = 0.8

    detection = EnergyBeatDetector(_config()).process(_frame(samples, start_time_seconds=10.0))

    assert len(detection.transients) == 1
    assert detection.transients[0].timestamp_seconds == pytest.approx(10.04)
    assert detection.transients[0].strength == pytest.approx(0.8)


def test_higher_sensitivity_detects_a_quieter_transient() -> None:
    from lights_audio_engine.detectors.energy import EnergyBeatDetector

    samples = np.zeros(100)
    samples[40:60] = 0.15

    low = EnergyBeatDetector(_config(sensitivity=0.0)).process(_frame(samples))
    high = EnergyBeatDetector(_config(sensitivity=1.0)).process(_frame(samples))

    assert low.transients == ()
    assert len(high.transients) == 1


def test_energy_detector_is_invariant_to_contiguous_input_chunking() -> None:
    from lights_audio_engine.detectors.energy import EnergyBeatDetector

    samples = np.zeros(100)
    samples[40:60] = 0.8
    whole_detector = EnergyBeatDetector(_config())
    chunked_detector = EnergyBeatDetector(_config())

    whole = whole_detector.process(_frame(samples)).transients
    first = chunked_detector.process(_frame(samples[:47])).transients
    second = chunked_detector.process(_frame(samples[47:], start_time_seconds=0.047)).transients

    assert first + second == whole


def test_energy_detector_reset_replays_the_same_transient() -> None:
    from lights_audio_engine.detectors.energy import EnergyBeatDetector

    samples = np.zeros(100)
    samples[40:60] = 0.8
    detector = EnergyBeatDetector(_config())

    first = detector.process(_frame(samples)).transients
    detector.reset()
    replay = detector.process(_frame(samples)).transients

    assert replay == first


def test_energy_detector_rejects_a_mismatched_sample_rate() -> None:
    from lights_audio_engine.detectors.energy import EnergyBeatDetector

    detector = EnergyBeatDetector(_config())

    with pytest.raises(ValueError, match="sample rate"):
        detector.process(_frame(np.zeros(100), sample_rate_hz=2_000))


def test_noop_drop_detector_never_invents_events() -> None:
    from lights_audio_engine.detectors.drop import NoOpDropDetector

    detector = NoOpDropDetector()

    assert detector.process(_frame(np.ones(100))) == ()
    detector.reset()
    assert detector.process(_frame(np.ones(100), start_time_seconds=0.1)) == ()
