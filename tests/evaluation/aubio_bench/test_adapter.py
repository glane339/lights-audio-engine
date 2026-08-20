from __future__ import annotations

import sys
import types

import numpy as np
import pytest


class _FakeOnset:
    def __init__(self, timestamps: list[float]) -> None:
        self._timestamps = iter(timestamps)
        self.calls: list[np.ndarray] = []
        self._last_timestamp = 0.0

    def __call__(self, samples: np.ndarray) -> float:
        self.calls.append(samples.copy())
        try:
            self._last_timestamp = next(self._timestamps)
        except StopIteration:
            return 0.0
        return 1.0

    def get_last_s(self) -> float:
        return self._last_timestamp


def _install_fake_aubio(
    monkeypatch: pytest.MonkeyPatch, timestamps: list[float]
) -> list[_FakeOnset]:
    instances: list[_FakeOnset] = []

    def onset(method: str, window_size: int, hop_size: int, sample_rate_hz: int) -> _FakeOnset:
        assert (method, window_size, hop_size, sample_rate_hz) == ("default", 1024, 256, 48_000)
        instance = _FakeOnset(timestamps.copy())
        instances.append(instance)
        return instance

    monkeypatch.setitem(sys.modules, "aubio", types.SimpleNamespace(onset=onset))
    return instances


def test_candidate_requires_aubio_only_when_constructed(monkeypatch: pytest.MonkeyPatch) -> None:
    from lights_audio_engine.evaluation.aubio_bench.adapter import (
        AubioOnsetCandidate,
        AubioUnavailableError,
    )

    monkeypatch.setitem(sys.modules, "aubio", None)
    with pytest.raises(AubioUnavailableError, match=r"\.\[aubio\]"):
        AubioOnsetCandidate()


def test_candidate_feeds_full_native_hops_in_causal_order(monkeypatch: pytest.MonkeyPatch) -> None:
    from lights_audio_engine.evaluation.aubio_bench.adapter import AubioOnsetCandidate
    from lights_audio_engine.models import AudioFrame

    instances = _install_fake_aubio(monkeypatch, [])
    candidate = AubioOnsetCandidate()

    samples = np.linspace(-1.0, 1.0, 512, dtype=np.float64)
    assert candidate.process(AudioFrame(samples[:300], 48_000, 0.0)) == ()
    assert candidate.process(AudioFrame(samples[300:], 48_000, 300 / 48_000)) == ()

    assert len(instances) == 1
    assert len(instances[0].calls) == 2
    np.testing.assert_array_equal(instances[0].calls[0], samples[:256].astype(np.float32))
    np.testing.assert_array_equal(instances[0].calls[1], samples[256:].astype(np.float32))


def test_candidate_uses_aubio_timestamp_and_runner_emission_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lights_audio_engine.evaluation.aubio_bench.adapter import AubioOnsetCandidate
    from lights_audio_engine.evaluation.runner import run_candidate
    from lights_audio_engine.models import AudioFrame

    _install_fake_aubio(monkeypatch, [0.003])
    run = run_candidate(
        AubioOnsetCandidate(),
        (AudioFrame(np.zeros(384, dtype=np.float64), 48_000, 0.0),),
    )

    assert len(run.detections) == 1
    detection = run.detections[0]
    assert detection.event.timestamp_seconds == pytest.approx(0.003)
    assert detection.event.strength == 1.0
    assert detection.emitted_stream_time_seconds == pytest.approx(384 / 48_000)
    assert detection.decision_latency_seconds == pytest.approx(384 / 48_000 - 0.003)


def test_reset_discards_partial_hop_and_recreates_aubio_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lights_audio_engine.evaluation.aubio_bench.adapter import AubioOnsetCandidate
    from lights_audio_engine.models import AudioFrame

    instances = _install_fake_aubio(monkeypatch, [0.004])
    candidate = AubioOnsetCandidate()
    candidate.process(AudioFrame(np.zeros(128, dtype=np.float64), 48_000, 0.0))
    candidate.reset()

    events = candidate.process(AudioFrame(np.zeros(256, dtype=np.float64), 48_000, 0.0))

    assert len(instances) == 2
    assert len(instances[0].calls) == 0
    assert events[0].timestamp_seconds == pytest.approx(0.004)
