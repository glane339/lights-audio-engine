from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest
from pytest import MonkeyPatch


def test_estimate_tempo_and_beats_raises_when_librosa_is_unavailable(
    monkeypatch: MonkeyPatch,
) -> None:
    from lights_audio_engine.evaluation.librosa_bench._librosa_backend import (
        LibrosaUnavailableError,
        estimate_tempo_and_beats,
    )

    monkeypatch.setitem(sys.modules, "librosa", None)
    with pytest.raises(LibrosaUnavailableError, match=r'pip install -e "\.\[librosa\]"'):
        estimate_tempo_and_beats(np.zeros(48_000, dtype=np.float64), 48_000)


def test_onset_envelope_raises_when_librosa_is_unavailable(monkeypatch: MonkeyPatch) -> None:
    from lights_audio_engine.evaluation.librosa_bench._librosa_backend import (
        LibrosaUnavailableError,
        onset_envelope,
    )

    monkeypatch.setitem(sys.modules, "librosa", None)
    with pytest.raises(LibrosaUnavailableError):
        onset_envelope(np.zeros(48_000, dtype=np.float64), 48_000)


def test_estimate_tempo_and_beats_passes_exact_sample_rate_no_default() -> None:
    librosa = pytest.importorskip("librosa")
    from lights_audio_engine.evaluation.librosa_bench._librosa_backend import (
        estimate_tempo_and_beats,
    )

    calls: list[int] = []
    original = librosa.beat.beat_track

    def _recording_beat_track(*, y: Any, sr: int, **kwargs: Any) -> tuple[Any, Any]:
        calls.append(sr)
        return original(y=y, sr=sr, **kwargs)

    import unittest.mock as mock

    samples = np.zeros(48_000, dtype=np.float64)
    with mock.patch.object(librosa.beat, "beat_track", _recording_beat_track):
        estimate_tempo_and_beats(samples, 44_100)

    assert calls == [44_100]


def test_real_librosa_returns_finite_tempo_and_ordered_beats() -> None:
    pytest.importorskip("librosa")
    from lights_audio_engine.evaluation.librosa_bench._librosa_backend import (
        estimate_tempo_and_beats,
        onset_envelope,
    )

    sample_rate_hz = 22_050
    duration_seconds = 4.0
    samples = np.zeros(round(duration_seconds * sample_rate_hz), dtype=np.float64)
    for beat_time_seconds in np.arange(0.5, duration_seconds, 0.5):
        start = round(beat_time_seconds * sample_rate_hz)
        samples[start : start + round(0.01 * sample_rate_hz)] = 0.8

    tempo_bpm, beat_times = estimate_tempo_and_beats(samples, sample_rate_hz)
    envelope = onset_envelope(samples, sample_rate_hz)

    assert tempo_bpm > 0.0
    assert list(beat_times) == sorted(beat_times)
    assert all(0.0 <= b <= duration_seconds for b in beat_times)
    assert envelope.hop_length > 0
    assert envelope.frame_rate_hz == pytest.approx(sample_rate_hz / envelope.hop_length)
    assert len(envelope.values) > 0
