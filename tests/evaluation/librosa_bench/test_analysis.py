from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from pytest import MonkeyPatch


def _write_artifact(
    tmp_path: Path, *, sample_rate_hz: int = 48_000, sample_count: int = 48_000
) -> Path:
    from lights_audio_engine.evaluation.artifact import write_artifact

    path = tmp_path / "track.npy"
    write_artifact(
        path, np.zeros(sample_count, dtype=np.float64), label="track", sample_rate_hz=sample_rate_hz
    )
    return path


def test_analyze_segment_passes_exact_sample_rate_with_no_default(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    from lights_audio_engine.evaluation.artifact import read_artifact
    from lights_audio_engine.evaluation.librosa_bench import analysis
    from lights_audio_engine.evaluation.librosa_bench._librosa_backend import OnsetEnvelope

    recorded: dict[str, int] = {}

    def fake_estimate(
        samples: npt.NDArray[np.float64], sample_rate_hz: int
    ) -> tuple[float, tuple[float, ...]]:
        recorded["tempo_sr"] = sample_rate_hz
        return 120.0, (0.5, 1.0)

    def fake_envelope(samples: npt.NDArray[np.float64], sample_rate_hz: int) -> OnsetEnvelope:
        recorded["envelope_sr"] = sample_rate_hz
        return OnsetEnvelope((0.1, 0.2), 512, sample_rate_hz / 512)

    monkeypatch.setattr(analysis, "estimate_tempo_and_beats", fake_estimate)
    monkeypatch.setattr(analysis, "onset_envelope", fake_envelope)
    result = analysis.analyze_segment(
        read_artifact(_write_artifact(tmp_path, sample_rate_hz=44_100))
    )
    assert (recorded["tempo_sr"], recorded["envelope_sr"], result.sample_rate_hz) == (
        44_100,
        44_100,
        44_100,
    )
    assert result.tempo_bpm == 120.0 and result.beat_times_seconds == (0.5, 1.0)


def test_analyze_segment_requires_explicit_index_for_discontinuous_artifact(tmp_path: Path) -> None:
    from lights_audio_engine.evaluation.artifact import SegmentInfo, read_artifact, write_artifact
    from lights_audio_engine.evaluation.librosa_bench.analysis import analyze_segment

    path = tmp_path / "split.npy"
    write_artifact(
        path,
        np.zeros(20, dtype=np.float64),
        label="split",
        sample_rate_hz=1_000,
        segments=(SegmentInfo(0, 10, 0.0, None), SegmentInfo(10, 10, 4.0, "overflow")),
    )
    with pytest.raises(ValueError, match="explicit segment index"):
        analyze_segment(read_artifact(path))


def test_analyze_segment_rejects_out_of_range_index(tmp_path: Path) -> None:
    from lights_audio_engine.evaluation.artifact import read_artifact
    from lights_audio_engine.evaluation.librosa_bench.analysis import analyze_segment

    with pytest.raises(ValueError, match="segment index is out of range"):
        analyze_segment(read_artifact(_write_artifact(tmp_path)), segment_index=5)


def test_analyze_segment_raises_on_non_finite_backend_result(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    from lights_audio_engine.evaluation.artifact import read_artifact
    from lights_audio_engine.evaluation.librosa_bench import analysis
    from lights_audio_engine.evaluation.librosa_bench._librosa_backend import OnsetEnvelope

    def fake_estimate(
        samples: npt.NDArray[np.float64], sample_rate_hz: int
    ) -> tuple[float, tuple[float, ...]]:
        return float("nan"), ()

    def fake_envelope(samples: npt.NDArray[np.float64], sample_rate_hz: int) -> OnsetEnvelope:
        return OnsetEnvelope((), 512, 1.0)

    monkeypatch.setattr(analysis, "estimate_tempo_and_beats", fake_estimate)
    monkeypatch.setattr(analysis, "onset_envelope", fake_envelope)
    with pytest.raises(analysis.LibrosaAnalysisError, match="finite"):
        analysis.analyze_segment(read_artifact(_write_artifact(tmp_path)))


def test_onset_envelope_roundtrips_through_sidecar_files(tmp_path: Path) -> None:
    from lights_audio_engine.evaluation.librosa_bench._librosa_backend import OnsetEnvelope
    from lights_audio_engine.evaluation.librosa_bench.analysis import (
        read_onset_envelope,
        write_onset_envelope,
    )

    envelope = OnsetEnvelope((0.1, 0.4, 0.2), 512, 86.1328125)
    path = tmp_path / "track.envelope.npy"
    write_onset_envelope(path, envelope)
    assert read_onset_envelope(path) == envelope


def test_analyze_segment_end_to_end_with_real_librosa(tmp_path: Path) -> None:
    pytest.importorskip("librosa")
    from lights_audio_engine.evaluation.artifact import read_artifact
    from lights_audio_engine.evaluation.librosa_bench.analysis import analyze_segment

    result = analyze_segment(read_artifact(_write_artifact(tmp_path, sample_count=48_000 * 2)))
    assert result.tempo_bpm >= 0.0
    assert result.onset_summary.frame_count == len(result.onset_envelope.values)
