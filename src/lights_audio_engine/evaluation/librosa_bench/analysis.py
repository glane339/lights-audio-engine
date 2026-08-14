from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lights_audio_engine.evaluation.artifact import PcmArtifact
from lights_audio_engine.evaluation.librosa_bench._librosa_backend import (
    OnsetEnvelope,
    estimate_tempo_and_beats,
    onset_envelope,
)


class LibrosaAnalysisError(ValueError):
    """An offline Librosa analysis result is invalid."""


@dataclass(frozen=True, slots=True)
class OnsetSummary:
    hop_length: int
    frame_rate_hz: float
    mean_strength: float
    max_strength: float
    frame_count: int


@dataclass(frozen=True, slots=True)
class LibrosaAnalysis:
    label: str
    segment_index: int
    sample_rate_hz: int
    tempo_bpm: float
    beat_times_seconds: tuple[float, ...]
    onset_summary: OnsetSummary
    onset_envelope: OnsetEnvelope


def _select_segment_samples(
    artifact: PcmArtifact, segment_index: int | None
) -> tuple[int, np.ndarray]:
    if segment_index is None:
        if len(artifact.segments) != 1:
            raise ValueError("a discontinuous artifact requires an explicit segment index")
        segment_index = 0
    if not 0 <= segment_index < len(artifact.segments):
        raise ValueError("segment index is out of range")
    segment = artifact.segments[segment_index]
    return segment_index, artifact.samples[
        segment.start_sample : segment.start_sample + segment.sample_count
    ]


def _validate(tempo_bpm: float, beat_times: tuple[float, ...], envelope: OnsetEnvelope) -> None:
    values = (tempo_bpm, *beat_times, *envelope.values, envelope.frame_rate_hz)
    if not all(np.isfinite(value) for value in values):
        raise LibrosaAnalysisError("Librosa analysis output must be finite")
    if (
        tempo_bpm < 0
        or any(value < 0 for value in beat_times)
        or any(value < 0 for value in envelope.values)
    ):
        raise LibrosaAnalysisError("Librosa analysis output must be non-negative")
    if envelope.hop_length <= 0 or envelope.frame_rate_hz <= 0:
        raise LibrosaAnalysisError("Librosa onset envelope metadata must be positive")


def analyze_segment(artifact: PcmArtifact, *, segment_index: int | None = None) -> LibrosaAnalysis:
    selected_index, samples = _select_segment_samples(artifact, segment_index)
    tempo_bpm, beat_times = estimate_tempo_and_beats(samples, artifact.sample_rate_hz)
    envelope = onset_envelope(samples, artifact.sample_rate_hz)
    _validate(tempo_bpm, beat_times, envelope)
    values = envelope.values
    summary = OnsetSummary(
        hop_length=envelope.hop_length,
        frame_rate_hz=envelope.frame_rate_hz,
        mean_strength=float(np.mean(values)) if values else 0.0,
        max_strength=max(values, default=0.0),
        frame_count=len(values),
    )
    return LibrosaAnalysis(
        label=artifact.label,
        segment_index=selected_index,
        sample_rate_hz=artifact.sample_rate_hz,
        tempo_bpm=tempo_bpm,
        beat_times_seconds=beat_times,
        onset_summary=summary,
        onset_envelope=envelope,
    )


def write_onset_envelope(path: Path, envelope: OnsetEnvelope) -> None:
    path = Path(path)
    if path.suffix.lower() != ".npy":
        raise LibrosaAnalysisError("onset envelope path must end in .npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(envelope.values, dtype=np.float64), allow_pickle=False)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "kind": "librosa_onset_envelope",
                "hop_length": envelope.hop_length,
                "frame_rate_hz": envelope.frame_rate_hz,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def read_onset_envelope(path: Path) -> OnsetEnvelope:
    path = Path(path)
    try:
        manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        values = np.load(path, allow_pickle=False)
        if manifest.get("kind") != "librosa_onset_envelope":
            raise ValueError
        return OnsetEnvelope(
            tuple(float(value) for value in values),
            int(manifest["hop_length"]),
            float(manifest["frame_rate_hz"]),
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise LibrosaAnalysisError(f"unable to read onset envelope: {error}") from error
