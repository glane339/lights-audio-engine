from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_authoritative_artifact_round_trips_exact_float64_and_replays_cadence(
    tmp_path: Path,
) -> None:
    from lights_audio_engine.evaluation.artifact import read_artifact, write_artifact
    from lights_audio_engine.evaluation.replay_source import ReplayAudioSource

    samples = np.array([0.0, -0.0, 0.1, -0.25, 1.0, -1.0, 1.0 / 3.0], dtype=np.float64)
    path = tmp_path / "capture.npy"
    write_artifact(
        path,
        samples,
        label="synthetic",
        sample_rate_hz=1_000,
        frame_lengths=(3, 4),
    )

    artifact = read_artifact(path)
    assert np.array_equal(artifact.samples, samples)
    assert artifact.label == "synthetic"
    assert artifact.frame_lengths == (3, 4)
    assert artifact.segments[0].start_sample == 0
    assert artifact.segments[0].sample_count == 7

    frames = tuple(ReplayAudioSource(artifact, block_size_frames=3).stream())
    assert [frame.samples.size for frame in frames] == [3, 3, 1]
    assert [frame.start_time_seconds for frame in frames] == [0.0, 0.003, 0.006]
    assert np.array_equal(np.concatenate([frame.samples for frame in frames]), samples)


def test_artifact_checksum_and_truncation_fail_informatively(tmp_path: Path) -> None:
    from lights_audio_engine.evaluation.artifact import ArtifactError, read_artifact, write_artifact

    path = tmp_path / "capture.npy"
    write_artifact(path, np.arange(8, dtype=np.float64) / 10, label="x", sample_rate_hz=1_000)
    path.write_bytes(path.read_bytes()[:-5])

    with pytest.raises(ArtifactError, match="checksum|read"):
        read_artifact(path)


def test_artifact_rejects_manifest_dtype_mismatch(tmp_path: Path) -> None:
    from lights_audio_engine.evaluation.artifact import ArtifactError, read_artifact, write_artifact

    path = tmp_path / "capture.npy"
    write_artifact(path, np.zeros(8, dtype=np.float64), label="x", sample_rate_hz=1_000)
    manifest_path = path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dtype"] = "float32"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactError, match="dtype"):
        read_artifact(path)


def test_artifact_rejects_inconsistent_segment_boundaries(tmp_path: Path) -> None:
    from lights_audio_engine.evaluation.artifact import ArtifactError, read_artifact, write_artifact

    path = tmp_path / "capture.npy"
    write_artifact(path, np.zeros(8, dtype=np.float64), label="x", sample_rate_hz=1_000)
    manifest_path = path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["segments"][0]["start_sample"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactError, match="segment"):
        read_artifact(path)


def test_replay_rejects_silent_stitching_across_discontinuities(tmp_path: Path) -> None:
    from lights_audio_engine.evaluation.artifact import SegmentInfo, read_artifact, write_artifact
    from lights_audio_engine.evaluation.replay_source import ReplayAudioSource

    path = tmp_path / "capture.npy"
    write_artifact(
        path,
        np.zeros(10, dtype=np.float64),
        label="split",
        sample_rate_hz=1_000,
        segments=(
            SegmentInfo(0, 4, 0.0, None),
            SegmentInfo(4, 6, 3.5, "overflow"),
        ),
    )
    artifact = read_artifact(path)

    with pytest.raises(ValueError, match="segment"):
        ReplayAudioSource(artifact, block_size_frames=3)

    frames = tuple(ReplayAudioSource(artifact, block_size_frames=3, segment_index=1).stream())
    assert [frame.start_time_seconds for frame in frames] == [0.0, 0.003]
    assert sum(frame.samples.size for frame in frames) == 6
