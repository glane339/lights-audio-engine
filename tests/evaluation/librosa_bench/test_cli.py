from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _write_pulse_artifact(tmp_path: Path, beat_times: tuple[float, ...] = (0.5, 1.0, 1.5)) -> Path:
    from lights_audio_engine.evaluation.artifact import write_artifact

    sample_count = round((max(beat_times) + 0.5) * 48_000)
    samples = np.zeros(sample_count, dtype=np.float64)
    for time_seconds in beat_times:
        start = round(time_seconds * 48_000)
        samples[start : start + 960] = 0.8
    path = tmp_path / "track.npy"
    write_artifact(path, samples, label="track", sample_rate_hz=48_000)
    return path


def test_cli_missing_librosa_exits_with_documented_message(tmp_path: Path, monkeypatch) -> None:
    import sys

    from lights_audio_engine.evaluation.librosa_bench.cli import main

    monkeypatch.setitem(sys.modules, "librosa", None)
    artifact = _write_pulse_artifact(tmp_path)
    output = tmp_path / "report.json"

    exit_code = main([str(artifact), "--output", str(output)])

    assert exit_code == 2
    assert not output.exists()


def test_cli_end_to_end_with_real_librosa_produces_advisory_report(tmp_path: Path) -> None:
    pytest.importorskip("librosa")
    from lights_audio_engine.evaluation.librosa_bench.cli import main

    artifact = _write_pulse_artifact(tmp_path)
    reference = tmp_path / "human.txt"
    reference.write_text("0.500\tbeat\n1.000\tbeat\n1.500\tbeat\n", encoding="utf-8")
    output = tmp_path / "report.json"
    audacity_path = tmp_path / "track.librosa-beats.txt"
    candidate_path = tmp_path / "track.librosa-candidate.txt"
    envelope_path = tmp_path / "track.envelope.npy"

    exit_code = main(
        [
            str(artifact),
            "--output",
            str(output),
            "--audacity-labels",
            str(audacity_path),
            "--candidate-reference",
            str(candidate_path),
            "--envelope-npy",
            str(envelope_path),
            "--human-reference",
            str(reference),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["advisory_only"] is True
    assert payload["production_candidate"] is False
    assert audacity_path.exists()
    assert candidate_path.exists()
    assert envelope_path.exists()
    assert payload["human_comparison"] is not None
    assert payload["onset_envelope_path"] == str(envelope_path)


def test_cli_reports_bad_artifact_with_exit_code_two(tmp_path: Path) -> None:
    from lights_audio_engine.evaluation.librosa_bench.cli import main

    missing = tmp_path / "does-not-exist.npy"
    exit_code = main([str(missing), "--output", str(tmp_path / "report.json")])

    assert exit_code == 2
