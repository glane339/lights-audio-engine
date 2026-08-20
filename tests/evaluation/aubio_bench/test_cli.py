from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_cli_missing_aubio_exits_with_documented_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys

    from lights_audio_engine.evaluation.artifact import write_artifact
    from lights_audio_engine.evaluation.aubio_bench.cli import main

    artifact = tmp_path / "track.npy"
    write_artifact(artifact, np.zeros(512, dtype=np.float64), label="track", sample_rate_hz=48_000)
    monkeypatch.setitem(sys.modules, "aubio", None)

    assert main([str(artifact), "--output", str(tmp_path / "report.json")]) == 2
    assert ".[aubio]" in capsys.readouterr().err
