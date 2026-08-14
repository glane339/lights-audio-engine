from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def test_evaluation_cli_runs_dataset_manifest_and_writes_explicit_report(tmp_path: Path) -> None:
    from lights_audio_engine.evaluation.artifact import SegmentInfo, write_artifact
    from lights_audio_engine.evaluation.cli import main

    samples = np.zeros(96_000, dtype=np.float64)
    samples[48_000 + 24_000 : 48_000 + 24_960] = 0.8
    artifact = tmp_path / "holdout.npy"
    reference = tmp_path / "holdout.txt"
    dataset = tmp_path / "dataset.json"
    output = tmp_path / "report.json"
    write_artifact(
        artifact,
        samples,
        label="holdout",
        sample_rate_hz=48_000,
        segments=(
            SegmentInfo(0, 48_000, 0.0, None),
            SegmentInfo(48_000, 48_000, 10.0, "overflow"),
        ),
    )
    reference.write_text("0.500\tbeat\n", encoding="utf-8")
    dataset.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tracks": [
                    {
                        "label": "holdout",
                        "split": "holdout",
                        "artifact_path": artifact.name,
                        "reference_path": reference.name,
                        "segment_index": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert main([str(dataset), "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["recommendation"] in {
        "baseline",
        "broadband",
        "multiband",
        "NO PRODUCTION CANDIDATE YET",
    }
    assert report["ranking"]
    assert {item["block_size_frames"] for item in report["candidate_reports"]} == {
        240,
        480,
        960,
    }
    assert all(item["tracks"][0]["split"] == "holdout" for item in report["candidate_reports"])
    assert all(item["tracks"][0]["segment_index"] == 1 for item in report["candidate_reports"])
    assert "p95_absolute_timing_error_seconds" in report["candidate_reports"][0]["holdout_metrics"]
    assert "p95_decision_latency_seconds" in report["candidate_reports"][0]["holdout_metrics"]
