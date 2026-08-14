"""Explicit JSON report serialization for M2C results."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from lights_audio_engine.evaluation.bakeoff import BakeoffReport


def write_report(path: Path, report: BakeoffReport) -> None:
    """Write the complete per-track and pooled bake-off report."""

    payload = asdict(report)
    payload["schema_version"] = 1
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
