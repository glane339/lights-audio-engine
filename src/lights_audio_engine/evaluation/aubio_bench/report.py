"""Advisory report serialization for causal Aubio replay observations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from lights_audio_engine.evaluation.aubio_bench.adapter import (
    HOP_SIZE_FRAMES,
    METHOD,
    WINDOW_SIZE_FRAMES,
)
from lights_audio_engine.evaluation.runner import CandidateRun


@dataclass(frozen=True, slots=True)
class AubioEvent:
    timestamp_seconds: float
    strength: float
    emitted_stream_time_seconds: float
    decision_latency_seconds: float


@dataclass(frozen=True, slots=True)
class AubioBenchmarkReport:
    kind: str
    advisory_only: bool
    production_candidate: bool
    ground_truth: bool
    label: str
    segment_index: int
    sample_rate_hz: int
    method: str
    window_size_frames: int
    hop_size_frames: int
    processing_seconds: float
    events: tuple[AubioEvent, ...]


def build_report(
    run: CandidateRun,
    *,
    label: str,
    segment_index: int,
    sample_rate_hz: int,
) -> AubioBenchmarkReport:
    """Build a report containing native causal event and emission observations."""

    return AubioBenchmarkReport(
        kind="aubio_causal_advisory_benchmark",
        advisory_only=True,
        production_candidate=False,
        ground_truth=False,
        label=label,
        segment_index=segment_index,
        sample_rate_hz=sample_rate_hz,
        method=METHOD,
        window_size_frames=WINDOW_SIZE_FRAMES,
        hop_size_frames=HOP_SIZE_FRAMES,
        processing_seconds=run.processing_seconds,
        events=tuple(
            AubioEvent(
                timestamp_seconds=item.event.timestamp_seconds,
                strength=item.event.strength,
                emitted_stream_time_seconds=item.emitted_stream_time_seconds,
                decision_latency_seconds=item.decision_latency_seconds,
            )
            for item in run.detections
        ),
    )


def write_report(path: Path, report: AubioBenchmarkReport) -> None:
    """Write a versioned advisory observation report."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": 1, **asdict(report)}, indent=2) + "\n", encoding="utf-8"
    )
