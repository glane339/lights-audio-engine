"""Command line entry point for the offline, advisory Librosa benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

from lights_audio_engine.evaluation.artifact import ArtifactError, read_artifact
from lights_audio_engine.evaluation.librosa_bench._librosa_backend import LibrosaUnavailableError
from lights_audio_engine.evaluation.librosa_bench.analysis import (
    LibrosaAnalysisError,
    analyze_segment,
    write_onset_envelope,
)
from lights_audio_engine.evaluation.librosa_bench.export import (
    write_audacity_labels,
    write_candidate_reference,
)
from lights_audio_engine.evaluation.librosa_bench.report import (
    build_report,
    score_against_human_reference,
    write_report,
)
from lights_audio_engine.evaluation.reference import ReferenceFormatError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lights_audio_engine.evaluation.librosa_bench",
        description="Run the offline, advisory Librosa benchmark for one artifact segment.",
    )
    parser.add_argument("artifact", type=Path, help="schema-versioned PCM artifact (.npy)")
    parser.add_argument("--output", type=Path, required=True, help="JSON report destination")
    parser.add_argument("--segment-index", type=int)
    parser.add_argument("--audacity-labels", type=Path)
    parser.add_argument("--candidate-reference", type=Path)
    parser.add_argument("--envelope-npy", type=Path)
    parser.add_argument("--human-reference", type=Path)
    parser.add_argument("--tolerance-ms", type=float, default=50.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one offline benchmark and return a developer-oriented exit code."""

    try:
        options = _parser().parse_args(argv)
        tolerance_seconds = float(options.tolerance_ms) / 1_000.0
        if tolerance_seconds < 0.0:
            raise ValueError("tolerance must be non-negative")
        artifact = read_artifact(cast(Path, options.artifact))
        analysis = analyze_segment(artifact, segment_index=options.segment_index)
        if options.audacity_labels is not None:
            write_audacity_labels(options.audacity_labels, analysis.beat_times_seconds)
        if options.candidate_reference is not None:
            write_candidate_reference(options.candidate_reference, analysis.beat_times_seconds)
        envelope_path = cast(Path | None, options.envelope_npy)
        if envelope_path is not None:
            write_onset_envelope(envelope_path, analysis.onset_envelope)
        human_reference = cast(Path | None, options.human_reference)
        human_comparison = (
            score_against_human_reference(
                human_reference,
                analysis.beat_times_seconds,
                tolerance_seconds=tolerance_seconds,
            )
            if human_reference is not None
            else None
        )
        report = build_report(
            analysis,
            human_comparison,
            onset_envelope_path=envelope_path,
        )
        write_report(cast(Path, options.output), report)
    except SystemExit as error:
        return int(error.code) if isinstance(error.code, int) else 2
    except (
        ArtifactError,
        ReferenceFormatError,
        LibrosaUnavailableError,
        LibrosaAnalysisError,
        OSError,
        ValueError,
    ) as error:
        print(f"Librosa benchmark error: {error}", file=sys.stderr)
        return 2
    return 0
