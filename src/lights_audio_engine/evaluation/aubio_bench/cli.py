"""Command line entry point for the causal, advisory Aubio benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

from lights_audio_engine.evaluation.artifact import ArtifactError, read_artifact
from lights_audio_engine.evaluation.aubio_bench.adapter import (
    AubioOnsetCandidate,
    AubioUnavailableError,
)
from lights_audio_engine.evaluation.aubio_bench.export import (
    write_sonic_visualiser_time_instants,
)
from lights_audio_engine.evaluation.aubio_bench.report import build_report, write_report
from lights_audio_engine.evaluation.replay_source import ReplayAudioSource
from lights_audio_engine.evaluation.runner import run_candidate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lights_audio_engine.evaluation.aubio_bench",
        description="Run the causal, advisory Aubio onset benchmark on one M2C artifact.",
    )
    parser.add_argument("artifact", type=Path, help="schema-versioned M2C .npy artifact")
    parser.add_argument("--output", type=Path, required=True, help="JSON report destination")
    parser.add_argument(
        "--sonic-visualiser",
        type=Path,
        help="optional Sonic Visualiser Time Instants output using the report event timestamps",
    )
    parser.add_argument("--segment-index", type=int)
    parser.add_argument(
        "--delivery-block-size",
        type=int,
        default=240,
        help="replay delivery size in frames; Aubio internally receives fixed 256-frame hops",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one causal replay benchmark and return a developer-oriented exit code."""

    try:
        options = _parser().parse_args(argv)
        if options.delivery_block_size <= 0:
            raise ValueError("delivery block size must be positive")
        artifact = read_artifact(cast(Path, options.artifact))
        segment_index = cast(int | None, options.segment_index)
        replay = ReplayAudioSource(
            artifact,
            block_size_frames=cast(int, options.delivery_block_size),
            segment_index=segment_index,
        )
        run = run_candidate(AubioOnsetCandidate(), replay.stream())
        selected_index = 0 if segment_index is None else segment_index
        report = build_report(
            run,
            label=artifact.label,
            segment_index=selected_index,
            sample_rate_hz=artifact.sample_rate_hz,
        )
        write_report(cast(Path, options.output), report)
        if options.sonic_visualiser is not None:
            write_sonic_visualiser_time_instants(
                cast(Path, options.sonic_visualiser),
                tuple(event.timestamp_seconds for event in report.events),
            )
    except SystemExit as error:
        return int(error.code) if isinstance(error.code, int) else 2
    except (ArtifactError, AubioUnavailableError, OSError, ValueError) as error:
        print(f"Aubio benchmark error: {error}", file=sys.stderr)
        return 2
    return 0
