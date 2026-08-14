"""Command line entry point for deterministic M2C bake-offs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

from lights_audio_engine.evaluation.artifact import ArtifactError
from lights_audio_engine.evaluation.bakeoff import DatasetSplit, TrackSpec, run_bakeoff
from lights_audio_engine.evaluation.reference import ReferenceFormatError
from lights_audio_engine.evaluation.report import write_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lights_audio_engine.evaluation",
        description="Run the deterministic offline M2C detector bake-off.",
    )
    parser.add_argument("dataset", type=Path, help="schema-versioned JSON dataset manifest")
    parser.add_argument("--output", type=Path, required=True, help="JSON report destination")
    parser.add_argument(
        "--block-sizes",
        type=int,
        nargs="+",
        default=(240, 480, 960),
        help="replay delivery sizes in frames",
    )
    parser.add_argument("--tolerance-ms", type=float, default=50.0)
    return parser


def _load_tracks(path: Path) -> tuple[TrackSpec, ...]:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
        raw = cast(dict[str, object], decoded) if isinstance(decoded, dict) else None
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            raise ValueError("dataset schema_version must be 1")
        items = raw.get("tracks")
        if not isinstance(items, list) or not items:
            raise ValueError("dataset tracks must be a non-empty list")
        tracks: list[TrackSpec] = []
        dataset_directory = path.parent
        for item in cast(list[object], items):
            if not isinstance(item, dict):
                raise ValueError("each dataset track must be an object")
            values = cast(dict[str, object], item)
            artifact_path = Path(str(values["artifact_path"]))
            reference_path = Path(str(values["reference_path"]))
            segment_index = values.get("segment_index")
            if segment_index is not None and (
                isinstance(segment_index, bool) or not isinstance(segment_index, int)
            ):
                raise ValueError("track segment_index must be an integer when supplied")
            tracks.append(
                TrackSpec(
                    label=str(values["label"]),
                    artifact_path=(
                        artifact_path
                        if artifact_path.is_absolute()
                        else dataset_directory / artifact_path
                    ),
                    reference_path=(
                        reference_path
                        if reference_path.is_absolute()
                        else dataset_directory / reference_path
                    ),
                    split=DatasetSplit(str(values["split"])),
                    segment_index=segment_index,
                )
            )
        return tuple(tracks)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid dataset manifest: {error}") from error


def main(argv: list[str] | None = None) -> int:
    """Run a manifest-defined bake-off and return a developer-oriented exit code."""

    try:
        options = _parser().parse_args(argv)
        block_sizes = tuple(cast(list[int], options.block_sizes))
        if any(size <= 0 for size in block_sizes):
            raise ValueError("block sizes must be positive")
        tolerance_seconds = float(options.tolerance_ms) / 1_000.0
        report = run_bakeoff(
            _load_tracks(cast(Path, options.dataset)),
            block_sizes=block_sizes,
            tolerance_seconds=tolerance_seconds,
        )
        write_report(cast(Path, options.output), report)
    except SystemExit as error:
        return int(error.code) if isinstance(error.code, int) else 2
    except (ArtifactError, ReferenceFormatError, OSError, ValueError) as error:
        print(f"M2C evaluation error: {error}", file=sys.stderr)
        return 2
    return 0
