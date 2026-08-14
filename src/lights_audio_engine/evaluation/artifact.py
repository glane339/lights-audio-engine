"""Exact float64 PCM artifacts and schema-versioned manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

SCHEMA_VERSION = 1


class ArtifactError(ValueError):
    """An authoritative PCM artifact is invalid or corrupt."""


@dataclass(frozen=True, slots=True)
class SegmentInfo:
    """One contiguous logical stream stored in the sample array."""

    start_sample: int
    sample_count: int
    original_start_time_seconds: float
    discontinuity_before: str | None


@dataclass(frozen=True, slots=True, eq=False)
class PcmArtifact:
    """Loaded exact detector-input samples and their interpretation metadata."""

    samples: npt.NDArray[np.float64]
    label: str
    sample_rate_hz: int
    frame_lengths: tuple[int, ...]
    segments: tuple[SegmentInfo, ...]
    checksum_sha256: str


def _manifest_path(sample_path: Path) -> Path:
    return sample_path.with_suffix(".json")


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ArtifactError(f"unable to read authoritative sample data: {error}") from error
    return digest.hexdigest()


def _int_value(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError
    return value


def _float_value(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError
    return float(value)


def _validate_loaded_metadata(
    samples: npt.NDArray[np.float64],
    frame_lengths: tuple[int, ...],
    segments: tuple[SegmentInfo, ...],
) -> None:
    if not bool(np.isfinite(samples).all()) or bool((np.abs(samples) > 1.0).any()):
        raise ArtifactError("authoritative samples must be finite normalized float64 values")
    if frame_lengths and (
        any(length <= 0 for length in frame_lengths) or sum(frame_lengths) != samples.size
    ):
        raise ArtifactError("artifact frame lengths do not match sample data")
    expected_start = 0
    for index, segment in enumerate(segments):
        if segment.start_sample != expected_start or segment.sample_count < 0:
            raise ArtifactError("artifact segment boundaries are not contiguous")
        if samples.size and segment.sample_count == 0:
            raise ArtifactError("artifact segment must not be empty")
        if segment.original_start_time_seconds < 0.0 or not np.isfinite(
            segment.original_start_time_seconds
        ):
            raise ArtifactError("artifact segment timing must be finite and non-negative")
        if index == 0 and segment.discontinuity_before is not None:
            raise ArtifactError("first artifact segment cannot follow a discontinuity")
        if index > 0 and segment.discontinuity_before is None:
            raise ArtifactError("later artifact segments require discontinuity metadata")
        expected_start += segment.sample_count
    if not segments or expected_start != samples.size:
        raise ArtifactError("artifact segments do not cover sample data")


def write_artifact(
    sample_path: Path,
    samples: npt.NDArray[np.float64],
    *,
    label: str,
    sample_rate_hz: int,
    frame_lengths: tuple[int, ...] = (),
    segments: tuple[SegmentInfo, ...] | None = None,
) -> None:
    """Write exact NumPy samples plus the adjacent JSON manifest."""

    path = Path(sample_path)
    if path.suffix.lower() != ".npy":
        raise ArtifactError("authoritative sample path must end in .npy")
    if samples.dtype != np.float64 or samples.ndim != 1:
        raise ArtifactError("authoritative samples must be a one-dimensional float64 array")
    if not bool(np.isfinite(samples).all()):
        raise ArtifactError("authoritative samples must be finite")
    if sample_rate_hz <= 0:
        raise ArtifactError("sample rate must be positive")
    if not label:
        raise ArtifactError("label must not be empty")
    if frame_lengths and sum(frame_lengths) != samples.size:
        raise ArtifactError("frame lengths must sum to sample count")
    actual_segments = segments or (SegmentInfo(0, int(samples.size), 0.0, None),)
    if sum(segment.sample_count for segment in actual_segments) != samples.size:
        raise ArtifactError("segment sample counts must cover the sample array")

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, samples, allow_pickle=False)
    checksum = _checksum(path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "label": label,
        "sample_rate_hz": sample_rate_hz,
        "sample_count": int(samples.size),
        "dtype": "float64",
        "frame_lengths": list(frame_lengths),
        "segments": [asdict(segment) for segment in actual_segments],
        "sample_data_sha256": checksum,
        "authoritative": True,
    }
    _manifest_path(path).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_artifact(sample_path: Path) -> PcmArtifact:
    """Load an artifact after verifying its schema, checksum, and array contract."""

    path = Path(sample_path)
    try:
        raw_manifest = json.loads(_manifest_path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"unable to read artifact manifest: {error}") from error
    if not isinstance(raw_manifest, dict):
        raise ArtifactError("artifact manifest must be a JSON object")
    manifest = cast(dict[str, object], raw_manifest)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactError("unsupported artifact schema version")
    if manifest.get("dtype") != "float64":
        raise ArtifactError("artifact manifest dtype must be float64")
    expected_checksum = manifest.get("sample_data_sha256")
    if not isinstance(expected_checksum, str) or _checksum(path) != expected_checksum:
        raise ArtifactError("authoritative sample checksum mismatch")
    try:
        loaded = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ArtifactError(f"unable to read authoritative sample data: {error}") from error
    if loaded.dtype != np.float64 or loaded.ndim != 1:
        raise ArtifactError("authoritative sample data must be one-dimensional float64")
    sample_count = manifest.get("sample_count")
    if sample_count != int(loaded.size):
        raise ArtifactError("artifact sample count does not match manifest")
    try:
        label = manifest["label"]
        sample_rate = manifest["sample_rate_hz"]
        raw_lengths = manifest["frame_lengths"]
        raw_segments = manifest["segments"]
        if not isinstance(label, str) or not isinstance(sample_rate, int):
            raise TypeError
        if not isinstance(raw_lengths, list) or not isinstance(raw_segments, list):
            raise TypeError
        frame_lengths = tuple(_int_value(value) for value in cast(list[object], raw_lengths))
        parsed_segments: list[SegmentInfo] = []
        for raw_segment in cast(list[object], raw_segments):
            if not isinstance(raw_segment, dict):
                raise TypeError
            segment = cast(dict[str, object], raw_segment)
            discontinuity = segment["discontinuity_before"]
            if discontinuity is not None and not isinstance(discontinuity, str):
                raise TypeError
            parsed_segments.append(
                SegmentInfo(
                    start_sample=_int_value(segment["start_sample"]),
                    sample_count=_int_value(segment["sample_count"]),
                    original_start_time_seconds=_float_value(
                        segment["original_start_time_seconds"]
                    ),
                    discontinuity_before=discontinuity,
                )
            )
        segments = tuple(parsed_segments)
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactError("artifact manifest metadata is malformed") from error
    _validate_loaded_metadata(loaded, frame_lengths, segments)
    loaded.setflags(write=False)
    return PcmArtifact(
        samples=loaded,
        label=label,
        sample_rate_hz=sample_rate,
        frame_lengths=frame_lengths,
        segments=segments,
        checksum_sha256=expected_checksum,
    )
