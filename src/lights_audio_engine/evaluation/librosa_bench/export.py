from __future__ import annotations

from os import replace
from pathlib import Path
from tempfile import NamedTemporaryFile

from lights_audio_engine.evaluation.reference import ReferenceFormatError, parse_reference


def _require_librosa_suffix(path: Path, suffix: str) -> Path:
    path = Path(path)
    if not str(path).endswith(suffix):
        raise ValueError(f"Librosa output path must use the {suffix} suffix")
    return path


def write_audacity_labels(
    path: Path, beat_times_seconds: tuple[float, ...], *, label_prefix: str = "librosa-beat"
) -> None:
    _require_librosa_suffix(path, ".librosa-beats.txt").write_text(
        "".join(
            f"{time:.6f}\t{time:.6f}\t{label_prefix}-{index}\n"
            for index, time in enumerate(beat_times_seconds, 1)
        ),
        encoding="utf-8",
    )


def write_candidate_reference(path: Path, beat_times_seconds: tuple[float, ...]) -> None:
    _require_librosa_suffix(path, ".librosa-candidate.txt").write_text(
        "".join(f"{time:.6f}\tbeat\n" for time in beat_times_seconds), encoding="utf-8"
    )


def convert_audacity_export_to_reference(audacity_path: Path, output_path: Path) -> None:
    times: list[float] = []
    for number, line in enumerate(Path(audacity_path).read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split("\t")
        if len(fields) != 3:
            raise ReferenceFormatError(f"line {number} must be a three-column Audacity label")
        try:
            start, end = float(fields[0]), float(fields[1])
        except ValueError as error:
            raise ReferenceFormatError(f"line {number} has an invalid timestamp") from error
        if abs(start - end) > 1e-9:
            raise ReferenceFormatError(f"line {number} is a range label, not a point label")
        times.append(start)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=output_path.parent,
    ) as temporary:
        temporary_path = Path(temporary.name)
        temporary.write("".join(f"{time:.6f}\tbeat\n" for time in times))
    try:
        parse_reference(temporary_path)
        replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
