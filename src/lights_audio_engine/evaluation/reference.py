"""Human-readable beat point-label parsing."""

from __future__ import annotations

from math import isfinite
from pathlib import Path


class ReferenceFormatError(ValueError):
    """A beat-reference label file is malformed."""


def parse_reference(path: Path) -> tuple[float, ...]:
    """Parse strictly increasing ``timestamp<TAB>label`` point events."""

    timestamps: list[float] = []
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = raw_line.split("\t")
        if len(fields) != 2 or not fields[1].strip():
            raise ReferenceFormatError(f"line {line_number} must be tab-separated point label")
        try:
            timestamp = float(fields[0])
        except ValueError as error:
            raise ReferenceFormatError(f"line {line_number} has an invalid timestamp") from error
        if not isfinite(timestamp):
            raise ReferenceFormatError(f"line {line_number} timestamp must be finite")
        if timestamp < 0.0:
            raise ReferenceFormatError(f"line {line_number} timestamp must be non-negative")
        if timestamps and timestamp <= timestamps[-1]:
            raise ReferenceFormatError("reference timestamps must be strictly increasing")
        timestamps.append(timestamp)
    return tuple(timestamps)
