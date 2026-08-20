"""Export exact Aubio benchmark event times for manual waveform review."""

from __future__ import annotations

from pathlib import Path


def write_sonic_visualiser_time_instants(
    path: Path, timestamps_seconds: tuple[float, ...], *, label: str = "aubio"
) -> None:
    """Write Sonic Visualiser Time Instants rows as timestamp/label pairs."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{timestamp:.6f}\t{label}\n" for timestamp in timestamps_seconds),
        encoding="utf-8",
    )
