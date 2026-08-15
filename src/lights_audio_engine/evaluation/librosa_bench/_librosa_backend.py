from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt


class LibrosaUnavailableError(ImportError):
    """Raised when the optional Librosa dependency is unavailable."""


@dataclass(frozen=True, slots=True)
class OnsetEnvelope:
    values: tuple[float, ...]
    hop_length: int
    frame_rate_hz: float


def _import_librosa() -> Any:
    try:
        import librosa  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        message = (
            "Librosa is required for offline benchmarking; "
            'install it with pip install -e ".[librosa]".'
        )
        raise LibrosaUnavailableError(message) from error
    return cast(Any, librosa)


def estimate_tempo_and_beats(
    samples: npt.NDArray[np.float64], sample_rate_hz: int
) -> tuple[float, tuple[float, ...]]:
    librosa = _import_librosa()
    tempo, beat_times = librosa.beat.beat_track(y=samples, sr=sample_rate_hz, units="time")
    normalized_tempo = float(np.asarray(tempo).reshape(-1)[0])
    normalized_beats = tuple(float(value) for value in np.asarray(beat_times).reshape(-1))
    return normalized_tempo, normalized_beats


def onset_envelope(samples: npt.NDArray[np.float64], sample_rate_hz: int) -> OnsetEnvelope:
    librosa = _import_librosa()
    hop_length = 512
    values = librosa.onset.onset_strength(y=samples, sr=sample_rate_hz, hop_length=hop_length)
    return OnsetEnvelope(
        values=tuple(float(value) for value in np.asarray(values).reshape(-1)),
        hop_length=hop_length,
        frame_rate_hz=sample_rate_hz / hop_length,
    )
