"""Causal trailing-window multiband novelty detector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from lights_audio_engine.evaluation.detectors.broadband_onset import (
    BroadbandOnsetConfig,
    CausalNoveltyDetector,
)


@dataclass(frozen=True, slots=True)
class MultibandOnsetConfig(BroadbandOnsetConfig):
    """Frozen M2C multiband parameters with a zero-padded 5 ms causal hop."""

    novelty_floor: float = 0.00002
    fft_size_frames: int = 960
    band_edges_hz: tuple[float, ...] = (40.0, 180.0, 2_000.0, 8_000.0, 16_000.0)


class MultibandOnsetDetector(CausalNoveltyDetector):
    """Fuse positive novelty across bands before one shared peak picker."""

    def __init__(self, config: MultibandOnsetConfig | None = None) -> None:
        self._multiband_config = config or MultibandOnsetConfig()
        super().__init__(self._multiband_config)

    def _reset_feature_state(self) -> None:
        pass

    def _feature(self, hop: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        fft_size = self._multiband_config.fft_size_frames
        spectrum = np.abs(np.fft.rfft(hop, n=fft_size)) / hop.size
        frequencies = np.fft.rfftfreq(fft_size, 1.0 / self._multiband_config.sample_rate_hz)
        features: list[float] = []
        edges = self._multiband_config.band_edges_hz
        for low, high in zip(edges[:-1], edges[1:], strict=True):
            band = spectrum[(frequencies >= low) & (frequencies < high)]
            features.append(float(np.sqrt(np.mean(np.square(band)))) if band.size else 0.0)
        return np.asarray(features, dtype=np.float64)

    def _strength_scale(self) -> float:
        return 0.005
