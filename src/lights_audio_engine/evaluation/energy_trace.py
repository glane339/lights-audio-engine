"""Observational per-window traces for the production energy detector."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from unittest.mock import patch

from lights_audio_engine.config import AudioEngineConfig
from lights_audio_engine.detectors import energy as energy_module
from lights_audio_engine.detectors.energy import EnergyBeatDetector
from lights_audio_engine.evaluation.artifact import read_artifact
from lights_audio_engine.evaluation.replay_source import ReplayAudioSource
from lights_audio_engine.models import AudioFrame, Float64Samples


@dataclass(frozen=True, slots=True)
class EnergyWindowTrace:
    """One completed production energy-detector window observation."""

    window_index: int
    start_time_seconds: float
    end_time_seconds: float
    rms: float
    baseline: float
    threshold: float
    is_active: bool


@dataclass(frozen=True, slots=True)
class EnergyTraceReport:
    """Evaluation-only observations from one sequential energy-detector replay."""

    artifact_label: str | None
    sample_rate_hz: int
    segment_index: int | None
    windows: tuple[EnergyWindowTrace, ...]


class _TracingEnergyBeatDetector(EnergyBeatDetector):
    """Observe production window values without changing production detector code."""

    def __init__(self, config: AudioEngineConfig) -> None:
        super().__init__(config)
        self._traced_windows: list[EnergyWindowTrace] = []
        self._last_window_rms = 0.0
        self._baseline_used_by_production = 0.0
        self._next_window_index = 0
        self._production_rms = EnergyBeatDetector._rms

    @property
    def traced_windows(self) -> tuple[EnergyWindowTrace, ...]:
        return tuple(self._traced_windows)

    def process(self, frame: AudioFrame):
        """Run unmodified production processing while observing its median call."""

        with (
            patch.object(energy_module, "median", self._observe_production_median),
            patch.object(
                EnergyBeatDetector,
                "_rms",
                staticmethod(self._observe_production_rms),
            ),
        ):
            return super().process(frame)

    def reset(self) -> None:
        """Reset both production state and evaluation-only observations."""

        super().reset()
        self._traced_windows.clear()
        self._last_window_rms = 0.0
        self._baseline_used_by_production = 0.0
        self._next_window_index = 0

    def _observe_production_rms(self, samples: Float64Samples) -> float:
        rms = self._production_rms(samples)
        self._last_window_rms = rms
        return rms

    def _threshold(self) -> float:
        self._baseline_used_by_production = 0.0
        threshold = super()._threshold()
        sample_rate_hz = self._config.expected_sample_rate_hz
        start_time_seconds = (
            self._stream_start_time_seconds or 0.0
        ) + self._next_window_index * self._window_size / sample_rate_hz
        self._traced_windows.append(
            EnergyWindowTrace(
                window_index=self._next_window_index,
                start_time_seconds=start_time_seconds,
                end_time_seconds=start_time_seconds + self._window_size / sample_rate_hz,
                rms=self._last_window_rms,
                baseline=self._baseline_used_by_production,
                threshold=threshold,
                is_active=self._last_window_rms >= threshold,
            )
        )
        self._next_window_index += 1
        return threshold

    def _observe_production_median(self, values: Iterable[float]) -> float:
        baseline = float(median(values))
        self._baseline_used_by_production = baseline
        return baseline


def trace_energy_frames(
    frames: Iterable[AudioFrame], *, config: AudioEngineConfig
) -> EnergyTraceReport:
    """Trace completed production windows while processing sequential audio frames."""

    detector = _TracingEnergyBeatDetector(config)
    for frame in frames:
        detector.process(frame)
    return EnergyTraceReport(
        artifact_label=None,
        sample_rate_hz=config.expected_sample_rate_hz,
        segment_index=None,
        windows=detector.traced_windows,
    )


def trace_energy_artifact(
    sample_path: Path,
    *,
    config: AudioEngineConfig | None = None,
    block_size_frames: int = 240,
    segment_index: int | None = None,
) -> EnergyTraceReport:
    """Replay one authoritative PCM artifact through the production detector trace."""

    artifact = read_artifact(sample_path)
    trace_config = config or AudioEngineConfig()
    report = trace_energy_frames(
        ReplayAudioSource(
            artifact,
            block_size_frames=block_size_frames,
            segment_index=segment_index,
        ).stream(),
        config=trace_config,
    )
    return EnergyTraceReport(
        artifact_label=artifact.label,
        sample_rate_hz=report.sample_rate_hz,
        segment_index=0 if segment_index is None else segment_index,
        windows=report.windows,
    )


def write_energy_trace_report(path: Path, report: EnergyTraceReport) -> None:
    """Write an evaluation-only energy trace as explicit JSON."""

    payload = asdict(report)
    payload["schema_version"] = 1
    payload["kind"] = "production_energy_window_trace"
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
