"""Observational per-window traces for the production energy detector."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from math import ceil
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Literal
from unittest.mock import patch

import numpy as np

from lights_audio_engine.config import AudioEngineConfig
from lights_audio_engine.detectors import energy as energy_module
from lights_audio_engine.detectors.energy import EnergyBeatDetector
from lights_audio_engine.evaluation.artifact import read_artifact
from lights_audio_engine.evaluation.replay_source import ReplayAudioSource
from lights_audio_engine.models import AudioFrame, Float64Samples

EnergyThresholdFormulation = Literal[
    "production-control",
    "c1-additive-margin",
    "c3-bounded-multiplicative",
    "c2-normalized-excess",
]

_FORMULATIONS: tuple[EnergyThresholdFormulation, ...] = (
    "production-control",
    "c1-additive-margin",
    "c3-bounded-multiplicative",
    "c2-normalized-excess",
)
_EVALUATION_SENSITIVITIES = (0.5, 0.7, 1.0)
_C2_BASELINE_FLOOR_HYPOTHESIS = 0.01


@dataclass(frozen=True, slots=True)
class EnergyCandidateWindow:
    """One evaluation-only candidate decision over a traced RMS window."""

    window_index: int
    start_time_seconds: float
    end_time_seconds: float
    rms: float
    baseline: float
    decision_statistic: float
    decision_boundary: float
    is_active: bool


@dataclass(frozen=True, slots=True)
class EnergyCandidateMetrics:
    """Summary metrics that preserve candidate and refractory counts separately."""

    decision_statistic_median: float | None
    decision_statistic_p95: float | None
    decision_statistic_maximum: float | None
    decision_boundary_median: float | None
    decision_boundary_p95: float | None
    decision_boundary_maximum: float | None
    crossing_window_count: int
    crossing_window_fraction: float
    rising_edge_count: int
    emitted_event_count: int
    refractory_suppressed_candidate_count: int
    emitted_interval_minimum_seconds: float | None
    emitted_interval_median_seconds: float | None
    emitted_interval_count_below_half_second: int
    active_window_count: int
    active_window_density: float
    candidate_processing_seconds: float
    full_replay_processing_seconds: float


@dataclass(frozen=True, slots=True)
class EnergyCandidateResult:
    """One formulation/sensitivity comparison result, not a production recommendation."""

    formulation: EnergyThresholdFormulation
    sensitivity: float
    decision_statistic_label: str
    decision_boundary_label: str
    evaluation_hypothesis: str
    windows: tuple[EnergyCandidateWindow, ...]
    metrics: EnergyCandidateMetrics


@dataclass(frozen=True, slots=True)
class EnergyCandidateReport:
    """Deterministic matrix of evaluation hypotheses over one energy trace."""

    artifact_label: str | None
    sample_rate_hz: int
    segment_index: int | None
    history_seconds: float
    max_bpm: float
    hypothesis_notice: str
    results: tuple[EnergyCandidateResult, ...]


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


def evaluate_energy_trace(
    trace: EnergyTraceReport,
    *,
    config: AudioEngineConfig,
    formulation: EnergyThresholdFormulation,
) -> EnergyCandidateResult:
    """Evaluate one threshold hypothesis over pre-window history from a production trace."""

    history_windows = max(1, ceil(config.energy_history_seconds / config.analysis_window_seconds))
    history: deque[float] = deque(maxlen=history_windows)
    windows: list[EnergyCandidateWindow] = []
    started = perf_counter()
    for trace_window in trace.windows:
        baseline = median(history) if history else 0.0
        statistic, boundary, statistic_label, boundary_label, hypothesis = _candidate_decision(
            trace_window.rms, baseline, config.sensitivity, formulation
        )
        windows.append(
            EnergyCandidateWindow(
                window_index=trace_window.window_index,
                start_time_seconds=trace_window.start_time_seconds,
                end_time_seconds=trace_window.end_time_seconds,
                rms=trace_window.rms,
                baseline=baseline,
                decision_statistic=statistic,
                decision_boundary=boundary,
                is_active=statistic >= boundary,
            )
        )
        history.append(trace_window.rms)
    processing_seconds = perf_counter() - started
    result_windows = tuple(windows)
    statistic_label = _statistic_label(formulation)
    boundary_label = _boundary_label(formulation)
    hypothesis = _hypothesis(formulation)
    return EnergyCandidateResult(
        formulation=formulation,
        sensitivity=config.sensitivity,
        decision_statistic_label=statistic_label,
        decision_boundary_label=boundary_label,
        evaluation_hypothesis=hypothesis,
        windows=result_windows,
        metrics=_candidate_metrics(result_windows, config.max_bpm, processing_seconds),
    )


def evaluate_energy_artifact(
    sample_path: Path,
    *,
    config: AudioEngineConfig | None = None,
    block_size_frames: int = 240,
    segment_index: int | None = None,
) -> EnergyCandidateReport:
    """Replay an artifact once, then evaluate the fixed formulation/sensitivity matrix."""

    base_config = config or AudioEngineConfig(max_bpm=240.0)
    trace_started = perf_counter()
    trace = trace_energy_artifact(
        sample_path,
        config=base_config,
        block_size_frames=block_size_frames,
        segment_index=segment_index,
    )
    trace_replay_processing_seconds = perf_counter() - trace_started
    results = tuple(
        replace(
            result,
            metrics=replace(
                result.metrics,
                full_replay_processing_seconds=(
                    trace_replay_processing_seconds + result.metrics.candidate_processing_seconds
                ),
            ),
        )
        for result in (
            evaluate_energy_trace(
                trace,
                config=replace(base_config, sensitivity=sensitivity),
                formulation=formulation,
            )
            for formulation in _FORMULATIONS
            for sensitivity in _EVALUATION_SENSITIVITIES
        )
    )
    return EnergyCandidateReport(
        artifact_label=trace.artifact_label,
        sample_rate_hz=trace.sample_rate_hz,
        segment_index=trace.segment_index,
        history_seconds=base_config.energy_history_seconds,
        max_bpm=base_config.max_bpm,
        hypothesis_notice=(
            "Evaluation hypotheses only; these constants and formulations are not "
            "production recommendations."
        ),
        results=results,
    )


def write_energy_candidate_report(path: Path, report: EnergyCandidateReport) -> None:
    """Write an explicit JSON report for evaluation-only threshold comparisons."""

    payload = asdict(report)
    payload["schema_version"] = 1
    payload["kind"] = "energy_threshold_candidate_evaluation"
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _candidate_decision(
    rms: float,
    baseline: float,
    sensitivity: float,
    formulation: EnergyThresholdFormulation,
) -> tuple[float, float, str, str, str]:
    absolute_gate = 0.30 - 0.25 * sensitivity
    if formulation == "production-control":
        multiplier = 3.0 - 1.5 * sensitivity
        return (
            rms,
            max(absolute_gate, baseline * multiplier),
            "rms",
            "rms_threshold",
            "Control: unchanged production threshold equation.",
        )
    if formulation == "c1-additive-margin":
        coefficient = 1.5 - sensitivity
        return (
            rms,
            absolute_gate + coefficient * baseline,
            "rms",
            "rms_threshold",
            "C1 evaluation hypothesis: absolute_gate + (1.5 - sensitivity) * baseline.",
        )
    if formulation == "c3-bounded-multiplicative":
        multiplier = 1.5 - 0.45 * sensitivity
        return (
            rms,
            max(absolute_gate, baseline * multiplier),
            "rms",
            "rms_threshold",
            "C3 evaluation hypothesis: max(absolute_gate, baseline * "
            "(1.5 - 0.45 * sensitivity)); multipliers are 1.275, 1.185, and 1.05 "
            "at sensitivities 0.5, 0.7, and 1.0.",
        )
    baseline_floor = _C2_BASELINE_FLOOR_HYPOTHESIS
    contrast = (rms - baseline) / max(baseline, baseline_floor)
    return (
        contrast,
        0.50 - 0.30 * sensitivity,
        "normalized_transient_excess",
        "contrast_threshold",
        "C2 evaluation hypothesis: (rms - baseline) / max(baseline, 0.01) >= "
        "(0.50 - 0.30 * sensitivity); 0.01 is an evaluation-only low-baseline floor.",
    )


def _statistic_label(formulation: EnergyThresholdFormulation) -> str:
    return "normalized_transient_excess" if formulation == "c2-normalized-excess" else "rms"


def _boundary_label(formulation: EnergyThresholdFormulation) -> str:
    return "contrast_threshold" if formulation == "c2-normalized-excess" else "rms_threshold"


def _hypothesis(formulation: EnergyThresholdFormulation) -> str:
    return _candidate_decision(0.0, 0.0, 0.5, formulation)[4]


def _candidate_metrics(
    windows: tuple[EnergyCandidateWindow, ...], max_bpm: float, processing_seconds: float
) -> EnergyCandidateMetrics:
    statistics = tuple(window.decision_statistic for window in windows)
    boundaries = tuple(window.decision_boundary for window in windows)
    active_windows = tuple(window for window in windows if window.is_active)
    rising_edges: list[EnergyCandidateWindow] = []
    previous_active = False
    for window in windows:
        if window.is_active and not previous_active:
            rising_edges.append(window)
        previous_active = window.is_active

    emitted_times: list[float] = []
    suppressed = 0
    minimum_interval = 60.0 / max_bpm
    for window in rising_edges:
        timestamp = window.start_time_seconds
        if not emitted_times or timestamp - emitted_times[-1] >= minimum_interval:
            emitted_times.append(timestamp)
        else:
            suppressed += 1
    intervals = tuple(
        current - previous
        for previous, current in zip(emitted_times, emitted_times[1:], strict=False)
    )
    return EnergyCandidateMetrics(
        decision_statistic_median=_distribution_value(statistics, 0.5),
        decision_statistic_p95=_distribution_value(statistics, 0.95),
        decision_statistic_maximum=max(statistics) if statistics else None,
        decision_boundary_median=_distribution_value(boundaries, 0.5),
        decision_boundary_p95=_distribution_value(boundaries, 0.95),
        decision_boundary_maximum=max(boundaries) if boundaries else None,
        crossing_window_count=len(active_windows),
        crossing_window_fraction=len(active_windows) / len(windows) if windows else 0.0,
        rising_edge_count=len(rising_edges),
        emitted_event_count=len(emitted_times),
        refractory_suppressed_candidate_count=suppressed,
        emitted_interval_minimum_seconds=min(intervals) if intervals else None,
        emitted_interval_median_seconds=median(intervals) if intervals else None,
        emitted_interval_count_below_half_second=sum(interval < 0.5 for interval in intervals),
        active_window_count=len(active_windows),
        active_window_density=len(active_windows) / len(windows) if windows else 0.0,
        candidate_processing_seconds=processing_seconds,
        full_replay_processing_seconds=processing_seconds,
    )


def _distribution_value(values: tuple[float, ...], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))
