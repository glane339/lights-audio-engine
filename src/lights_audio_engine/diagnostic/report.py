"""Small offline report for diagnostic beat-characterization JSONL logs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from statistics import median, quantiles
from typing import cast


class ReportError(ValueError):
    """A JSONL log cannot be interpreted safely."""


@dataclass(frozen=True, slots=True)
class StreamAnalysis:
    """Compact metrics for one logical stream within one session log."""

    path: Path
    label: str
    stream_ordinal: int
    beat_count: int
    observed_duration_seconds: float
    median_interval_seconds: float | None
    interval_iqr_seconds: float | None
    long_gap_count: int
    short_interval_count: int
    bpm_median: float | None
    bpm_min: float | None
    bpm_max: float | None
    tempo_relation: str | None
    mean_process_seconds: float | None
    max_process_seconds: float | None


def _number(record: dict[str, object], name: str, *, line: str) -> float:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportError(f"{line}: {name} must be numeric")
    parsed = float(value)
    if not isfinite(parsed):
        raise ReportError(f"{line}: {name} must be finite")
    return parsed


def _non_negative_number(record: dict[str, object], name: str, *, line: str) -> float:
    value = _number(record, name, line=line)
    if value < 0.0:
        raise ReportError(f"{line}: {name} must be non-negative")
    return value


def _positive_number(record: dict[str, object], name: str, *, line: str) -> float:
    value = _number(record, name, line=line)
    if value <= 0.0:
        raise ReportError(f"{line}: {name} must be positive")
    return value


def _integer(
    record: dict[str, object],
    name: str,
    *,
    line: str,
    minimum: int = 0,
) -> int:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReportError(f"{line}: {name} must be an integer of at least {minimum}")
    return value


def _string(record: dict[str, object], name: str, *, line: str) -> str:
    value = record.get(name)
    if not isinstance(value, str):
        raise ReportError(f"{line}: {name} must be a string")
    return value


def _stream_ordinal(record: dict[str, object], *, line: str) -> int:
    return _integer(record, "stream_ordinal", line=line)


def _validate_session(record: dict[str, object], *, line: str) -> None:
    _string(record, "label", line=line)
    _string(record, "wall_clock_start", line=line)
    _integer(record, "device_index", line=line)
    _string(record, "device_name", line=line)
    _integer(record, "sample_rate_hz", line=line, minimum=1)
    _integer(record, "channels", line=line, minimum=1)
    _integer(record, "block_size_frames", line=line, minimum=1)
    sensitivity = _number(record, "sensitivity", line=line)
    if not 0.0 <= sensitivity <= 1.0:
        raise ReportError(f"{line}: sensitivity must be between 0.0 and 1.0")
    min_bpm = _positive_number(record, "min_bpm", line=line)
    max_bpm = _positive_number(record, "max_bpm", line=line)
    if max_bpm <= min_bpm:
        raise ReportError(f"{line}: max_bpm must be greater than min_bpm")
    analysis_window = _positive_number(record, "analysis_window_seconds", line=line)
    energy_history = _positive_number(record, "energy_history_seconds", line=line)
    if energy_history < analysis_window:
        raise ReportError(
            f"{line}: energy_history_seconds must be at least analysis_window_seconds"
        )
    _integer(record, "bpm_history_size", line=line, minimum=3)


def _validate_beat(record: dict[str, object], *, line: str) -> None:
    _non_negative_number(record, "host_time_seconds", line=line)
    _non_negative_number(record, "beat_timestamp_seconds", line=line)
    _stream_ordinal(record, line=line)
    _integer(record, "beat_index", line=line)
    interval = record.get("interval_seconds")
    if interval is not None:
        _positive_number(record, "interval_seconds", line=line)
    strength = _number(record, "strength", line=line)
    if not 0.0 <= strength <= 1.0:
        raise ReportError(f"{line}: strength must be between 0.0 and 1.0")
    if record.get("bpm") is not None:
        _positive_number(record, "bpm", line=line)


def _validate_discontinuity(record: dict[str, object], *, line: str) -> None:
    _non_negative_number(record, "host_time_seconds", line=line)
    _stream_ordinal(record, line=line)
    reason = _string(record, "reason", line=line)
    if reason not in {"overflow", "dropped_block", "sample_rate_change", "restart"}:
        raise ReportError(f"{line}: reason is not recognized")
    detail = record.get("detail")
    if detail is not None and not isinstance(detail, str):
        raise ReportError(f"{line}: detail must be a string or null")
    _integer(record, "frames_emitted_before", line=line)


def _validate_summary(record: dict[str, object], *, line: str) -> None:
    _non_negative_number(record, "host_time_seconds", line=line)
    _stream_ordinal(record, line=line)
    if record.get("stream_time_seconds") is not None:
        _non_negative_number(record, "stream_time_seconds", line=line)
    frame_count = _integer(record, "frame_count", line=line, minimum=1)
    stream_frame_count = _integer(record, "stream_frame_count", line=line, minimum=1)
    if stream_frame_count < frame_count:
        raise ReportError(f"{line}: stream_frame_count must be at least frame_count")
    mean_level = _number(record, "mean_input_level", line=line)
    max_level = _number(record, "max_input_level", line=line)
    if not 0.0 <= mean_level <= max_level <= 1.0:
        raise ReportError(f"{line}: input levels must satisfy 0 <= mean <= max <= 1")
    mean_process = _non_negative_number(record, "mean_process_seconds", line=line)
    max_process = _non_negative_number(record, "max_process_seconds", line=line)
    if max_process < mean_process:
        raise ReportError(f"{line}: max_process_seconds must be at least mean_process_seconds")


def _validate_record(record: dict[str, object], *, line: str) -> None:
    record_type = record["type"]
    if record_type == "session":
        _validate_session(record, line=line)
    elif record_type == "beat":
        _validate_beat(record, line=line)
    elif record_type == "discontinuity":
        _validate_discontinuity(record, line=line)
    else:
        _validate_summary(record, line=line)


def _load(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ReportError(f"{path}: {error}") from error
    for line_number, raw_line in enumerate(lines, start=1):
        context = f"{path}:{line_number}"
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ReportError(f"{context}: invalid JSON: {error.msg}") from error
        if not isinstance(value, dict):
            raise ReportError(f"{context}: each record must be a JSON object")
        record = cast(dict[str, object], value)
        if record.get("schema") != 1:
            raise ReportError(f"{context}: unsupported schema")
        if record.get("type") not in {"session", "beat", "discontinuity", "summary"}:
            raise ReportError(f"{context}: unsupported record type")
        if records and record.get("type") == "session":
            raise ReportError(f"{context}: session record is only allowed first")
        record["_line_context"] = context
        _validate_record(record, line=context)
        records.append(record)
    if not records or records[0].get("type") != "session":
        raise ReportError(f"{path}: first record must be a session record")
    return records[0], records[1:]


def _iqr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    first, _, third = quantiles(values, n=4, method="inclusive")
    return third - first


def _tempo_relation(observed_bpm: float | None, expected_bpm: float | None) -> str | None:
    if observed_bpm is None or expected_bpm is None:
        return None
    candidates = (
        (expected_bpm, "on-tempo"),
        (expected_bpm / 2.0, "half-time"),
        (expected_bpm * 2.0, "double-time"),
    )
    target, label = min(candidates, key=lambda candidate: abs(observed_bpm - candidate[0]))
    if abs(observed_bpm - target) / target <= 0.12:
        return label
    return "other"


def analyze_file(path: Path, *, expected_bpm: float | None = None) -> tuple[StreamAnalysis, ...]:
    """Parse and summarize each logical stream in one schema-v1 log."""

    if expected_bpm is not None and (not isfinite(expected_bpm) or expected_bpm <= 0.0):
        raise ReportError("expected BPM must be finite and positive")
    session, records = _load(path)
    label = cast(str, session["label"])
    stream_starts: dict[int, float] = {0: 0.0}
    for record in records:
        if record.get("type") == "discontinuity":
            context = cast(str, record["_line_context"])
            ordinal = _stream_ordinal(record, line=context)
            stream_starts[ordinal + 1] = _number(record, "host_time_seconds", line=context)
    ordinals = sorted(
        {
            _stream_ordinal(record, line=cast(str, record["_line_context"]))
            for record in records
            if record.get("type") != "session"
        }
    )
    analyses: list[StreamAnalysis] = []
    for ordinal in ordinals:
        stream_records = [record for record in records if record.get("stream_ordinal") == ordinal]
        beats = [record for record in stream_records if record.get("type") == "beat"]
        summaries = [record for record in stream_records if record.get("type") == "summary"]
        intervals: list[float] = []
        bpm_values: list[float] = []
        host_times: list[float] = []
        for record in stream_records:
            context = cast(str, record["_line_context"])
            host_times.append(_number(record, "host_time_seconds", line=context))
        for beat in beats:
            context = cast(str, beat["_line_context"])
            interval = beat.get("interval_seconds")
            if interval is not None:
                parsed_interval = _number(beat, "interval_seconds", line=context)
                if parsed_interval <= 0.0:
                    raise ReportError(f"{context}: interval_seconds must be positive or null")
                intervals.append(parsed_interval)
            bpm = beat.get("bpm")
            if bpm is not None:
                parsed_bpm = _number(beat, "bpm", line=context)
                if parsed_bpm <= 0.0:
                    raise ReportError(f"{context}: bpm must be positive or null")
                bpm_values.append(parsed_bpm)
        median_interval = median(intervals) if intervals else None
        derived_bpm = 60.0 / median_interval if median_interval is not None else None
        long_gaps = (
            sum(interval > median_interval * 1.5 for interval in intervals)
            if median_interval is not None
            else 0
        )
        short_intervals = (
            sum(interval < median_interval * 0.75 for interval in intervals)
            if median_interval is not None
            else 0
        )
        weighted_process_total = 0.0
        process_frames = 0
        process_maxima: list[float] = []
        for summary in summaries:
            context = cast(str, summary["_line_context"])
            frames = _integer(summary, "frame_count", line=context, minimum=1)
            mean_process = _number(summary, "mean_process_seconds", line=context)
            max_process = _number(summary, "max_process_seconds", line=context)
            weighted_process_total += mean_process * frames
            process_frames += frames
            process_maxima.append(max_process)
        analyses.append(
            StreamAnalysis(
                path=path,
                label=label,
                stream_ordinal=ordinal,
                beat_count=len(beats),
                observed_duration_seconds=(
                    max(host_times) - stream_starts.get(ordinal, min(host_times))
                    if host_times
                    else 0.0
                ),
                median_interval_seconds=median_interval,
                interval_iqr_seconds=_iqr(intervals) if intervals else None,
                long_gap_count=long_gaps,
                short_interval_count=short_intervals,
                bpm_median=median(bpm_values) if bpm_values else None,
                bpm_min=min(bpm_values) if bpm_values else None,
                bpm_max=max(bpm_values) if bpm_values else None,
                tempo_relation=_tempo_relation(derived_bpm, expected_bpm),
                mean_process_seconds=(
                    weighted_process_total / process_frames if process_frames else None
                ),
                max_process_seconds=max(process_maxima) if process_maxima else None,
            )
        )
    return tuple(analyses)


def _format(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _print_analysis(analysis: StreamAnalysis) -> None:
    print(f"{analysis.path} [{analysis.label or 'unlabeled'}] stream {analysis.stream_ordinal}")
    print(
        f"  beats={analysis.beat_count} observed_host_duration="
        f"{analysis.observed_duration_seconds:.3f}s median_interval="
        f"{_format(analysis.median_interval_seconds)}s iqr="
        f"{_format(analysis.interval_iqr_seconds)}s"
    )
    print(
        f"  suspicious_long_gaps={analysis.long_gap_count} "
        f"potential_short_doubles={analysis.short_interval_count}"
    )
    print(
        f"  bpm_median={_format(analysis.bpm_median)} "
        f"bpm_range={_format(analysis.bpm_min)}..{_format(analysis.bpm_max)} "
        f"expected_relation={analysis.tempo_relation or 'n/a'}"
    )
    print(
        f"  process_mean={_format(analysis.mean_process_seconds, 6)}s "
        f"process_max={_format(analysis.max_process_seconds, 6)}s"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize beat-characterization JSONL logs.")
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--expected-bpm", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Print per-stream metrics and return a report-oriented exit code."""

    try:
        options = _parser().parse_args(argv)
        expected_bpm = cast(float | None, options.expected_bpm)
        logs = cast(list[Path], options.logs)
        analyses = [
            analysis for path in logs for analysis in analyze_file(path, expected_bpm=expected_bpm)
        ]
    except (ReportError, OSError) as error:
        print(f"Report error: {error}", file=sys.stderr)
        return 2
    for analysis in analyses:
        _print_analysis(analysis)
    if len(logs) > 1:
        print("Comparison")
        print("  log | stream | beats | median interval | long gaps | short doubles | relation")
        for analysis in analyses:
            print(
                f"  {analysis.label or analysis.path.stem} | {analysis.stream_ordinal} | "
                f"{analysis.beat_count} | {_format(analysis.median_interval_seconds)}s | "
                f"{analysis.long_gap_count} | {analysis.short_interval_count} | "
                f"{analysis.tempo_relation or 'n/a'}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
