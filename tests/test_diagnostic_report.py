from __future__ import annotations

import json
from pathlib import Path

import pytest

from lights_audio_engine.diagnostic.report import ReportError, analyze_file, main


def _session_record(label: str) -> dict[str, object]:
    return {
        "type": "session",
        "schema": 1,
        "label": label,
        "wall_clock_start": "2026-08-13T23:15:00+00:00",
        "device_index": 12,
        "device_name": "Microphone (Realtek Audio)",
        "sample_rate_hz": 48_000,
        "channels": 1,
        "block_size_frames": 960,
        "sensitivity": 0.5,
        "min_bpm": 50.0,
        "max_bpm": 240.0,
        "analysis_window_seconds": 0.02,
        "energy_history_seconds": 2.0,
        "bpm_history_size": 8,
    }


def _write_log(path: Path, intervals: list[float], *, bpm: float) -> None:
    records: list[dict[str, object]] = [_session_record(path.stem)]
    timestamp = 0.0
    records.append(
        {
            "type": "beat",
            "schema": 1,
            "host_time_seconds": 0.1,
            "beat_timestamp_seconds": timestamp,
            "stream_ordinal": 0,
            "beat_index": 0,
            "interval_seconds": None,
            "strength": 0.8,
            "bpm": None,
        }
    )
    for index, interval in enumerate(intervals, start=1):
        timestamp += interval
        records.append(
            {
                "type": "beat",
                "schema": 1,
                "host_time_seconds": timestamp + 0.1,
                "beat_timestamp_seconds": timestamp,
                "stream_ordinal": 0,
                "beat_index": index,
                "interval_seconds": interval,
                "strength": 0.8,
                "bpm": bpm,
            }
        )
    records.append(
        {
            "type": "summary",
            "schema": 1,
            "host_time_seconds": timestamp + 0.2,
            "stream_ordinal": 0,
            "stream_time_seconds": timestamp,
            "frame_count": 50,
            "stream_frame_count": 50,
            "mean_input_level": 0.1,
            "max_input_level": 0.3,
            "mean_process_seconds": 0.001,
            "max_process_seconds": 0.003,
        }
    )
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def test_steady_log_reports_expected_interval_and_processing_stats(tmp_path: Path) -> None:
    path = tmp_path / "steady.jsonl"
    _write_log(path, [0.5, 0.5, 0.5, 0.5], bpm=120.0)

    (analysis,) = analyze_file(path, expected_bpm=120.0)

    assert analysis.beat_count == 5
    assert analysis.observed_duration_seconds == pytest.approx(2.2)
    assert analysis.median_interval_seconds == 0.5
    assert analysis.interval_iqr_seconds == 0.0
    assert analysis.long_gap_count == 0
    assert analysis.short_interval_count == 0
    assert analysis.bpm_median == 120.0
    assert analysis.tempo_relation == "on-tempo"
    assert analysis.mean_process_seconds == 0.001
    assert analysis.max_process_seconds == 0.003


def test_report_flags_long_gaps_and_short_potential_doubles(tmp_path: Path) -> None:
    gaps = tmp_path / "gaps.jsonl"
    doubles = tmp_path / "doubles.jsonl"
    _write_log(gaps, [0.5, 0.5, 0.5, 1.2], bpm=120.0)
    _write_log(doubles, [0.5, 0.5, 0.5, 0.2], bpm=120.0)

    (gap_analysis,) = analyze_file(gaps)
    (double_analysis,) = analyze_file(doubles)

    assert gap_analysis.long_gap_count == 1
    assert double_analysis.short_interval_count == 1


@pytest.mark.parametrize(
    ("interval", "bpm", "relation"),
    [(1.0, 60.0, "half-time"), (0.25, 240.0, "double-time")],
)
def test_expected_bpm_identifies_obvious_tempo_relation(
    tmp_path: Path,
    interval: float,
    bpm: float,
    relation: str,
) -> None:
    path = tmp_path / f"{relation}.jsonl"
    _write_log(path, [interval, interval, interval], bpm=bpm)

    (analysis,) = analyze_file(path, expected_bpm=120.0)

    assert analysis.tempo_relation == relation


def test_malformed_jsonl_fails_with_file_and_line_context(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text(json.dumps(_session_record("bad")) + "\nnot-json\n", encoding="utf-8")

    with pytest.raises(ReportError, match=r"broken\.jsonl:2: invalid JSON"):
        analyze_file(path)


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (
            '{"type":"beat","schema":1,"host_time_seconds":0.1,'
            '"beat_timestamp_seconds":0.0,"stream_ordinal":0,"beat_index":0,'
            '"interval_seconds":null,"bpm":null}',
            "strength must be numeric",
        ),
        (
            '{"type":"summary","schema":1,"host_time_seconds":1.0,'
            '"stream_ordinal":0,"stream_time_seconds":1.0,"frame_count":50,'
            '"stream_frame_count":50,'
            '"mean_input_level":0.1,"max_input_level":0.2,'
            '"mean_process_seconds":NaN,"max_process_seconds":0.003}',
            "mean_process_seconds must be finite",
        ),
    ],
)
def test_malformed_typed_record_fails_instead_of_producing_metrics(
    tmp_path: Path,
    record: str,
    message: str,
) -> None:
    path = tmp_path / "malformed-record.jsonl"
    path.write_text(
        json.dumps(_session_record("bad")) + "\n" + record + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ReportError, match=message):
        analyze_file(path)


def test_duplicate_session_record_fails_clearly(tmp_path: Path) -> None:
    path = tmp_path / "duplicate-session.jsonl"
    session = json.dumps(_session_record("duplicate"))
    path.write_text(session + "\n" + session + "\n", encoding="utf-8")

    with pytest.raises(ReportError, match=r"duplicate-session\.jsonl:2: session record"):
        analyze_file(path)


def test_non_finite_expected_bpm_fails_clearly(tmp_path: Path) -> None:
    path = tmp_path / "steady.jsonl"
    _write_log(path, [0.5, 0.5, 0.5], bpm=120.0)

    with pytest.raises(ReportError, match="expected BPM must be finite and positive"):
        analyze_file(path, expected_bpm=float("nan"))


def test_multiple_logs_print_compact_comparison(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_log(first, [0.5, 0.5, 0.5], bpm=120.0)
    _write_log(second, [1.0, 1.0, 1.0], bpm=60.0)

    exit_code = main([str(first), str(second), "--expected-bpm", "120"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Comparison" in captured.out
    assert "first" in captured.out
    assert "second" in captured.out
    assert "half-time" in captured.out
