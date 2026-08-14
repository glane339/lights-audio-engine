from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from lights_audio_engine.evaluation.bakeoff import BakeoffReport
from lights_audio_engine.evaluation.candidates import Candidate, DetectedOnset
from lights_audio_engine.models import AudioFrame


def _write_track(
    tmp_path: Path,
    label: str,
    beat_times: tuple[float, ...],
    *,
    split: str,
):
    from lights_audio_engine.evaluation.artifact import write_artifact
    from lights_audio_engine.evaluation.bakeoff import DatasetSplit, TrackSpec

    sample_count = round((max(beat_times, default=0.5) + 0.25) * 48_000)
    samples = np.zeros(sample_count, dtype=np.float64)
    for timestamp in beat_times:
        start = round(timestamp * 48_000)
        samples[start : start + 960] = 0.8
    artifact_path = tmp_path / f"{label}.npy"
    reference_path = tmp_path / f"{label}.txt"
    write_artifact(
        artifact_path,
        samples,
        label=label,
        sample_rate_hz=48_000,
        frame_lengths=(sample_count,),
    )
    reference_path.write_text(
        "".join(f"{timestamp:.6f}\tbeat\n" for timestamp in beat_times), encoding="utf-8"
    )
    return TrackSpec(label, artifact_path, reference_path, DatasetSplit(split))


def _accuracy_signature(report: BakeoffReport) -> tuple[object, ...]:
    candidate_reports = report.candidate_reports
    return tuple(
        (
            candidate.candidate_name,
            candidate.block_size_frames,
            tuple(
                (
                    track.label,
                    tuple(d.event.timestamp_seconds for d in track.detections),
                    track.metrics.true_positives,
                    track.metrics.false_positives,
                    track.metrics.false_negatives,
                    track.metrics.f1,
                    track.metrics.longest_missed_run,
                )
                for track in candidate.tracks
            ),
            candidate.holdout_metrics.f1,
        )
        for candidate in candidate_reports
    )


def test_synthetic_candidate_a_end_to_end_is_exact_and_repeatable(tmp_path: Path) -> None:
    from lights_audio_engine.evaluation.bakeoff import run_bakeoff
    from lights_audio_engine.evaluation.candidates import BaselineCandidate

    track = _write_track(tmp_path, "holdout-pulses", (0.5, 1.0, 1.5), split="holdout")
    factories = {"baseline": BaselineCandidate}

    first = run_bakeoff((track,), candidate_factories=factories, block_sizes=(240, 480, 960))
    second = run_bakeoff((track,), candidate_factories=factories, block_sizes=(240, 480, 960))

    assert _accuracy_signature(first) == _accuracy_signature(second)
    for candidate in first.candidate_reports:
        assert candidate.tracks[0].metrics.precision == 1.0
        assert candidate.tracks[0].metrics.recall == 1.0
        assert candidate.tracks[0].metrics.f1 == 1.0


def test_bakeoff_exposes_per_track_pooled_holdout_gaps_and_ranking(tmp_path: Path) -> None:
    from lights_audio_engine.evaluation.bakeoff import run_bakeoff

    development = _write_track(tmp_path, "development", (0.5, 1.0), split="development")
    holdout_good = _write_track(tmp_path, "holdout-good", (0.5, 1.0), split="holdout")
    holdout_gap = _write_track(tmp_path, "holdout-gap", (0.5, 1.0, 1.5), split="holdout")

    class ScheduledCandidate:
        def __init__(self, schedule: tuple[float, ...]) -> None:
            self.schedule = schedule
            self.next_index = 0

        def process(self, frame: AudioFrame) -> tuple[DetectedOnset, ...]:
            frame_end = frame.start_time_seconds + frame.samples.size / frame.sample_rate_hz
            events: list[DetectedOnset] = []
            while (
                self.next_index < len(self.schedule) and self.schedule[self.next_index] < frame_end
            ):
                timestamp = self.schedule[self.next_index]
                if timestamp >= frame.start_time_seconds:
                    events.append(DetectedOnset(timestamp, 1.0))
                self.next_index += 1
            return tuple(events)

        def reset(self) -> None:
            self.next_index = 0

    factories: dict[str, Callable[[], Candidate]] = {
        "good": lambda: ScheduledCandidate((0.5, 1.0, 1.5)),
        "gappy": lambda: ScheduledCandidate((0.5,)),
        "double": lambda: ScheduledCandidate((0.5, 0.51, 1.0, 1.5)),
    }
    report = run_bakeoff(
        (development, holdout_good, holdout_gap),
        candidate_factories=factories,
        block_sizes=(240,),
    )

    by_name = {candidate.candidate_name: candidate for candidate in report.candidate_reports}
    assert by_name["gappy"].holdout_metrics.longest_missed_run == 2
    assert by_name["double"].holdout_metrics.short_doubles == 2
    assert [track.metrics.recall for track in by_name["gappy"].tracks] == pytest.approx(
        (0.5, 0.5, 1 / 3)
    )
    assert by_name["good"].holdout_metrics.recall == 1.0
    assert report.ranking[0] == "good"
    assert report.recommendation == "good"
    assert {track.split.value for track in by_name["good"].tracks} == {
        "development",
        "holdout",
    }


def test_quality_gate_returns_no_candidate_for_track_catastrophe(tmp_path: Path) -> None:
    from lights_audio_engine.evaluation.bakeoff import run_bakeoff

    holdout = _write_track(tmp_path, "holdout", (0.5, 1.0, 1.5, 1.8), split="holdout")

    class OneEvent:
        def __init__(self) -> None:
            self.emitted = False

        def process(self, frame: AudioFrame) -> tuple[DetectedOnset, ...]:
            if not self.emitted and frame.start_time_seconds <= 0.5 < (
                frame.start_time_seconds + frame.samples.size / frame.sample_rate_hz
            ):
                self.emitted = True
                return (DetectedOnset(0.5, 1.0),)
            return ()

        def reset(self) -> None:
            self.emitted = False

    report = run_bakeoff((holdout,), candidate_factories={"bad": OneEvent}, block_sizes=(240,))
    assert report.recommendation == "NO PRODUCTION CANDIDATE YET"


def test_single_segment_track_without_segment_index_preserves_existing_behavior(
    tmp_path: Path,
) -> None:
    from lights_audio_engine.evaluation.bakeoff import run_bakeoff
    from lights_audio_engine.evaluation.candidates import BaselineCandidate

    track = _write_track(tmp_path, "single", (0.5,), split="holdout")

    report = run_bakeoff(
        (track,), candidate_factories={"baseline": BaselineCandidate}, block_sizes=(240,)
    )

    assert report.candidate_reports[0].tracks[0].metrics.f1 == 1.0


def test_bakeoff_requires_segment_selection_for_discontinuous_artifact(tmp_path: Path) -> None:
    from lights_audio_engine.evaluation.artifact import SegmentInfo, write_artifact
    from lights_audio_engine.evaluation.bakeoff import DatasetSplit, TrackSpec, run_bakeoff

    artifact_path = tmp_path / "split.npy"
    reference_path = tmp_path / "split.txt"
    write_artifact(
        artifact_path,
        np.zeros(20, dtype=np.float64),
        label="split",
        sample_rate_hz=1_000,
        segments=(SegmentInfo(0, 10, 0.0, None), SegmentInfo(10, 10, 4.0, "overflow")),
    )
    reference_path.write_text("", encoding="utf-8")
    track = TrackSpec("split", artifact_path, reference_path, DatasetSplit.HOLDOUT)

    with pytest.raises(ValueError, match="explicit segment index"):
        run_bakeoff(
            (track,), candidate_factories={"baseline": lambda: _NoEvents()}, block_sizes=(5,)
        )


def test_bakeoff_evaluates_explicit_first_segment(tmp_path: Path) -> None:
    from lights_audio_engine.evaluation.artifact import SegmentInfo, write_artifact
    from lights_audio_engine.evaluation.bakeoff import DatasetSplit, TrackSpec, run_bakeoff

    artifact_path = tmp_path / "split.npy"
    reference_path = tmp_path / "split.txt"
    samples = np.zeros(20, dtype=np.float64)
    samples[5] = 0.8
    write_artifact(
        artifact_path,
        samples,
        label="split",
        sample_rate_hz=1_000,
        segments=(SegmentInfo(0, 10, 0.0, None), SegmentInfo(10, 10, 4.0, "overflow")),
    )
    reference_path.write_text("0.005\tbeat\n", encoding="utf-8")
    track = TrackSpec("split", artifact_path, reference_path, DatasetSplit.HOLDOUT, segment_index=0)

    report = run_bakeoff(
        (track,), candidate_factories={"pulse": _PulseCandidate}, block_sizes=(10,)
    )

    evaluated = report.candidate_reports[0].tracks[0]
    assert evaluated.segment_index == 0
    assert evaluated.metrics.f1 == 1.0


def test_bakeoff_evaluates_selected_segment_with_segment_relative_references(
    tmp_path: Path,
) -> None:
    from lights_audio_engine.evaluation.artifact import SegmentInfo, write_artifact
    from lights_audio_engine.evaluation.bakeoff import DatasetSplit, TrackSpec, run_bakeoff

    artifact_path = tmp_path / "split.npy"
    reference_path = tmp_path / "split.txt"
    samples = np.zeros(400, dtype=np.float64)
    samples[25] = 0.9
    samples[200 + 50] = 0.8
    write_artifact(
        artifact_path,
        samples,
        label="split",
        sample_rate_hz=1_000,
        segments=(SegmentInfo(0, 200, 0.0, None), SegmentInfo(200, 200, 8.0, "overflow")),
    )
    reference_path.write_text("0.050\tbeat\n", encoding="utf-8")
    track = TrackSpec("split", artifact_path, reference_path, DatasetSplit.HOLDOUT, segment_index=1)

    report = run_bakeoff(
        (track,),
        candidate_factories={"stateful": _FirstSegmentPoisonsCandidate},
        block_sizes=(100,),
    )

    evaluated = report.candidate_reports[0].tracks[0]
    assert evaluated.segment_index == 1
    assert tuple(item.event.timestamp_seconds for item in evaluated.detections) == (0.05,)
    assert evaluated.metrics.f1 == 1.0


@pytest.mark.parametrize("segment_index", (-1, 2))
def test_bakeoff_rejects_invalid_selected_segment_index(tmp_path: Path, segment_index: int) -> None:
    from lights_audio_engine.evaluation.artifact import SegmentInfo, write_artifact
    from lights_audio_engine.evaluation.bakeoff import DatasetSplit, TrackSpec, run_bakeoff

    artifact_path = tmp_path / "split.npy"
    reference_path = tmp_path / "split.txt"
    write_artifact(
        artifact_path,
        np.zeros(20, dtype=np.float64),
        label="split",
        sample_rate_hz=1_000,
        segments=(SegmentInfo(0, 10, 0.0, None), SegmentInfo(10, 10, 4.0, "overflow")),
    )
    reference_path.write_text("", encoding="utf-8")
    track = TrackSpec("split", artifact_path, reference_path, DatasetSplit.HOLDOUT, segment_index)

    with pytest.raises(ValueError, match="segment index is out of range"):
        run_bakeoff(
            (track,), candidate_factories={"baseline": lambda: _NoEvents()}, block_sizes=(5,)
        )


class _NoEvents:
    def process(self, frame: AudioFrame) -> tuple[DetectedOnset, ...]:
        return ()

    def reset(self) -> None:
        pass


class _FirstSegmentPoisonsCandidate:
    def __init__(self) -> None:
        self.poisoned = False
        self.emitted = False

    def process(self, frame: AudioFrame) -> tuple[DetectedOnset, ...]:
        if bool((frame.samples == 0.9).any()):
            self.poisoned = True
        if not self.poisoned and not self.emitted and bool((frame.samples == 0.8).any()):
            self.emitted = True
            return (DetectedOnset(0.05, 1.0),)
        return ()

    def reset(self) -> None:
        self.poisoned = False
        self.emitted = False


class _PulseCandidate:
    def process(self, frame: AudioFrame) -> tuple[DetectedOnset, ...]:
        if bool((frame.samples == 0.8).any()):
            return (DetectedOnset(0.005, 1.0),)
        return ()

    def reset(self) -> None:
        pass
