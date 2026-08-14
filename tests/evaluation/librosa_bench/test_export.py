from __future__ import annotations

from pathlib import Path

import pytest


def test_write_audacity_labels_uses_point_label_convention(tmp_path: Path) -> None:
    from lights_audio_engine.evaluation.librosa_bench.export import write_audacity_labels

    path = tmp_path / "beats.librosa-beats.txt"
    write_audacity_labels(path, (0.5, 1.0, 1.5))
    assert [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines()] == [
        ["0.500000", "0.500000", "librosa-beat-1"],
        ["1.000000", "1.000000", "librosa-beat-2"],
        ["1.500000", "1.500000", "librosa-beat-3"],
    ]


def test_write_candidate_reference_is_parseable_by_the_real_m2c_parser(tmp_path: Path) -> None:
    from lights_audio_engine.evaluation.librosa_bench.export import write_candidate_reference
    from lights_audio_engine.evaluation.reference import parse_reference

    path = tmp_path / "beats.librosa-candidate.txt"
    write_candidate_reference(path, (0.5, 1.0, 1.5))
    assert parse_reference(path) == (0.5, 1.0, 1.5)


def test_convert_audacity_export_round_trips_to_valid_reference(tmp_path: Path) -> None:
    from lights_audio_engine.evaluation.librosa_bench.export import (
        convert_audacity_export_to_reference,
    )
    from lights_audio_engine.evaluation.reference import parse_reference

    source, output = tmp_path / "reviewed.txt", tmp_path / "promoted-reference.txt"
    source.write_text("0.500000\t0.500000\tbeat\n1.000000\t1.000000\tbeat\n", encoding="utf-8")
    convert_audacity_export_to_reference(source, output)
    assert parse_reference(output) == (0.5, 1.0)


def test_convert_audacity_export_rejects_range_labels(tmp_path: Path) -> None:
    from lights_audio_engine.evaluation.librosa_bench.export import (
        convert_audacity_export_to_reference,
    )
    from lights_audio_engine.evaluation.reference import ReferenceFormatError

    source = tmp_path / "reviewed.txt"
    source.write_text("0.500000\t0.900000\tchorus\n", encoding="utf-8")
    with pytest.raises(ReferenceFormatError, match="range label"):
        convert_audacity_export_to_reference(source, tmp_path / "out.txt")
