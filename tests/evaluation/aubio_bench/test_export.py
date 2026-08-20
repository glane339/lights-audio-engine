from __future__ import annotations

from pathlib import Path


def test_write_sonic_visualiser_time_instants_uses_report_timestamps(tmp_path: Path) -> None:
    from lights_audio_engine.evaluation.aubio_bench.export import (
        write_sonic_visualiser_time_instants,
    )

    output = tmp_path / "aubio.sv.txt"
    write_sonic_visualiser_time_instants(output, (0.125, 1.5, 2.75))

    assert output.read_text(encoding="utf-8") == (
        "0.125000\taubio\n1.500000\taubio\n2.750000\taubio\n"
    )
