from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

_PYPROJECT = Path(__file__).parents[1] / "pyproject.toml"


def test_core_dependencies_exclude_librosa_and_extra_is_declared() -> None:
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    core_dependencies = data["project"]["dependencies"]
    assert not any(dep.lower().startswith("librosa") for dep in core_dependencies)

    optional = data["project"]["optional-dependencies"]
    assert "librosa" in optional
    assert any(dep.lower().startswith("librosa") for dep in optional["librosa"])
    assert not any(dep.lower().startswith("librosa") for dep in optional.get("dev", []))


def _run_isolated_import(code: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    return subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True, env=environment
    )


def test_bakeoff_import_does_not_pull_in_librosa_bench() -> None:
    result = _run_isolated_import(
        "import sys\n"
        "import lights_audio_engine.evaluation.bakeoff\n"
        "assert 'lights_audio_engine.evaluation.librosa_bench' not in sys.modules\n"
    )
    assert result.returncode == 0, result.stderr


def test_candidates_import_does_not_pull_in_librosa_bench() -> None:
    result = _run_isolated_import(
        "import sys\n"
        "import lights_audio_engine.evaluation.candidates\n"
        "assert 'lights_audio_engine.evaluation.librosa_bench' not in sys.modules\n"
    )
    assert result.returncode == 0, result.stderr


def test_evaluation_cli_import_does_not_pull_in_librosa_bench() -> None:
    result = _run_isolated_import(
        "import sys\n"
        "import lights_audio_engine.evaluation.cli\n"
        "assert 'lights_audio_engine.evaluation.librosa_bench' not in sys.modules\n"
    )
    assert result.returncode == 0, result.stderr


def test_librosa_bench_import_does_not_require_librosa_to_be_installed() -> None:
    result = _run_isolated_import(
        "import sys\n"
        "sys.modules['librosa'] = None\n"
        "import lights_audio_engine.evaluation.librosa_bench.cli\n"
    )
    assert result.returncode == 0, result.stderr


def test_librosa_bench_cli_import_does_not_pull_in_production_m2c_modules() -> None:
    result = _run_isolated_import(
        "import sys\n"
        "import lights_audio_engine.evaluation.librosa_bench.cli\n"
        "assert 'lights_audio_engine.evaluation.bakeoff' not in sys.modules\n"
        "assert 'lights_audio_engine.evaluation.candidates' not in sys.modules\n"
        "assert 'lights_audio_engine.evaluation.cli' not in sys.modules\n"
    )
    assert result.returncode == 0, result.stderr
