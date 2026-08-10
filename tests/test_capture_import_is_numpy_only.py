from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run_isolated_import(code: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    return subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_package_root_import_does_not_cross_into_capture_or_optional_audio_runtime() -> None:
    result = _run_isolated_import(
        "import sys\n"
        "sys.modules['sounddevice'] = None\n"
        "import lights_audio_engine\n"
        "assert 'lights_audio_engine.capture' not in sys.modules\n"
    )

    assert result.returncode == 0, result.stderr


def test_capture_import_succeeds_without_optional_audio_runtime() -> None:
    result = _run_isolated_import(
        "import sys\nsys.modules['sounddevice'] = None\nimport lights_audio_engine.capture\n"
    )

    assert result.returncode == 0, result.stderr
