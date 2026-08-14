from __future__ import annotations

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
