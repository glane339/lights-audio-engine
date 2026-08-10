from dataclasses import FrozenInstanceError
from math import inf, nan

import pytest


def test_configuration_defaults_are_valid_and_bounded() -> None:
    from lights_audio_engine.config import AudioEngineConfig

    config = AudioEngineConfig()

    assert config.expected_sample_rate_hz == 48_000
    assert config.sensitivity == 0.5
    assert config.min_bpm == 50.0
    assert config.max_bpm == 240.0
    assert config.analysis_window_seconds == 0.02
    assert config.energy_history_seconds == 2.0
    assert config.bpm_history_size == 8


@pytest.mark.parametrize("sensitivity", [-0.01, 1.01, nan, inf])
def test_configuration_rejects_invalid_sensitivity(sensitivity: float) -> None:
    from lights_audio_engine.config import AudioEngineConfig

    with pytest.raises(ValueError, match="sensitivity"):
        AudioEngineConfig(sensitivity=sensitivity)


@pytest.mark.parametrize(
    ("min_bpm", "max_bpm"),
    [(0.0, 120.0), (120.0, 120.0), (121.0, 120.0), (nan, 120.0), (60.0, inf)],
)
def test_configuration_rejects_invalid_bpm_range(min_bpm: float, max_bpm: float) -> None:
    from lights_audio_engine.config import AudioEngineConfig

    with pytest.raises(ValueError, match="BPM"):
        AudioEngineConfig(min_bpm=min_bpm, max_bpm=max_bpm)


@pytest.mark.parametrize("sample_rate_hz", [0, -1, 48_000.0, True])
def test_configuration_rejects_invalid_sample_rate(sample_rate_hz: object) -> None:
    from lights_audio_engine.config import AudioEngineConfig

    with pytest.raises((TypeError, ValueError), match="sample rate"):
        AudioEngineConfig(expected_sample_rate_hz=sample_rate_hz)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("analysis_window_seconds", 0.0),
        ("energy_history_seconds", 0.01),
        ("bpm_history_size", 2),
    ],
)
def test_configuration_rejects_invalid_analysis_history(field: str, value: float) -> None:
    from lights_audio_engine.config import AudioEngineConfig

    kwargs: dict[str, float | int] = {field: value}
    with pytest.raises((TypeError, ValueError)):
        AudioEngineConfig(**kwargs)  # type: ignore[arg-type]


def test_configuration_is_frozen() -> None:
    from lights_audio_engine.config import AudioEngineConfig

    config = AudioEngineConfig()

    with pytest.raises(FrozenInstanceError):
        config.sensitivity = 0.75  # type: ignore[misc]
