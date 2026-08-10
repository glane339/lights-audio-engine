def test_public_api_exports_only_the_m0_m1_boundary() -> None:
    import lights_audio_engine

    assert lights_audio_engine.__all__ == [
        "AudioAnalysisResult",
        "AudioEngine",
        "AudioEngineConfig",
        "AudioFrame",
        "BeatEvent",
        "DropEvent",
    ]


def test_primary_types_are_importable_from_package_root() -> None:
    from lights_audio_engine import (
        AudioAnalysisResult,
        AudioEngine,
        AudioEngineConfig,
        AudioFrame,
        BeatEvent,
        DropEvent,
    )

    assert AudioEngine(AudioEngineConfig()) is not None
    assert AudioFrame is not None
    assert AudioAnalysisResult is not None
    assert BeatEvent is not None
    assert DropEvent is not None
