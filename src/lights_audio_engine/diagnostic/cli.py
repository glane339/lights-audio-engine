"""Temporary command line for observing the engine against live audio."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Protocol, cast

from lights_audio_engine.capture import AudioSource, Discontinuity, run_engine
from lights_audio_engine.capture.live_source import (
    DiscontinuityObserver,
    SoundDeviceAudioSource,
)
from lights_audio_engine.config import AudioEngineConfig
from lights_audio_engine.diagnostic.session_log import (
    JsonlSessionLogger,
    LoggedAudioEngine,
    LoggedAudioSource,
    SessionInfo,
)
from lights_audio_engine.engine import AudioEngine
from lights_audio_engine.probe.backend import SoundDeviceBackend
from lights_audio_engine.probe.diagnostics import resolve_input_device
from lights_audio_engine.probe.models import (
    BackendUnavailableError,
    DeviceSelectionError,
    ProbeSnapshot,
)


class InventoryBackend(Protocol):
    """Small inventory surface needed by this disposable CLI."""

    def snapshot(self) -> ProbeSnapshot: ...


class SourceFactory(Protocol):
    """Construct the reusable source without coupling tests to sounddevice."""

    def __call__(
        self,
        device_index: int,
        *,
        sample_rate_hz: int,
        channels: int,
        block_size_frames: int,
        on_discontinuity: DiscontinuityObserver | None,
    ) -> AudioSource: ...


BackendFactory = Callable[[], InventoryBackend]


@dataclass(frozen=True, slots=True)
class _Options:
    device: str
    sample_rate_hz: int
    channels: int
    sensitivity: float
    min_bpm: float
    max_bpm: float
    log_jsonl: Path | None
    label: str


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


def _sensitivity(value: str) -> float:
    parsed = _finite_float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0.0 and 1.0")
    return parsed


def _positive_float(value: str) -> float:
    parsed = _finite_float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lights_audio_engine.diagnostic",
        description="Temporary live AUX diagnostic for the existing audio engine.",
    )
    parser.add_argument("--device", required=True, help="current input index or unique name text")
    parser.add_argument("--sample-rate", type=_positive_int, default=48_000)
    parser.add_argument("--channels", type=_positive_int, default=1)
    parser.add_argument("--sensitivity", type=_sensitivity, default=0.5)
    parser.add_argument("--min-bpm", type=_positive_float, default=50.0)
    parser.add_argument("--max-bpm", type=_positive_float, default=240.0)
    parser.add_argument("--log-jsonl", type=Path)
    parser.add_argument("--label", default="")
    return parser


def _options(namespace: argparse.Namespace) -> _Options:
    values = cast(dict[str, object], vars(namespace))
    return _Options(
        device=cast(str, values["device"]),
        sample_rate_hz=cast(int, values["sample_rate"]),
        channels=cast(int, values["channels"]),
        sensitivity=cast(float, values["sensitivity"]),
        min_bpm=cast(float, values["min_bpm"]),
        max_bpm=cast(float, values["max_bpm"]),
        log_jsonl=cast(Path | None, values["log_jsonl"]),
        label=cast(str, values["label"]),
    )


def _print_discontinuity(boundary: Discontinuity) -> None:
    detail = f" ({boundary.detail})" if boundary.detail else ""
    print(
        f"Discontinuity: {boundary.reason.value} after "
        f"{boundary.frames_emitted_before} frame(s){detail}",
        file=sys.stderr,
    )


def _run(
    options: _Options,
    *,
    source_factory: SourceFactory,
    backend_factory: BackendFactory,
) -> int:
    device = resolve_input_device(backend_factory().snapshot(), options.device)
    config = AudioEngineConfig(
        expected_sample_rate_hz=options.sample_rate_hz,
        sensitivity=options.sensitivity,
        min_bpm=options.min_bpm,
        max_bpm=options.max_bpm,
    )
    logger = (
        JsonlSessionLogger(
            options.log_jsonl,
            SessionInfo(
                label=options.label,
                device_index=device.index,
                device_name=device.name,
                sample_rate_hz=options.sample_rate_hz,
                channels=options.channels,
                block_size_frames=960,
                sensitivity=config.sensitivity,
                min_bpm=config.min_bpm,
                max_bpm=config.max_bpm,
                analysis_window_seconds=config.analysis_window_seconds,
                energy_history_seconds=config.energy_history_seconds,
                bpm_history_size=config.bpm_history_size,
            ),
        )
        if options.log_jsonl is not None
        else None
    )
    try:
        source = source_factory(
            device.index,
            sample_rate_hz=options.sample_rate_hz,
            channels=options.channels,
            block_size_frames=960,
            on_discontinuity=_print_discontinuity,
        )
    except BaseException:
        if logger is not None:
            logger.close()
        raise
    active_source: AudioSource = LoggedAudioSource(source, logger) if logger is not None else source
    engine = LoggedAudioEngine(config, logger) if logger is not None else AudioEngine(config)
    print(
        f"Listening: [{device.index}] {device.name}, {options.sample_rate_hz} Hz, "
        f"{options.channels} channel(s), 960-frame reads"
    )
    last_bpm: float | None = None
    try:
        for result in run_engine(active_source, engine):
            for event in result.beat_events:
                print(
                    f"Beat {event.beat_index}: t={event.timestamp_seconds:.6f}s "
                    f"strength={event.strength:.6f}"
                )
            if result.bpm is not None and result.bpm != last_bpm:
                print(f"BPM: {result.bpm:.3f}")
                last_bpm = result.bpm
    finally:
        try:
            active_source.close()
        finally:
            if logger is not None:
                logger.close()
    return 0


def main(
    argv: list[str] | None = None,
    source_factory: SourceFactory = SoundDeviceAudioSource,
    backend_factory: BackendFactory = SoundDeviceBackend,
) -> int:
    """Run until interrupted and return a developer-oriented exit code."""

    try:
        namespace = _parser().parse_args(argv)
    except SystemExit as error:
        return int(error.code) if isinstance(error.code, int) else 2
    try:
        return _run(
            _options(namespace),
            source_factory=source_factory,
            backend_factory=backend_factory,
        )
    except BackendUnavailableError as error:
        print(f"Backend unavailable: {error}", file=sys.stderr)
        return 3
    except DeviceSelectionError as error:
        print(f"Device selection error: {error}", file=sys.stderr)
        return 4
    except (TypeError, ValueError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Live diagnostic interrupted.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError) as error:
        print(f"Live diagnostic failure: {error}", file=sys.stderr)
        return 6
