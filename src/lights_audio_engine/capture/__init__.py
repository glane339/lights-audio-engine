"""Hardware-independent production audio capture boundary."""

from lights_audio_engine.capture.assembler import FrameAssembler
from lights_audio_engine.capture.channels import (
    AverageChannels,
    ChannelPolicy,
    MonoPassthrough,
    SelectChannel,
)
from lights_audio_engine.capture.driver import run_engine
from lights_audio_engine.capture.models import (
    CaptureConfig,
    CaptureError,
    CaptureItem,
    Discontinuity,
    DiscontinuityReason,
    FrameAssemblyError,
)
from lights_audio_engine.capture.source import AudioSource

__all__ = [
    "AverageChannels",
    "AudioSource",
    "CaptureConfig",
    "CaptureError",
    "CaptureItem",
    "ChannelPolicy",
    "Discontinuity",
    "DiscontinuityReason",
    "FrameAssembler",
    "FrameAssemblyError",
    "MonoPassthrough",
    "SelectChannel",
    "run_engine",
]
