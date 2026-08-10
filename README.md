# Lights Audio Engine

_Deterministic party-music analysis for low-latency reactive lighting._

---

## 📋 Purpose and scope

`lights-audio-engine` converts sequential, normalized mono audio frames into typed,
immutable analysis results for reactive lighting. Its musical target spans modern electronic
music, mainstream pop and dance-pop, party-oriented hip-hop, and older dance, funk, disco,
and pop recordings. It is not tuned only for four-on-the-floor material.

The M0/M1 foundation provides:

- Defensive audio-frame and configuration validation
- A sensitivity-aware short-term-energy transient detector
- Sample-derived beat timestamps and monotonic beat indexes
- A bounded, median interval BPM estimate across a default `50–240 BPM` range
- An explicit `DropEvent` model with an honest no-op M1 detector
- Deterministic reset and replay behavior

The stable engine deliberately does not own audio-device capture, scenes, presets, fixtures,
DMX, sACN/E1.31, Art-Net, WLED, LedFx control, ILDA, lasers, networking, a GUI, or persistent
show state. M1.5 adds only an experimental hardware-diagnostic sidecar; production capture
and the other concerns belong to the downstream Lights App or later, separately scoped
milestones.

## 🔧 Development setup

### Prerequisites

| Requirement | Version | Check command |
| --- | ---: | --- |
| Python | `>=3.11` | `py --version` |
| uv | Current supported release | `uv --version` |

From PowerShell:

```powershell
uv sync --extra dev
```

Run the complete local quality baseline:

```powershell
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run basedpyright
```

## 🎙️ Hardware probe (experimental)

M1.5 includes a developer-oriented Windows audio probe. It inventories PortAudio devices,
checks input formats, and captures incremental signal diagnostics without retaining raw PCM.
It does not produce `AudioFrame`, implement an `AudioSource`, or extend the stable package API.

Install the optional backend alongside development dependencies:

```powershell
uv sync --extra dev --extra probe
# Alternative editable install:
python -m pip install -e ".[probe]"
```

Run the four diagnostic commands:

```powershell
uv run python -m lights_audio_engine.probe devices
uv run python -m lights_audio_engine.probe check --device 3
uv run python -m lights_audio_engine.probe capture --device 3 --duration 3
uv run python -m lights_audio_engine.probe report
```

Device indexes and names are observations from the current PortAudio enumeration, not stable
identifiers. See the [hardware probe checklist](docs/hardware-probe-checklist.md) before using
the probe with the Lights hardware.

## 📦 Public API

The supported package-root imports are:

```python
from lights_audio_engine import (
    AudioAnalysisResult,
    AudioEngine,
    AudioEngineConfig,
    AudioFrame,
    BeatEvent,
    DropEvent,
)
```

Minimal use:

```python
import numpy as np

from lights_audio_engine import AudioEngine, AudioEngineConfig, AudioFrame

config = AudioEngineConfig(expected_sample_rate_hz=48_000, sensitivity=0.7)
engine = AudioEngine(config)

samples = np.zeros(4_800, dtype=np.float64)
samples[960:1_920] = 0.8
result = engine.process(
    AudioFrame(
        samples=samples,
        sample_rate_hz=48_000,
        start_time_seconds=0.0,
    )
)

for beat in result.beat_events:
    print(beat.beat_index, beat.timestamp_seconds, beat.strength)
```

Frames after the first must be contiguous in sample time. Call `engine.reset()` before
starting a new logical stream or replaying from an earlier timestamp.

`BeatEvent.strength` is normalized detector-relative transient energy. It is useful for
relative lighting reactivity, but it is neither calibrated musical salience nor confidence.

## 🔄 Tempo behavior

The default tempo range is `50.0–240.0 BPM`, and both bounds remain configurable. The M1
estimator reports the tempo directly observed in adjacent beat intervals: transients spaced
`0.3` seconds apart produce approximately `200 BPM`, not an automatic `100 BPM` half-time
interpretation. Half/double-time candidate ranking requires richer evidence and is deferred.

## ⚙️ Package layout

```text
src/lights_audio_engine/
├── __init__.py          # Deliberate public boundary
├── config.py            # Frozen, validated engine configuration
├── engine.py            # Streaming orchestration, beat indexes, and BPM state
├── models.py            # Immutable input, event, and result models
├── probe/               # Experimental hardware diagnostic sidecar
└── detectors/
    ├── energy.py        # Minimal deterministic energy/transient detector
    └── drop.py          # Explicit no-op M1 drop detector

tests/                   # Synthetic contract, detector, and engine tests
docs/architecture.md     # Boundaries, timing, state, and detector decisions
```

## 📚 Architecture

See [Architecture](docs/architecture.md) for the data flow, timing model, sensitivity
semantics, BPM calculation, validation rules, and downstream ownership boundary.

## ⚠️ Current limitations

- The energy detector is an M1 architecture proof, not production-quality beat tracking
- Detection remains amplitude-dependent; sensitivity improves reactivity but is not gain control
- Spectral flux and sub-bass, bass, midrange, and high-frequency energy are not yet computed
- Input is normalized floating-point mono PCM only
- The configured sample rate is fixed for an engine instance
- BPM preserves observed intervals but has no phase, downbeat, bar, or half/double-time model
- Drop detection intentionally emits no events pending a measurable definition and fixtures
- File decoding, pacing, production live capture, and reconnect behavior are deferred

Future analysis can add band energy, onset strength, rhythmic phase, and broad musical
intensity behind the existing detector and immutable-result boundaries. Builds, drops,
breakdowns, and bar/downbeat estimates require dedicated evidence rather than placeholder
fields. Low latency and stable sample timing matter because lighting must react near the
perceived musical event to remain useful.
