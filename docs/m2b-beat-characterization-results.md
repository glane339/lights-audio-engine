# M2B Live Beat Characterization Results

## Hardware setup

- Windows 11 lights laptop
- Windows WASAPI input device 12: Microphone (Realtek(R) Audio)
- 48 kHz mono
- 960-frame reads
- Sensitivity: 0.5
- BPM range: 50-240
- Expected test tempo: 140 BPM
- Real AUX/splitter input path
- Test material: "I Don't Trust a Soul" by Disco Lines and Ship Wrek

## Normal-level runs

### R1
- Duration: 98.281 s
- Beats detected: 25
- Median interval: 0.428 s
- IQR: 1.077 s
- Suspicious long gaps: 8
- Potential short doubles: 0
- Median BPM: 142.864
- BPM range: 97.333-151.268
- Mean processing time: 0.151 ms
- Max processing time: 12.338 ms
- Discontinuities: 0

### R2
- Duration: 46.594 s
- Beats detected: 9
- Median interval: 0.424 s
- IQR: 0.034 s
- Suspicious long gaps: 1
- Potential short doubles: 0
- Median BPM: 143.813
- BPM range: 139.003-144.094
- Mean processing time: 0.146 ms
- Max processing time: 13.995 ms
- Discontinuities: 0

### R3
- Duration: 53.360 s
- Beats detected: 13
- Median interval: 0.420 s
- IQR: 0.866 s
- Suspicious long gaps: 3
- Potential short doubles: 0
- Median BPM: 146.446
- BPM range: 142.180-152.737
- Mean processing time: 0.153 ms
- Max processing time: 12.848 ms
- Discontinuities: 0

## Low-level run

### Low R1
- Duration: 72.125 s
- Beats detected: 10
- Median interval: 0.427 s
- IQR: 0.011 s
- Suspicious long gaps: 2
- Potential short doubles: 0
- Median BPM: 139.765
- BPM range: 138.448-141.440
- Mean processing time: 0.145 ms
- Max processing time: 1.215 ms
- Discontinuities: 0

## Findings

The detector consistently identifies approximately correct beat spacing when it is actively detecting beats, but repeatedly stops producing beat events for long periods.

Lowering input level did not resolve the failure.

No capture discontinuities occurred in any of the four runs.

The evidence therefore points to detector behavior rather than capture instability. The current fixed-window broadband RMS threshold-crossing detector should not be considered production-quality for real music.

Software processing cost is very low on average, so CPU processing time is not presently the primary performance limitation.

## Decision

M2B characterization is complete.

The next milestone should evaluate improved low-latency onset detection using deterministic replay of identical captured PCM before replacing the production detector.
