# Experiment 02 — PPG Beat Detector Benchmark

## Detectors

1. Current HealthSense detector:
   - Butterworth 0.5–8 Hz
   - z-score
   - scipy.find_peaks

2. NeuroKit2 Elgendi detector

## Results

Current detector:

- AF median F1: 0.8548
- Non-AF median F1: 0.9908
- AF/Non-AF gap: 0.1359
- AF median HR MAE: 4.70 bpm
- Non-AF median HR MAE: 1.10 bpm

NeuroKit2 Elgendi:

- AF median F1: 0.8372
- Non-AF median F1: 0.9921
- AF/Non-AF gap: 0.1549
- AF median HR MAE: 4.65 bpm
- Non-AF median HR MAE: 1.00 bpm

## Decision

Do not replace the current detector with Elgendi.

Elgendi slightly improves Non-AF performance but decreases AF
beat-detection performance and increases the AF vs Non-AF performance gap.

The validation alignment method must be audited before benchmarking
additional beat detectors.
