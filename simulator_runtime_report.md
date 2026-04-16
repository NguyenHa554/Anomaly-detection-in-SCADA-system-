# Simulator Runtime Report

## Summary

The current system was replayed with the simulator on `SWaT.csv` using the new production-oriented thresholds.

- Early clean normal data no longer raises the previous false alerts.
- On the attacked test portion, the ensemble detector still hits all 6 attack intervals.
- Stage-level tracing is uneven, especially for `P4` and `P6`.

## Attack Interval Hits

Measured on the 6 real attack intervals in the attacked SWaT test segment:

| Stage | Attack intervals detected |
| --- | --- |
| P2 | 5 / 6 |
| P3 | 5 / 6 |
| P4 | 4 / 6 |
| P5 | 6 / 6 |
| P6 | 3 / 6 |
| Ensemble (any stage alerts) | 6 / 6 |

## Current Thresholds

From `ml_models/thresholds.json`:

| Stage | Threshold T | Window W |
| --- | ---: | ---: |
| P2 | 7.229 | 10 |
| P3 | 5.686 | 10 |
| P4 | 8.369 | 10 |
| P5 | 9.217 | 10 |
| P6 | 58.288 | 10 |

## Test-Set Metrics

| Stage | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| P2 | 0.822 | 0.817 | 0.819 |
| P3 | 0.797 | 0.745 | 0.770 |
| P4 | 0.744 | 0.142 | 0.239 |
| P5 | 0.779 | 0.624 | 0.693 |
| P6 | 0.997 | 0.460 | 0.630 |

## Interpretation

- The system-level alert is suitable as a safer deployment baseline because it still alerts on all 6 attack intervals.
- The new thresholds are better for normal data than the old test-tuned thresholds.
- Per-stage localization is not fully strong yet, especially for `P4` and `P6`, which lose recall under the safer threshold setting.
