# E012 Radial Frequency Localization Results

Generated from the frozen E012 full run. Interpretation must remain frequency-localization within calls `6..13`; no claim that middle frequencies cause memorization is supported.

## E011 Positive-Control Comparison

E011 radial `G <- M, r<=11` recorded `761/1000` memorized samples and FID-1K `38.0190`. The exactly corresponding E012 `111` condition recorded `761/1000` and FID-1K `38.0190`. Any discrepancy must be investigated rather than tuned away.

## Condition Results

| Condition | Bands | Modes | Mem count | Mem rate | Delta vs G | 95% CI | FID-1K |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| 000_none | none | 0 | 0 | 0.00% | 0.00 pp | [0.00, 0.00] pp | 34.9014 |
| 100_B1 | B1 | 113 | 95 | 9.50% | 9.50 pp | [7.70, 11.30] pp | 47.1730 |
| 010_B2 | B2 | 140 | 0 | 0.00% | 0.00 pp | [0.00, 0.00] pp | 49.7917 |
| 001_B3 | B3 | 124 | 0 | 0.00% | 0.00 pp | [0.00, 0.00] pp | 37.5288 |
| 110_B1_B2 | B1+B2 | 253 | 622 | 62.20% | 62.20 pp | [59.20, 65.20] pp | 40.2796 |
| 101_B1_B3 | B1+B3 | 237 | 241 | 24.10% | 24.10 pp | [21.50, 26.80] pp | 54.0809 |
| 011_B2_B3 | B2+B3 | 264 | 0 | 0.00% | 0.00 pp | [0.00, 0.00] pp | 61.6959 |
| 111_B1_B2_B3 | B1+B2+B3 | 377 | 761 | 76.10% | 76.10 pp | [73.40, 78.70] pp | 38.0190 |
| control_B4 | B4 | 152 | 0 | 0.00% | 0.00 pp | [0.00, 0.00] pp | 35.2534 |

## Factorial Contrasts

| Contrast | Estimate | 95% CI |
| --- | ---: | --- |
| B1 | 42.98 pp | [41.05, 44.90] pp |
| B2 | 26.17 pp | [24.88, 27.47] pp |
| B3 | 7.12 pp | [6.42, 7.83] pp |
| B1xB2 | 52.35 pp | [49.75, 54.95] pp |
| B1xB3 | 14.25 pp | [12.85, 15.65] pp |
| B2xB3 | -0.35 pp | [-2.00, 1.30] pp |
| B1xB2xB3 | -0.70 pp | [-4.00, 2.60] pp |

## Full-Mask Overlap

| Condition | Memorized | Intersection | Recovery of 111 | Jaccard |
| --- | ---: | ---: | ---: | ---: |
| 100_B1 | 95 | 95 | 0.1248 | 0.1248 |
| 010_B2 | 0 | 0 | 0.0000 | 0.0000 |
| 001_B3 | 0 | 0 | 0.0000 | 0.0000 |
| 110_B1_B2 | 622 | 622 | 0.8173 | 0.8173 |
| 101_B1_B3 | 241 | 241 | 0.3167 | 0.3167 |
| 011_B2_B3 | 0 | 0 | 0.0000 | 0.0000 |

## NN Ratio Diagnostics

| Condition | Mean | Median | Q25 | Q75 |
| --- | ---: | ---: | ---: | ---: |
| 000_none | 0.957135 | 0.968896 | 0.941226 | 0.986193 |
| 100_B1 | 0.502254 | 0.468579 | 0.391833 | 0.566722 |
| 010_B2 | 0.949970 | 0.966086 | 0.931153 | 0.984728 |
| 001_B3 | 0.955297 | 0.967600 | 0.939614 | 0.985814 |
| 110_B1_B2 | 0.358973 | 0.304288 | 0.246343 | 0.391324 |
| 101_B1_B3 | 0.450137 | 0.408809 | 0.337677 | 0.504049 |
| 011_B2_B3 | 0.947230 | 0.964819 | 0.927041 | 0.984424 |
| 111_B1_B2_B3 | 0.303458 | 0.238862 | 0.190375 | 0.323829 |
| control_B4 | 0.956072 | 0.968388 | 0.939255 | 0.985863 |
