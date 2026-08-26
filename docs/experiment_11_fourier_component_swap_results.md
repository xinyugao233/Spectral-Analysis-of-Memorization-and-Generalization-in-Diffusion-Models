# E011: Fourier-Component Swap Results

## Status

`COMPLETED`. Slurm job `16820230` completed with exit code `0:0` in
`00:38:17` at repository commit
`fe3c857dff0093f8ac02b8c60894c27421944a7f`.

The run evaluated 8,000 unique `(try, condition, seed)` records with zero
failures. All nearest-neighbor values were finite. Seeds `50000..50999`, the
two frozen checkpoints, pure-Euler sampler, calls `6..13`, masks, and strict
`d1NN < d2NN / 3` rule were unchanged.

## Results

| Mask | Condition | Memorized | Rate | FID-1K |
| --- | --- | ---: | ---: | ---: |
| Radial disk `r <= 11` | `G` | 0/1,000 | 0.000 | 34.9014 |
| Radial disk `r <= 11` | `M` | 870/1,000 | 0.870 | 43.4477 |
| Radial disk `r <= 11` | `G <- M` | 761/1,000 | 0.761 | 38.0190 |
| Radial disk `r <= 11` | `M <- G` | 0/1,000 | 0.000 | 97.6654 |
| Matched square `K = 9` | `G` | 0/1,000 | 0.000 | 34.9014 |
| Matched square `K = 9` | `M` | 870/1,000 | 0.870 | 43.4477 |
| Matched square `K = 9` | `G <- M` | 737/1,000 | 0.737 | 38.4852 |
| Matched square `K = 9` | `M <- G` | 0/1,000 | 0.000 | 99.3803 |

Paired effects use 100,000 bootstrap resamples over aligned latent seeds with
bootstrap seed `0`.

| Mask | Effect | Estimate | 95% CI |
| --- | --- | ---: | ---: |
| Radial disk | `Delta_G = Mem(G <- M) - Mem(G)` | 0.761 | [0.734, 0.787] |
| Radial disk | `Delta_M = Mem(M <- G) - Mem(M)` | -0.870 | [-0.890, -0.849] |
| Matched square | `Delta_G` | 0.737 | [0.709, 0.764] |
| Matched square | `Delta_M` | -0.870 | [-0.890, -0.849] |

The paired disk-minus-square comparison is `0.024` with 95% CI
`[0.015, 0.034]` for `Delta_G`, and exactly `0.000` with CI `[0.000, 0.000]`
for `Delta_M`.

## Validation

- Expected and observed records: `8,000`; unique keys: `8,000`.
- Failure rows: `0`; all nearest-neighbor records are finite.
- Each of the eight conditions contains exactly 1,000 seeds spanning
  `50000..50999`.
- Baseline records are identical across the radial and square tries.
- The preceding smoke run established exact same-seed repeat determinism for
  all eight conditions. The full 8,000-record run was not duplicated.
- All eight official EDM FID invocations returned exit code `0`.
- Disk effective mode count: `377`; square effective mode count: `361` at
  `K = 9`.

## Interpretation And Limits

Replacing the recipient denoiser prediction's selected Fourier coefficients
during calls `6..13` changed the memorization rate strongly in the donor's
direction under both masks. The mode-count-matched square produced a slightly
smaller `Delta_G`; `Delta_M` was identical under the two masks.

This intervention does not establish that middle frequencies cause
memorization. The models have asymmetric baselines and differ in training-set
size. The intervention replaces a broad selected coefficient region in a
whole denoiser prediction across a fixed time window. In addition, `M <- G`
has substantially worse descriptive FID-1K (`97.6654` and `99.3803`) than the
baselines, so its zero memorization rate cannot be separated from output
degradation. FID-1K is descriptive only.

## Evidence

- Compact tables and validation: [`results/experiment_11/`](../results/experiment_11/)
- Figures: [`figures/experiment_11/`](../figures/experiment_11/)
- Frozen protocol: [`experiment_11_fourier_component_swap_protocol.md`](experiment_11_fourier_component_swap_protocol.md)
- External raw output: `/home/xggh8/data/zw-lab/e011_fourier_component_swap`
- Slurm stdout: `/home/xggh8/data/zw-lab/slurm-e011-fft-swap-16820230.out`
- Slurm stderr: `/home/xggh8/data/zw-lab/slurm-e011-fft-swap-16820230.err`

The external directory retains the 8,000 raw float64 samples, 8,000 FID PNGs,
per-sample CSV, FID logs, and compact outputs. Raw samples and per-sample rows
are not committed to Git.
