# E011: Wild-Middle Fourier-Component Swap Protocol

> Presentation clarification (2026-11-02): the frozen disk (377 modes) and
> square (361 modes) are approximately mode-count-matched. Protocol settings
> and original results are unchanged.

Status: **FROZEN BEFORE INFERENCE**

## Research Question

When two compatible diffusion denoisers evaluate the same sampler state, does
replacing the recipient's wild-middle-associated Fourier coefficients with the
donor's coefficients move final memorization behavior toward the donor?

E011 is a causal intervention on the denoiser prediction. It is not a new
definition of the CALM wild middle and does not assume that frequency
components cause memorization.

## Existing Evidence

The paper-derived sigma-frequency report maps the paper's wild-middle regime
to sampler calls `6..13` on the repository's 18-call schedule and observes an
information boundary spanning radial frequencies through `r=11`. This is a
diagnostic correspondence. E010 separately found that a whole-denoiser swap at
high-frequency-geometry-derived times suppressed memorization, but E010 did
not swap Fourier components.

## Hypothesis

Let `G` be the E010 generalizing endpoint and `M` the E010 memorizing endpoint.
For shared seeds, define

```text
Delta_G = Mem(G <- M) - Mem(G)
Delta_M = Mem(M <- G) - Mem(M)
```

The reciprocal-transfer hypothesis predicts `Delta_G > 0` and `Delta_M < 0`
when the donor supplies the selected denoiser Fourier coefficients during the
wild-middle calls. Directions are not encoded as pass/fail expectations; both
effects and confidence intervals are reported regardless of sign.

## Frozen Models

E011 reuses the exact E010 asymmetric model pair:

- `M`: EDM-1K at 12,000 kimg, SHA-256
  `e5a7debafcd19191d6557f645216bfcb2e7589922396fd08130e76e3f5388b0a`.
- `G`: EDM-50K at 40,000 kimg, SHA-256
  `a355ea67605dea3e2e663e94eb23416ffeb7679757088a68dc6228c03da5a92b`.

The complete paths and compatibility evidence remain frozen in
[`e010_model_pair_manifest.json`](../data/e010_model_pair_manifest.json).
The models are not baseline matched, and dataset size is not isolated as a
causal variable.

## Sampler And Shared-State Intervention

- Pure Euler sampling on the frozen 18-call EDM schedule.
- No churn and no Heun correction.
- Identical initial latent for all conditions sharing a seed.
- Unconditional inference with `class_labels=None`.
- Unclamped, unquantized model outputs for memorization evaluation.

At an intervention call, both models evaluate the same current state `x_t` and
noise level `sigma_t`. For recipient prediction `D_base`, donor prediction
`D_donor`, and binary mask `B`, E011 uses

```text
F(D_hybrid) = (1 - B) F(D_base) + B F(D_donor)
```

and inserts `D_hybrid` into the otherwise unchanged Euler update. E011 does
not maintain two diverged trajectories and does not modify the noisy state
directly.

## Frozen Time Window

The intervention applies by sampler-call index, never by a rounded
floating-point comparison:

```text
calls: 6..13 inclusive
sigma: 8.400935309099816 .. 0.13951646873101678
```

These calls operationalize the report's rounded `0.14 <= sigma <= 8.4`
wild-middle interval. Outside calls `6..13`, the recipient model runs normally.

## Frozen Fourier Convention

- Channelwise two-dimensional `torch.fft.rfft2` and `torch.fft.irfft2`.
- `norm="ortho"` in both directions.
- `ky = torch.fft.fftfreq(H) * H`.
- `kx = torch.fft.rfftfreq(W) * W`.
- No special DC treatment.
- Image shape `3 x 32 x 32`.

### Try 01: Radial Disk

The primary mask is

```text
B_disk = 1[sqrt(kx^2 + ky^2) <= 11].
```

### Try 02: Matched Square

The robustness mask is

```text
B_square = 1[max(abs(kx), abs(ky)) <= K].
```

Stored `rfft2` coefficients at horizontal frequency `kx=0` or `kx=16`
represent one full-spectrum coefficient. Other stored horizontal frequencies
represent a conjugate pair and count twice. Exhaustive integer search over
`K=0..16` selects `K=9`:

| Mask | Stored coefficients | Effective full-spectrum modes |
| --- | ---: | ---: |
| Disk `r<=11` | 200 | 377 |
| Square `K=9` | 190 | 361 |

The absolute difference is `16` modes and the relative difference against the
disk is `16/377 = 0.0424403183`. Across three channels the corresponding
effective counts are `1,131` and `1,083`. No continuous-area approximation is
used.

## Conditions And Seeds

Each try contains four conditions:

```text
G
M
G <- M
M <- G
```

All conditions use shared latent seeds `50000..50999`, giving 1,000 paired
quadruples per try and 8,000 records across both tries. Repository manifests
contain no prior evaluation seed range overlapping these seeds. The same seed
range is intentionally reused between Try 01 and Try 02 for paired mask-shape
comparisons.

## Evaluation

### Primary Memorization Metric

- Frozen CIFAR-10 1K reference subset in `[-1,1]`.
- Deterministic CPU `float64` direct distances.
- Stable `(distance, reference position)` ordering.
- Strict rule `d1NN < d2NN / 3`.
- 100,000 paired bootstrap resamples over latent seeds with bootstrap seed `0`.

E011 reports both condition rates, both reciprocal treatment effects, and
their paired 95% percentile intervals. It also reports paired differences
between disk and square treatment effects. No sign is hard-coded as a formal
success label.

### Descriptive Image Quality

FID-1K is computed from the 1,000 generated PNGs per condition with the
official NVlabs EDM `fid.py` implementation and official CIFAR-10 reference
statistics. Conversion to uint8 is used only for FID; raw unclamped float64
samples remain authoritative for nearest-neighbor evaluation. The EDM source
commit, `fid.py` SHA-256, detector URL, reference URL, and retrieved artifact
hashes are recorded. FID-1K is descriptive and cannot establish quality
equivalence.

## Required Outputs

External outputs retain:

- one per-sample CSV with exactly 8,000 unique `(try, condition, seed)` rows;
- generated float64 samples and FID PNGs, separated by try and condition;
- per-condition FID logs and metrics;
- compact condition and paired-effect summaries;
- mask comparison and numerical validation records;
- complete run provenance and zero-row failure table;
- concise figures and a report with separate Try 01 and Try 02 sections.

Git receives only compact reviewable summaries, figures, manifests, and the
results document. Generated samples and raw per-sample rows remain external.

## Numerical And Execution Gates

Before full execution:

1. FFT round-trip passes.
2. Empty mask returns the base prediction.
3. Full mask returns the donor prediction.
4. Self-swap returns the original prediction.
5. Hybrid behavior outside calls `6..13` equals the base behavior.
6. Disk and square masks match their exact grid definitions.
7. Weighted mode counts and `K=9` selection match the frozen manifest.
8. Same-seed sampling is deterministic.
9. Checkpoint, reference archive, subset, config, and manifest hashes pass.
10. The output path is collision-safe and the execution checkout is clean and
    pinned to the submitted commit.

The Slurm smoke uses both mask types, all four conditions, and at least two
shared seeds. It must produce finite outputs, complete metadata, nearest-two
records, valid FID input images, a successful FID invocation, and a visual
montage before either 1,000-seed try can run.

## Stop Conditions

Stop without a full run if any identity, numerical, determinism, output-path,
nearest-neighbor, FID-pipeline, or paired-alignment gate fails. Do not change
seeds, masks, time calls, checkpoints, sampler, or metric after seeing a
failure or result. Do not implement the sigma-by-frequency heatmap in E011.

## Interpretation Rules

E011 may show that replacing selected denoiser-prediction Fourier components
changes memorization behavior in the donor's direction and whether that result
is robust to a mode-count-matched square. It cannot establish that middle
frequencies cause memorization, that orientation explains disk/square
differences, that dataset size causes the result, or that the selected
time-frequency region is universal.

## Compute And Reproduction

All checkpoint loading, generation, nearest-neighbor evaluation, and FID run
through Slurm. The login node is limited to source synchronization, hash
inspection, submission, and monitoring. Caches and temporary files use the
job temporary directory; finalized raw outputs are staged under
`/home/xggh8/data/zw-lab/e011_fourier_component_swap`.
