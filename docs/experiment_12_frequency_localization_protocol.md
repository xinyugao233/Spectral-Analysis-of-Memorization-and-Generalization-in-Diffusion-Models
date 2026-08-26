# E012: Radial Frequency Localization Protocol

**Status:** IMPLEMENTING; full execution requires explicit authorization after
the guarded smoke passes.

## Research Question

Within the already successful E011 wild-middle sampler calls `6..13`, which
subregions of the radial donor spectrum `r <= 11` are sufficient or necessary
for transferring memorization from the memorizing model `M` to the
generalizing model `G`, and do those subregions interact?

E012 localizes frequency while holding diffusion time fixed. It does not test
a sigma/time sweep and cannot establish that any frequency band causes
memorization outside the frozen intervention window.

## Frozen Parent Evidence

E011 used sampler implementation commit
`fe3c857dff0093f8ac02b8c60894c27421944a7f` with the same model pair, seeds,
sampler, and calls and observed `G <- M` memorization of `761/1000` for the
radial `r <= 11` mask. E012 preserves E011's validated sampler and evaluator
semantics. The reverse `M <- G` direction is excluded because its severe FID
degradation makes it a less interpretable localization target.

## Models, Sampler, And Evaluation

- Base `G`: `edm_50k_040000`, SHA-256
  `a355ea67605dea3e2e663e94eb23416ffeb7679757088a68dc6228c03da5a92b`.
- Donor `M`: `edm_1k_012000`, SHA-256
  `e5a7debafcd19191d6557f645216bfcb2e7589922396fd08130e76e3f5388b0a`.
- Pure-Euler 18-call sampler with float64 state, no churn, clamping,
  quantization, or Heun correction.
- Intervention calls `6..13`, corresponding to sigma values
  `8.400935309099816`, `5.315194521796382`, `3.256821519765537`,
  `1.9233398370400518`, `1.088170636545279`, `0.5853481231945422`,
  `0.29644228447915727`, and `0.13951646873101678`.
- Both denoisers evaluate the same current hybrid state. The Fourier mask acts
  on their denoiser predictions, never on `x_t`. Outside calls `6..13`, only
  `G` is evaluated.
- Seeds `50000..50999`, deterministic CPU float64 nearest-neighbor distances,
  stable `(distance, reference position)` ordering, and strict criterion
  `d1NN < d2NN / 3`.
- Official hash-pinned NVlabs EDM FID evaluator; FID-1K is descriptive only.
- Paired percentile bootstrap over latent seeds with 100,000 resamples and
  bootstrap seed `0`.

## Frozen Fourier Bands

Coordinates are the actual 32x32 `rfft2` grid:

```text
ky = torch.fft.fftfreq(32) * 32
kx = torch.fft.rfftfreq(32) * 32
r  = sqrt(kx^2 + ky^2)
```

All transforms use channelwise `rfft2/irfft2` with `norm="ortho"`.

| Band | Definition | Stored coefficients | Effective full-spectrum modes |
| --- | --- | ---: | ---: |
| B1 | `0 <= r <= 6` | 63 | 113 |
| B2 | `6 < r <= 9` | 73 | 140 |
| B3 | `9 < r <= 11` | 64 | 124 |
| B4 control | `11 < r <= 13` | 78 | 152 |

B1, B2, and B3 are pairwise disjoint and their exact Boolean union equals the
E011 mask `r <= 11`; their effective counts sum to `377`. B4 is an adjacent
control with a similar, not identical, mode count and is excluded from the
factorial decomposition.

## Conditions

The frozen registry is:

```text
000_none
100_B1
010_B2
001_B3
110_B1_B2
101_B1_B3
011_B2_B3
111_B1_B2_B3
control_B4
```

All nine conditions use the same 1,000 seeds, giving exactly 9,000 full-run
records. `000_none` is pure G. `111_B1_B2_B3` must be exactly equivalent to
the E011 radial `G <- M, r <= 11` implementation.

## Outcomes And Analyses

For every condition, report memorization count/rate, paired effect relative to
`000_none`, 95% paired bootstrap interval, FID-1K and delta FID relative to G,
and mean/median/25th/75th percentiles of `d1NN/d2NN`.

The factorial analysis reports paired main effects B1, B2, and B3; two-way
interactions B1xB2, B1xB3, and B2xB3; and the B1xB2xB3 interaction. Standard
2^3 contrast scaling is frozen: an order-k contrast averages matched finite
differences over the remaining `3-k` factors. These are contrasts on the
binary memorization outcome, not mechanistic or additive decompositions.

For every single- and pair-band condition, compare its memorized seed set with
`111`: intersection count, fraction of `111` recovered, Jaccard similarity,
and seeds memorized only by the sub-band condition.

Required figures show condition rates, factorial effects, overlap with `111`,
FID-1K, and the radial-band geometry. No sigma-frequency heatmap is permitted.

## Execution Gates

Before full execution, tests must establish exact band geometry, counts,
disjointness, union equality, `000`/G equivalence, `111`/E011 equivalence,
same-state evaluation, no donor evaluation outside calls `6..13`, deterministic
generation, exact seeds/record count, hash consistency, and collision-safe
outputs.

The smoke uses both seeds `50000,50001` and all nine conditions. It must
produce 18 finite records, exact repeated outputs, exact `000`/E011-G and
`111`/E011-radial equivalence for the first seed, successful FID invocation,
correct mask metadata, a nine-condition montage, and validation status `pass`.

The launcher has separate `smoke`, `full`, and `resume` modes. Passing smoke
does not start the full run. Full execution requires a later explicit user
authorization and writes to the immutable external directory
`/home/xggh8/data/zw-lab/e012_frequency_localization`.

## Stop And Interpretation Rules

Stop on any identity, geometry, equivalence, determinism, output, FID, or
record-count failure. Do not tune masks, calls, seeds, models, or metrics after
observing results. If B4 is similarly strong, report that this weakens
localization to `r <= 11`. E012 may describe localization, distributed effects,
or interactions within the frozen window; it must not claim that middle
frequencies cause memorization.
