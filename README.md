# Spectral Diffusion Playground

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Tests: 206 passing](https://img.shields.io/badge/tests-206%20passing-2ea44f)](#installation-and-reproduction)
[![License: MIT](https://img.shields.io/badge/license-MIT-0b7285)](LICENSE)
[![Tag: portfolio-v1](https://img.shields.io/badge/tag-portfolio--v1-c2410c)](https://github.com/xinyugao233/spectral-diffusion-playground/tree/portfolio-v1)

A research portfolio using frequency-resolved data geometry to ask **when
along a diffusion trajectory an intervention can change memorization**.

## Latest Results: Fourier Memorization Transfer

E011 and E012 extend the earlier geometry-guided timing analysis with
frequency-selective Fourier swapping.

### E011: Whole-Denoiser Prediction Transfer

```text
G baseline:                    0.0%
G <- M, radial r <= 11:       76.1%
paired effect:               +76.1 pp
95% CI:                [+73.4, +78.7]
FID-1K:              34.9014 -> 38.0190
```

During sampler calls `6..13` (`sigma` approximately `8.40` down to `0.14`),
the base and donor models receive the same current sampler state. Selected
Fourier coefficients from the memorizing model `M` are copied into the
generalizing model's `G` **denoiser prediction**, not into `x_t` or completed
images. The `r <= 11` mask contains low-to-intermediate radial frequencies; it
is not a narrow middle-frequency ring.

![E011 Fourier memorization transfer](figures/experiment_11/experiment_11_memorization_rates.png)

### E012: Radial Frequency Localization

```text
B1:                   9.5%
B2:                   0.0%
B3:                   0.0%
B1+B2:               62.2%
B1+B2+B3:            76.1%
adjacent B4:          0.0%
```

The `B1 x B2` factorial interaction is `+52.35 pp` with 95% CI
`[+49.75, +54.95]`. `B1+B2` recovers `81.7%` of the full-mask memorized
seeds. These results show strong frequency-selective transfer and an
interaction-dominated radial localization under this intervention. They do
not establish that particular frequencies universally cause diffusion
memorization.

![E012 radial frequency localization](figures/experiment_12/experiment_12_memorization_rates.png)
**Quick links:** [Main Result](#main-result) ·
[Method](#from-shell-geometry-to-intervention) ·
[Detailed Experiments](#detailed-research-record) ·
[Reproduction](#installation-and-reproduction)

## Main Result

> **Swapping the whole denoiser at the high-frequency-derived sampled sigma
> values `1.9233` and `1.0882` suppressed memorization more strongly than both
> neighboring width-matched controls.**

```text
high target final memorization rate:       141/256 = 55.08%
neighboring final memorization rates:      167/256 = 65.23%
                                          170/256 = 66.41%

target effect - mean(control effects):                 10.74%
paired bootstrap 95% CI:                      [6.84%, 14.84%]
```

![Low- and high-frequency-derived whole-denoiser swap results](figures/experiment_10/low_high_directional_swap_table.png)

[Open the publication-quality PDF](figures/experiment_10/low_high_directional_swap_table.pdf)

Only high-derived suppression passed the preregistered influence criterion:
**`HIGH_DERIVED_SUPPRESSION_SUPPORTED`**. Low-derived suppression was not
supported. Induction was floor-limited by the EDM-50K recipient's `0/256`
no-swap baseline under the frozen E010 seeds.

![Target-minus-control contrasts with paired confidence intervals](figures/experiment_10/target_control_contrasts.png)

Frequency-resolved geometry determines **when** the whole denoiser is swapped;
frequency is not itself the manipulated model component. This is a timing
result for a whole-denoiser intervention, not evidence that high-frequency
components themselves cause memorization.

## From Shell Geometry To Intervention

### Why A Candidate Danger Region Should Exist

Our starting idea is simple: memorization should be most likely when the
training set both supervises a broad region of noisy space and gives the model
a clear direction toward one particular training example. Shell coverage
`C_sigma` tells us roughly how broadly the training-set geometry supervises
noisy space, while maximum posterior weight `W_sigma` tells us roughly whether
the model knows which individual training example to approach.

At large `sigma`, coverage can be broad but the posterior is scattered: many
training examples remain possible, but none supplies a strong direction. In
the middle, coverage and posterior concentration can both be high, combining
broad supervision with a clear example-specific direction. This is the
candidate memorization-danger regime. At very small `sigma`, the posterior can
be concentrated while the shell-covered region is small: the nearby example
may be clear, but the supervised region is restricted. This is a motivating
geometric hypothesis, not a proved causal mechanism or a validated universal
danger zone.

This analysis is a paper-derived clean-room extension of
[*Two Calm Ends and the Wild Middle: A Geometric Picture of Memorization in
Diffusion Models*](https://arxiv.org/abs/2602.17846) and does not claim exact
reproduction of the paper's unavailable executed code.

### Find Low- And High-Frequency Candidate Regions

We ask whether this candidate regime appears at different points of the
diffusion trajectory for coarse and fine image structure. The Fourier cutoff
is frozen first at `r = 4`, giving complementary low-frequency (`r <= 4`) and
high-frequency (`r > 4`) measurement subspaces. Here `r` says **what frequency
subspace is measured**, while `sigma` says **where the sampler is along the
diffusion trajectory**. The cutoff is an operational CIFAR-10 choice, with
sensitivity checks at `r = 3,5`, rather than a universal semantic boundary.

Coverage and posterior concentration are then computed independently as
`C_low(sigma), W_low(sigma)` and `C_high(sigma), W_high(sigma)`. On the frozen
18-point schedule, the low-frequency geometry selects the sampled candidate
`sigma = 3.2568`; the high-frequency geometry selects the sampled candidates
`sigma = 1.9233, 1.0882`. These are sampled locations, not evidence that every
continuous value between them satisfies the selection rule.

![Four frequency-restricted geometry curves and frozen targets](figures/experiment_10/geometry_targets.png)

### Test The Predicted Sigma Locations

The recipient model runs normally until it reaches a geometry-selected sigma
value. At that point the donor temporarily replaces the **whole denoiser**;
the recipient resumes immediately afterward, and final memorization is
compared with preregistered neighboring controls. Sigma itself is not swapped,
and neither low- nor high-frequency components are swapped in isolation.

```text
-> E004: freeze r = 4
-> E004B: low/high coverage and posterior geometry
        |
        +---------------------+
        |                     |
   r <= 4: low           r > 4: high
        |                     |
 C_low, W_low           C_high, W_high
        |                     |
 sigma = 3.2568         sigma = 1.9233, 1.0882
        +----------+----------+
                   |
-> E010: whole-denoiser swaps
                   |
        neighboring controls
                   |
-> high-derived suppression supported
```

---

## Detailed Research Record

The sections below preserve the full experiment-by-experiment methods,
results, reproduction details, and negative evidence. They are not required
to understand the main narrative above.

The first four experiments establish the Fourier foundations and freeze the
operational cutoff before any model intervention is evaluated.

### E001: From Pixels To Frequency Space

**Question.** What information does a channelwise 2D Fourier transform expose,
and can the original image be recovered? E001 computes the centered FFT,
linear and log-magnitude spectra, and inverse FFT with orthonormal
normalization. The reconstruction matches the input up to numerical precision.

![Original image, Fourier spectra, and inverse reconstruction](figures/understanding_images_in_fourier_space_default_fft_reference_rgb.png)

- **Purpose:** Build intuition for the reversible pixel-space to frequency-space
  transformation.
- **Main artifact:** [RGB FFT visualization](figures/understanding_images_in_fourier_space_default_fft_reference_rgb.png)
  and [RGB/grayscale comparison](figures/understanding_images_in_fourier_space_default_fft_reference_rgb_and_grayscale.png).
- **Takeaway:** Fourier space reorganizes image variation by spatial frequency;
  it does not discard information by itself.

```bash
python experiments/01_fft_visualization.py
python experiments/01_fft_visualization.py --grayscale
```

### E002: How Gaussian Noise Changes Frequency Content

**Question.** How does additive white Gaussian noise alter an image and its
spectrum? E002 evaluates five frozen noise levels and compares pixel-space
corruption with log spectra and radial energy distributions. Increasing noise
raises energy broadly across spatial frequencies and makes the normalized
DC-excluded profile more uniform.

![Gaussian noise in pixel and frequency space](figures/how_gaussian_noise_changes_frequency_content_default_fft_reference_seed0_grid.png)

- **Purpose:** Connect pixel-space corruption to frequency-space behavior.
- **Main artifact:** [Noise/frequency grid](figures/how_gaussian_noise_changes_frequency_content_default_fft_reference_seed0_grid.png)
  and [normalized radial distribution](figures/how_gaussian_noise_changes_frequency_content_default_fft_reference_seed0_normalized_radial_distribution.png).
- **Takeaway:** White noise contributes broadly rather than occupying one
  narrow frequency band.

```bash
python experiments/02_noise_vs_frequency.py
```

### E003: From Coarse Structure To Fine Variation

**Question.** What becomes visible as progressively larger centered frequency
regions are retained? E003 applies nested circular low-pass masks and exact
complementary high-pass masks. The low-pass reconstruction and signed residual
sum back to the input, while the reconstruction-error curve quantifies how
rapidly image content is recovered.

![Low-pass masks and cumulative image reconstruction](figures/where_image_information_lives_grid.png)

![Complementary high-frequency residuals](figures/high_frequency_residuals.png)

- **Purpose:** Establish the complementary projection operators used later.
- **Main artifact:** [Reconstruction grid](figures/where_image_information_lives_grid.png),
  [high-frequency residuals](figures/high_frequency_residuals.png), and
  [error curve](figures/reconstruction_error_vs_frequency_radius.png).
- **Takeaway:** Increasing radius progressively restores image variation, but
  frequency bands remain operational measurements rather than semantic labels.

```bash
python experiments/03_frequency_decomposition.py
```

### E004: Freezing An Operational Frequency Split

**Question.** Which centered radial cutoff provides a useful and auditable split
on 32 x 32 CIFAR-10 images? E004 freezes 20 examples, six candidate radii, exact
display conventions, and reconstruction/energy diagnostics before review. A
single disclosed reviewer selected `r = 4` as the smallest visually acceptable
cutoff; `r = 3` and `r = 5` remain the primary sensitivity cutoffs.

![Representative E004 cutoff montage for CIFAR-10 classes 0 and 1](figures/experiment_04_cutoff_montage_classes_0_1.png)

- **Purpose:** Freeze an operational frequency split before examining denoiser
  curves.
- **Main artifact:** [Five class-grouped cutoff montages](figures/README.md#e004-operational-frequency-cutoff)
  and the [decision record](docs/experiment_04_frequency_cutoff_decision.md).
- **Takeaway:** `r = 4` is a documented experimental choice, not a universal
  boundary between structure and detail.

The originally planned two-independent-reviewer scoring procedure was not
completed. No inter-rater or blinded-review claim is made.

### E004A: Full-Space Coverage And Posterior Concentration

The paper's geometric picture is defined by two quantities that are distinct
from the repository's Fourier residual energies.

For training set `D = {x_i}`, paper Eq. (3) assigns empirical-posterior weight

```text
w_i(y,sigma) = exp(-||y-x_i||^2 / (2 sigma^2))
               / sum_j exp(-||y-x_j||^2 / (2 sigma^2)).
```

Definition 4.1 averages the largest weight for noisy **training** queries:

```text
W_sigma(D) = E_{X ~ p_D, Z} max_i w_i(X + sigma Z, sigma).
```

Definition 4.6 measures whether a noisy **held-out** query lies in the union of
training-centered Gaussian shells:

```text
C_sigma(p,D) = P(X + sigma Z in union_i S_sigma(x_i)),  X ~ p.
```

Here `p` is the data distribution, not a scalar shell-probability parameter.
The shell radii use the paper's `c = 5` convention and include both annulus
boundaries. Coverage is an exact binary union-of-annuli event; it is not a
nearest-neighbor-distance substitute.

![Gaussian-shell coverage and maximum posterior weight across noise levels](figures/experiment_04a/coverage_and_max_posterior_weight.png)

The relationship motivates three qualitative regimes:

- **Small noise:** posterior concentration is high, but held-out shell coverage
  is limited.
- **Medium noise:** coverage and posterior concentration can be simultaneously
  high, producing a clean-room candidate geometric region under the explicit
  thresholds `q_C = q_W = 0.8`.
- **Large noise:** coverage is broad, but posterior concentration is weak.

The paper does not provide a universal numerical boundary for these regimes.
The clean-room protocol froze exploratory thresholds `q_W = q_C = 0.8` before
execution; sampled full-space point estimates are high-high at
`sigma in {2,3,4,5}`. Continuous shading from `2` to `5` is visual only, and
decisions use observed grid points without interpolation.

Evaluating the same definitions directly on the exact 18-point E006 sampler
grid selects indices `{8,9}`, at `sigma in {3.2568215, 1.9233398}`, under both
the point-estimate and 95% lower-bound rules. This is the **E004A clean-room
geometric high-high region** at `q_C=q_W=0.8`, not a universal paper boundary.

![E004A geometry evaluated on the exact E006 schedule](figures/experiment_04a/e006_grid_geometry_alignment.png)

This is a **paper-derived clean-room reproduction**, not an exact numerical
reproduction. It uses the first 1,000 canonical CIFAR-10 training and test
images, seed `0`, and a frozen 20-point sigma grid because the paper's exact
subset, seeds, grid, and executed Figure 3 code were unavailable. See the
[source audit](docs/paper_geometry_source_audit.md), [frozen protocol](docs/experiment_04a_paper_geometry_protocol.md),
and [result record](docs/experiment_04a_paper_geometry_results.md).

The committed estimates can also be regenerated end to end from a local
CIFAR-10 archive using the frozen configuration:

```bash
python experiments/04a_paper_geometry_curves.py \
  --compute \
  --dataset-root /path/to/cifar10 \
  --output-dir results/experiment_04a_reproduction \
  --device auto
```

This mode computes both curves from images and deterministic Gaussian draws;
it does not use committed estimates as numerical inputs. `--plot-only` retains
the lightweight figure-regeneration path, and `--validate-only` checks an
existing result directory without recomputation.

The recorded end-to-end CPU run completed in 3.31 seconds with 441.8 MiB peak
resident memory. Every sigma passed the tolerance frozen before execution;
maximum absolute differences were `1.11e-16` for coverage and `4.72e-16` for
maximum posterior weight. The sampled high-high set remained `{2,3,4,5}`.
See the [fresh reproduction artifacts](results/experiment_04a_reproduction/)
and [comparison record](results/experiment_04a_reproduction/reproduction_comparison.csv).

This is an end-to-end local regeneration under the frozen clean-room
configuration. It is independent of the committed curve estimates, but uses
the same subset, seed, sigma grid, estimator, normalization, and Gaussian
draws. The near-exact agreement establishes deterministic reproducibility, not
robustness across alternative subsets or seeds.

### E004B: When Do Low And High Geometry Become Distinctive?

**Question.** Does the paper-derived coverage/concentration geometry occupy
the frozen low- and high-frequency subspaces at the same noise levels? E004B
uses the same CIFAR-10 examples, 18-point schedule, Gaussian corruptions, and
definitions as E004A, then computes both quantities after exact complementary
Fourier projection.

E004B draws two curves in the low-frequency subspace and two curves in the
high-frequency subspace: coverage and maximum posterior weight in each space:

```text
C_low_sigma(p,D)
W_low_sigma(D)
C_high_sigma(p,D)
W_high_sigma(D)
```

This distinction is structural:

```text
E004B = frequency-restricted data geometry
E005  = frequency-restricted denoising residual energy
```

![Low-frequency coverage and posterior concentration](figures/experiment_04b/low_frequency_coverage_and_posterior.png)

![High-frequency coverage and posterior concentration](figures/experiment_04b/high_frequency_coverage_and_posterior.png)

On the frozen 18-point schedule, the conservative rule selects a sampled noise
level only when the 95% lower confidence bounds for both quantities reach
`0.8`:

```text
D_b^grid = {sigma_i : LCB(C_b(sigma_i)) >= 0.8
                       and LCB(W_b(sigma_i)) >= 0.8}
```

At the primary cutoff `r=4`, the exact real subspace ranks are 147 (low) and
2,925 (high). Applying the frozen `q_C=q_W=0.8` lower-confidence-bound rule
selects the low-band candidate at `sigma=3.2568215` and the high-band candidate
at sampled values `sigma={1.9233398,1.0881706}`. For exact implementation and
reproduction, these map to sampler calls `{8}` and `{9,10}`, respectively.
Point-estimate classification agrees at the primary cutoff.

**Candidate-region interpretation:** at `r=4`, the joint low-band candidate is
the single sampled location `sigma=3.2568`; the joint high-band candidate is
the two-point set `sigma={1.9233,1.0882}`. These are descriptive sampled-grid
objects, not interpolated continuous zones. Writing the visual span as
approximately `1.09-1.92` must not imply that unmeasured values between those
points passed the rule. The large rank difference between the subspaces
prevents attributing their ordering to frequency organization alone.

At `r=4`, the low- and high-frequency projectors have substantially different
ranks, `147` and `2925`. E004B therefore measures the geometry of the actual
operational Fourier decomposition, but it does not isolate frequency
organization from subspace dimension, covariance, or energy structure.
Rank- and power-matched control subspaces would be required before attributing
the observed ordering specifically to frequency.

The low-band joint high-coverage/high-posterior target occurs one sampler
index earlier than the high-band target. This statement concerns the joint
high-high regions. Coverage alone does not exhibit the same ordering: the
high-band coverage threshold persists to lower sigma, while low-band posterior
concentration persists farther toward high noise.

The low target is unchanged at `r=3,4,5`. The high target is `{9,10}` at
`r=3,4`, while its lower-bound sensitivity result narrows to `{10}` at `r=5`.
This dependence remains visible and does not revise `r=4`.

- **Purpose:** Extend the paper geometry into preregistered complementary
  frequency subspaces without using denoiser outputs.
- **Main artifact:** [Band comparison](figures/experiment_04b/low_high_geometry_comparison.png),
  [cutoff sensitivity](figures/experiment_04b/frequency_geometry_cutoff_sensitivity.png),
  and [validated results](docs/experiment_04b_frequency_restricted_geometry_results.md).
- **Takeaway:** Under the frozen clean-room setup, the low-band candidate is
  `sigma=3.2568`, while the high-band candidate is the next two sampled
  lower-noise values `sigma={1.9233,1.0882}`. This is descriptive geometry,
  not causal or memorization evidence.

```bash
python experiments/04b_frequency_restricted_geometry.py \
  --compute \
  --dataset-root /path/to/cifar10 \
  --output-dir results/experiment_04b_reproduction \
  --figure-dir figures/experiment_04b_reproduction \
  --device cpu \
  --cutoffs 3 4 5
```

### E005: Supporting Residual Dynamics

E005 projects a fixed denoiser's residual into exact complementary Fourier
bands. Under the frozen 20%-to-80% rule at `r=4`, the low-frequency residual
transition is `5..11` (`sigma=12.9101..0.585348`) and the high-frequency
transition is `11..14` (`sigma=0.585348..0.0599473`). This is a descriptive
coarse-to-fine residual pattern, not the E004B geometry rule and not a claim
about when either band was learned.

See the [protocol](docs/experiment_05_spectral_residual_protocol.md),
[model provenance](docs/experiment_05_clean_room_models.md), and
[results](docs/experiment_05_spectral_residual_results.md).

### E006: Historical Exploratory Swaps

**E006** was formally `INCONCLUSIVE`. It tested whole-denoiser swaps aligned
with the earlier E005 residual windows, but the EDM-50K
no-swap baseline was degenerate at `0/256`. Descriptively, the low residual
window passed its frozen influence test in both directions and the high
residual window passed in neither. E006 did not identify a memorization danger
zone and did not test the later E004B target.

See the [protocol](docs/experiment_06_transition_swap_protocol.md),
[results](docs/experiment_06_transition_window_swap_results.md), and
[region-definition audit](docs/danger_zone_definition_audit.md).

**E007-E009** document blocked, retired, and negative paths that motivated the
final directional E010 design. Their individual records follow.

### E007: Blocked Full-Space Intervention

E007 preserves a proposed full-space geometry intervention but is
`PROPOSED — BLOCKED BY KNOWN BASELINE DEGENERACY`; it was not executed. The
[blocked protocol](docs/experiment_07_geometry_aligned_swap_protocol.md)
preserves the design and its stopping condition.

### E008: Retired Symmetric-Pair Design

E008 is `RETIRED_UNEXECUTED`. Its baseline-only preflight found no eligible
model pair, yielding `BLOCKED_NO_ELIGIBLE_PAIR` before any swap. See the
[baseline preflight](docs/experiment_08_checkpoint_preflight.md),
[preflight results](docs/experiment_08_checkpoint_preflight_results.md),
[frozen swap protocol](docs/experiment_08_frequency_geometry_swap_protocol.md),
and [retirement decision](docs/experiment_08_retirement_decision.md).

### E009: Negative Model-Pair Search

E009 searched matched 2K/5K/10K trajectories and extended the 5K lineage
through 30K kimg. Every Stage B 5K checkpoint scored `0/128`, yielding
`BLOCKED_NO_ELIGIBLE_5K_THROUGH_30K`. See the
[Stage A protocol](docs/experiment_09_intermediate_dataset_training_design.md),
[Stage A results](docs/experiment_09_stage_a_results.md),
[Stage B protocol](docs/experiment_09_stage_b_protocol.md), and
[Stage B results](docs/experiment_09_stage_b_results.md).

### E010: Directional Whole-Denoiser Intervention

This is the full E010 record. E010 tests whether intervention at the
E004B-selected sigma locations changes final-sample memorization more than
intervention at neighboring controls. At each listed sigma value, the donor
replaces the recipient's **whole denoiser**. The experiment does not replace
sigma, low-frequency coefficients, high-frequency coefficients, or any
isolated Fourier component.

The low target is the single sampled value `sigma=3.2568` (sampler call `{8}`).
The high target is the two sampled values `sigma={1.9233,1.0882}` (sampler
calls `{9,10}`). The same 256 latent seeds are used across each target and its
preregistered neighboring controls.

The two directional questions are:

- **Suppression:** memorizing recipient + empirically generalizing,
  floor-baseline donor. Can the donor suppress memorization?
- **Induction:** empirically generalizing recipient + memorizing donor. Can the
  donor induce memorization?

![Frozen low/high targets and neighboring whole-denoiser swap controls](figures/experiment_10/condition_map.png)

```text
memorizing no-swap baseline:               215/256 = 83.98%
generalizing no-swap baseline:               0/256 observed memorized samples

high target final memorization rate:       141/256 = 55.08%
neighboring final memorization rates:      167/256 = 65.23%
                                          170/256 = 66.41%

high target suppression effect:                       28.91%
neighboring suppression effects:            18.75%, 17.58%
target effect - mean(control effects):                 10.74%
paired bootstrap 95% CI:                      [6.84%, 14.84%]
```

![Suppression-direction memorization rates](figures/experiment_10/suppression_rates.png)

![Induction-direction memorization rates](figures/experiment_10/induction_rates.png)

Only high-derived suppression passed the preregistered influence criterion:
**`HIGH_DERIVED_SUPPRESSION_SUPPORTED`**. Low-derived suppression was not
supported. Induction was not supported and was floor-limited by the EDM-50K
recipient: no memorized samples were observed for its `0/256` no-swap baseline
under the frozen E010 seeds.

At the low-derived candidate `sigma = 3.2568`, the final memorization rate was
`188/256 = 73.44%`; the later neighboring control was lower at
`182/256 = 71.09%`. The low location therefore was not selectively special
under the frozen criterion. At the high-derived sampled locations, the final
rate was `141/256 = 55.08%`, below both controls at `65.23%` and `66.41%`.
The strongest evidence is not merely a lower target rate; it is that the
suppression effect exceeded both neighboring controls and its paired confidence
interval excluded zero.

Complete intervention definitions, both directions, and the frozen decision
rule remain in the
[E010 protocol](docs/experiment_10_directional_memorization_transfer_protocol.md),
[analysis plan](docs/experiment_10_directional_analysis_plan.md), and
[validated results](docs/experiment_10_directional_memorization_transfer_results.md).

### E011: Fourier-Component Memorization Transfer

E011 tests same-state Fourier swapping on denoiser predictions rather than on
the current noisy state or completed images. During sampler calls `6..13`,
selected coefficients from the memorizing model are copied into the
generalizing model over the radial `r <= 11` mask. The generalizing baseline
was `0.0%`, while `G <- M` reached `76.1%` with a paired effect of `+76.1 pp`
and 95% CI `[+73.4,+78.7]`. A mode-count-matched square mask reached `73.7%`.
The reverse direction has a substantial quality caveat. See the
[E011 protocol](docs/experiment_11_fourier_component_swap_protocol.md),
[results](docs/experiment_11_fourier_component_swap_results.md),
[compact results](results/experiment_11/), and
[figures](figures/experiment_11/).

### E012: Radial Frequency Localization

E012 partitions the radial `r <= 11` mask into B1, B2, and B3 and tests their
factorial combinations while holding the intervention timing fixed. The
observed rates were `B1=9.5%`, `B2=0.0%`, `B3=0.0%`, `B1+B2=62.2%`,
`B1+B2+B3=76.1%`, and adjacent `B4=0.0%`. The B1 x B2 interaction was
`+52.35 pp`. This localizes the tested transfer to an interaction among radial
bands; it does not show that any frequency band universally causes
memorization. See the
[E012 protocol](docs/experiment_12_frequency_localization_protocol.md),
[results](docs/experiment_12_frequency_localization_results.md),
[compact results](results/experiment_12/), and
[figures](figures/experiment_12/).
## Mathematical Core

The paper geometry first measures full-space coverage and concentration:

```text
C_sigma(p,D) = P(X + sigma Z in union_i S_sigma(x_i))
W_sigma(D)   = E max_i w_i(X + sigma Z, sigma)
```

E004B evaluates the same definitions on projected data and projected Gaussian
corruptions. Its shell dimensions are the exact real projector ranks:

```text
C_sigma^b(p,D), W_sigma^b(D),  b in {low, high}
d_low(r=4) = 147, d_high(r=4) = 2925
```

E005 then asks a different question by applying complementary Fourier
projections to a fixed denoiser's residual.

E005 applies complementary Fourier projections directly to

```text
e_sigma = m_sigma(X + sigma Z) - X
```

and measures

```text
E_full = ||e_sigma||_2^2
E_low  = ||P_low,r e_sigma||_2^2
E_high = ||P_high,r e_sigma||_2^2
```

The channelwise 2D FFT uses `norm="ortho"`; the high-frequency mask is the exact
complement of the centered low-frequency mask. The resulting band energies sum
to the full residual energy within the frozen tolerance.

## Documentation And Results

| Experiment | Protocol and result narrative | Compact results | Figures |
| --- | --- | --- | --- |
| E004 | [Protocol](docs/experiment_04_frequency_cutoff_protocol.md) · [Reviewer instructions](docs/experiment_04_reviewer_instructions.md) · [Decision](docs/experiment_04_frequency_cutoff_decision.md) | [Results index](results/README.md#e004-operational-frequency-cutoff) | [Montages](figures/README.md#e004-operational-frequency-cutoff) |
| E004A | [Source audit](docs/paper_geometry_source_audit.md) · [Protocol](docs/experiment_04a_paper_geometry_protocol.md) · [Results](docs/experiment_04a_paper_geometry_results.md) | [`results/experiment_04a/`](results/experiment_04a/) | [`figures/experiment_04a/`](figures/experiment_04a/) |
| E004B | [Protocol](docs/experiment_04b_frequency_restricted_geometry_protocol.md) · [Results](docs/experiment_04b_frequency_restricted_geometry_results.md) | [`results/experiment_04b/`](results/experiment_04b/) | [`figures/experiment_04b/`](figures/experiment_04b/) |
| E005 | [Protocol](docs/experiment_05_spectral_residual_protocol.md) · [Model provenance](docs/experiment_05_clean_room_models.md) · [Results](docs/experiment_05_spectral_residual_results.md) | [`results/experiment_05/`](results/experiment_05/) | [`figures/experiment_05/`](figures/experiment_05/) |
| E006 | [Protocol](docs/experiment_06_transition_swap_protocol.md) · [Results](docs/experiment_06_transition_window_swap_results.md) | [`results/experiment_06/`](results/experiment_06/) | [`figures/experiment_06/`](figures/experiment_06/) |
| E007 | [Blocked proposed protocol](docs/experiment_07_geometry_aligned_swap_protocol.md) | Blocked; not executed | Not generated |
| E008 | [Baseline preflight](docs/experiment_08_checkpoint_preflight.md) · [Results](docs/experiment_08_checkpoint_preflight_results.md) · [Frozen swap protocol](docs/experiment_08_frequency_geometry_swap_protocol.md) · [Retirement](docs/experiment_08_retirement_decision.md) | [`results/experiment_08_preflight/`](results/experiment_08_preflight/) | [`figures/experiment_08_preflight/`](figures/experiment_08_preflight/) |
| E009 | [Stage A protocol](docs/experiment_09_intermediate_dataset_training_design.md) · [Stage A results](docs/experiment_09_stage_a_results.md) · [Stage B protocol](docs/experiment_09_stage_b_protocol.md) · [Stage B results](docs/experiment_09_stage_b_results.md) | [`results/experiment_09_stage_a/`](results/experiment_09_stage_a/) · [`results/experiment_09_stage_b/`](results/experiment_09_stage_b/) | [`figures/experiment_09_stage_a/`](figures/experiment_09_stage_a/) · [`figures/experiment_09_stage_b/`](figures/experiment_09_stage_b/) |
| E010 | [Protocol](docs/experiment_10_directional_memorization_transfer_protocol.md) · [Analysis](docs/experiment_10_directional_analysis_plan.md) · [Results](docs/experiment_10_directional_memorization_transfer_results.md) | [`results/experiment_10/`](results/experiment_10/) | [`figures/experiment_10/`](figures/experiment_10/) |
| E011 | [Protocol](docs/experiment_11_fourier_component_swap_protocol.md) · [Results](docs/experiment_11_fourier_component_swap_results.md) | [`results/experiment_11/`](results/experiment_11/) | [`figures/experiment_11/`](figures/experiment_11/) |
| E012 | [Protocol](docs/experiment_12_frequency_localization_protocol.md) · [Results](docs/experiment_12_frequency_localization_results.md) | [`results/experiment_12/`](results/experiment_12/) | [`figures/experiment_12/`](figures/experiment_12/) |

See the [documentation index](docs/README.md), [results index](results/README.md),
and [figures index](figures/README.md) for the complete navigation map.

## Installation And Reproduction

Python 3.11 or newer is required for the reusable local experiments.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python -m unittest discover tests
```

E004, E004A, and E004B require an existing CIFAR-10 root. E005, E006, and E010
require the frozen external archive, matched EDM checkpoints, and recorded
Hellbender environment. Exact hashes, configurations, paths, and Slurm commands
are recorded in their protocol and result documents. No experiment downloads
data or checkpoints implicitly.

## Reproducibility And Provenance

Scientific choices were frozen before evaluation. Key commits are:

| Milestone | Commit |
| --- | --- |
| E004 cutoff implementation | `a745cf1805deea0691fc3c43a591315b8a63984a` |
| E004 cutoff decision | `59b558e` |
| E005 evaluator | `b16c3a9c8224755cc2a5a52b0f1aacff44a63da7` |
| E005 results | `52d6889` |
| E006 protocol | `068c7e3a745fb51b1d2416524b7e29f70b0b5f08` |
| E006 executed implementation | `ae0febb9b983c50c5946d61423fda72358887523` |
| E006 results | `df06e4fe3d9350988a5882b8d17db45c8ef6645f` |
| E010 protocol and implementation | `cb6c17208dab9ef8af80135ea6ead40cd2a439fc` |
| E010 results | `6461316e6599c3c085ca8e189f541c68d4e7736a` |
| E010 merge | `30ad164f846d72721e58c2599e3bcd6aee43c957` |

Historical E005/E006 checkpoint pair:

```text
EDM-1K:  8e53dd93177c0144d38508c5634ae9ffbce303b6c8209af65085d376ce9026a1
EDM-50K: a355ea67605dea3e2e663e94eb23416ffeb7679757088a68dc6228c03da5a92b
```

Frozen E010 checkpoint pair:

```text
Memorizing EDM-1K 12K:                  e5a7debafcd19191d6557f645216bfcb2e7589922396fd08130e76e3f5388b0a
Empirically generalizing EDM-50K 40K:  a355ea67605dea3e2e663e94eb23416ffeb7679757088a68dc6228c03da5a92b
```

Only compact summaries, manifests, validation records, and final figures are
committed. Large raw artifacts remain external so the repository stays
reviewable and practical to clone:

```text
E005: /home/xggh8/data/zw-lab/e005_spectral_residual_curves
E006: /home/xggh8/data/zw-lab/e006_transition_window_swaps
E010: /home/xggh8/data/zw-lab/e010_directional_memorization_transfer
```

## Repository Layout

```text
spectral-diffusion-playground/
├── assets/       # deterministic examples and documented image provenance
├── configs/      # frozen geometry and model-experiment configurations
├── data/         # small versioned manifests, never downloaded datasets
├── docs/         # protocols, provenance records, and result narratives
├── experiments/  # executable experiment entry points and evaluators
├── figures/      # curated, reviewable figures
├── results/      # compact machine-readable outputs
├── scripts/      # guarded preflight and Slurm launchers
├── src/          # reusable FFT, filtering, evaluation, and plotting code
└── tests/        # numerical identities, determinism, schemas, and safeguards
```

### Scope Of The Result

E010 establishes a selective timing association for one asymmetric model pair
under a whole-denoiser intervention. It does **not** establish:

- high-frequency-component or fine-detail causality;
- dataset-size causality;
- a universal memorization danger zone;
- a continuous danger interval between sparsely sampled sigma values;
- a general impossibility of induction; or
- generalization beyond the tested pair, sampler, seeds, and memorization
  criterion.

Frequency-resolved geometry determines **when** the whole denoiser is swapped;
frequency is not itself the manipulated model component.

## Limitations

- E004 used one disclosed qualitative reviewer; the planned two-reviewer
  scoring procedure was not completed.
- The cutoff is CIFAR-10-specific and operational, not a universal semantic
  boundary.
- E004A is a clean-room reproduction using deterministic first-1K subsets,
  seed `0`, and a new 20-point grid. The paper's exact Figure 3 subset, seeds,
  grid, and executed code were unavailable.
- The E004A high-high thresholds are preregistered clean-room diagnostics, not
  universal boundaries supplied by the paper.
- E004B uses the same deterministic subset and corruption draws; it establishes
  reproducibility under that freeze, not robustness across datasets or seeds.
- E004B's high-band lower-confidence target narrows at `r=5`, so the result is
  not frequency-scale invariant.
- E004B uses an 18-point sigma grid. Its low candidate is only one sampled
  point, and its high candidate is two sampled points; denser local evaluation
  would be required to resolve continuous boundaries.
- E004B's low/high ranks are `147/2925` at `r=4`; rank, covariance, and power
  differences confound a frequency-only interpretation until matched controls
  are evaluated.
- E005 transition windows depend on the frozen clean-room model, schedule, and
  cutoff family; they do not identify when a model learned either band.
- E006 uses 256 seeds and a strict pixel-space criterion. Its degenerate
  EDM-50K baseline triggers the frozen safeguard and prevents a directional
  conclusion.
- E010 uses intentionally asymmetric baselines. Its EDM-50K floor limits
  induction sensitivity, and its whole-denoiser swaps do not identify an
  isolated frequency-component or dataset-size effect.
- E004A-E006 are paper-derived clean-room experiments, not exact reproductions
  of the paper's unavailable executed code.

## Citation And License

If you use this repository, cite it as software using [`CITATION.cff`](CITATION.cff)
and record the exact Git commit. Cite the grounding paper separately. Released
under the [MIT License](LICENSE).
