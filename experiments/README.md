# Experiment Entry Points

Each numbered script answers one narrow question and is executable from the
repository root. Reusable numerical and plotting logic belongs in
`src/spectral_diffusion_playground/`; scripts orchestrate frozen inputs and
write explicit outputs.

| Script | Role | Status |
| --- | --- | --- |
| `01_fft_visualization.py` | Reversible image/Fourier visualization | Complete |
| `02_noise_vs_frequency.py` | Gaussian noise and radial spectral energy | Complete |
| `03_frequency_decomposition.py` | Complementary low/high reconstruction | Complete |
| `04_frequency_cutoff.py` | Deterministic CIFAR-10 cutoff review packet | Complete |
| `04a_paper_geometry_curves.py` | Compute, validate, or plot the paper-derived clean-room coverage/concentration baseline | Complete |
| `04b_frequency_restricted_geometry.py` | Compute coverage/concentration separately in frozen low/high Fourier subspaces | Complete |
| `05_spectral_residual_curves.py` | Orthogonal fixed-sigma residual energies | Complete |
| `06_transition_window_swaps.py` | Historical spectral-window swaps | Complete; exploratory outcome `INCONCLUSIVE` |
| E007 protocol only | Geometry-aligned whole-denoiser swaps | Proposed; blocked by known baseline degeneracy |
| `08_checkpoint_baseline_preflight.py` | No-swap checkpoint eligibility preflight | Complete; `BLOCKED_NO_ELIGIBLE_PAIR` |
| E008 swap protocol | Frequency-geometry whole-denoiser swaps | `RETIRED_UNEXECUTED`; historical `BLOCKED_NO_ELIGIBLE_PAIR` |
| E009 protocols and evaluators | Intermediate-dataset model search | Complete; no eligible >=5K checkpoint through Stage B |
| `10_directional_memorization_transfer.py` | Directional whole-denoiser swaps over E004B-derived intervals | Complete; `HIGH_DERIVED_SUPPRESSION_SUPPORTED` |
| `11_fourier_component_swap.py` | Shared-state denoiser-prediction Fourier-component swaps | Ready for guarded smoke; no result yet |

## Local Foundations

```bash
python experiments/01_fft_visualization.py
python experiments/02_noise_vs_frequency.py
python experiments/03_frequency_decomposition.py
python experiments/04_frequency_cutoff.py --dataset-root /path/to/cifar10
python experiments/04a_paper_geometry_curves.py
python experiments/04a_paper_geometry_curves.py --validate-only
python experiments/04a_paper_geometry_curves.py --compute \
  --dataset-root /path/to/cifar10 \
  --output-dir results/experiment_04a_reproduction \
  --device auto
python experiments/04a_paper_geometry_curves.py --compute-e006-grid \
  --dataset-root /path/to/cifar10 \
  --device auto
python experiments/04b_frequency_restricted_geometry.py --compute \
  --dataset-root /path/to/cifar10 \
  --output-dir results/experiment_04b_reproduction \
  --figure-dir figures/experiment_04b_reproduction \
  --device cpu --cutoffs 3 4 5
```

## Model Experiments

E005 and E006 require the frozen external archive, clean-room checkpoints, and
recorded execution environment. Use the guarded Slurm launchers only after
verifying every hash and output-path collision gate:

```text
scripts/e005_eval_spectral_residuals.slurm
scripts/e006_eval_transition_swaps.slurm
scripts/e010_directional_transfer.slurm
scripts/e011_fourier_component_swap.slurm
```

Exact commands and identities are in the
[E005 results](../docs/experiment_05_spectral_residual_results.md) and
[E006 protocol](../docs/experiment_06_transition_swap_protocol.md).

E007 is protocol-only and has not been executed. Its geometry-aligned target
is frozen, but the historical E006 model pair is blocked by the known EDM-50K
`0/256` baseline. A baseline-only nondegenerate-model preflight is required by
[`docs/experiment_07_geometry_aligned_swap_protocol.md`](../docs/experiment_07_geometry_aligned_swap_protocol.md).

E008 swaps remain protocol-only and are now retired. The baseline preflight
found no eligible EDM-50K checkpoint among the 21 available snapshots. It
preregistered whole-denoiser swaps over the E004B low target `8..8` and high
target `9..10`; no E008 swap or confirmatory computation occurred. See
[`docs/experiment_08_frequency_geometry_swap_protocol.md`](../docs/experiment_08_frequency_geometry_swap_protocol.md).

E010 is a distinct asymmetric-baseline directional intervention. Its
[protocol](../docs/experiment_10_directional_memorization_transfer_protocol.md)
and [analysis plan](../docs/experiment_10_directional_analysis_plan.md) are
frozen before inference. It does not satisfy E008's baseline-matching gate.

E011 is a new frequency-component intervention using the same E010 model pair.
Its [protocol](../docs/experiment_11_fourier_component_swap_protocol.md)
freezes a radial `r<=11` try and a weighted-mode-count-matched square `K=9`
try over calls `6..13`. It has no results until the guarded smoke and full run
pass.

The separate [checkpoint preflight](../docs/experiment_08_checkpoint_preflight.md)
evaluates no-swap pilot baselines only. Its interface cannot accept a donor
model or swap window, and its reserved confirmatory seeds are untouched. The
[results](../docs/experiment_08_checkpoint_preflight_results.md) record
`BLOCKED_NO_ELIGIBLE_PAIR`.

The canonical order is documented in
[`docs/canonical_experiment_pipeline.md`](../docs/canonical_experiment_pipeline.md).
E004A selects the full-space target `8..9`; E004B independently selects
band-specific geometry targets; E005 provides residual-energy interpretation;
E006 is historical exploratory evidence; E007 remains proposed and E008 is
retired unexecuted.

## Output Discipline

- Write compact numerical summaries under `results/`.
- Write final figures under `figures/`.
- Keep downloaded datasets, checkpoints, generated samples, and large raw
  tables outside Git.
- Never adjust frozen cutoffs, windows, seeds, or decision rules after seeing
  model results.
