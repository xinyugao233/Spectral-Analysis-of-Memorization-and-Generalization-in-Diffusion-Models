# Results Index

This directory contains compact, machine-readable outputs intended for code
review, plotting, and provenance checks. Large raw arrays, generated samples,
downloaded datasets, and checkpoints remain outside Git.

The [canonical pipeline registry](canonical_experiment_pipeline.json) records
the ordered E004-E008 roles, full-space and band-specific geometry targets,
the distinction between selection and spectral interpretation, and the
current intervention blockers.

## E009 Stage B Warm-Start Smoke

The [smoke validation](experiment_09_stage_b/smoke_validation.json),
[initialization record](experiment_09_stage_b/warm_start_initialization.json),
and [run manifest](experiment_09_stage_b/run_manifest.json) document the
successful 12K-to-13K warm-start gate. Large model artifacts remain external
under `/home/xggh8/data/zw-lab/e009_stage_b_edm5k_30000kimg`.

## E009 Stage B Baseline Gate

[`experiment_09_stage_b/`](experiment_09_stage_b/) now also contains the
frozen 24-checkpoint inventory, all 3,072 same-seed baseline records,
checkpoint summaries, zero-row failure table, deterministic smoke validation,
provenance manifests, final validation, and null pair-selection record.

All 18 warm-start 5K checkpoints at 13K through 30K scored `0/128`; five of
six EDM-1K checkpoints were eligible. The formal outcome is
`BLOCKED_NO_ELIGIBLE_5K_THROUGH_30K`. E008 and confirmatory seeds remain
untouched. See the [Stage B results document](../docs/experiment_09_stage_b_results.md).

## E004: Operational Frequency Cutoff

Files stored directly under `results/`:

| File | Purpose |
| --- | --- |
| `experiment_04_manifest.json` | Dataset, FFT, mask, and execution identity |
| `experiment_04_image_manifest.csv` | Frozen CIFAR-10 examples and indices |
| `experiment_04_cutoff_measurements.csv` | Per-image reconstruction diagnostics |
| `experiment_04_cutoff_energy.csv` | Retained and complementary energy fractions |
| `experiment_04_cutoff_review.csv` | Canonical review-row ordering |
| `experiment_04_reviewer_A.csv` | Blank reviewer template |
| `experiment_04_reviewer_B.csv` | Blank reviewer template |

The templates remain blank because the planned two-independent-reviewer
scoring procedure was not completed. The final `r = 4` decision is documented
as a single-reviewer qualitative result in
[the decision record](../docs/experiment_04_frequency_cutoff_decision.md).

## E004A: Paper Coverage-Concentration Geometry

[`experiment_04a/`](experiment_04a/) contains the clean-room coverage curve,
maximum-posterior-weight curve, validation record, and provenance manifest.
The result is imported from the validated full-space gate in the shared
research-context hub; the source hashes and deviations from the unavailable
paper execution are recorded in the manifest and
[source audit](../docs/paper_geometry_source_audit.md).

Curve rows use the stable schema `sigma_index`, `sigma`, `metric`, `estimate`,
`ci95_low`, `ci95_high`, `training_examples`, `query_examples`, `shell_c`,
`subset_sha256`, `seed`, and `status`. All source values are finite.

[`experiment_04a_reproduction/`](experiment_04a_reproduction/) contains a
fresh end-to-end local computation from the hash-verified CIFAR-10 archive.
It adds Monte Carlo standard errors, a per-sigma committed-versus-fresh
comparison, complete execution provenance, and regenerated figures. Both
metrics agree at every sigma under the tolerance frozen before execution.

The three `e006_grid_geometry.*` files in
[`experiment_04a/`](experiment_04a/) evaluate both E004A quantities directly
on the exact 18-point E006 schedule. The point-estimate and lower-bound
high-high sets are both indices `{8,9}` at `q_C=q_W=0.8`; no interpolation or
gap filling is used. Canonical distinctions among all historical regions are
stored in [`region_definition_registry.json`](region_definition_registry.json).

## E004B: Frequency-Restricted Geometry

[`experiment_04b/`](experiment_04b/) contains 108 finite curve rows, the
primary target summary, cutoff sensitivity, numerical projection validation,
and complete local provenance. At `r=4`, the lower-confidence-bound targets
are low `{8}` and high `{9,10}`. The high lower-bound target narrows to `{10}`
at sensitivity cutoff `r=5`; this is retained as a documented limitation.

## E005: Spectral Residual Curves

[`experiment_05/`](experiment_05/) contains aggregated curves, numerical
identity validation, transition-window extraction, and the compact provenance
manifest.

The 2,304,000-row per-sample residual table remains external at
`/home/xggh8/data/zw-lab/e005_spectral_residual_curves`. Its hash and
reproduction command are recorded in the
[E005 results document](../docs/experiment_05_spectral_residual_results.md).

## E006: Historical Spectral-Window Swaps

[`experiment_06/`](experiment_06/) contains condition summaries, paired
comparisons, the frozen `INCONCLUSIVE` outcome, validation and failure records,
qualitative-selection metadata, and the compact provenance manifest.

Generated samples, nearest-neighbor rows, and per-sample rows remain external
at `/home/xggh8/data/zw-lab/e006_transition_window_swaps`. They are excluded
from Git intentionally. See the
[E006 results document](../docs/experiment_06_transition_window_swap_results.md).

E006's internal `paper_medium_reference` key is retained for provenance. It is
a literature-derived Table 1 / Figure 10 compatibility condition, not an
E004A-derived geometry window.

## E008 Baseline Preflight

[`experiment_08_preflight/`](experiment_08_preflight/) contains the frozen
42-checkpoint inventory, all `5,376` per-seed no-swap pilot records, the
checkpoint summary, zero-row failure table, run manifest, validation record,
and blocked pair-selection outcome.

The formal result is `BLOCKED_NO_ELIGIBLE_PAIR`: six EDM-1K checkpoints were
eligible, but every one of the 21 EDM-50K checkpoints was `0/128`.
Confirmatory seeds `0..255` remain reserved and no E008 swap output exists.
E008 is now `RETIRED_UNEXECUTED`; the preserved historical outcome is
`BLOCKED_NO_ELIGIBLE_PAIR`. See the
[results document](../docs/experiment_08_checkpoint_preflight_results.md) and
[retirement decision](../docs/experiment_08_retirement_decision.md).

## E009 Stage A Baseline Gate

[`experiment_09_stage_a/`](experiment_09_stage_a/) contains the frozen
39-checkpoint inventory, checkpoint-pool manifest, identity sidecar, all 4,992
per-seed no-swap records, checkpoint summaries, zero-row failure table,
validation, smoke result, and pair-selection outcome. Inventory job `15720448`
accepted the complete 2K, 5K, and 10K `0..12K` trajectories. Pilot array
`15720492` completed with zero failures. Only 2K at 12K kimg was eligible;
therefore the formal outcome is `PROVISIONAL_2K_ONLY_STAGE_B_REQUIRED` and no
pair was selected. See the [results document](../docs/experiment_09_stage_a_results.md).

## E010 Directional Memorization Transfer

[`experiment_10/`](experiment_10/) contains the 14-condition summary, paired
transition counts, four preregistered directional contrasts, formal outcome,
zero-row failure table, validation, and compact provenance manifest. The sole
supported label is `HIGH_DERIVED_SUPPRESSION_SUPPORTED`.

The 3,584-row per-sample table and generated samples remain external at
`/home/xggh8/data/zw-lab/e010_directional_memorization_transfer`. E010 uses
asymmetric baselines and does not execute or revise E008. See the
[E010 results document](../docs/experiment_10_directional_memorization_transfer_results.md).

## E011 Fourier-Component Swap

[`experiment_11/`](experiment_11/) contains the compact eight-condition
summary, paired effects, radial-versus-square comparison, FID-1K records,
execution manifest, and validation record. Raw samples, generated FID images,
and per-sample nearest-neighbor rows remain external. The radial `G <- M`
condition reached `76.1%` memorization and the matched square reached `73.7%`.

## E012 Radial Frequency Localization

[`experiment_12/`](experiment_12/) contains the nine-condition summary,
factorial contrasts, full-mask overlap, nearest-neighbor diagnostics, FID-1K
records, execution manifest, and validation record. The full `r <= 11`
condition exactly reproduces E011 at `76.1%`; B1+B2 reaches `62.2%`, and the
dominant B1 x B2 interaction is `+52.35` percentage points. The 5.1 MB
per-sample table, 9,000 raw samples, and 9,000 FID images remain external.

## Artifact Policy

Commit an output only when it is small, deterministic, reviewable, and needed
to verify a reported conclusion. External raw artifacts must be identified by
storage path, row count or shape, SHA-256, and a reproduction command in the
corresponding manifest or results document.
