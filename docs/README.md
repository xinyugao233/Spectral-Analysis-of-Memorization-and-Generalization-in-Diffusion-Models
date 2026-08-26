# Documentation Index

Protocols freeze decisions before execution. Result documents report the
validated run without changing those decisions. Provenance documents connect
external data and checkpoints to exact hashes and commits.

Start with the [canonical experimental pipeline](canonical_experiment_pipeline.md).
It freezes the order E004 cutoff -> E004A full-space geometry -> E004B
frequency-restricted geometry -> E005 spectral interpretation -> historical
E006 exploration -> proposed E007/E008 interventions.

## E004: Operational CIFAR-10 Cutoff

1. [Frozen cutoff-selection protocol](experiment_04_frequency_cutoff_protocol.md)
2. [Reviewer instructions](experiment_04_reviewer_instructions.md)
3. [Final single-reviewer decision](experiment_04_frequency_cutoff_decision.md)

Result: operational reference `r = 4`, primary sensitivity `r = 3, 5`, and
optional extended sensitivity `r = 6`.

## E004A: Paper Coverage-Concentration Geometry

1. [Paper definition and source audit](paper_geometry_source_audit.md)
2. [Frozen clean-room protocol](experiment_04a_paper_geometry_protocol.md)
3. [Validated clean-room results](experiment_04a_paper_geometry_results.md)

Result: the paper's qualitative small-, medium-, and large-noise geometry is
recovered in a deterministic clean-room setup. The sampled primary-threshold
high-high region is `sigma in {2,3,4,5}`; it is not defined by E005.
Direct evaluation on the exact E006 schedule selects indices `{8,9}` under
both the point-estimate and 95% lower-bound rules at `q_C=q_W=0.8`.

## E004B: Frequency-Restricted Geometry

1. [Frozen clean-room protocol](experiment_04b_frequency_restricted_geometry_protocol.md)
2. [Validated clean-room results](experiment_04b_frequency_restricted_geometry_results.md)

Result at `r=4`: low-band target `{8}` and high-band target `{9,10}` under
the primary lower-confidence-bound rule. These are subspace coverage and
posterior-concentration curves, not E005 denoising residual energies.
E004B is complete as a descriptive result. Its low/high ranks are `147/2925`,
so rank, covariance, and energy differences prevent attribution to frequency
organization alone without matched controls.

## E005: Spectral Residual Curves

1. [Frozen residual-energy protocol](experiment_05_spectral_residual_protocol.md)
2. [Clean-room model and checkpoint provenance](experiment_05_clean_room_models.md)
3. [Validated residual-curve results](experiment_05_spectral_residual_results.md)

Result: an ordered low-frequency then high-frequency residual transition under
the frozen clean-room setup.

## E006: Historical Spectral-Window Swaps

1. [Frozen whole-denoiser swap protocol](experiment_06_transition_swap_protocol.md)
2. [Validated swap results](experiment_06_transition_window_swap_results.md)

Formal outcome: `INCONCLUSIVE` because the EDM-50K no-swap baseline was
degenerate at `0/256`. Descriptively, the E005 low-frequency spectral
transition passed the frozen influence test in both directions; the E005
high-frequency spectral transition did not.

## Region Definitions And Proposed E007

1. [Canonical region-definition audit](danger_zone_definition_audit.md)
2. [Blocked geometry-aligned E007 protocol](experiment_07_geometry_aligned_swap_protocol.md)

The audit distinguishes the literature-derived paper medium reference, E005
spectral transitions, and E004A clean-room geometric high-high set. E007 is
`PROPOSED — BLOCKED BY KNOWN BASELINE DEGENERACY` and does not alter
historical E006. The target remains indices `8..9`, but the historical E006
pair cannot support an informative bidirectional decision because its EDM-50K
baseline is already `0/256`.

The main scientific chain remains incomplete until E007 or an equivalent
preregistered geometry-aligned swap is executed with a nondegenerate model
pair. E004A selects the candidate region; E005 interprets its spectral
location; historical E006 does not substitute for that final test.

## Retired E008

[E008](experiment_08_frequency_geometry_swap_protocol.md) freezes only the
design for whole-denoiser swaps over the E004B low target `8..8` and high
target `9..10`. It is now `RETIRED_UNEXECUTED`; its primary bidirectional
design never passed the model-pair gate.

The [baseline-only checkpoint preflight](experiment_08_checkpoint_preflight.md)
froze the existing-snapshot inventory, disjoint pilot seeds, strict
eligibility interval, and deterministic pair-selection rule. Its
[validated results](experiment_08_checkpoint_preflight_results.md) are
`BLOCKED_NO_ELIGIBLE_PAIR`: six EDM-1K checkpoints passed, but all 21 EDM-50K
checkpoints were `0/128`. This gate did not execute E008. The
[retirement decision](experiment_08_retirement_decision.md) records that no
further E008 training or execution is planned.

The [frozen E009 Stage A protocol](experiment_09_intermediate_dataset_training_design.md)
uses nested class-balanced 2K/5K/10K subsets, matched 12K-kimg EDM runs, 1K
checkpoint cadence, new pilot seeds, and prospective stopping/pair-selection
rules. The Stage A training array completed at 12K kimg for all three roles.
Inventory job `15720448` then froze and accepted all 39 expected persistent
checkpoints. The subsequent baseline-only pilot is documented below.

The dated [E009 evaluation execution amendment](experiment_09_stage_a_evaluation_execution.md)
records the post-training inventory, smoke, and baseline-only pilot gates
without changing the frozen scientific settings.

The [validated Stage A results](experiment_09_stage_a_results.md) report all
4,992 no-swap records. Only the 2K 12K-kimg checkpoint was eligible; no 5K or
10K checkpoint qualified, so the frozen outcome is
`PROVISIONAL_2K_ONLY_STAGE_B_REQUIRED`. This triggered the separately frozen
Stage B warm-start extension; E008 remained unexecuted.

The [frozen Stage B protocol](experiment_09_stage_b_protocol.md) specifies a
single matched-exposure warm-start extension of the existing 5K state from
12K to 30K kimg. The [validated Stage B results](experiment_09_stage_b_results.md)
report 3,072 no-swap records with zero failures. Every 5K checkpoint scored
`0/128`, so the formal outcome is `BLOCKED_NO_ELIGIBLE_5K_THROUGH_30K` and
E008 remains unexecuted.

## E010: Directional Memorization Transfer

1. [Frozen protocol](experiment_10_directional_memorization_transfer_protocol.md)
2. [Frozen analysis plan](experiment_10_directional_analysis_plan.md)

E010 is a separate asymmetric-baseline experiment using existing checkpoints.
It tests whole-denoiser suppression and induction over the E004B-derived low
target `{8}` and high target `{9,10}` plus neighboring controls. It does not
revise or execute E008, and it does not claim frequency-component or
dataset-size causality. The
[validated results](experiment_10_directional_memorization_transfer_results.md)
support only high-derived suppression under the frozen criterion.

## E011: Wild-Middle Fourier-Component Swaps

1. [Frozen protocol](experiment_11_fourier_component_swap_protocol.md)

E011 is ready for guarded smoke validation. It reuses the E010 model pair but
intervenes on selected Fourier coefficients of the two models' denoiser
predictions evaluated on one shared state. Try 01 uses the radial disk
`r<=11`; Try 02 uses the closest weighted-mode-count square, `K=9`. Both use
calls `6..13` and seeds `50000..50999`. No E011 result exists yet.

## Pipeline Audit

[The frequency-resolved pipeline audit](frequency_resolved_pipeline_audit.md)
checks the four E004B curves, independent low/high target selection, the E005
distinction, proposed E008 controls, whole-denoiser intervention semantics, and
the baseline-pair gate. It changes no scientific artifacts.

## Reading Order

New readers should start with the [root README](../README.md), then the
canonical pipeline, E004 decision, E004A source audit and results, E005
results, historical E006 results, and blocked E007 protocol.
Reviewers auditing a specific experiment should read its protocol before its
result document and then inspect the linked compact [`results/`](../results/)
and [`figures/`](../figures/) directories.
