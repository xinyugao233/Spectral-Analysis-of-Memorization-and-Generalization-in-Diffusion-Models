# Third-party notices and scope

The repository's MIT license covers original project materials for which that
license is established. It does not relicense NVIDIA derivatives or external
checkpoints, datasets, or the grounding paper.

`src/spectral_diffusion_playground/e009_warm_start.py` explicitly derives from
NVlabs EDM `training/training_loop.py` at commit
`008a4e5316c8e3bfe61a62f874bddba254295afb`, copyright 2022 NVIDIA CORPORATION & AFFILIATES.
Its existing CC BY-NC-SA 4.0 notice is preserved. The source adaptation adds
separate EMA loading and extended deterministic state management; the current
maintenance changes only the obsolete sampler superclass call.

Verified upstream on 2026-11-02:
- [Pinned source](https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/training/training_loop.py)
- [Bundled upstream license](LICENSE-EDM.txt)
- [Pinned license](https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/LICENSE.txt)

Package metadata therefore describes the distribution as `MIT AND CC-BY-NC-SA-4.0`.
This records existing scope; it is not a new relicensing decision.

The project implements geometry analysis, experimental registries, intervention
logic, statistics, tests and reporting. EDM architecture/training infrastructure,
external trained weights and CIFAR-10 are reused. Project-level descriptions do
not establish individual authorship of each component. Confirm personal
contribution statements and checkpoint redistribution rights before publication.

E011/E012 record a dirty external EDM tree. The upstream commit alone cannot
identify executed source. See the maintenance artifact inventory for recoverability;
a currently accessible patch does not by itself prove historical identity.
