# Initial local maintenance verification — 2026-11-02

This record predates publication. See the [final review](publication-review.md)
and [maintenance PR](https://github.com/xinyugao233/Spectral-Analysis-of-Memorization-and-Generalization-in-Diffusion-Models/pull/12) for subsequent publication and hosted checks.

Local maintenance, CPU analysis and presentation are implemented. Authentic
per-seed statistics were reproduced; model inference/FID were not rerun.
Full historical execution provenance remains incomplete because the executed
dirty EDM patch was not recorded. Public releases and hosted CI remain pending.

## Baseline

Read-only `git ls-remote ... refs/heads/main` confirmed that GitHub main still
matched the starting commit below at handoff; no newer public work was replaced.

- Source checkout: `<original-checkout>`.
- Original branch `main`, commit
  `055a8d4ba533ab1cba34b6e2ffedf88e051f8f89`.
- Original untracked `scripts/render_sigma_frequency_correspondence_report.py`
  was preserved and excluded from the dedicated maintenance clone.
- Maintenance branch: `codex/public-maintenance`.
- Baseline Python 3.13.2 / Torch 2.7.1 / torchvision 0.22.1:
  `PYTHONPATH=src <original-checkout>/.venv/bin/python -m unittest discover -s tests`
  ran **232 tests in 2.228 s**, with **one README-content failure** and a sampler
  superclass deprecation warning. psutil happened to exist in that environment
  but was undeclared. The audited newer-Torch errors were not observed in this
  older baseline environment; the obsolete call was verified in source.
- Existing metadata allowed Python >=3.11 and unconstrained Torch. Only the
  explicitly tested review stack below is certified by this maintenance.

## Checklist

| Item | Status | Evidence/result |
|---|---|---|
| S1 | DONE locally | Declared psutil; minimal superclass fix; fixed-seed sampler/RNG tests pass; pinned review dependencies and CPU CI |
| S2 | DONE | Removed README heading/wording order assertions; retained scientific registry safeguards; local links checked |
| S3 | DONE | Eight CRLF/LF explanations independently hash-verified; separate exact-byte public manifest/verifier; corrupt fixtures fail |
| S4 | DONE | CPU-only aggregate and authentic-row modes; all three experiments' paired tables reproduced; no inference imports |
| S5 | DONE | E010/E011/E012 intervention distinctions, bank/seed/CI scope, corrected induction explanation, mode-count and quality caveats |
| S6 attribution | DONE | Pinned upstream license checked; bundled license and precise scope notices; project-level authorship wording |
| S6 historical dirty source | BLOCKED | Current patch captured; contemporaneous executed patch/tree identity unavailable |
| R2 | DONE | 177-line front page, labeled explanatory vector/PNG figure, counts/denominators and source map |
| R3 | DONE locally | Documentation links, CI badge pointing to the actual staged workflow, metadata ready for review |
| External evidence | VERIFIED/STAGED | Original rows/failure tables/NPZ/report downloaded and verified; selected checkpoints/archive/FID dependencies hashed remotely |
| Publication | BLOCKED | Explicit authorization and attribution/redistribution review still required |

## Environment and verification

macOS 26.6.2 arm64, Python 3.13.2, Torch 2.15.0, torchvision 0.29.2,
NumPy 2.5.4, psutil 7.2.2. A fresh isolated install used declared dependencies
with `constraints-review.txt`; no manual psutil step was needed. `pip check`
reported no broken requirements. [Install log](fresh-install.log).

```bash
python -m pip install -c constraints-review.txt -e .
python -m unittest discover -s tests
python scripts/verify_artifacts.py
python scripts/check_document_links.py
python scripts/analyze_results.py --mode aggregate --output-dir outputs/aggregate-review
python scripts/analyze_results.py --mode paired --input-dir /path/to/original-rows --output-dir outputs/paired-review
```

**236 tests passed** in the fresh review environment.
[Full suite output](unittest-final.log). Sampler/RNG regression checks are
included; analytic table tests independently compare the offline statistics to
the frozen implementation and check label equality/degenerate distances,
incomplete pairing, duplicates, invalid distances and byte/value corruption.
A separate import check confirmed table analysis does not import Torch.

Public checks distinguish available bytes from missing external originals:
24 exact-byte checks: 18 VERIFIED, 6 MISSING_EXTERNAL. With the recovered external
artifacts, every listed output verifies: 24 VERIFIED checks (full logs retained locally).
The runtime verifier never silently normalizes line endings.

Aggregate consistency: checked against public tables (report retained locally), about 0.0013 s after
imports. Authentic paired analysis: checked against public paired tables (report retained locally), about 6.4 s;
all computed comparisons passed at absolute tolerance `1e-12`. The fixed
100,000-resample/seed-0 bootstrap and E012 scaling were retained. Full data
coverage is 3,584 / 8,000 / 9,000 observations for E010/E011/E012.
Main reproduced quantities include E011 +76.1 pp [73.4,78.7] and E012 B1×B2
+52.35 pp [49.75,54.95], with B1+B2 overlap 622/761. These are generation-seed
statistics for the fixed model pair, not new training or FID evidence.

All tracked original configs, registries, result values, manifests and figures
remain byte-identical. New presentation assets were visually inspected and
have [separate provenance](presentation.md). Document corrections retain the
historical protocols and identify their interpretation clarifications.

## Metadata staged, not applied

Description: “Controlled denoiser and Fourier interventions studying CIFAR-10
memorization, with frozen protocols and paired-seed analysis.”
Topics: `diffusion-models`, `memorization`, `fourier-analysis`, `pytorch`,
`reproducible-research`, `cifar10`.
CPU CI is staged and its real workflow URL is used; **hosted status is pending**.
No repository rename, push, PR, publication, upload, or compute job occurred.

Remaining risks: no exact historical dirty EDM snapshot; no new model/FID
execution; conditional intervals cannot establish model/dataset robustness;
publication and redistribution remain unresolved. [Artifact details](artifacts.md).

## Publication preparation

Original local commits and unredacted audit files are retained outside the
publication branch. Public log copies redact machine-specific paths; hashes,
commands and measured results are retained. New maintenance documents use
portable artifact identities. Historical manifests and completed experiment
records retain their original paths to preserve provenance.

[Independent final publication checks](publication-review.md).
