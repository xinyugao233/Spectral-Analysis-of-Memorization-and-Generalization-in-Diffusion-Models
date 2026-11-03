# Offline analysis contract

`table_analysis.py` imports only the standard library and NumPy. Neither table
mode imports Torch, loads a model, uses CUDA, downloads data or invokes inference.
The scientific hypothesis/intervention and frozen controls are unchanged:
E010 whole-denoiser timing, E011 same-state Fourier prediction swaps, E012 band
subsets at calls 6..13. No training-corruption or loss-weight changes are made.

Aggregate mode checks tracked counts, rates, denominators and uniqueness. It
cannot reconstruct paired outcomes, uncertainty or overlap from marginals.
Paired mode requires exact original per-sample file bytes named in each run
manifest. It verifies SHA-256 and size before parsing, then validates condition
and seed coverage, unique pairing, finite nonnegative ordered distances, strict
`d1 < d2/3` labels, ratio consistency and distinct nearest-neighbor identities.
The original-byte requirement also pins row metadata and observation provenance.

Equality is nonmemorized. For d1=d2=0 the historical label is false and ratio
is +infinity. Such a ratio is recognized by row validation rather than converted
into a positive label; E012 finite-ratio summary reproduction still requires
finite inputs, as the original evaluator did. No degenerate distances occur in
the recovered historical data. Analytic fixture tests cover both boundary cases.

E010 uses 14 conditions × 256 seeds (40000..40255). E011 uses two masks × four
conditions × 1,000 seeds; E012 uses nine conditions × 1,000 seeds (50000..50999).
The command recomputes the observed condition summaries and paired effects,
E010 target/control contrasts, E011 disk/square differences, E012 factorial
contrasts, full-mask overlap and NN diagnostics. It does not recompute FID.

Bootstrap uses `numpy.random.default_rng(0)`, 100,000 resamples, chunks of
1,000, with 2.5/97.5 percentile intervals. This exactly mirrors the frozen
executable implementation. A k-factor E012 contrast divides the signed
8-cell sum by `2**(3-k)`; the two-factor denominator is 2. Do not silently
substitute conventional differently scaled effect coding.

Computed columns are compared to frozen tables with absolute tolerance 1e-12
and zero relative tolerance. CSV output uses explicit UTF-8/LF and new output
directories. Synthetic fixtures validate algorithms only; they never count as
reproduced experiments. Full mode cannot bypass original hashes or coverage.
The authentic-row reproduction report is retained in local audit staging.
The [verification record](verification.md) summarizes the checked comparisons.

The observation units are generation seeds for a fixed trained model pair.
Percentile intervals are conditional on that pair and bank; zero empirical
bootstrap spread is not proof of a zero population effect.
