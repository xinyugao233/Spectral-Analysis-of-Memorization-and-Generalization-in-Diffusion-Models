# Artifact inventory, 2026-11-02

Original run manifests remain byte-identical. The [public artifact manifest](public_artifacts.json)
records exact public hashes/sizes alongside original hashes/sizes and source paths.
Eight E011/E012 CSV mismatches are exactly explained by CRLF→LF serialization:
replacing each public LF by CRLF reproduces the recorded original checksum.
The verifier **does not normalize** its input; any other byte/value change fails.
Future derived tables use explicit LF serialization; Git preserves committed
E011/E012 table bytes with `-text` rules.

## Recovered original observations

[Recovered identities](recovered_artifacts.json) give exact original hashes,
sizes and source locations for all three per-sample CSVs, three failure tables,
E010 generated-sample NPZ and E012 original report. Their bytes are staged
under `../external-evidence/spectral`, outside Git, with `SHA256SUMS`.
They are available but not published. Their source runs are E010, E011 job
16820230 and E012 job 16822015; exact code and environment identities remain
in the corresponding [original manifests](../../results/README.md).

The original observation counts are 3,584 / 8,000 / 9,000. All original hashes
match. Reproduction of paired table statistics passed from these authentic
rows; this does not rerun inference or FID. The release candidate holds the
per-sample/failure CSVs, original manifests and usage instructions. Permission
and attribution for publication remain owner decisions.

```bash
python scripts/verify_artifacts.py --external-dir /path/to/original-artifacts --require-external
python scripts/analyze_results.py --mode paired --input-dir /path/to/original-rows --output-dir outputs/paired-new
```

The first command also checks the external NPZ/report listed in the manifests;
a minimal rows-only release is sufficient for paired analysis, not for that
all-artifact check. Missing files are explicitly reported, never verified.

## Verified external dependencies (not downloaded or executed)

The [external dependency inventory](external_dependencies.json) records paths,
expected hashes, sizes, role and next verification commands. Read-only checks
on Hellbender verified:

- M checkpoint: `e5a7debafcd19191d6557f645216bfcb2e7589922396fd08130e76e3f5388b0a`, 223,159,918 bytes.
- G checkpoint: `a355ea67605dea3e2e663e94eb23416ffeb7679757088a68dc6228c03da5a92b`, 223,159,918 bytes.
- CIFAR archive: `795cdc1444465ae4e19e25a0615d05ba0a0e83caa5db6b1b811deaf4c7910dfa`, 164,500,138 bytes.
- Pinned EDM `fid.py`: `704be0de2c77c0090fdb9787268360ef9d438ca83c6aca14aa6d4f51dcff364d`.
- Inception detector and FID reference: recorded cache files still exist for
  E011/E012 and match their original SHA-256 and sizes.

Raw samples/FID image trees remain external. Their manifest tree hashes are
identity claims; tree-wide byte verification and new model/FID execution were
not performed. They are unnecessary for the table-only reproduction command.

## Executed EDM source identity: incomplete

Both E011/E012 record upstream commit
`008a4e5316c8e3bfe61a62f874bddba254295afb` with a **dirty** worktree.
The current authorized EDM checkout still has that HEAD and three tracked
modifications. Its current patch is retained in private local staging as
2026-11-02 inspection evidence, **not** proof of the historical executed patch.
No historical patch hash/source snapshot was found in the run manifests.
Exact historical source identity remains blocked on such a snapshot or
contemporaneous patch. Replacing it with clean upstream would not resolve it.
The patch is excluded from this public change; no third-party recovered source is uploaded.
