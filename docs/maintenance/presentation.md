# Derived presentation assets

`figures/presentation_v2/fourier_intervention.png` and its SVG show a schematic
of the same-state E011/E012 prediction operation and a chart of the exact E012
condition summary. No generated image substitutes for experimental evidence.
All conditions, counts, denominators and band limits are retained, including
zero counts. The footer preserves the reverse-direction quality caveat.
The original figures, numerical artifacts and run manifests remain unchanged.

Regenerate to a **new** directory:

```bash
python scripts/render_public_explanation.py --output-dir outputs/explanation-new
```

The maintenance image was visually inspected for labels, clipping, scales and
legibility. Its derivation and exact output hashes are recorded in
[presentation.json](presentation.json). Diagram arrows denote computational
flow, not an additional causal claim.
