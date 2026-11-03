"""CPU-only table analysis. Imports NumPy, never Torch, EDM, CUDA, or datasets."""

from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def condition_key(row):
    return (row.get("try_id", ""), row["condition_id"])


def validate_rows(rows, *, conditions, seeds, experiment):
    """Require complete unique pairing and the historical strict distance label.

    d1=d2=0 is nonmemorized; its stored ratio is +inf, as in the original
    evaluator. Distances must be finite/nonnegative and ordered independently.
    """
    expected = {(key, seed) for key in conditions for seed in seeds}
    observed = {}
    for row in rows:
        required = {
            "condition_id",
            "sample_seed",
            "d1nn",
            "d2nn",
            "memorized",
            "status",
            "experiment_id",
        }
        if not required <= row.keys():
            raise ValueError(
                f"Missing required fields: {sorted(required - row.keys())}"
            )
        key, seed = condition_key(row), int(row["sample_seed"])
        pair = key, seed
        if pair not in expected or pair in observed:
            raise ValueError(f"Unexpected or duplicate condition/seed: {pair}")
        if row["status"] != "ok" or row["experiment_id"] != experiment:
            raise ValueError(f"Failed record or wrong experiment: {pair}")
        d1, d2 = float(row["d1nn"]), float(row["d2nn"])
        if not (math.isfinite(d1) and math.isfinite(d2) and 0 <= d1 <= d2):
            raise ValueError(f"Invalid nearest-neighbor distances: {pair}")
        label = int(row["memorized"])
        if label not in (0, 1) or label != int(d1 < d2 / 3):
            raise ValueError(f"Strict d1 < d2/3 label mismatch: {pair}")
        if "d1nn_over_d2nn" in row:
            ratio = float(row["d1nn_over_d2nn"])
            expected_ratio = d1 / d2 if d2 > 0 else math.inf
            if not math.isclose(ratio, expected_ratio, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Ratio mismatch: {pair}")
        if {"d1nn_reference_index", "d2nn_reference_index"} <= row.keys():
            first, second = (
                int(row["d1nn_reference_index"]),
                int(row["d2nn_reference_index"]),
            )
            if first == second or min(first, second) < 0:
                raise ValueError(f"Invalid nearest-neighbor identities: {pair}")
        observed[pair] = label
    if observed.keys() != expected:
        missing = sorted(expected - observed.keys())
        raise ValueError(
            f"Incomplete pairing: missing {len(missing)} rows, first={missing[:3]}"
        )
    return {
        key: np.array([observed[key, seed] for seed in seeds], dtype=np.float64)
        for key in conditions
    }


def interval(values, *, resamples=100000, seed=0):
    """Same RNG, chunking and percentile estimator as frozen E010–E012 code."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or not data.size or not np.isfinite(data).all() or resamples <= 0:
        raise ValueError(
            "Expected a finite nonempty paired vector and positive resamples"
        )
    generator = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for offset in range(0, resamples, 1000):
        count = min(1000, resamples - offset)
        indices = generator.integers(0, data.size, size=(count, data.size))
        means[offset : offset + count] = data[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def factorial_vectors(outcomes):
    definitions = {
        "B1": (0,),
        "B2": (1,),
        "B3": (2,),
        "B1xB2": (0, 1),
        "B1xB3": (0, 2),
        "B2xB3": (1, 2),
        "B1xB2xB3": (0, 1, 2),
    }
    result = {}
    for name, active in definitions.items():
        total = np.zeros_like(next(iter(outcomes.values())), dtype=np.float64)
        for key, values in outcomes.items():
            bits = key[1][:3]
            if bits == "con":
                continue
            sign = np.prod([1.0 if bits[i] == "1" else -1.0 for i in active])
            total += sign * values
        result[name] = total / 2 ** (3 - len(active))
    return result


def compare_rows(actual, frozen, keys, *, tolerance=1e-12):
    """Compare explicitly computed columns only; FID is not recomputed here."""
    keyed = {tuple(row[k] for k in keys): row for row in actual}
    expected_keys = {tuple(row[k] for k in keys) for row in frozen}
    if set(keyed) != expected_keys:
        raise ValueError(f"Comparison key coverage mismatch: {keys}")
    for frozen_row in frozen:
        actual_row = keyed[tuple(frozen_row[k] for k in keys)]
        for field, value in actual_row.items():
            if field in keys:
                continue
            expected = frozen_row[field]
            if isinstance(value, bool):
                equal = str(value).lower() == expected.lower()
            elif isinstance(value, (int, float)):
                equal = math.isclose(
                    value, float(expected), rel_tol=0, abs_tol=tolerance
                )
            elif isinstance(value, list):
                equal = value == json.loads(expected)
            else:
                equal = str(value) == expected
            if not equal:
                raise ValueError(
                    f"Frozen comparison mismatch {keys}: {field}: {value} != {expected}"
                )


def aggregate_check(root, number):
    root = Path(root)
    directory = root / f"results/experiment_{number}"
    rows = read_csv(directory / f"experiment_{number}_condition_summary.csv")
    seed_info = json.loads((root / f"data/e0{number}_seed_manifest.json").read_text())
    seen = set()
    for row in rows:
        key = condition_key(row)
        if key in seen:
            raise ValueError(f"Duplicate aggregate condition: {key}")
        seen.add(key)
        count, n = int(row["memorized_count"]), int(row["sample_count"])
        if (
            n != seed_info["seed_count"]
            or not 0 <= count <= n
            or not math.isclose(
                count / n, float(row["memorization_rate"]), abs_tol=1e-12
            )
        ):
            raise ValueError("Inconsistent count/rate/denominator")
    registry = json.loads(
        (root / f"data/e0{number}_condition_manifest.json").read_text()
    )
    expected = (
        {
            (trial["try_id"], c["condition_id"])
            for trial in registry["tries"]
            for c in trial["conditions"]
        }
        if number == 11
        else {("", c["condition_id"]) for c in registry["conditions"]}
    )
    if seen != expected:
        raise ValueError("Aggregate condition coverage differs from frozen registry")
    return {
        "experiment": f"E0{number}",
        "scope": "Aggregate consistency only; paired observations and intervals NOT reproduced",
        "conditions": rows,
    }


def paired_analysis(root, number, input_path):
    root, input_path = Path(root), Path(input_path)
    directory = root / f"results/experiment_{number}"
    manifest = json.loads(
        (directory / f"experiment_{number}_manifest.json").read_text()
    )
    name = f"experiment_{number}_per_sample.csv"
    identity = manifest["outputs"][name]
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Missing authentic {name}; expected SHA256 {identity['sha256']}; source {manifest['output_dir']}/{name}"
        )
    raw = input_path.read_bytes()
    if (
        hashlib.sha256(raw).hexdigest() != identity["sha256"]
        or len(raw) != identity["size_bytes"]
    ):
        raise ValueError(f"Original per-sample identity mismatch: {input_path}")
    seeds_info = json.loads((root / f"data/e0{number}_seed_manifest.json").read_text())
    seeds = list(range(seeds_info["seed_start"], seeds_info["seed_end"] + 1))
    registry = json.loads(
        (root / f"data/e0{number}_condition_manifest.json").read_text()
    )
    if number == 11:
        conditions = [
            (trial["try_id"], condition["condition_id"])
            for trial in registry["tries"]
            for condition in trial["conditions"]
        ]
    else:
        conditions = [("", c["condition_id"]) for c in registry["conditions"]]
    source_rows = read_csv(input_path)
    vectors = validate_rows(
        source_rows, conditions=conditions, seeds=seeds, experiment=f"E0{number}"
    )
    tables = {}
    summary = []
    for key, values in vectors.items():
        row = {
            "condition_id": key[1],
            "memorized_count": int(values.sum()),
            "sample_count": len(values),
            "memorization_rate": float(values.mean()),
        }
        if key[0]:
            row["try_id"] = key[0]
        summary.append(row)
    tables["condition_summary"] = (
        summary,
        ["try_id", "condition_id"] if number == 11 else ["condition_id"],
    )
    cache = {}

    def estimate(values):
        cache_key = np.asarray(values, dtype=np.float64).tobytes()
        if cache_key not in cache:
            cache[cache_key] = interval(values)
        low, high = cache[cache_key]
        return {"estimate": float(values.mean()), "ci95_low": low, "ci95_high": high}

    if number == 10:
        analyses = []
        for prefix, direction, sign in [
            ("A", "suppression", -1),
            ("B", "induction", 1),
        ]:
            base = vectors["", prefix + "0"]
            for band, indices in [("low", (1, 2, 3)), ("high", (4, 5, 6))]:
                before, target, after = [
                    sign * (vectors["", prefix + str(i)] - base) for i in indices
                ]
                e = estimate(target - (before + after) / 2)
                analyses.append(
                    {
                        "direction": direction,
                        "band": band,
                        "before_effect": float(before.mean()),
                        "target_effect": float(target.mean()),
                        "after_effect": float(after.mean()),
                        "contrast": e["estimate"],
                        "contrast_ci95_low": e["ci95_low"],
                        "contrast_ci95_high": e["ci95_high"],
                        "criterion_pass": bool(
                            target.mean() > 0
                            and target.mean() > before.mean()
                            and target.mean() > after.mean()
                            and e["estimate"] > 0
                            and e["ci95_low"] > 0
                        ),
                    }
                )
        tables["directional_analysis"] = (analyses, ["direction", "band"])
        transitions = []
        for prefix in ("A", "B"):
            base = vectors["", prefix + "0"].astype(bool)
            for index in range(1, 7):
                condition_id = prefix + str(index)
                swapped = vectors["", condition_id].astype(bool)
                transitions.append(
                    {
                        "condition_id": condition_id,
                        "memorized_to_non_memorized": int((base & ~swapped).sum()),
                        "non_memorized_to_memorized": int((~base & swapped).sum()),
                        "memorized_to_memorized": int((base & swapped).sum()),
                        "non_memorized_to_non_memorized": int((~base & ~swapped).sum()),
                    }
                )
        tables["paired_transitions"] = (transitions, ["condition_id"])
    elif number == 11:
        analyses, comparisons, effects = [], [], {}
        for trial, mask in [
            ("try01_radial_r11", "disk"),
            ("try02_square_matched", "square"),
        ]:
            for recipient, hybrid, name in [
                ("G", "G_from_M", "Delta_G"),
                ("M", "M_from_G", "Delta_M"),
            ]:
                base, treated = vectors[trial, recipient], vectors[trial, hybrid]
                difference = treated - base
                effects[mask, name] = difference
                e = estimate(difference)
                analyses.append(
                    {
                        "try_id": trial,
                        "effect_name": name,
                        "baseline_rate": float(base.mean()),
                        "hybrid_rate": float(treated.mean()),
                        "paired_effect": e["estimate"],
                        "ci95_low": e["ci95_low"],
                        "ci95_high": e["ci95_high"],
                    }
                )
        for name in ("Delta_G", "Delta_M"):
            disk, square = effects["disk", name], effects["square", name]
            e = estimate(disk - square)
            comparisons.append(
                {
                    "effect_name": name,
                    "disk_effect": float(disk.mean()),
                    "square_effect": float(square.mean()),
                    "disk_minus_square_effect": e["estimate"],
                    "ci95_low": e["ci95_low"],
                    "ci95_high": e["ci95_high"],
                }
            )
        tables["paired_effects"] = (analyses, ["try_id", "effect_name"])
        tables["mask_comparison"] = (comparisons, ["effect_name"])
    else:
        for row in summary:
            e = estimate(vectors["", row["condition_id"]] - vectors["", "000_none"])
            row.update(
                delta_vs_G=e["estimate"],
                ci95_low=e["ci95_low"],
                ci95_high=e["ci95_high"],
            )
        contrasts = [
            {"contrast": name, **estimate(values)}
            for name, values in factorial_vectors(vectors).items()
        ]
        tables["factorial_contrasts"] = (contrasts, ["contrast"])
        overlap = []
        full = vectors["", "111_B1_B2_B3"].astype(bool)
        for key, values in vectors.items():
            if key[1] in ("000_none", "111_B1_B2_B3", "control_B4"):
                continue
            selected = values.astype(bool)
            intersection = int((selected & full).sum())
            union = int((selected | full).sum())
            only = np.array(seeds)[selected & ~full].tolist()
            overlap.append(
                {
                    "condition_id": key[1],
                    "memorized_count": int(selected.sum()),
                    "intersection_with_111": intersection,
                    "recovery_of_111": intersection / int(full.sum())
                    if full.sum()
                    else 0.0,
                    "jaccard": intersection / union if union else 1.0,
                    "subband_only_count": len(only),
                    "subband_only_seeds": only,
                }
            )
        tables["full_mask_overlap"] = (overlap, ["condition_id"])
        diagnostics = []
        for key in vectors:
            ratios = np.array(
                [
                    float(row["d1nn_over_d2nn"])
                    for row in source_rows
                    if condition_key(row) == key
                ]
            )
            if not np.isfinite(ratios).all():
                raise ValueError("E012 ratio summaries require finite ratios")
            diagnostics.append(
                {
                    "condition_id": key[1],
                    "mean": float(ratios.mean()),
                    "median": float(np.median(ratios)),
                    "q25": float(np.quantile(ratios, 0.25)),
                    "q75": float(np.quantile(ratios, 0.75)),
                }
            )
        tables["nn_diagnostics"] = (diagnostics, ["condition_id"])
    for name, (rows, keys) in tables.items():
        compare_rows(
            rows, read_csv(directory / f"experiment_{number}_{name}.csv"), keys
        )
    return {
        "experiment": f"E0{number}",
        "scope": "Authentic per-seed paired reproduction; no new inference or FID",
        "source_sha256": identity["sha256"],
        "records": len(source_rows),
        "bootstrap_resamples": 100000,
        "bootstrap_seed": 0,
        "absolute_tolerance": 1e-12,
        "numpy": np.__version__,
        "tables": {name: rows for name, (rows, _) in tables.items()},
        "comparison": "PASS",
    }
