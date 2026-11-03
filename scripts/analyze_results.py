"""Analyze tracked aggregates or authentic per-sample rows without model inference."""

import argparse
import csv
import json
import time
from pathlib import Path
from spectral_diffusion_playground.table_analysis import (
    aggregate_check,
    paired_analysis,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["aggregate", "paired"], required=True)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--experiment", type=int, choices=[10, 11, 12])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output_dir}")
    if args.mode == "paired" and args.input_dir is None:
        parser.error(
            "--input-dir containing authentic experiment_10/11/12_per_sample.csv is required for paired mode"
        )
    start = time.perf_counter()
    results = []
    for number in [args.experiment] if args.experiment else [10, 11, 12]:
        result = (
            aggregate_check(root, number)
            if args.mode == "aggregate"
            else paired_analysis(
                root, number, args.input_dir / f"experiment_{number}_per_sample.csv"
            )
        )
        results.append(result)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "mode": args.mode,
        "duration_seconds": time.perf_counter() - start,
        "results": results,
    }
    (args.output_dir / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    for result in results:
        for name, rows in result.get("tables", {}).items():
            with (args.output_dir / f"{result['experiment']}_{name}.csv").open(
                "x", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(rows[0]), lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(rows)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "output": str(args.output_dir.resolve()),
                "duration_seconds": report["duration_seconds"],
                "scope": [r["scope"] for r in results],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
