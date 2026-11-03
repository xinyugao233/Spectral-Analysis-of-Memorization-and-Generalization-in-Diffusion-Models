"""Verify available public artifacts; report missing external evidence separately."""

import argparse
import json
from pathlib import Path
from spectral_diffusion_playground.artifact_verification import verify_entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-dir", type=Path)
    parser.add_argument("--require-external", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "docs/maintenance/public_artifacts.json").read_text())
    results = verify_entries(root, manifest, args.external_dir)
    print(json.dumps(results, indent=2))
    return int(
        any(
            r["status"] in ("MISMATCH", "MISSING_PUBLIC")
            or (args.require_external and r["status"] == "MISSING_EXTERNAL")
            for r in results
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
