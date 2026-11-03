"""Verify exact public bytes separately from original external run artifacts."""

import hashlib
from pathlib import Path


def verify_entries(root, manifest, external_dir=None):
    results = []
    for entry in manifest["entries"]:
        path = Path(root) / entry["path"]
        expected = entry["public_sha256"]
        size = entry["public_size_bytes"]
        representation = "public"
        if expected is None:
            representation = "external original"
            path = Path(external_dir) / path.name if external_dir else path
            expected, size = entry["original_sha256"], entry["original_size_bytes"]
        if not path.is_file():
            status = (
                "MISSING_EXTERNAL"
                if representation == "external original"
                else "MISSING_PUBLIC"
            )
        else:
            raw = path.read_bytes()
            status = (
                "VERIFIED"
                if len(raw) == size and hashlib.sha256(raw).hexdigest() == expected
                else "MISMATCH"
            )
        results.append(
            {
                "path": str(path),
                "status": status,
                "representation": representation,
                "expected_sha256": expected,
                "source": entry["source_location"],
            }
        )
    return results
