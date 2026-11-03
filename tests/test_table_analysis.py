"""Protect scientific table contracts while allowing editorial rewrites."""

import hashlib
import tempfile
import unittest
from pathlib import Path
import numpy as np
from spectral_diffusion_playground.table_analysis import (
    validate_rows,
    interval,
    factorial_vectors,
)
from spectral_diffusion_playground.artifact_verification import verify_entries


class TableAnalysisTests(unittest.TestCase):
    def rows(self):
        return [
            {
                "experiment_id": "FIXTURE",
                "condition_id": c,
                "sample_seed": str(seed),
                "d1nn": "1",
                "d2nn": "3",
                "memorized": "0",
                "status": "ok",
            }
            for c in ("a", "b")
            for seed in (1, 2)
        ]

    def validate(self, rows):
        return validate_rows(
            rows, conditions=[("", "a"), ("", "b")], seeds=[1, 2], experiment="FIXTURE"
        )

    def test_strict_equality_and_degenerate_distance_labels(self):
        rows = self.rows()
        rows[0].update(d1nn="0", d2nn="0", d1nn_over_d2nn="inf")
        rows[1].update(d1nn="0", d2nn="3", memorized="1")
        values = self.validate(rows)
        np.testing.assert_array_equal(values["", "a"], [0, 1])
        rows[2]["memorized"] = "1"
        with self.assertRaisesRegex(ValueError, "label mismatch"):
            self.validate(rows)

    def test_duplicate_missing_nonfinite_negative_and_ordering(self):
        cases = [self.rows()[:-1], self.rows() + self.rows()[:1]]
        for d1, d2 in [("nan", "3"), ("-1", "3"), ("4", "3"), ("1", "inf")]:
            rows = self.rows()
            rows[0].update(d1nn=d1, d2nn=d2)
            cases.append(rows)
        for rows in cases:
            with self.assertRaises(ValueError):
                self.validate(rows)

    def test_statistics_match_frozen_implementation_on_synthetic_pairs(self):
        from spectral_diffusion_playground.e010_directional_transfer import (
            bootstrap_mean_interval,
        )
        from spectral_diffusion_playground.e012_frequency_localization import (
            factorial_contrast_vectors,
        )

        values = np.array([-1, 0, 1, 0, 1], dtype=float)
        self.assertEqual(
            interval(values, resamples=1234),
            bootstrap_mean_interval(values, resamples=1234),
        )
        ids = [
            "000_none",
            "100_B1",
            "010_B2",
            "001_B3",
            "110_B1_B2",
            "101_B1_B3",
            "011_B2_B3",
            "111_B1_B2_B3",
            "control_B4",
        ]
        rng = np.random.default_rng(9)
        outcomes = {key: rng.integers(0, 2, 5).astype(float) for key in ids}
        actual = factorial_vectors(
            {("", key): value for key, value in outcomes.items()}
        )
        expected = factorial_contrast_vectors(outcomes)
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name])
        # An isolated 110 hit gives a B1×B2 finite difference of 1/2, not 1/4.
        isolated = {("", key): np.array([float(key == "110_B1_B2")]) for key in ids}
        self.assertEqual(factorial_vectors(isolated)["B1xB2"][0], 0.5)

    def test_verifier_detects_value_and_byte_corruption_and_missing_assets(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "table.csv"
            raw = b"n,value\n1,0.5\n"
            path.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            entry = {
                "path": "table.csv",
                "public_sha256": digest,
                "public_size_bytes": len(raw),
                "original_sha256": digest,
                "original_size_bytes": len(raw),
                "source_location": "fixture",
            }
            manifest = {"entries": [entry]}
            self.assertEqual(verify_entries(temp, manifest)[0]["status"], "VERIFIED")
            path.write_bytes(raw.replace(b"0.5", b"0.6"))
            self.assertEqual(verify_entries(temp, manifest)[0]["status"], "MISMATCH")
            path.write_bytes(raw.replace(b"\n", b"\r\n"))
            self.assertEqual(verify_entries(temp, manifest)[0]["status"], "MISMATCH")
            entry.update(path="missing.csv", public_sha256=None, public_size_bytes=None)
            self.assertEqual(
                verify_entries(temp, manifest)[0]["status"], "MISSING_EXTERNAL"
            )
