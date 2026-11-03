"""Consistency gates for E004A, E005, E006, and proposed E007 regions."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spectral_diffusion_playground.region_definitions import (
    contiguous_components,
    high_high_indices,
)


FROZEN_E005_E006_ARTIFACTS = {
    "docs/experiment_05_spectral_residual_results.md": "5eeb6e56429a879040134670d8cca23f43087520b5d8bfacea3e1fc46513eb82",
    "docs/experiment_06_transition_window_swap_results.md": "76dadf6d9482b2c9c5f24161df816b30631c5eaecdd1a38b91ab0884165902ff",
    "figures/experiment_06/experiment_06_generated_nn_pairs.png": "8db3a5341e13ed7e37f58378af0eec561a3ae060142e36210345037f71893877",
    "figures/experiment_06/experiment_06_memorization_rates.png": "227555026a7c801dc077b59e928409685bb8b3351c37a535afbd3388bec29871",
    "figures/experiment_06/experiment_06_paired_changes.png": "2f4079a3f4f4a336e463909a910bd3cb6a1c62888c3583e14eebc453475bb3d8",
    "figures/experiment_06/experiment_06_paper_medium_reference.png": "bb216ae24f32469fb601cc439f5406f47c8c62cdb79d418c6fac0206c620de0b",
    "figures/experiment_06/experiment_06_ratio_distributions.png": "2c42c5a1b604a418b654c5bf3c390dcc371eb98940d9720fa5ccb914531b3f29",
    "figures/experiment_06/experiment_06_transition_vs_controls.png": "c36e311ca202f5b3e353e7c502c1d09dcdfa9b4c2d4a14d5fdabfd4be3a8d7c5",
    "results/experiment_05/experiment_05_aggregated_curves.csv": "1f184e2c3a19f29b73edcdcf07e4872bb54bef89c1351673f9ee9882ee59ccb0",
    "results/experiment_05/experiment_05_identity_validation.json": "911126ea61ddc02a6440762df9066117b49703c5b3960b8d60032553f4c3a940",
    "results/experiment_05/experiment_05_manifest.json": "3d5b1ade232eee5ba80150e35c53b2f33eb354c7b6bb06a21bdf39dfc34c99e8",
    "results/experiment_05/experiment_05_transition_windows.json": "aa588d071716e81694ea467f282947cc9949834ffdc8011abe847d0344cbd6bf",
    "results/experiment_06/experiment_06_condition_summary.csv": "4f2bc0744cb169629c0369bcce23c7598d79f55a94969183c50c3055fe1770db",
    "results/experiment_06/experiment_06_failures.csv": "e78f3d5343ced34e72dc711006e04b6bcce28884da36a79e788019a65727f8bf",
    "results/experiment_06/experiment_06_manifest.json": "961dc56f20810cf22c01a4d53b0abcc269c751651c11da4ea168570529061886",
    "results/experiment_06/experiment_06_outcome.json": "f2467e528d68c8329ecaf41ee22427046bcde7a0372fa6336f9d75025abe01f9",
    "results/experiment_06/experiment_06_paired_comparisons.csv": "32036a89ed75f4896f5d56a96657522561389faa5951685cc0ffa73b5dbb914d",
    "results/experiment_06/experiment_06_qualitative_sample_manifest.json": "36b3784959b28a9ab4ee579d7851f0bf537decd4a6e25380c96df70fcce0ee7a",
    "results/experiment_06/experiment_06_validation.json": "dee057793e6c95d201fcfc3ba4e73da6f335a0b80bc229aeeb138edec8bb5be5",
}


class RegionDefinitionMathTests(unittest.TestCase):
    """Verify classification uses evaluated points and preserves gaps."""

    def test_high_high_classification_does_not_fill_gaps(self) -> None:
        indices = high_high_indices(
            [0.9, 0.1, 0.9],
            [0.9, 0.9, 0.9],
            q_coverage=0.8,
            q_posterior_weight=0.8,
        )
        self.assertEqual(indices, [0, 2])
        self.assertEqual(contiguous_components(indices), [[0], [2]])

    def test_nonfinite_geometry_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            high_high_indices(
                [0.9, float("nan")],
                [0.9, 0.9],
                q_coverage=0.8,
                q_posterior_weight=0.8,
            )


class RegionDefinitionRepositoryTests(unittest.TestCase):
    """Keep machine-readable definitions aligned with scientific wording."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = json.loads(
            (REPO_ROOT / "results" / "region_definition_registry.json").read_text()
        )
        cls.pipeline = json.loads(
            (REPO_ROOT / "results" / "canonical_experiment_pipeline.json").read_text()
        )
        cls.manifest = json.loads(
            (
                REPO_ROOT
                / "results"
                / "experiment_04a"
                / "e006_grid_geometry_manifest.json"
            ).read_text()
        )
        with (REPO_ROOT / "results" / "experiment_04a" / "e006_grid_geometry.csv").open(
            newline=""
        ) as handle:
            cls.rows = list(csv.DictReader(handle))
        cls.readme = (REPO_ROOT / "README.md").read_text()
        cls.research_record = (REPO_ROOT / "docs" / "research_record.md").read_text()
        cls.e006_protocol = (
            REPO_ROOT / "docs" / "experiment_06_transition_swap_protocol.md"
        ).read_text()
        cls.e007_protocol = (
            REPO_ROOT / "docs" / "experiment_07_geometry_aligned_swap_protocol.md"
        ).read_text()
        cls.canonical_pipeline = (
            REPO_ROOT / "docs" / "canonical_experiment_pipeline.md"
        ).read_text()

    def test_exact_e006_grid_schema_and_classification(self) -> None:
        expected_fields = {
            "sigma_index",
            "sigma",
            "coverage_estimate",
            "coverage_ci95_low",
            "coverage_ci95_high",
            "posterior_weight_estimate",
            "posterior_weight_ci95_low",
            "posterior_weight_ci95_high",
            "coverage_ge_q_c",
            "posterior_weight_ge_q_w",
            "high_high_point_estimate",
            "high_high_lower_bound",
        }
        self.assertEqual(len(self.rows), 18)
        self.assertEqual(set(self.rows[0]), expected_fields)
        point = [
            int(row["sigma_index"])
            for row in self.rows
            if row["high_high_point_estimate"] == "True"
        ]
        lower = [
            int(row["sigma_index"])
            for row in self.rows
            if row["high_high_lower_bound"] == "True"
        ]
        self.assertEqual(point, [8, 9])
        self.assertEqual(lower, [8, 9])
        self.assertEqual(point, self.manifest["clean_room_geometry_high_high_indices"])
        self.assertEqual(
            lower,
            self.manifest["clean_room_geometry_high_high_lower_bound_indices"],
        )

    def test_registry_keeps_sources_distinct(self) -> None:
        paper = self.registry["paper_reported_medium_reference"]
        low = self.registry["e005_low_spectral_transition"]
        high = self.registry["e005_high_spectral_transition"]
        geometry = self.registry["e004a_clean_room_geometry_high_high"]
        self.assertEqual(paper["source"], "paper Table 1 / Figure 10")
        self.assertFalse(paper["defines_original_danger_zone"])
        self.assertFalse(low["defines_original_danger_zone"])
        self.assertFalse(high["defines_original_danger_zone"])
        self.assertEqual(geometry["indices"], [8, 9])
        self.assertIn("q_C=q_W=0.8", geometry["source"])
        self.assertFalse(geometry["defines_universal_paper_boundary"])

    def test_registry_records_blocked_e007_target(self) -> None:
        target = self.registry["e007_geometry_aligned_swap_target"]
        self.assertEqual(target["indices"], [8, 9])
        self.assertEqual(target["execution_status"], "blocked")
        self.assertEqual(
            target["blocker"],
            "historical EDM-50K no-swap baseline is 0/256",
        )
        self.assertFalse(target["historical_e006_reinterpreted"])

    def test_canonical_pipeline_has_seven_ordered_stages(self) -> None:
        expected = ["E004", "E004A", "E004B", "E005", "E006", "E007", "E008"]
        observed = [
            self.pipeline[f"stage_{index}_{suffix}"]["experiment"]
            for index, suffix in (
                (1, "cutoff"),
                (2, "geometry"),
                (3, "frequency_restricted_geometry"),
                (4, "spectral_interpretation"),
                (5, "historical_swap"),
                (6, "geometry_swap"),
                (7, "frequency_geometry_swap"),
            )
        ]
        self.assertEqual(observed, expected)

    def test_geometry_alone_selects_the_candidate_target(self) -> None:
        geometry = self.pipeline["stage_2_geometry"]
        band_geometry = self.pipeline["stage_3_frequency_restricted_geometry"]
        spectral = self.pipeline["stage_4_spectral_interpretation"]
        historical = self.pipeline["stage_5_historical_swap"]
        self.assertEqual(geometry["selection_metrics"], ["C_sigma", "W_sigma"])
        self.assertEqual(geometry["primary_rule"], "95% lower confidence bounds")
        self.assertEqual(geometry["target_indices"], [8, 9])
        self.assertFalse(band_geometry["uses_e005_for_selection"])
        self.assertFalse(spectral["defines_danger_zone"])
        self.assertFalse(historical["tested_geometry_defined_target"])

    def test_cutoff_role_is_spectral_not_geometric(self) -> None:
        cutoff = self.pipeline["stage_1_cutoff"]
        self.assertEqual(cutoff["reference_cutoff"], 4)
        self.assertEqual(cutoff["sensitivity_cutoffs"], [3, 5])
        self.assertIn(
            "The cutoff does not enter the definitions of `C_sigma(p,D)` or "
            "`W_sigma(D)`.",
            self.canonical_pipeline,
        )

    def test_readme_links_to_accessible_research_record(self) -> None:
        # Protect access to the detailed evidence, not editorial order or wording.
        self.assertIn("docs/research_record.md", self.readme)
        self.assertTrue((Path(__file__).resolve().parents[1] / "docs/research_record.md").is_file())

    def test_e006_is_historical_not_final(self) -> None:
        historical = self.pipeline["stage_5_historical_swap"]
        self.assertEqual(
            historical["role"], "exploratory spectral-aligned intervention"
        )
        self.assertEqual(historical["formal_outcome"], "INCONCLUSIVE")
        self.assertIn("**E006** was formally `INCONCLUSIVE`", self.research_record)
        self.assertNotIn("E006: Final Geometry-Aligned", self.research_record)

    def test_e007_is_the_blocked_full_space_stage(self) -> None:
        final_stage = self.pipeline["stage_6_geometry_swap"]
        self.assertEqual(final_stage["target_indices"], [8, 9])
        self.assertEqual(final_stage["pre_control"], [6, 7])
        self.assertEqual(final_stage["post_control"], [10, 11])
        self.assertEqual(final_stage["status"], "blocked")
        self.assertIn("**E007-E009** document blocked, retired", self.research_record)
        self.assertNotIn("E007 has been executed", self.research_record)

    def test_readme_preserves_inconclusive_historical_account(self) -> None:
        self.assertIn("formally `INCONCLUSIVE`", self.research_record)
        self.assertIn("did not identify a", self.research_record)
        self.assertIn("memorization danger zone", self.research_record)
        self.assertNotIn("causal model-swap intervention", self.research_record)
        self.assertIn("q_C=q_W=0.8", self.research_record)

    def test_paper_reference_is_not_described_as_e004a_derived(self) -> None:
        required = (
            "E006 tested spectral-aligned windows and a literature-derived paper "
            "medium\nreference. It did not preregister a window from locally "
            "computed coverage and\nposterior-weight curves, because E004A did "
            "not yet exist."
        )
        self.assertIn(required, self.e006_protocol)

    def test_e006_formal_outcome_remains_inconclusive(self) -> None:
        outcome = json.loads(
            (
                REPO_ROOT / "results" / "experiment_06" / "experiment_06_outcome.json"
            ).read_text()
        )
        self.assertEqual(outcome["outcome"], "INCONCLUSIVE")

    def test_frozen_e005_e006_artifacts_are_byte_identical(self) -> None:
        for relative_path, expected_hash in FROZEN_E005_E006_ARTIFACTS.items():
            with self.subTest(path=relative_path):
                actual_hash = hashlib.sha256(
                    (REPO_ROOT / relative_path).read_bytes()
                ).hexdigest()
                self.assertEqual(actual_hash, expected_hash)

    def test_e007_is_blocked_and_unexecuted(self) -> None:
        self.assertIn(
            "PROPOSED — BLOCKED BY KNOWN BASELINE DEGENERACY",
            self.e007_protocol,
        )
        self.assertIn("has not been executed", self.e007_protocol)
        self.assertNotIn("Status: completed", self.e007_protocol)

    def test_e007_records_known_degenerate_baseline(self) -> None:
        self.assertIn("EDM-50K no-swap baseline", self.e007_protocol)
        self.assertIn("`0/256`", self.e007_protocol)
        self.assertIn(
            "must not be permitted to produce a non-`INCONCLUSIVE` classification",
            self.e007_protocol,
        )
        self.assertNotIn("Hellbender", self.e007_protocol)
        self.assertNotIn("compute access", self.e007_protocol.lower())

    def test_e007_preserves_geometry_target_and_preflight_separation(self) -> None:
        self.assertIn("exactly indices `8..9`", self.e007_protocol)
        self.assertIn(
            "`3.256821519765537` and `1.9233398370400518`",
            self.e007_protocol,
        )
        self.assertIn("C_sigma lower 95% bound >= 0.8", self.e007_protocol)
        self.assertIn("W_sigma lower 95% bound >= 0.8", self.e007_protocol)
        self.assertIn("pilot seeds `10000..10127`", self.e007_protocol)
        self.assertIn("confirmatory seeds `0..255`", self.e007_protocol)
        self.assertIn("Do not generate or inspect geometry-target", self.e007_protocol)

    def test_e007_optional_direction_has_no_primary_classification(self) -> None:
        self.assertIn("Optional One-Direction Descriptive Analysis", self.e007_protocol)
        self.assertIn(
            "must not receive a\n`YES/PARTIAL/MIXED/NO` classification",
            self.e007_protocol,
        )

    def test_alignment_figure_exists(self) -> None:
        path = (
            REPO_ROOT
            / "figures"
            / "experiment_04a"
            / "e006_grid_geometry_alignment.png"
        )
        self.assertTrue(path.is_file())
        self.assertGreater(path.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
