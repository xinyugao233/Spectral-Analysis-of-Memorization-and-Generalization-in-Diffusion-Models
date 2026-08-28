"""Tests for E004B frequency-restricted Gaussian-shell geometry."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spectral_diffusion_playground.filters import create_frequency_mask
from spectral_diffusion_playground.frequency_restricted_geometry import (
    IMAGE_DIMENSION,
    band_gaussian_shell_membership,
    band_maximum_posterior_weight,
    band_posterior_weights,
    band_projector_rank,
    evaluate_frequency_restricted_geometry,
    mask_is_conjugate_symmetric,
    project_to_band,
    summarize_targets,
)
from spectral_diffusion_playground.paper_geometry import (
    gaussian_shell_membership,
    maximum_posterior_weight,
)
from spectral_diffusion_playground.paper_geometry_evaluation import index_text_sha256

FROZEN_PRIOR_ARTIFACTS = {
    "docs/experiment_04a_paper_geometry_protocol.md": "b43a4e88e82f757ab839c47c134cabbb23dfa9803333a83ccfff29025b8d95ee",
    "docs/experiment_04a_paper_geometry_results.md": "be0010b20da5ff51de4f48ecdd3c759e9674b983d3c7e896f332cf917916cd40",
    "docs/experiment_07_geometry_aligned_swap_protocol.md": "de4d8919d3ad385b9ca3bcbbd4715df6fbfe79bb68c91429b716317bc153a126",
    "figures/experiment_04a/coverage_and_max_posterior_weight.png": "5045d667f4ec518c63948c0f8164721a72f1c603ee27a2b57ba121193b08f15f",
    "figures/experiment_04a/e006_grid_geometry_alignment.png": "278b2d4f3fbea761e28359c5c251d943a296cfc4b451ddaac0b6cc62ce87d625",
    "results/experiment_04a/coverage_curve.csv": "77b474c8962bb225e1ddf7438a73bb7e79ca64175ad6840a6f638d30df2c9260",
    "results/experiment_04a/e006_grid_geometry.csv": "e35a4273cc07b11c8baae5950f79edddaa23a0b9dccf45d4002ede3164dc5264",
    "results/experiment_04a/e006_grid_geometry_manifest.json": "0cd4799536b33ae5175b82daf2cc54a6d2d9660ea3cfac167caf644ac456683b",
    "results/experiment_04a/e006_grid_geometry_validation.json": "1c6112a0c9e4c37bd2a93550fe0c8a29a45d86dd7f04554547bff7e0af98a9a0",
    "results/experiment_04a/geometry_manifest.json": "e82e82566ab77bc96aec9ba07e431e53c9f917ca786bad240530c3f7c2f88fa3",
    "results/experiment_04a/geometry_validation.json": "24db46766da47e41e612f4cd97c4657d066ffd49152aeb694687a9c022c33e67",
    "results/experiment_04a/max_posterior_weight_curve.csv": "d12ddaa3d3a0ab0509f0e8e9522c758e59483bdfe766aac2b89b67653e2ad89b",
}


def _load_experiment_module():
    """Load the numeric experiment entrypoint for direct mode tests."""
    path = REPO_ROOT / "experiments" / "04b_frequency_restricted_geometry.py"
    experiments_root = str(path.parent)
    if experiments_root not in sys.path:
        sys.path.insert(0, experiments_root)
    spec = importlib.util.spec_from_file_location("experiment_04b", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load E004B experiment module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FrequencyProjectorTests(unittest.TestCase):
    """Validate exact complementary orthogonal Fourier projectors."""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.values = self.rng.normal(size=(4, IMAGE_DIMENSION))

    def test_masks_are_binary_complementary_symmetric_and_include_dc(self) -> None:
        for radius in (3, 4, 5):
            low = create_frequency_mask(32, 32, radius)
            high = 1.0 - low
            self.assertTrue(np.all(np.isin(low, (0.0, 1.0))))
            self.assertTrue(np.all(np.isin(high, (0.0, 1.0))))
            np.testing.assert_array_equal(low + high, np.ones((32, 32)))
            self.assertTrue(mask_is_conjugate_symmetric(low))
            self.assertTrue(mask_is_conjugate_symmetric(high))
            self.assertEqual(low[16, 16], 1.0)

    def test_exact_band_ranks(self) -> None:
        expected = {3: (87, 2985), 4: (147, 2925), 5: (243, 2829)}
        for radius, (low_rank, high_rank) in expected.items():
            self.assertEqual(band_projector_rank(radius, "low"), low_rank)
            self.assertEqual(band_projector_rank(radius, "high"), high_rank)
            self.assertEqual(low_rank + high_rank, IMAGE_DIMENSION)

    def test_linearity_idempotence_orthogonality_and_reconstruction(self) -> None:
        left, right = self.values[:2], self.values[2:]
        for radius in (3, 4, 5):
            low = project_to_band(self.values, radius, "low")
            high = project_to_band(self.values, radius, "high")
            np.testing.assert_allclose(low + high, self.values, atol=2e-15)
            np.testing.assert_allclose(
                project_to_band(left + right, radius, "low"),
                project_to_band(left, radius, "low")
                + project_to_band(right, radius, "low"),
                atol=2e-15,
            )
            np.testing.assert_allclose(
                project_to_band(low, radius, "low"), low, atol=2e-15
            )
            np.testing.assert_allclose(
                project_to_band(high, radius, "high"), high, atol=2e-15
            )
            np.testing.assert_allclose(np.sum(low * high, axis=1), 0.0, atol=2e-13)

    def test_parseval_and_imaginary_residual(self) -> None:
        low, low_imaginary = project_to_band(
            self.values, 4, "low", return_imaginary_residual=True
        )
        high, high_imaginary = project_to_band(
            self.values, 4, "high", return_imaginary_residual=True
        )
        np.testing.assert_allclose(
            np.square(low).sum(axis=1) + np.square(high).sum(axis=1),
            np.square(self.values).sum(axis=1),
            rtol=2e-15,
            atol=2e-12,
        )
        self.assertLessEqual(max(low_imaginary, high_imaginary), 1e-15)

    def test_same_underlying_noise_projects_to_exact_complements(self) -> None:
        noise = self.rng.standard_normal(self.values.shape, dtype=np.float32).astype(
            np.float64
        )
        low = project_to_band(noise, 4, "low")
        high = project_to_band(noise, 4, "high")
        np.testing.assert_allclose(low + high, noise, atol=2e-15)


class FrequencyRestrictedMetricTests(unittest.TestCase):
    """Validate projected posterior and rank-aware shell definitions."""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(17)

    def test_band_posterior_weights_are_exactly_normalized(self) -> None:
        queries = self.rng.normal(size=(5, 12))
        references = self.rng.normal(size=(7, 12))
        weights = band_posterior_weights(queries, references, sigma=0.8)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-15)

    def test_shell_radii_use_explicit_band_rank(self) -> None:
        references = np.zeros((1, 5))
        query = np.asarray([[4.5, 0.0, 0.0, 0.0, 0.0]])
        rank_two = band_gaussian_shell_membership(
            query,
            references,
            sigma=1.0,
            band_dimension=2,
        )
        storage_five = gaussian_shell_membership(query, references, sigma=1.0)
        self.assertNotEqual(bool(rank_two[0]), bool(storage_five[0]))

    def test_identity_low_projector_matches_full_space_estimators(self) -> None:
        references = self.rng.normal(size=(6, IMAGE_DIMENSION))
        clean = self.rng.normal(size=(4, IMAGE_DIMENSION))
        noise = self.rng.normal(size=clean.shape)
        sigma = 0.7
        projected_references = project_to_band(references, 23, "low")
        projected_queries = project_to_band(clean + sigma * noise, 23, "low")
        np.testing.assert_allclose(projected_references, references, atol=2e-15)
        np.testing.assert_allclose(projected_queries, clean + sigma * noise, atol=2e-15)
        self.assertAlmostEqual(
            band_maximum_posterior_weight(
                projected_queries, projected_references, sigma
            ),
            maximum_posterior_weight(clean + sigma * noise, references, sigma),
            places=14,
        )
        np.testing.assert_array_equal(
            band_gaussian_shell_membership(
                projected_queries,
                projected_references,
                sigma,
                band_dimension=IMAGE_DIMENSION,
            ),
            gaussian_shell_membership(clean + sigma * noise, references, sigma),
        )
        np.testing.assert_allclose(project_to_band(clean, 23, "high"), 0.0, atol=0.0)

    def test_evaluator_is_deterministic_and_does_not_gap_fill(self) -> None:
        training = self.rng.normal(size=(5, IMAGE_DIMENSION))
        test = self.rng.normal(size=(4, IMAGE_DIMENSION))
        kwargs = {
            "cutoffs": [4],
            "sigmas": [2.0, 1.0, 0.5],
            "shell_c": 5.0,
            "posterior_draws": 2,
            "coverage_draws": 2,
            "bootstrap_replicates": 10,
            "seed": 3,
            "query_batch_size": 2,
            "reference_batch_size": 3,
        }
        first = evaluate_frequency_restricted_geometry(training, test, **kwargs)
        second = evaluate_frequency_restricted_geometry(training, test, **kwargs)
        self.assertEqual(first, second)
        synthetic_targets = {
            "cutoffs": {
                "4": {
                    "low": {
                        "band_dimension": 147,
                        "point_estimate_indices": [0, 2],
                        "point_estimate_components": [[0], [2]],
                        "lower_bound_indices": [0, 2],
                        "lower_bound_components": [[0], [2]],
                    },
                    "high": {
                        "band_dimension": 2925,
                        "point_estimate_indices": [],
                        "point_estimate_components": [],
                        "lower_bound_indices": [],
                        "lower_bound_components": [],
                    },
                }
            }
        }
        summary = summarize_targets(
            synthetic_targets, primary_cutoff=4, q_coverage=0.8, q_weight=0.8
        )
        self.assertEqual(summary["low_lower_bound_indices"], [0, 2])
        self.assertEqual(summary["low_band_geometry_target_indices"], [0, 2])
        self.assertEqual(summary["connected_components"]["low_lower_bound"], [[0], [2]])
        self.assertEqual(summary["high_lower_bound_indices"], [])
        self.assertEqual(summary["high_band_geometry_target_indices"], [])


class FrequencyRestrictedRepositoryTests(unittest.TestCase):
    """Protect prior experiments and E004B/E008 execution boundaries."""

    def test_config_is_tied_to_frozen_e004a_dataset(self) -> None:
        module = _load_experiment_module()
        config = module.load_e004b_config(
            REPO_ROOT / "configs" / "e004b_frequency_restricted_geometry.json"
        )
        self.assertEqual(
            config["dataset"]["training_subset_indices"], list(range(1000))
        )
        self.assertEqual(config["dataset"]["test_subset_indices"], list(range(1000)))
        self.assertEqual(
            index_text_sha256(config["dataset"]["training_subset_indices"]),
            "8db91b2ee25d579493dbc2ca66417cc945e215b5424349884013834d43df7ac4",
        )

    def test_compute_mode_does_not_read_committed_estimates(self) -> None:
        module = _load_experiment_module()
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        compute_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "compute"
        )
        called_names = {
            node.func.id
            for node in ast.walk(compute_node)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertNotIn("read_curve_csv", called_names)

    def test_frozen_e004a_and_e007_artifacts_are_byte_identical(self) -> None:
        for relative_path, expected_hash in FROZEN_PRIOR_ARTIFACTS.items():
            with self.subTest(path=relative_path):
                actual_hash = hashlib.sha256(
                    (REPO_ROOT / relative_path).read_bytes()
                ).hexdigest()
                self.assertEqual(actual_hash, expected_hash)

    def test_existing_frozen_e005_e006_hash_gate_remains_active(self) -> None:
        source = (REPO_ROOT / "tests" / "test_region_definitions.py").read_text()
        self.assertIn("test_frozen_e005_e006_artifacts_are_byte_identical", source)

    def test_committed_e004b_results_match_frozen_targets(self) -> None:
        results = REPO_ROOT / "results" / "experiment_04b"
        summary = json.loads((results / "band_target_summary.json").read_text())
        validation = json.loads(
            (results / "frequency_restricted_geometry_validation.json").read_text()
        )
        manifest = json.loads(
            (results / "frequency_restricted_geometry_manifest.json").read_text()
        )
        self.assertEqual(summary["low_band_geometry_target_indices"], [8])
        self.assertEqual(summary["high_band_geometry_target_indices"], [9, 10])
        self.assertEqual(validation["status"], "pass")
        self.assertEqual(manifest["validation"]["row_count"], 108)
        self.assertFalse(manifest["scientific_scope"]["e008_executed"])

    def test_canonical_registries_distinguish_geometry_and_residuals(self) -> None:
        pipeline = json.loads(
            (REPO_ROOT / "results" / "canonical_experiment_pipeline.json").read_text()
        )
        stage = pipeline["stage_3_frequency_restricted_geometry"]
        self.assertEqual(stage["low_target_indices"], [8])
        self.assertEqual(stage["high_target_indices"], [9, 10])
        self.assertFalse(stage["uses_e005_for_selection"])
        self.assertEqual(
            pipeline["stage_7_frequency_geometry_swap"]["status"],
            "proposed_not_executed",
        )

    def test_e008_protocol_is_blocked_and_unexecuted(self) -> None:
        protocol = (
            REPO_ROOT / "docs" / "experiment_08_frequency_geometry_swap_protocol.md"
        ).read_text()
        self.assertIn("PROPOSED — NOT EXECUTED", protocol)
        self.assertIn("EDM-50K no-swap baseline is `0/256`", protocol)
        self.assertIn("Low `8..8`", protocol)
        self.assertIn("High `9..10`", protocol)

    def test_documentation_states_four_curves_and_rank_limitations(self) -> None:
        research_record = (REPO_ROOT / "docs" / "research_record.md").read_text()
        results = (
            REPO_ROOT
            / "docs"
            / "experiment_04b_frequency_restricted_geometry_results.md"
        ).read_text()
        protocol = (
            REPO_ROOT
            / "docs"
            / "experiment_04b_frequency_restricted_geometry_protocol.md"
        ).read_text()
        self.assertIn("two curves in the low-frequency subspace", research_record)
        self.assertIn("two curves in the\nhigh-frequency subspace", research_record)
        for quantity in (
            "C_low_sigma(p,D)",
            "W_low_sigma(D)",
            "C_high_sigma(p,D)",
            "W_high_sigma(D)",
        ):
            self.assertIn(quantity, research_record)
        self.assertIn("E004B = frequency-restricted data geometry", research_record)
        self.assertIn(
            "E005  = frequency-restricted denoising residual energy",
            research_record,
        )
        self.assertIn("ranks, `147` and `2925`", research_record)
        self.assertIn("does not isolate frequency", research_record)
        self.assertIn(
            "subspace dimension, covariance, or energy structure", research_record
        )
        self.assertIn(
            "Coverage alone does not exhibit the same ordering", research_record
        )
        self.assertNotIn(
            "low-frequency coverage occurs earlier", research_record.lower()
        )
        self.assertNotIn("low frequencies cause memorization", research_record.lower())
        self.assertIn("### Required Future Controls", results)
        self.assertIn("rank-matched random subspaces", results)
        self.assertIn("rank-and-power-matched", results)
        self.assertIn("optionally whitened", results)
        self.assertIn("repeated dataset subsets and Gaussian seeds", results)
        self.assertIn("Distances are neither divided by band\nrank", protocol)
        self.assertIn("Sigma is not rescaled by band energy", protocol)


if __name__ == "__main__":
    unittest.main()
