"""Protocol and numerical tests for E012 radial frequency localization."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from spectral_diffusion_playground.e011_fourier_component_swap import (
    FourierSwapCondition,
    disk_mask,
    sample_condition as sample_e011_condition,
)
from spectral_diffusion_playground.e012_frequency_localization import (
    BASE_MODEL,
    DONOR_MODEL,
    EXPECTED_RECORDS,
    SAMPLE_SEEDS,
    RadialBandCondition,
    condition_mode_count,
    denoiser_prediction_for_step,
    factorial_contrast_vectors,
    frozen_conditions,
    mask_for_condition,
    overlap_with_full,
    radial_band_masks,
    ratio_summary,
    sample_condition,
    summarize_factorial_contrasts,
    validate_band_geometry,
    validate_condition_registry,
)

ROOT = Path(__file__).resolve().parents[1]


class RecordingDenoiser(torch.nn.Module):
    """Deterministic affine denoiser that records input tensors."""

    def __init__(self, scale: float, bias: float) -> None:
        super().__init__()
        self.scale = scale
        self.bias = bias
        self.inputs: list[torch.Tensor] = []

    def forward(
        self,
        state: torch.Tensor,
        sigma: torch.Tensor,
        class_labels: None = None,
    ) -> torch.Tensor:
        del sigma, class_labels
        self.inputs.append(state.detach().clone())
        return state * self.scale + self.bias


class E012FrequencyLocalizationTests(unittest.TestCase):
    """Protect E012 band geometry, sampler behavior, and frozen contracts."""

    def test_exact_band_geometry_counts_disjointness_and_union(self) -> None:
        masks = radial_band_masks()
        ky = torch.fft.fftfreq(32) * 32
        kx = torch.fft.rfftfreq(32) * 32
        grid_y, grid_x = torch.meshgrid(ky, kx, indexing="ij")
        radius = torch.sqrt(grid_x.square() + grid_y.square())
        expected = {
            "B1": (radius >= 0) & (radius <= 6),
            "B2": (radius > 6) & (radius <= 9),
            "B3": (radius > 9) & (radius <= 11),
            "B4": (radius > 11) & (radius <= 13),
        }
        for name in expected:
            self.assertTrue(torch.equal(masks[name], expected[name]))
        for left, right in (("B1", "B2"), ("B1", "B3"), ("B2", "B3")):
            self.assertFalse(torch.logical_and(masks[left], masks[right]).any())
        self.assertTrue(
            torch.equal(masks["B1"] | masks["B2"] | masks["B3"], disk_mask(32, 32, 11))
        )
        self.assertEqual(
            validate_band_geometry(),
            {
                "B1": {"stored_coefficients": 63, "effective_full_spectrum_modes": 113},
                "B2": {"stored_coefficients": 73, "effective_full_spectrum_modes": 140},
                "B3": {"stored_coefficients": 64, "effective_full_spectrum_modes": 124},
                "B4": {"stored_coefficients": 78, "effective_full_spectrum_modes": 152},
            },
        )

    def test_condition_registry_seeds_and_record_count(self) -> None:
        conditions = frozen_conditions()
        validate_condition_registry(conditions)
        self.assertEqual(len(conditions), 9)
        self.assertEqual(SAMPLE_SEEDS, tuple(range(50000, 51000)))
        self.assertEqual(len(conditions) * len(SAMPLE_SEEDS), EXPECTED_RECORDS)
        self.assertEqual(condition_mode_count(conditions[0]), 0)
        self.assertEqual(condition_mode_count(conditions[7]), 377)
        self.assertEqual(condition_mode_count(conditions[8]), 152)

    def test_000_is_g_and_outside_window_does_not_evaluate_donor(self) -> None:
        base = RecordingDenoiser(0.25, 0.1)
        donor = RecordingDenoiser(0.75, -0.2)
        state = torch.ones((1, 3, 32, 32), dtype=torch.float64)
        sigma = torch.ones((1,), dtype=torch.float64)
        none = frozen_conditions()[0]
        actual = denoiser_prediction_for_step(
            {BASE_MODEL: base, DONOR_MODEL: donor},
            none,
            state,
            sigma,
            6,
            mask_for_condition(none, 32, 32),
        )
        self.assertTrue(torch.equal(actual, state * 0.25 + 0.1))
        self.assertEqual((len(base.inputs), len(donor.inputs)), (1, 0))
        hybrid = frozen_conditions()[7]
        denoiser_prediction_for_step(
            {BASE_MODEL: base, DONOR_MODEL: donor},
            hybrid,
            state,
            sigma,
            5,
            mask_for_condition(hybrid, 32, 32),
        )
        self.assertEqual((len(base.inputs), len(donor.inputs)), (2, 0))

    def test_inside_window_uses_exact_same_state(self) -> None:
        base = RecordingDenoiser(0.25, 0.1)
        donor = RecordingDenoiser(0.75, -0.2)
        state = torch.arange(3 * 32 * 32, dtype=torch.float64).reshape(1, 3, 32, 32)
        condition = frozen_conditions()[3]
        denoiser_prediction_for_step(
            {BASE_MODEL: base, DONOR_MODEL: donor},
            condition,
            state,
            torch.ones((1,), dtype=torch.float64),
            6,
            mask_for_condition(condition, 32, 32),
        )
        self.assertTrue(torch.equal(base.inputs[0], donor.inputs[0]))
        self.assertTrue(torch.equal(base.inputs[0], state))

    def test_000_and_111_match_e011_implementations(self) -> None:
        latent = torch.randn(
            (1, 3, 32, 32), generator=torch.Generator().manual_seed(19)
        )
        networks = {
            BASE_MODEL: RecordingDenoiser(0.1, 0.01),
            DONOR_MODEL: RecordingDenoiser(0.2, -0.02),
        }
        actual_g = sample_condition(networks, frozen_conditions()[0], latent.clone())
        expected_g = sample_e011_condition(
            networks,
            FourierSwapCondition("try01_radial_r11", "disk", "G", BASE_MODEL, None),
            latent.clone(),
        )
        self.assertTrue(torch.equal(actual_g, expected_g))
        actual_full = sample_condition(networks, frozen_conditions()[7], latent.clone())
        expected_full = sample_e011_condition(
            networks,
            FourierSwapCondition(
                "try01_radial_r11", "disk", "G_from_M", BASE_MODEL, DONOR_MODEL
            ),
            latent.clone(),
        )
        self.assertTrue(torch.equal(actual_full, expected_full))

    def test_sampler_is_repeat_deterministic(self) -> None:
        latent = torch.randn(
            (1, 3, 32, 32), generator=torch.Generator().manual_seed(20)
        )
        condition = frozen_conditions()[6]
        first = sample_condition(
            {
                BASE_MODEL: RecordingDenoiser(0.1, 0),
                DONOR_MODEL: RecordingDenoiser(0.2, 0),
            },
            condition,
            latent,
        )
        second = sample_condition(
            {
                BASE_MODEL: RecordingDenoiser(0.1, 0),
                DONOR_MODEL: RecordingDenoiser(0.2, 0),
            },
            condition,
            latent.clone(),
        )
        self.assertTrue(torch.equal(first, second))

    def test_factorial_contrasts_use_standard_scaling(self) -> None:
        outcomes = {}
        for condition in frozen_conditions()[:8]:
            bits = condition.factorial_bits
            assert bits is not None
            outcomes[condition.condition_id] = np.asarray(
                [bits[0] + 2 * bits[1] + 4 * bits[2] + 3 * bits[0] * bits[1]],
                dtype=np.float64,
            )
        vectors = factorial_contrast_vectors(outcomes)
        self.assertEqual(float(vectors["B1"][0]), 2.5)
        self.assertEqual(float(vectors["B2"][0]), 3.5)
        self.assertEqual(float(vectors["B3"][0]), 4.0)
        self.assertEqual(float(vectors["B1xB2"][0]), 3.0)
        self.assertEqual(float(vectors["B1xB3"][0]), 0.0)
        self.assertEqual(float(vectors["B2xB3"][0]), 0.0)
        self.assertEqual(float(vectors["B1xB2xB3"][0]), 0.0)
        self.assertEqual(len(summarize_factorial_contrasts(outcomes, resamples=100)), 7)

    def test_overlap_and_ratio_summaries(self) -> None:
        overlap = overlap_with_full(
            np.asarray([1, 1, 0, 0]),
            np.asarray([1, 0, 1, 0]),
            (10, 11, 12, 13),
        )
        self.assertEqual(overlap["intersection_with_111"], 1)
        self.assertEqual(overlap["recovery_of_111"], 0.5)
        self.assertEqual(overlap["jaccard"], 1 / 3)
        self.assertEqual(overlap["subband_only_seeds"], [11])
        summary = ratio_summary(np.asarray([0.1, 0.2, 0.3, 0.4]))
        self.assertEqual(summary["median"], 0.25)
        self.assertAlmostEqual(summary["q25"], 0.175)
        self.assertAlmostEqual(summary["q75"], 0.325)

    def test_config_manifest_hashes(self) -> None:
        config = json.loads(
            (ROOT / "configs/e012_frequency_localization.json").read_text()
        )
        for path_key, hash_key in (
            ("model_pair_manifest", "model_pair_manifest_sha256"),
            ("condition_manifest", "condition_manifest_sha256"),
            ("seed_manifest", "seed_manifest_sha256"),
            ("mask_manifest", "mask_manifest_sha256"),
        ):
            path = ROOT / config["inputs"][path_key]
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                config["inputs"][hash_key],
            )
        self.assertEqual(config["execution"]["expected_records"], 9000)

    def test_launcher_is_guarded_and_does_not_auto_submit_full(self) -> None:
        launcher = (ROOT / "scripts/e012_frequency_localization.slurm").read_text()
        self.assertIn("E012_REPO_ROOT", launcher)
        self.assertIn("E012_REPO_COMMIT", launcher)
        self.assertIn('git -C "$REPO_ROOT" status --porcelain=v1', launcher)
        self.assertIn("CUBLAS_WORKSPACE_CONFIG=:4096:8", launcher)
        self.assertIn("--gres=gpu:L40S:1", launcher)
        self.assertIn("python -m torch.distributed.run --help", launcher)
        self.assertNotIn("BASH_SOURCE", launcher)
        self.assertFalse(
            any(line.lstrip().startswith("sbatch ") for line in launcher.splitlines())
        )
        self.assertIn('if [ -n "$OUTPUT_DIR" ] && [ -e "$OUTPUT_DIR" ]', launcher)
        entrypoint = (ROOT / "experiments/12_frequency_localization.py").read_text()
        self.assertIn("Refusing existing output directory", entrypoint)


if __name__ == "__main__":
    unittest.main()
