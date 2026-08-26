"""Numerical and protocol tests for the frozen E011 intervention."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from spectral_diffusion_playground.e011_fourier_component_swap import (
    DISK_RADIUS,
    EXPECTED_RECORDS,
    INTERVENTION_CALLS,
    SAMPLE_SEEDS,
    SQUARE_CUTOFF,
    denoiser_prediction_for_step,
    disk_mask,
    effective_mode_count,
    frozen_conditions,
    hybridize_denoiser_predictions,
    mask_for_condition,
    paired_mask_effect_difference,
    paired_memorization_effect,
    sample_condition,
    select_matched_square_cutoff,
    square_mask,
    validate_condition_registry,
)

ROOT = Path(__file__).resolve().parents[1]


class AffineDenoiser(torch.nn.Module):
    """Small deterministic denoiser used only for sampler invariance tests."""

    def __init__(self, scale: float, bias: float) -> None:
        super().__init__()
        self.scale = scale
        self.bias = bias
        self.calls = 0

    def forward(
        self,
        state: torch.Tensor,
        sigma: torch.Tensor,
        class_labels: None = None,
    ) -> torch.Tensor:
        del sigma, class_labels
        self.calls += 1
        return state * self.scale + self.bias


class E011FourierSwapTests(unittest.TestCase):
    """Protect Fourier geometry, sampler behavior, and frozen identities."""

    def test_fft_round_trip(self) -> None:
        generator = torch.Generator().manual_seed(4)
        values = torch.randn((2, 3, 32, 32), generator=generator, dtype=torch.float64)
        reconstructed = torch.fft.irfft2(
            torch.fft.rfft2(values, norm="ortho"),
            s=values.shape[-2:],
            norm="ortho",
        )
        self.assertTrue(torch.allclose(values, reconstructed, atol=1e-12, rtol=1e-12))

    def test_empty_mask_returns_base_prediction(self) -> None:
        generator = torch.Generator().manual_seed(5)
        base = torch.randn((2, 3, 32, 32), generator=generator, dtype=torch.float64)
        donor = torch.randn((2, 3, 32, 32), generator=generator, dtype=torch.float64)
        mask = torch.zeros((1, 1, 32, 17), dtype=torch.bool)
        actual = hybridize_denoiser_predictions(base, donor, mask)
        self.assertTrue(torch.allclose(actual, base, atol=1e-12, rtol=1e-12))

    def test_full_mask_returns_donor_prediction(self) -> None:
        generator = torch.Generator().manual_seed(6)
        base = torch.randn((2, 3, 32, 32), generator=generator, dtype=torch.float64)
        donor = torch.randn((2, 3, 32, 32), generator=generator, dtype=torch.float64)
        mask = torch.ones((1, 1, 32, 17), dtype=torch.bool)
        actual = hybridize_denoiser_predictions(base, donor, mask)
        self.assertTrue(torch.allclose(actual, donor, atol=1e-12, rtol=1e-12))

    def test_self_swap_returns_original_prediction(self) -> None:
        generator = torch.Generator().manual_seed(7)
        prediction = torch.randn(
            (1, 3, 32, 32), generator=generator, dtype=torch.float64
        )
        mask = disk_mask(32, 32)[None, None]
        actual = hybridize_denoiser_predictions(prediction, prediction, mask)
        self.assertTrue(torch.allclose(actual, prediction, atol=1e-12, rtol=1e-12))

    def test_outside_window_evaluates_only_base(self) -> None:
        condition = next(
            item
            for item in frozen_conditions()
            if item.try_id == "try01_radial_r11" and item.condition_id == "G_from_M"
        )
        base = AffineDenoiser(0.25, 0.1)
        donor = AffineDenoiser(0.75, -0.2)
        networks = {condition.base_model: base, condition.donor_model: donor}
        state = torch.ones((1, 3, 32, 32), dtype=torch.float64)
        sigma = torch.ones((1,), dtype=torch.float64)
        mask = mask_for_condition(condition, 32, 32)
        actual = denoiser_prediction_for_step(
            networks, condition, state, sigma, 5, mask
        )
        expected = state * 0.25 + 0.1
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual((base.calls, donor.calls), (1, 0))

    def test_inside_window_uses_shared_state_for_both_models(self) -> None:
        condition = next(
            item
            for item in frozen_conditions()
            if item.try_id == "try02_square_matched" and item.condition_id == "M_from_G"
        )
        base = AffineDenoiser(0.25, 0.1)
        donor = AffineDenoiser(0.75, -0.2)
        networks = {condition.base_model: base, condition.donor_model: donor}
        state = torch.arange(3 * 32 * 32, dtype=torch.float64).reshape(1, 3, 32, 32)
        sigma = torch.ones((1,), dtype=torch.float64)
        mask = mask_for_condition(condition, 32, 32)
        actual = denoiser_prediction_for_step(
            networks, condition, state, sigma, 6, mask
        )
        expected = hybridize_denoiser_predictions(
            state * 0.25 + 0.1, state * 0.75 - 0.2, mask
        )
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual((base.calls, donor.calls), (1, 1))

    def test_mask_geometry_and_weighted_counts(self) -> None:
        ky = torch.fft.fftfreq(32) * 32
        kx = torch.fft.rfftfreq(32) * 32
        grid_y, grid_x = torch.meshgrid(ky, kx, indexing="ij")
        expected_disk = torch.sqrt(grid_x.square() + grid_y.square()) <= 11
        expected_square = torch.maximum(grid_x.abs(), grid_y.abs()) <= 9
        actual_disk = disk_mask(32, 32, DISK_RADIUS)
        actual_square = square_mask(32, 32, SQUARE_CUTOFF)
        self.assertTrue(torch.equal(actual_disk, expected_disk))
        self.assertTrue(torch.equal(actual_square, expected_square))
        self.assertEqual(int(actual_disk.sum()), 200)
        self.assertEqual(int(actual_square.sum()), 190)
        self.assertEqual(effective_mode_count(actual_disk, 32), 377)
        self.assertEqual(effective_mode_count(actual_square, 32), 361)

    def test_square_matching_is_exhaustive_and_frozen(self) -> None:
        match = select_matched_square_cutoff(32, 32)
        self.assertEqual(match["square_cutoff"], 9)
        self.assertEqual(match["disk_effective_mode_count"], 377)
        self.assertEqual(match["square_effective_mode_count"], 361)
        self.assertEqual(match["absolute_count_difference"], 16)

    def test_sampler_is_deterministic(self) -> None:
        condition = next(
            item
            for item in frozen_conditions()
            if item.try_id == "try01_radial_r11" and item.condition_id == "G_from_M"
        )
        latent = torch.randn((1, 3, 32, 32), generator=torch.Generator().manual_seed(8))
        first = sample_condition(
            {
                condition.base_model: AffineDenoiser(0.1, 0.0),
                condition.donor_model: AffineDenoiser(0.2, 0.0),
            },
            condition,
            latent,
        )
        second = sample_condition(
            {
                condition.base_model: AffineDenoiser(0.1, 0.0),
                condition.donor_model: AffineDenoiser(0.2, 0.0),
            },
            condition,
            latent.clone(),
        )
        self.assertTrue(torch.equal(first, second))

    def test_condition_registry_seed_range_and_record_count(self) -> None:
        conditions = frozen_conditions()
        validate_condition_registry(conditions)
        self.assertEqual(len(conditions), 8)
        self.assertEqual(INTERVENTION_CALLS, tuple(range(6, 14)))
        self.assertEqual(SAMPLE_SEEDS, tuple(range(50000, 51000)))
        self.assertEqual(len(conditions) * len(SAMPLE_SEEDS), EXPECTED_RECORDS)

    def test_seed_range_does_not_overlap_prior_evaluation_ranges(self) -> None:
        current = set(SAMPLE_SEEDS)
        for prior in (
            range(256),
            range(10000, 10128),
            range(20000, 20128),
            range(30000, 30128),
            range(40000, 40256),
        ):
            self.assertFalse(current.intersection(prior))

    def test_paired_effects_do_not_assume_direction(self) -> None:
        base = np.asarray([0, 0, 1, 1], dtype=np.int8)
        hybrid = np.asarray([1, 0, 0, 1], dtype=np.int8)
        summary = paired_memorization_effect(base, hybrid, resamples=500)
        self.assertEqual(summary["paired_effect"], 0.0)
        inverse = paired_memorization_effect(hybrid, base, resamples=500)
        self.assertEqual(inverse["paired_effect"], 0.0)

    def test_mask_effect_comparison_is_paired(self) -> None:
        base = np.asarray([0, 0, 0, 0], dtype=np.int8)
        disk = np.asarray([1, 1, 0, 0], dtype=np.int8)
        square = np.asarray([1, 0, 0, 0], dtype=np.int8)
        summary = paired_mask_effect_difference(
            base, disk, base.copy(), square, resamples=500
        )
        self.assertEqual(summary["disk_effect"], 0.5)
        self.assertEqual(summary["square_effect"], 0.25)
        self.assertEqual(summary["disk_minus_square_effect"], 0.25)

    def test_config_manifest_hashes_and_frozen_fid_source(self) -> None:
        config = json.loads(
            (ROOT / "configs/e011_fourier_component_swap.json").read_text()
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
        self.assertEqual(
            config["fid"]["fid_py_sha256"],
            "704be0de2c77c0090fdb9787268360ef9d438ca83c6aca14aa6d4f51dcff364d",
        )

    def test_slurm_launcher_is_guarded_and_quota_safe(self) -> None:
        launcher = (ROOT / "scripts/e011_fourier_component_swap.slurm").read_text()
        self.assertIn("E011_REPO_ROOT", launcher)
        self.assertIn("E011_REPO_COMMIT", launcher)
        self.assertIn('git -C "$REPO_ROOT" status --porcelain=v1', launcher)
        self.assertIn('DNNLIB_CACHE_DIR="${TMP_ROOT}/dnnlib"', launcher)
        self.assertIn("CUBLAS_WORKSPACE_CONFIG=:4096:8", launcher)
        self.assertIn("--gres=gpu:L40S:1", launcher)
        self.assertIn("python_executable=", launcher)
        self.assertIn("torch_version=", launcher)
        self.assertIn("torch_file=", launcher)
        self.assertIn("python -m torch.distributed.run --help", launcher)
        entrypoint = (ROOT / "experiments/11_fourier_component_swap.py").read_text()
        self.assertIn('sys.executable,\n            "-m",', entrypoint)
        self.assertIn('"torch.distributed.run"', entrypoint)
        self.assertNotIn("E011_TORCHRUN", launcher + entrypoint)
        self.assertNotIn("BASH_SOURCE", launcher)

    def test_full_output_is_immutable_without_explicit_resume(self) -> None:
        launcher = (ROOT / "scripts/e011_fourier_component_swap.slurm").read_text()
        self.assertIn(
            'if [ -n "$OUTPUT_DIR" ] && [ -e "$OUTPUT_DIR" ] && [ "$MODE" != "resume" ]',
            launcher,
        )
        self.assertIn(
            "Refusing existing output directory",
            (ROOT / "experiments/11_fourier_component_swap.py").read_text(),
        )


if __name__ == "__main__":
    unittest.main()
