"""Frozen radial-band intervention and factorial utilities for E012."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Final, Mapping, Sequence

import numpy as np
import torch

from spectral_diffusion_playground.e006_transition_swaps import SIGMA_GRID
from spectral_diffusion_playground.e010_directional_transfer import (
    bootstrap_mean_interval,
)
from spectral_diffusion_playground.e011_fourier_component_swap import (
    INTERVENTION_CALLS,
    disk_mask,
    effective_mode_count,
    frequency_coordinates,
    hybridize_denoiser_predictions,
)

EXPERIMENT_ID: Final[str] = "E012"
RUN_ID: Final[str] = "e012_frequency_localization"
SAMPLE_SEEDS: Final[tuple[int, ...]] = tuple(range(50000, 51000))
SMOKE_SEEDS: Final[tuple[int, ...]] = SAMPLE_SEEDS[:2]
EXPECTED_RECORDS: Final[int] = 9_000
BOOTSTRAP_RESAMPLES: Final[int] = 100_000
BOOTSTRAP_SEED: Final[int] = 0
BASE_MODEL: Final[str] = "edm_50k_040000"
DONOR_MODEL: Final[str] = "edm_1k_012000"

BAND_LIMITS: Final[dict[str, tuple[float, float, bool]]] = {
    "B1": (0.0, 6.0, True),
    "B2": (6.0, 9.0, False),
    "B3": (9.0, 11.0, False),
    "B4": (11.0, 13.0, False),
}


@dataclass(frozen=True)
class RadialBandCondition:
    """One frozen G-recipient radial-band condition."""

    condition_id: str
    bands: tuple[str, ...]

    @property
    def is_hybrid(self) -> bool:
        """Return whether any donor coefficients are selected."""
        return bool(self.bands)

    @property
    def factorial_bits(self) -> tuple[int, int, int] | None:
        """Return B1/B2/B3 membership, excluding the B4 control."""
        if "B4" in self.bands:
            return None
        return tuple(int(name in self.bands) for name in ("B1", "B2", "B3"))


def frozen_conditions() -> tuple[RadialBandCondition, ...]:
    """Return the nine preregistered E012 conditions in display order."""
    return (
        RadialBandCondition("000_none", ()),
        RadialBandCondition("100_B1", ("B1",)),
        RadialBandCondition("010_B2", ("B2",)),
        RadialBandCondition("001_B3", ("B3",)),
        RadialBandCondition("110_B1_B2", ("B1", "B2")),
        RadialBandCondition("101_B1_B3", ("B1", "B3")),
        RadialBandCondition("011_B2_B3", ("B2", "B3")),
        RadialBandCondition("111_B1_B2_B3", ("B1", "B2", "B3")),
        RadialBandCondition("control_B4", ("B4",)),
    )


def validate_condition_registry(conditions: Sequence[RadialBandCondition]) -> None:
    """Fail unless the condition registry is exactly the frozen registry."""
    if tuple(conditions) != frozen_conditions():
        raise ValueError("Condition registry differs from the frozen E012 registry")
    identifiers = [condition.condition_id for condition in conditions]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("E012 condition identifiers must be unique")


def radial_band_masks(
    height: int = 32,
    width: int = 32,
    *,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """Return exact disjoint masks B1-B4 on the actual rFFT grid."""
    ky, kx = frequency_coordinates(height, width, device=device)
    radius = torch.sqrt(kx.square() + ky.square())
    masks: dict[str, torch.Tensor] = {}
    for name, (lower, upper, include_lower) in BAND_LIMITS.items():
        lower_test = radius >= lower if include_lower else radius > lower
        masks[name] = torch.logical_and(lower_test, radius <= upper)
    return masks


def validate_band_geometry(
    height: int = 32,
    width: int = 32,
    *,
    device: torch.device | str | None = None,
) -> dict[str, dict[str, int]]:
    """Validate frozen geometry and return stored/effective mode counts."""
    masks = radial_band_masks(height, width, device=device)
    expected = {
        "B1": (63, 113),
        "B2": (73, 140),
        "B3": (64, 124),
        "B4": (78, 152),
    }
    counts = {
        name: {
            "stored_coefficients": int(mask.sum().item()),
            "effective_full_spectrum_modes": effective_mode_count(mask, width),
        }
        for name, mask in masks.items()
    }
    for name, (stored, effective) in expected.items():
        if counts[name] != {
            "stored_coefficients": stored,
            "effective_full_spectrum_modes": effective,
        }:
            raise ValueError(f"Unexpected E012 geometry for {name}: {counts[name]}")
    factorial = (masks["B1"], masks["B2"], masks["B3"])
    if any(
        torch.logical_and(left, right).any().item()
        for left, right in combinations(factorial, 2)
    ):
        raise ValueError("E012 factorial bands are not pairwise disjoint")
    union = factorial[0] | factorial[1] | factorial[2]
    if not torch.equal(union, disk_mask(height, width, 11, device=device)):
        raise ValueError("E012 factorial union does not equal E011 r<=11")
    return counts


def mask_for_condition(
    condition: RadialBandCondition,
    height: int,
    width: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return one condition's union mask with NCHW-broadcast dimensions."""
    masks = radial_band_masks(height, width, device=device)
    selected = torch.zeros_like(masks["B1"])
    for name in condition.bands:
        if name not in masks:
            raise ValueError(f"Unknown E012 band: {name}")
        selected |= masks[name]
    return selected[None, None]


def condition_mode_count(condition: RadialBandCondition, width: int = 32) -> int:
    """Return effective full-spectrum modes selected by one condition."""
    return effective_mode_count(mask_for_condition(condition, 32, width)[0, 0], width)


def denoiser_prediction_for_step(
    networks: Mapping[str, torch.nn.Module],
    condition: RadialBandCondition,
    state: torch.Tensor,
    sigma: torch.Tensor,
    step_index: int,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Evaluate G and conditionally insert M coefficients on the same state."""
    base = networks[BASE_MODEL](state, sigma, class_labels=None)
    if not condition.is_hybrid or step_index not in INTERVENTION_CALLS:
        return base
    donor = networks[DONOR_MODEL](state, sigma, class_labels=None)
    return hybridize_denoiser_predictions(base, donor, mask)


def sample_condition(
    networks: Mapping[str, torch.nn.Module],
    condition: RadialBandCondition,
    latent: torch.Tensor,
) -> torch.Tensor:
    """Run one deterministic E012 18-call pure-Euler trajectory."""
    if latent.ndim != 4 or latent.shape[-2:] != (32, 32):
        raise ValueError("E012 latent must have NCHW shape ending in 32x32")
    state = latent.to(torch.float64) * SIGMA_GRID[0]
    mask = mask_for_condition(condition, 32, 32, device=state.device)
    schedule = SIGMA_GRID + (0.0,)
    with torch.no_grad():
        for step, (sigma_value, sigma_next) in enumerate(
            zip(schedule[:-1], schedule[1:])
        ):
            sigma = torch.full(
                (state.shape[0],), sigma_value, dtype=torch.float64, device=state.device
            )
            denoised = denoiser_prediction_for_step(
                networks, condition, state, sigma, step, mask
            ).to(torch.float64)
            if not torch.isfinite(denoised).all().item():
                raise RuntimeError(f"Nonfinite denoiser output at call {step}")
            state = (
                state + (sigma_next - sigma_value) * (state - denoised) / sigma_value
            )
    if not torch.isfinite(state).all().item():
        raise RuntimeError("Nonfinite final E012 sample")
    return state


def paired_effect(
    baseline: np.ndarray,
    treated: np.ndarray,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Return treated-minus-baseline paired effect and percentile interval."""
    base = np.asarray(baseline, dtype=np.float64)
    value = np.asarray(treated, dtype=np.float64)
    if base.shape != value.shape or base.ndim != 1 or base.size == 0:
        raise ValueError("Paired outcomes must be aligned nonempty vectors")
    difference = value - base
    low, high = bootstrap_mean_interval(difference, seed=seed, resamples=resamples)
    return {"estimate": float(difference.mean()), "ci95_low": low, "ci95_high": high}


def factorial_contrast_vectors(
    outcomes: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return per-seed standard 2^3 main and interaction contrasts."""
    by_bits = {
        condition.factorial_bits: np.asarray(
            outcomes[condition.condition_id], dtype=np.float64
        )
        for condition in frozen_conditions()
        if condition.factorial_bits is not None
    }
    if len(by_bits) != 8 or len({array.shape for array in by_bits.values()}) != 1:
        raise ValueError("Factorial outcomes must contain eight aligned conditions")
    definitions = {
        "B1": (0,),
        "B2": (1,),
        "B3": (2,),
        "B1xB2": (0, 1),
        "B1xB3": (0, 2),
        "B2xB3": (1, 2),
        "B1xB2xB3": (0, 1, 2),
    }
    result: dict[str, np.ndarray] = {}
    for name, active in definitions.items():
        total = np.zeros_like(next(iter(by_bits.values())), dtype=np.float64)
        for bits, values in by_bits.items():
            sign = np.prod([1.0 if bits[index] else -1.0 for index in active])
            total += sign * values
        result[name] = total / (2 ** (3 - len(active)))
    return result


def summarize_factorial_contrasts(
    outcomes: Mapping[str, np.ndarray],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[dict[str, float | str]]:
    """Summarize paired factorial contrasts with percentile intervals."""
    summaries: list[dict[str, float | str]] = []
    for name, values in factorial_contrast_vectors(outcomes).items():
        low, high = bootstrap_mean_interval(values, seed=seed, resamples=resamples)
        summaries.append(
            {
                "contrast": name,
                "estimate": float(values.mean()),
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    return summaries


def overlap_with_full(
    condition: np.ndarray,
    full: np.ndarray,
    seeds: Sequence[int],
) -> dict[str, object]:
    """Return paired memorized-seed overlap diagnostics against condition 111."""
    selected = np.asarray(condition, dtype=bool)
    reference = np.asarray(full, dtype=bool)
    seed_array = np.asarray(seeds, dtype=np.int64)
    if selected.shape != reference.shape or selected.shape != seed_array.shape:
        raise ValueError("Overlap inputs must be aligned")
    intersection = int(np.logical_and(selected, reference).sum())
    union = int(np.logical_or(selected, reference).sum())
    full_count = int(reference.sum())
    subband_only = seed_array[np.logical_and(selected, ~reference)].tolist()
    return {
        "memorized_count": int(selected.sum()),
        "intersection_with_111": intersection,
        "recovery_of_111": intersection / full_count if full_count else 0.0,
        "jaccard": intersection / union if union else 1.0,
        "subband_only_count": len(subband_only),
        "subband_only_seeds": subband_only,
    }


def ratio_summary(values: np.ndarray) -> dict[str, float]:
    """Return frozen continuous nearest-neighbor ratio summaries."""
    ratios = np.asarray(values, dtype=np.float64)
    if ratios.ndim != 1 or ratios.size == 0 or not np.isfinite(ratios).all():
        raise ValueError("NN ratio values must be finite and nonempty")
    return {
        "mean": float(ratios.mean()),
        "median": float(np.median(ratios)),
        "q25": float(np.quantile(ratios, 0.25)),
        "q75": float(np.quantile(ratios, 0.75)),
    }
