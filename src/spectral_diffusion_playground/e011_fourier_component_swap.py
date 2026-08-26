"""Frozen Fourier-component intervention utilities for Experiment 11."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping, Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from spectral_diffusion_playground.e006_transition_swaps import SIGMA_GRID
from spectral_diffusion_playground.e010_directional_transfer import (
    bootstrap_mean_interval,
)

EXPERIMENT_ID: Final[str] = "E011"
RUN_ID: Final[str] = "e011_fourier_component_swap"
SAMPLE_SEEDS: Final[tuple[int, ...]] = tuple(range(50000, 51000))
SMOKE_SEEDS: Final[tuple[int, ...]] = SAMPLE_SEEDS[:2]
INTERVENTION_CALLS: Final[tuple[int, ...]] = tuple(range(6, 14))
DISK_RADIUS: Final[int] = 11
SQUARE_CUTOFF: Final[int] = 9
BOOTSTRAP_RESAMPLES: Final[int] = 100_000
BOOTSTRAP_SEED: Final[int] = 0
EXPECTED_RECORDS: Final[int] = 8_000

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FourierSwapCondition:
    """One baseline or shared-state Fourier-component swap condition."""

    try_id: str
    mask_type: str
    condition_id: str
    base_model: str
    donor_model: str | None

    @property
    def is_hybrid(self) -> bool:
        """Return whether this condition evaluates a donor model."""
        return self.donor_model is not None


def frozen_conditions() -> tuple[FourierSwapCondition, ...]:
    """Return both frozen four-condition E011 tries in execution order."""
    conditions: list[FourierSwapCondition] = []
    for try_id, mask_type in (
        ("try01_radial_r11", "disk"),
        ("try02_square_matched", "square"),
    ):
        conditions.extend(
            (
                FourierSwapCondition(try_id, mask_type, "G", "edm_50k_040000", None),
                FourierSwapCondition(try_id, mask_type, "M", "edm_1k_012000", None),
                FourierSwapCondition(
                    try_id,
                    mask_type,
                    "G_from_M",
                    "edm_50k_040000",
                    "edm_1k_012000",
                ),
                FourierSwapCondition(
                    try_id,
                    mask_type,
                    "M_from_G",
                    "edm_1k_012000",
                    "edm_50k_040000",
                ),
            )
        )
    return tuple(conditions)


def validate_condition_registry(
    conditions: Sequence[FourierSwapCondition],
) -> None:
    """Fail unless conditions exactly match the frozen E011 registry."""
    if tuple(conditions) != frozen_conditions():
        raise ValueError("Condition registry differs from the frozen E011 registry")
    keys = [(item.try_id, item.condition_id) for item in conditions]
    if len(keys) != len(set(keys)):
        raise ValueError("E011 condition keys must be unique within each try")


def frequency_coordinates(
    height: int,
    width: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return actual vertical and stored horizontal FFT coordinates."""
    if height < 1 or width < 2:
        raise ValueError("FFT image dimensions are invalid")
    ky = torch.fft.fftfreq(height, device=device, dtype=dtype) * height
    kx = torch.fft.rfftfreq(width, device=device, dtype=dtype) * width
    return torch.meshgrid(ky, kx, indexing="ij")


def disk_mask(
    height: int,
    width: int,
    radius: float = DISK_RADIUS,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return ``1[sqrt(kx^2 + ky^2) <= radius]`` on an rFFT grid."""
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("Disk radius must be finite and nonnegative")
    ky, kx = frequency_coordinates(height, width, device=device)
    return torch.sqrt(kx.square() + ky.square()) <= radius


def square_mask(
    height: int,
    width: int,
    cutoff: float = SQUARE_CUTOFF,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return ``1[max(abs(kx), abs(ky)) <= cutoff]`` on an rFFT grid."""
    if not np.isfinite(cutoff) or cutoff < 0:
        raise ValueError("Square cutoff must be finite and nonnegative")
    ky, kx = frequency_coordinates(height, width, device=device)
    return torch.maximum(kx.abs(), ky.abs()) <= cutoff


def rfft2_full_spectrum_weights(
    height: int,
    width: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return each stored coefficient's multiplicity in the full spectrum."""
    _, kx = frequency_coordinates(height, width, device=device)
    self_conjugate_horizontal = kx == 0
    if width % 2 == 0:
        self_conjugate_horizontal = torch.logical_or(
            self_conjugate_horizontal, kx == width // 2
        )
    return torch.where(
        self_conjugate_horizontal,
        torch.ones_like(kx, dtype=torch.int64),
        torch.full_like(kx, 2, dtype=torch.int64),
    )


def effective_mode_count(mask: torch.Tensor, width: int) -> int:
    """Count full-spectrum modes represented by one two-dimensional rFFT mask."""
    values = torch.as_tensor(mask, dtype=torch.bool)
    if values.ndim != 2 or values.shape[1] != width // 2 + 1:
        raise ValueError("Mask shape is incompatible with the supplied width")
    weights = rfft2_full_spectrum_weights(
        int(values.shape[0]), width, device=values.device
    )
    return int(weights[values].sum().item())


def select_matched_square_cutoff(
    height: int,
    width: int,
    radius: float = DISK_RADIUS,
    candidates: Sequence[int] = tuple(range(17)),
) -> dict[str, int | float]:
    """Select the smallest square cutoff closest in weighted mode count."""
    if not candidates or any(value < 0 for value in candidates):
        raise ValueError("Square candidates must be nonempty and nonnegative")
    target_mask = disk_mask(height, width, radius)
    target_count = effective_mode_count(target_mask, width)
    records = []
    for cutoff in candidates:
        candidate_mask = square_mask(height, width, cutoff)
        count = effective_mode_count(candidate_mask, width)
        records.append((abs(count - target_count), int(cutoff), count))
    difference, cutoff, count = min(records)
    return {
        "disk_effective_mode_count": target_count,
        "square_effective_mode_count": count,
        "square_cutoff": cutoff,
        "absolute_count_difference": difference,
        "relative_count_difference": difference / target_count,
    }


def mask_for_condition(
    condition: FourierSwapCondition,
    height: int,
    width: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Construct the frozen mask for one condition's try."""
    if condition.mask_type == "disk":
        mask = disk_mask(height, width, DISK_RADIUS, device=device)
    elif condition.mask_type == "square":
        mask = square_mask(height, width, SQUARE_CUTOFF, device=device)
    else:
        raise ValueError(f"Unknown E011 mask type: {condition.mask_type}")
    return mask[None, None, :, :]


def hybridize_denoiser_predictions(
    base_prediction: torch.Tensor,
    donor_prediction: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Replace masked base-prediction Fourier coefficients with donor values."""
    if base_prediction.shape != donor_prediction.shape:
        raise ValueError("Base and donor predictions must have identical shapes")
    if base_prediction.ndim != 4:
        raise ValueError("Predictions must use NCHW layout")
    if base_prediction.device != donor_prediction.device:
        raise ValueError("Base and donor predictions must share a device")
    base_fft = torch.fft.rfft2(base_prediction, norm="ortho")
    donor_fft = torch.fft.rfft2(donor_prediction, norm="ortho")
    selected = torch.as_tensor(mask, dtype=torch.bool, device=base_fft.device)
    if selected.shape[-2:] != base_fft.shape[-2:]:
        raise ValueError("Mask shape is incompatible with prediction FFT shape")
    hybrid_fft = torch.where(selected, donor_fft, base_fft)
    return torch.fft.irfft2(
        hybrid_fft,
        s=base_prediction.shape[-2:],
        norm="ortho",
    )


def denoiser_prediction_for_step(
    networks: Mapping[str, torch.nn.Module],
    condition: FourierSwapCondition,
    state: torch.Tensor,
    sigma: torch.Tensor,
    step_index: int,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the frozen baseline or shared-state hybrid denoiser."""
    base = networks[condition.base_model](state, sigma, class_labels=None)
    if not condition.is_hybrid or step_index not in INTERVENTION_CALLS:
        return base
    if condition.donor_model is None:
        raise RuntimeError("Hybrid E011 condition is missing a donor")
    donor = networks[condition.donor_model](state, sigma, class_labels=None)
    return hybridize_denoiser_predictions(base, donor, mask)


def sample_condition(
    networks: Mapping[str, torch.nn.Module],
    condition: FourierSwapCondition,
    latent: torch.Tensor,
) -> torch.Tensor:
    """Run one deterministic 18-call pure-Euler E011 trajectory."""
    if latent.ndim != 4 or latent.shape[-2:] != (32, 32):
        raise ValueError("E011 latent must have NCHW shape ending in 32x32")
    state = latent.to(torch.float64) * SIGMA_GRID[0]
    mask = mask_for_condition(
        condition, int(state.shape[-2]), int(state.shape[-1]), device=state.device
    )
    schedule = SIGMA_GRID + (0.0,)
    with torch.no_grad():
        for step, (sigma_value, sigma_next) in enumerate(
            zip(schedule[:-1], schedule[1:])
        ):
            sigma = torch.full(
                (state.shape[0],),
                sigma_value,
                dtype=torch.float64,
                device=state.device,
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
        raise RuntimeError("Nonfinite final E011 sample")
    return state


def paired_memorization_effect(
    baseline: np.ndarray,
    hybrid: np.ndarray,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Return hybrid-minus-baseline memorization effect and paired interval."""
    base = np.asarray(baseline, dtype=np.float64)
    treated = np.asarray(hybrid, dtype=np.float64)
    if base.shape != treated.shape or base.ndim != 1 or base.size == 0:
        raise ValueError("Paired outcomes must be aligned nonempty vectors")
    effects = treated - base
    low, high = bootstrap_mean_interval(effects, seed=seed, resamples=resamples)
    return {
        "baseline_rate": float(base.mean()),
        "hybrid_rate": float(treated.mean()),
        "paired_effect": float(effects.mean()),
        "ci95_low": low,
        "ci95_high": high,
    }


def paired_mask_effect_difference(
    disk_baseline: np.ndarray,
    disk_hybrid: np.ndarray,
    square_baseline: np.ndarray,
    square_hybrid: np.ndarray,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Compare paired treatment effects between disk and square masks."""
    disk_effect = np.asarray(disk_hybrid, dtype=np.float64) - np.asarray(
        disk_baseline, dtype=np.float64
    )
    square_effect = np.asarray(square_hybrid, dtype=np.float64) - np.asarray(
        square_baseline, dtype=np.float64
    )
    if disk_effect.shape != square_effect.shape or disk_effect.ndim != 1:
        raise ValueError("Disk and square effects must align by seed")
    difference = disk_effect - square_effect
    low, high = bootstrap_mean_interval(difference, seed=seed, resamples=resamples)
    return {
        "disk_effect": float(disk_effect.mean()),
        "square_effect": float(square_effect.mean()),
        "disk_minus_square_effect": float(difference.mean()),
        "ci95_low": low,
        "ci95_high": high,
    }


def sigma_grid() -> tuple[float, ...]:
    """Expose the immutable 18-call sampler schedule."""
    return tuple(SIGMA_GRID)
