#!/usr/bin/env python3
"""Run E011 shared-state Fourier-component denoiser swaps."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import pickle
import platform
import re
import socket
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch

from _bootstrap import REPO_ROOT
from spectral_diffusion_playground.e006_transition_swaps import (
    SIGMA_GRID,
    generated_sample_hash,
)
from spectral_diffusion_playground.e010_directional_transfer import nearest_two_cpu
from spectral_diffusion_playground.e011_fourier_component_swap import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DISK_RADIUS,
    EXPECTED_RECORDS,
    EXPERIMENT_ID,
    INTERVENTION_CALLS,
    SAMPLE_SEEDS,
    SMOKE_SEEDS,
    SQUARE_CUTOFF,
    FourierSwapCondition,
    disk_mask,
    effective_mode_count,
    frozen_conditions,
    paired_mask_effect_difference,
    paired_memorization_effect,
    sample_condition,
    select_matched_square_cutoff,
    square_mask,
    validate_condition_registry,
)

DEFAULT_CONFIG = REPO_ROOT / "configs/e011_fourier_component_swap.json"
PER_SAMPLE = "experiment_11_per_sample.csv"
FAILURES = "experiment_11_failures.csv"
FID_METRICS = "experiment_11_fid_metrics.json"


def parse_args() -> argparse.Namespace:
    """Parse the guarded E011 command-line interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=("preflight", "smoke", "full", "summarize", "plot-only"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--edm-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Return a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def dump_json(path: Path, payload: object) -> None:
    """Write deterministic strict JSON."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(
    path: Path, rows: Sequence[Mapping[str, object]], header: Sequence[str]
) -> None:
    """Write rows with a fixed field order."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV records."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def git_commit() -> str:
    """Return the current repository commit."""
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def validate_execution_commit() -> str:
    """Require the exact clean checkout supplied by the launcher."""
    expected = os.environ.get("E011_REPO_COMMIT", "")
    if len(expected) != 40:
        raise RuntimeError("E011_REPO_COMMIT must contain the pinned commit")
    observed = git_commit()
    if observed != expected:
        raise RuntimeError(f"Execution commit mismatch: {observed} != {expected}")
    status = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain=v1"], text=True
    )
    if status:
        raise RuntimeError("E011 requires a clean repository checkout")
    return observed


def condition_from_record(
    try_id: str, mask_type: str, record: Mapping[str, Any]
) -> FourierSwapCondition:
    """Construct one condition from the frozen JSON registry."""
    return FourierSwapCondition(
        try_id=try_id,
        mask_type=mask_type,
        condition_id=str(record["condition_id"]),
        base_model=str(record["base_model"]),
        donor_model=(
            str(record["donor_model"])
            if record.get("donor_model") is not None
            else None
        ),
    )


def validate_static_inputs(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate config, manifests, condition registry, masks, and seeds."""
    config = load_json(config_path)
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("Unexpected E011 configuration")
    observed: dict[str, Any] = {"config_sha256": sha256_file(config_path)}
    manifests: dict[str, dict[str, Any]] = {}
    for name, path_key, hash_key in (
        ("model_pair", "model_pair_manifest", "model_pair_manifest_sha256"),
        ("conditions", "condition_manifest", "condition_manifest_sha256"),
        ("seeds", "seed_manifest", "seed_manifest_sha256"),
        ("masks", "mask_manifest", "mask_manifest_sha256"),
    ):
        path = REPO_ROOT / config["inputs"][path_key]
        digest = sha256_file(path)
        if digest != config["inputs"][hash_key]:
            raise RuntimeError(f"Frozen E011 {name} hash mismatch")
        manifests[name] = load_json(path)
        observed[f"{name}_sha256"] = digest

    registered = []
    for try_record in manifests["conditions"]["tries"]:
        registered.extend(
            condition_from_record(
                str(try_record["try_id"]),
                str(try_record["mask_type"]),
                item,
            )
            for item in try_record["conditions"]
        )
    validate_condition_registry(registered)
    if manifests["conditions"]["intervention_call_indices"] != list(INTERVENTION_CALLS):
        raise RuntimeError("E011 intervention calls changed")
    seeds = manifests["seeds"]
    if (
        seeds["seed_start"] != SAMPLE_SEEDS[0]
        or seeds["seed_end"] != SAMPLE_SEEDS[-1]
        or seeds["expected_record_count"] != EXPECTED_RECORDS
    ):
        raise RuntimeError("E011 seed manifest changed")

    match = select_matched_square_cutoff(32, 32)
    mask_manifest = manifests["masks"]
    if (
        mask_manifest["disk"]["radius"] != DISK_RADIUS
        or mask_manifest["square"]["chosen_cutoff"] != SQUARE_CUTOFF
        or match["disk_effective_mode_count"]
        != mask_manifest["disk"]["effective_full_spectrum_modes"]
        or match["square_effective_mode_count"]
        != mask_manifest["square"]["effective_full_spectrum_modes"]
    ):
        raise RuntimeError("E011 mask manifest changed")
    return config, {"hashes": observed, "manifests": manifests, "mask_match": match}


def validate_external_inputs(
    config: Mapping[str, Any], frozen: Mapping[str, Any], edm_root: Path
) -> dict[str, Any]:
    """Hash checkpoints, dataset inputs, and the official EDM FID evaluator."""
    pair = frozen["manifests"]["model_pair"]
    checkpoints: dict[str, str] = {}
    identities: dict[str, dict[str, int]] = {}
    if str(edm_root) not in sys.path:
        sys.path.insert(0, str(edm_root))
    for role in ("memorizing_model", "generalizing_model"):
        record = pair[role]
        path = Path(record["canonical_path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Checkpoint identity mismatch: {path}")
        if path.stat().st_size != record["size_bytes"]:
            raise RuntimeError(f"Checkpoint size mismatch: {path}")
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        network = payload.get("ema")
        if network is None:
            raise RuntimeError(f"EMA missing from checkpoint: {path}")
        identity = {
            "img_resolution": int(getattr(network, "img_resolution", -1)),
            "img_channels": int(getattr(network, "img_channels", -1)),
            "label_dim": int(getattr(network, "label_dim", -1)),
        }
        if identity != {"img_resolution": 32, "img_channels": 3, "label_dim": 0}:
            raise RuntimeError(f"Incompatible E011 model identity: {identity}")
        checkpoints[record["model_id"]] = str(path)
        identities[record["model_id"]] = identity
        del network, payload

    reference = config["reference"]
    archive = Path(reference["archive"])
    subset = REPO_ROOT / reference["subset_indices"]
    if sha256_file(archive) != reference["archive_sha256"]:
        raise RuntimeError("CIFAR-10 archive hash mismatch")
    if sha256_file(subset) != reference["subset_text_sha256"]:
        raise RuntimeError("Reference subset text hash mismatch")
    indices = np.loadtxt(subset, dtype=np.int64)
    int_hash = hashlib.sha256(
        np.asarray(indices, dtype="<i8").tobytes(order="C")
    ).hexdigest()
    if indices.shape != (1000,) or int_hash != reference["subset_int64_le_sha256"]:
        raise RuntimeError("Reference subset content mismatch")

    fid_path = edm_root / "fid.py"
    if sha256_file(fid_path) != config["fid"]["fid_py_sha256"]:
        raise RuntimeError("Official EDM fid.py hash mismatch")
    edm_commit = subprocess.check_output(
        ["git", "-C", str(edm_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if edm_commit != config["fid"]["source_commit"]:
        raise RuntimeError("EDM source commit mismatch")
    edm_status = subprocess.check_output(
        ["git", "-C", str(edm_root), "status", "--porcelain=v1"], text=True
    )
    return {
        "checkpoints": checkpoints,
        "model_identities": identities,
        "archive_sha256": reference["archive_sha256"],
        "subset_text_sha256": reference["subset_text_sha256"],
        "subset_int64_le_sha256": int_hash,
        "edm_commit": edm_commit,
        "edm_worktree_clean": not bool(edm_status),
        "fid_py_path": str(fid_path),
        "fid_py_sha256": config["fid"]["fid_py_sha256"],
    }


def configure_determinism() -> None:
    """Enable the existing deterministic E010 generation settings."""
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def load_networks(
    frozen: Mapping[str, Any], device: torch.device, edm_root: Path
) -> dict[str, torch.nn.Module]:
    """Load the exact E010 EMA model pair."""
    if str(edm_root) not in sys.path:
        sys.path.insert(0, str(edm_root))
    result: dict[str, torch.nn.Module] = {}
    pair = frozen["manifests"]["model_pair"]
    for role in ("memorizing_model", "generalizing_model"):
        record = pair[role]
        with Path(record["canonical_path"]).open("rb") as handle:
            network = pickle.load(handle)["ema"].to(device).eval()
        result[record["model_id"]] = network
    return result


def seeded_latent(seed: int, device: torch.device) -> torch.Tensor:
    """Create one isolated device-local latent from a frozen integer seed."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(
        (1, 3, 32, 32), generator=generator, device=device, dtype=torch.float32
    )


def load_reference(config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Load the frozen CIFAR-10 1K bank in NCHW float64 ``[-1,1]``."""
    indices = np.loadtxt(
        REPO_ROOT / config["reference"]["subset_indices"], dtype=np.int64
    )
    images = np.empty((1000, 3, 32, 32), dtype=np.float64)
    with zipfile.ZipFile(config["reference"]["archive"]) as archive:
        names = sorted(name for name in archive.namelist() if name.endswith(".png"))
        for position, index in enumerate(indices):
            with archive.open(names[int(index)]) as handle:
                image = np.asarray(
                    PIL.Image.open(handle).convert("RGB"), dtype=np.float64
                )
            images[position] = np.transpose(image / 127.5 - 1.0, (2, 0, 1))
    return images, indices


def per_sample_header() -> list[str]:
    """Return the immutable E011 per-sample schema."""
    return [
        "experiment_id",
        "try_id",
        "mask_type",
        "condition_id",
        "base_model",
        "donor_model",
        "base_checkpoint_sha256",
        "donor_checkpoint_sha256",
        "intervention_call_indices",
        "intervention_sigma_values",
        "swap_target",
        "shared_state",
        "fft_normalization",
        "mask_parameter",
        "effective_mode_count",
        "sample_seed",
        "generated_sample_hash",
        "d1nn",
        "d2nn",
        "d1nn_over_d2nn",
        "d1nn_reference_index",
        "d2nn_reference_index",
        "memorized",
        "sampler",
        "status",
        "error",
    ]


def condition_key(condition: FourierSwapCondition) -> str:
    """Return a filesystem-safe unique condition key."""
    return f"{condition.try_id}__{condition.condition_id}"


def condition_mode_count(condition: FourierSwapCondition) -> int:
    """Return the frozen per-channel effective mode count."""
    if condition.mask_type == "disk":
        return effective_mode_count(disk_mask(32, 32), 32)
    return effective_mode_count(square_mask(32, 32), 32)


def sample_to_uint8(sample: np.ndarray) -> np.ndarray:
    """Convert an unclamped NCHW sample to HWC uint8 for FID only."""
    image = np.transpose(np.asarray(sample, dtype=np.float64), (1, 2, 0))
    return np.clip(image * 127.5 + 128.0, 0.0, 255.0).astype(np.uint8)


def evaluate_condition(
    networks: Mapping[str, torch.nn.Module],
    condition: FourierSwapCondition,
    seeds: Sequence[int],
    reference: np.ndarray,
    reference_indices: np.ndarray,
    frozen: Mapping[str, Any],
    device: torch.device,
    output_dir: Path,
) -> list[dict[str, object]]:
    """Generate, store, and deterministically score one E011 condition."""
    pair = frozen["manifests"]["model_pair"]
    hashes = {
        pair["memorizing_model"]["model_id"]: pair["memorizing_model"]["sha256"],
        pair["generalizing_model"]["model_id"]: pair["generalizing_model"]["sha256"],
    }
    raw_dir = output_dir / "raw_samples" / condition.try_id / condition.condition_id
    fid_dir = output_dir / "fid_images" / condition.try_id / condition.condition_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    fid_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for seed in seeds:
        try:
            latent = seeded_latent(seed, device)
            sample_tensor = sample_condition(networks, condition, latent)
            sample = (
                sample_tensor[0].detach().cpu().numpy().astype(np.float64, copy=False)
            )
            d1, d2, first, second = nearest_two_cpu(sample, reference)
            ratio = d1 / d2 if d2 > 0.0 else float("inf")
            np.save(raw_dir / f"seed_{seed}.npy", sample)
            PIL.Image.fromarray(sample_to_uint8(sample), mode="RGB").save(
                fid_dir / f"seed_{seed}.png"
            )
            row = {
                "experiment_id": EXPERIMENT_ID,
                "try_id": condition.try_id,
                "mask_type": condition.mask_type,
                "condition_id": condition.condition_id,
                "base_model": condition.base_model,
                "donor_model": condition.donor_model or "",
                "base_checkpoint_sha256": hashes[condition.base_model],
                "donor_checkpoint_sha256": (
                    hashes[condition.donor_model] if condition.donor_model else ""
                ),
                "intervention_call_indices": json.dumps(
                    list(INTERVENTION_CALLS), separators=(",", ":")
                ),
                "intervention_sigma_values": json.dumps(
                    [SIGMA_GRID[index] for index in INTERVENTION_CALLS],
                    separators=(",", ":"),
                ),
                "swap_target": "denoiser_prediction",
                "shared_state": "true",
                "fft_normalization": "ortho",
                "mask_parameter": (
                    f"r<={DISK_RADIUS}"
                    if condition.mask_type == "disk"
                    else f"K<={SQUARE_CUTOFF}"
                ),
                "effective_mode_count": condition_mode_count(condition),
                "sample_seed": seed,
                "generated_sample_hash": generated_sample_hash(sample),
                "d1nn": f"{d1:.17g}",
                "d2nn": f"{d2:.17g}",
                "d1nn_over_d2nn": f"{ratio:.17g}",
                "d1nn_reference_index": int(reference_indices[first]),
                "d2nn_reference_index": int(reference_indices[second]),
                "memorized": int(d1 < d2 / 3.0),
                "sampler": "pure_euler_18_call_no_churn",
                "status": "ok",
                "error": "",
            }
        except Exception as error:
            row = {key: "" for key in per_sample_header()}
            row.update(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "try_id": condition.try_id,
                    "mask_type": condition.mask_type,
                    "condition_id": condition.condition_id,
                    "base_model": condition.base_model,
                    "donor_model": condition.donor_model or "",
                    "sample_seed": seed,
                    "memorized": 0,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        rows.append(row)
    return rows


def cached_url_artifact(url: str) -> dict[str, object] | None:
    """Describe a dnnlib-cached URL artifact when present."""
    cache_root = os.environ.get("DNNLIB_CACHE_DIR")
    if not cache_root:
        return None
    prefix = hashlib.md5(url.encode("utf-8")).hexdigest()  # nosec B324
    matches = glob.glob(str(Path(cache_root) / "downloads" / f"{prefix}_*"))
    if len(matches) != 1:
        return None
    path = Path(matches[0])
    return {
        "url": url,
        "cache_path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def directory_artifact(path: Path) -> dict[str, object]:
    """Return a deterministic digest and size summary for an artifact tree."""
    digest = hashlib.sha256()
    file_count = 0
    size_bytes = 0
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file_path.relative_to(path).as_posix()
        file_digest = sha256_file(file_path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
        file_count += 1
        size_bytes += file_path.stat().st_size
    return {
        "path": str(path),
        "file_count": file_count,
        "size_bytes": size_bytes,
        "tree_sha256": digest.hexdigest(),
    }


def run_official_fid(
    output_dir: Path,
    config: Mapping[str, Any],
    edm_root: Path,
    mode: str,
) -> dict[str, Any]:
    """Run the hash-pinned official EDM FID evaluator on generated PNGs."""
    conditions = frozen_conditions()
    selected = conditions if mode == "full" else (conditions[0],)
    image_count = (
        int(config["fid"]["full_image_count"])
        if mode == "full"
        else int(config["fid"]["smoke_image_count"])
    )
    records = []
    log_dir = output_dir / "fid_logs"
    log_dir.mkdir(exist_ok=True)
    for condition in selected:
        key = condition_key(condition)
        image_dir = (
            output_dir / "fid_images" / condition.try_id / condition.condition_id
        )
        pngs = sorted(image_dir.glob("*.png"))
        if len(pngs) != image_count:
            raise RuntimeError(f"FID image count mismatch for {key}: {len(pngs)}")
        command = [
            "torchrun",
            "--standalone",
            "--nproc_per_node=1",
            str(edm_root / "fid.py"),
            "calc",
            f"--images={image_dir}",
            f"--ref={config['fid']['reference_url']}",
            f"--num={image_count}",
            "--seed=0",
            "--batch=64",
        ]
        completed = subprocess.run(
            command,
            cwd=edm_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_path = log_dir / f"{key}.log"
        log_path.write_text(completed.stdout, encoding="utf-8")
        numeric_lines = [
            line.strip()
            for line in completed.stdout.splitlines()
            if re.fullmatch(r"[-+0-9.eE]+", line.strip())
        ]
        fid_value = float(numeric_lines[-1]) if numeric_lines else None
        if completed.returncode != 0:
            raise RuntimeError(f"Official EDM FID failed for {key}: {log_path}")
        if mode == "full" and (fid_value is None or not np.isfinite(fid_value)):
            raise RuntimeError(f"Full FID-1K is nonfinite for {key}")
        records.append(
            {
                "try_id": condition.try_id,
                "condition_id": condition.condition_id,
                "image_count": image_count,
                "fid_1k": fid_value,
                "return_code": completed.returncode,
                "command": command,
                "log_path": str(log_path),
                "log_sha256": sha256_file(log_path),
            }
        )
    payload = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "mode": mode,
        "role": "descriptive_only",
        "fid_py_sha256": config["fid"]["fid_py_sha256"],
        "source_commit": config["fid"]["source_commit"],
        "reference": cached_url_artifact(config["fid"]["reference_url"]),
        "detector": cached_url_artifact(config["fid"]["detector_url"]),
        "records": records,
    }
    dump_json(output_dir / FID_METRICS, payload)
    return payload


def build_manifest(
    mode: str,
    output_dir: Path,
    commit: str,
    config_path: Path,
    frozen: Mapping[str, Any],
    external: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Build pre-execution E011 provenance."""
    seeds = SAMPLE_SEEDS if mode == "full" else SMOKE_SEEDS
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "mode": mode,
        "repository_commit": commit,
        "config_path": str(config_path),
        "config_sha256": frozen["hashes"]["config_sha256"],
        "frozen_input_hashes": frozen["hashes"],
        "external_provenance": external,
        "try_ids": ["try01_radial_r11", "try02_square_matched"],
        "condition_count": len(frozen_conditions()),
        "sample_seed_range": [seeds[0], seeds[-1]],
        "expected_records": len(seeds) * len(frozen_conditions()),
        "frequency_component_swap": True,
        "whole_denoiser_swap": False,
        "swap_target": "denoiser_prediction",
        "shared_state": True,
        "intervention_call_indices": list(INTERVENTION_CALLS),
        "intervention_sigma_values": [SIGMA_GRID[i] for i in INTERVENTION_CALLS],
        "training_executed": False,
        "device": str(device),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "output_dir": str(output_dir),
        "started_at_unix": time.time(),
    }


def expected_keys(seeds: Sequence[int]) -> set[tuple[str, str, int]]:
    """Return the exact registered result keys for a seed cohort."""
    return {
        (condition.try_id, condition.condition_id, seed)
        for condition in frozen_conditions()
        for seed in seeds
    }


def execute_run(
    mode: str,
    output_dir: Path,
    config_path: Path,
    config: Mapping[str, Any],
    frozen: Mapping[str, Any],
    external: Mapping[str, Any],
    commit: str,
    edm_root: Path,
    resume: bool,
) -> None:
    """Run the E011 smoke or full paired evaluation."""
    if output_dir.exists() and not resume:
        raise RuntimeError(f"Refusing existing output directory: {output_dir}")
    if resume and mode != "full":
        raise RuntimeError("Resume is allowed only for the frozen full run")
    output_dir.mkdir(parents=True, exist_ok=resume)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("E011 inference requires CUDA")
    configure_determinism()
    manifest_path = output_dir / "experiment_11_manifest.json"
    if resume:
        manifest = load_json(manifest_path)
        if (
            manifest["repository_commit"] != commit
            or manifest["config_sha256"] != frozen["hashes"]["config_sha256"]
            or manifest["mode"] != "full"
        ):
            raise RuntimeError("Existing E011 output is incompatible with resume")
    else:
        manifest = build_manifest(
            mode, output_dir, commit, config_path, frozen, external, device
        )
        dump_json(manifest_path, manifest)

    seeds = SAMPLE_SEEDS if mode == "full" else SMOKE_SEEDS
    reference, reference_indices = load_reference(config)
    networks = load_networks(frozen, device, edm_root)
    rows = (
        [dict(row) for row in read_rows(output_dir / PER_SAMPLE)]
        if resume and (output_dir / PER_SAMPLE).is_file()
        else []
    )
    successful = {
        (str(row["try_id"]), str(row["condition_id"]), int(row["sample_seed"]))
        for row in rows
        if row["status"] == "ok"
    }
    for condition in frozen_conditions():
        missing = [
            seed
            for seed in seeds
            if (condition.try_id, condition.condition_id, seed) not in successful
        ]
        if not missing:
            continue
        new_rows = evaluate_condition(
            networks,
            condition,
            missing,
            reference,
            reference_indices,
            frozen,
            device,
            output_dir,
        )
        missing_set = set(missing)
        rows = [
            row
            for row in rows
            if not (
                row["try_id"] == condition.try_id
                and row["condition_id"] == condition.condition_id
                and int(row["sample_seed"]) in missing_set
            )
        ]
        rows.extend(new_rows)
        rows.sort(
            key=lambda row: (
                str(row["try_id"]),
                str(row["condition_id"]),
                int(row["sample_seed"]),
            )
        )
        write_csv(output_dir / PER_SAMPLE, rows, per_sample_header())
        failures = [row for row in rows if row["status"] != "ok"]
        write_csv(output_dir / FAILURES, failures, per_sample_header())

    keys = {
        (str(row["try_id"]), str(row["condition_id"]), int(row["sample_seed"]))
        for row in rows
    }
    failures = [row for row in rows if row["status"] != "ok"]
    if keys != expected_keys(seeds) or len(rows) != len(keys) or failures:
        raise RuntimeError("E011 generation keys or failure gate did not pass")

    fid_payload = run_official_fid(output_dir, config, edm_root, mode)
    if mode == "smoke":
        validate_smoke(
            output_dir,
            networks,
            reference,
            reference_indices,
            frozen,
            device,
            rows,
            fid_payload,
        )
    else:
        summarize(output_dir)

    manifest["completed_at_unix"] = time.time()
    manifest["observed_records"] = len(rows)
    manifest["failure_count"] = len(failures)
    manifest["external_artifacts"] = {
        "raw_samples": directory_artifact(output_dir / "raw_samples"),
        "fid_images": directory_artifact(output_dir / "fid_images"),
    }
    manifest["outputs"] = {
        path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in sorted(output_dir.glob("experiment_11_*"))
        if path.name != "experiment_11_manifest.json"
    }
    dump_json(manifest_path, manifest)


def validate_smoke(
    output_dir: Path,
    networks: Mapping[str, torch.nn.Module],
    reference: np.ndarray,
    reference_indices: np.ndarray,
    frozen: Mapping[str, Any],
    device: torch.device,
    rows: Sequence[Mapping[str, object]],
    fid_payload: Mapping[str, Any],
) -> None:
    """Apply exact E011 smoke gates and write a review montage."""
    repeat_conditions = frozen_conditions()
    repeat_matches = {}
    for condition in repeat_conditions:
        repeat = evaluate_condition(
            networks,
            condition,
            (SMOKE_SEEDS[0],),
            reference,
            reference_indices,
            frozen,
            device,
            output_dir / "repeat_validation",
        )[0]
        original = next(
            row
            for row in rows
            if row["try_id"] == condition.try_id
            and row["condition_id"] == condition.condition_id
            and int(row["sample_seed"]) == SMOKE_SEEDS[0]
        )
        repeat_matches[condition_key(condition)] = dict(original) == repeat

    baseline_equal = {}
    for condition_id in ("G", "M"):
        disk = next(
            row
            for row in rows
            if row["try_id"] == "try01_radial_r11"
            and row["condition_id"] == condition_id
            and int(row["sample_seed"]) == SMOKE_SEEDS[0]
        )
        square = next(
            row
            for row in rows
            if row["try_id"] == "try02_square_matched"
            and row["condition_id"] == condition_id
            and int(row["sample_seed"]) == SMOKE_SEEDS[0]
        )
        baseline_equal[condition_id] = (
            disk["generated_sample_hash"] == square["generated_sample_hash"]
        )

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for axis, condition in zip(axes.flat, frozen_conditions()):
        path = (
            output_dir
            / "fid_images"
            / condition.try_id
            / condition.condition_id
            / f"seed_{SMOKE_SEEDS[0]}.png"
        )
        axis.imshow(PIL.Image.open(path))
        axis.set_title(f"{condition.try_id}\n{condition.condition_id}", fontsize=8)
        axis.axis("off")
    fig.suptitle("E011 smoke: shared seed across masks and conditions")
    fig.tight_layout()
    fig.savefig(figure_dir / "experiment_11_smoke_montage.png", dpi=180)
    plt.close(fig)

    status = bool(
        len(rows) == len(frozen_conditions()) * len(SMOKE_SEEDS)
        and all(repeat_matches.values())
        and all(baseline_equal.values())
        and fid_payload["records"]
    )
    validation = {
        "status": "pass" if status else "fail",
        "record_count": len(rows),
        "failure_count": sum(row["status"] != "ok" for row in rows),
        "exact_repeat_matches": repeat_matches,
        "baseline_hashes_equal_across_tries": baseline_equal,
        "fid_invocation_count": len(fid_payload["records"]),
        "disk_mode_count": effective_mode_count(disk_mask(32, 32), 32),
        "square_mode_count": effective_mode_count(square_mask(32, 32), 32),
        "square_cutoff": SQUARE_CUTOFF,
        "montage": str(figure_dir / "experiment_11_smoke_montage.png"),
    }
    dump_json(output_dir / "experiment_11_smoke_validation.json", validation)
    if not status:
        raise RuntimeError("E011 smoke validation failed")


def grouped_vectors(
    rows: Sequence[Mapping[str, str]],
) -> dict[tuple[str, str], np.ndarray]:
    """Return memorization vectors aligned to the frozen seed order."""
    result = {}
    for condition in frozen_conditions():
        selected = sorted(
            (
                row
                for row in rows
                if row["try_id"] == condition.try_id
                and row["condition_id"] == condition.condition_id
            ),
            key=lambda row: int(row["sample_seed"]),
        )
        result[(condition.try_id, condition.condition_id)] = np.asarray(
            [int(row["memorized"]) for row in selected], dtype=np.int8
        )
    return result


def fid_lookup(payload: Mapping[str, Any]) -> dict[tuple[str, str], float]:
    """Map full-run FID records to condition keys."""
    result = {}
    for row in payload["records"]:
        value = row["fid_1k"]
        if value is None or not np.isfinite(float(value)):
            raise RuntimeError("E011 full FID record is missing or nonfinite")
        result[(str(row["try_id"]), str(row["condition_id"]))] = float(value)
    return result


def summarize(output_dir: Path) -> None:
    """Validate and summarize a complete E011 full run without inference."""
    rows = read_rows(output_dir / PER_SAMPLE)
    failures = [row for row in rows if row["status"] != "ok"]
    keys = {
        (row["try_id"], row["condition_id"], int(row["sample_seed"])) for row in rows
    }
    finite = all(
        np.isfinite(float(row["d1nn"]))
        and np.isfinite(float(row["d2nn"]))
        and np.isfinite(float(row["d1nn_over_d2nn"]))
        for row in rows
        if row["status"] == "ok"
    )
    if (
        len(rows) != EXPECTED_RECORDS
        or keys != expected_keys(SAMPLE_SEEDS)
        or failures
        or not finite
    ):
        raise RuntimeError("E011 full record validation failed")

    baseline_fields = (
        "generated_sample_hash",
        "d1nn",
        "d2nn",
        "d1nn_over_d2nn",
        "d1nn_reference_index",
        "d2nn_reference_index",
        "memorized",
    )
    by_key = {
        (row["try_id"], row["condition_id"], int(row["sample_seed"])): row
        for row in rows
    }
    baseline_equal_across_tries = all(
        all(
            by_key[("try01_radial_r11", condition_id, seed)][field]
            == by_key[("try02_square_matched", condition_id, seed)][field]
            for field in baseline_fields
        )
        for condition_id in ("G", "M")
        for seed in SAMPLE_SEEDS
    )
    if not baseline_equal_across_tries:
        raise RuntimeError("E011 baselines differ between mask tries")

    vectors = grouped_vectors(rows)
    fid_payload = load_json(output_dir / FID_METRICS)
    fids = fid_lookup(fid_payload)
    if len(fids) != len(frozen_conditions()):
        raise RuntimeError("E011 requires one FID-1K value per condition")

    summaries = []
    analyses = []
    for condition in frozen_conditions():
        values = vectors[(condition.try_id, condition.condition_id)]
        summaries.append(
            {
                "try_id": condition.try_id,
                "mask_type": condition.mask_type,
                "condition_id": condition.condition_id,
                "memorized_count": int(values.sum()),
                "sample_count": int(values.size),
                "memorization_rate": f"{values.mean():.17g}",
                "fid_1k_descriptive": f"{fids[(condition.try_id, condition.condition_id)]:.17g}",
            }
        )
    for try_id, mask_type in (
        ("try01_radial_r11", "disk"),
        ("try02_square_matched", "square"),
    ):
        for recipient, hybrid, effect_name in (
            ("G", "G_from_M", "Delta_G"),
            ("M", "M_from_G", "Delta_M"),
        ):
            result = paired_memorization_effect(
                vectors[(try_id, recipient)],
                vectors[(try_id, hybrid)],
                seed=BOOTSTRAP_SEED,
                resamples=BOOTSTRAP_RESAMPLES,
            )
            analyses.append(
                {
                    "try_id": try_id,
                    "mask_type": mask_type,
                    "effect_name": effect_name,
                    **result,
                }
            )

    comparisons = []
    for recipient, hybrid, effect_name in (
        ("G", "G_from_M", "Delta_G"),
        ("M", "M_from_G", "Delta_M"),
    ):
        result = paired_mask_effect_difference(
            vectors[("try01_radial_r11", recipient)],
            vectors[("try01_radial_r11", hybrid)],
            vectors[("try02_square_matched", recipient)],
            vectors[("try02_square_matched", hybrid)],
            seed=BOOTSTRAP_SEED,
            resamples=BOOTSTRAP_RESAMPLES,
        )
        comparisons.append({"effect_name": effect_name, **result})

    write_csv(
        output_dir / "experiment_11_condition_summary.csv",
        summaries,
        list(summaries[0]),
    )
    write_csv(
        output_dir / "experiment_11_paired_effects.csv",
        analyses,
        list(analyses[0]),
    )
    write_csv(
        output_dir / "experiment_11_mask_comparison.csv",
        comparisons,
        list(comparisons[0]),
    )
    validation = {
        "status": "pass",
        "expected_records": EXPECTED_RECORDS,
        "observed_records": len(rows),
        "unique_keys": len(keys),
        "failure_count": len(failures),
        "finite_nearest_neighbor_records": finite,
        "condition_counts": {
            f"{try_id}/{condition_id}": int(values.size)
            for (try_id, condition_id), values in vectors.items()
        },
        "seed_range": [SAMPLE_SEEDS[0], SAMPLE_SEEDS[-1]],
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "disk_mode_count": effective_mode_count(disk_mask(32, 32), 32),
        "square_mode_count": effective_mode_count(square_mask(32, 32), 32),
        "fid_1k_role": "descriptive_only",
        "formal_directional_pass_fail_label": None,
        "baseline_records_equal_across_tries": baseline_equal_across_tries,
    }
    dump_json(output_dir / "experiment_11_validation.json", validation)
    generate_figures(output_dir, summaries, analyses, comparisons)


def generate_figures(
    output_dir: Path,
    summaries: Sequence[Mapping[str, object]],
    analyses: Sequence[Mapping[str, object]],
    comparisons: Sequence[Mapping[str, object]],
) -> None:
    """Generate compact E011 review figures from validated summaries."""
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, (try_id, title) in zip(
        axes,
        (
            ("try01_radial_r11", "Try 01: radial disk r<=11"),
            ("try02_square_matched", "Try 02: matched square K=9"),
        ),
    ):
        selected = [row for row in summaries if row["try_id"] == try_id]
        axis.bar(
            [str(row["condition_id"]) for row in selected],
            [float(row["memorization_rate"]) for row in selected],
            color=("#416f8a", "#a45f45", "#6d9fb8", "#c58a73"),
        )
        axis.set(title=title, ylabel="Memorization rate", ylim=(0, 1))
        axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figure_dir / "experiment_11_memorization_rates.png", dpi=200)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.8))
    labels = [f"{row['mask_type']}\n{row['effect_name']}" for row in analyses]
    values = np.asarray([float(row["paired_effect"]) for row in analyses])
    low = np.asarray([float(row["ci95_low"]) for row in analyses])
    high = np.asarray([float(row["ci95_high"]) for row in analyses])
    axis.errorbar(
        labels,
        values,
        yerr=np.vstack((values - low, high - values)),
        fmt="o",
        capsize=5,
        color="#234f62",
    )
    axis.axhline(0, color="black", linewidth=1)
    axis.set(
        ylabel="Hybrid minus recipient memorization rate",
        title="Paired Fourier-component treatment effects (95% bootstrap CI)",
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "experiment_11_paired_effects.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for axis, effect_name in zip(axes, ("Delta_G", "Delta_M")):
        row = next(item for item in comparisons if item["effect_name"] == effect_name)
        axis.bar(
            ["disk", "square"],
            [float(row["disk_effect"]), float(row["square_effect"])],
            color=("#416f8a", "#d09555"),
        )
        axis.axhline(0, color="black", linewidth=1)
        axis.set(title=effect_name, ylabel="Paired effect")
    fig.suptitle("Mode-count-matched mask comparison")
    fig.tight_layout()
    fig.savefig(figure_dir / "experiment_11_mask_comparison.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, (try_id, title) in zip(
        axes,
        (
            ("try01_radial_r11", "Radial disk"),
            ("try02_square_matched", "Matched square"),
        ),
    ):
        selected = [row for row in summaries if row["try_id"] == try_id]
        axis.bar(
            [str(row["condition_id"]) for row in selected],
            [float(row["fid_1k_descriptive"]) for row in selected],
            color="#657a55",
        )
        axis.set(title=title, ylabel="FID-1K (descriptive only)")
        axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figure_dir / "experiment_11_fid_1k.png", dpi=200)
    plt.close(fig)


def main() -> None:
    """Validate frozen inputs and execute only the requested E011 mode."""
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ and args.mode not in (
        "summarize",
        "plot-only",
    ):
        raise RuntimeError("E011 preflight and inference must run through Slurm")
    config_path = args.config.resolve()
    config, frozen = validate_static_inputs(config_path)
    edm_root = (args.edm_root or Path(config["execution"]["edm_root"])).resolve()
    if args.mode in ("summarize", "plot-only"):
        if args.output_dir is None:
            raise RuntimeError("E011 summarization requires --output-dir")
        summarize(args.output_dir.resolve())
        return
    commit = validate_execution_commit()
    external = validate_external_inputs(config, frozen, edm_root)
    if args.mode == "preflight":
        print(
            json.dumps(
                {"status": "pass", "commit": commit, **frozen["hashes"], **external},
                indent=2,
            )
        )
        return
    if args.output_dir is None:
        raise RuntimeError("E011 inference requires --output-dir")
    execute_run(
        args.mode,
        args.output_dir.resolve(),
        config_path,
        config,
        frozen,
        external,
        commit,
        edm_root,
        args.resume,
    )


if __name__ == "__main__":
    main()
