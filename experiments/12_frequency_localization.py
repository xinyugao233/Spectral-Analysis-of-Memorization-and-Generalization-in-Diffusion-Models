#!/usr/bin/env python3
"""Run E012 radial localization of E011's G-from-M transfer effect."""

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
    FourierSwapCondition,
    disk_mask,
    sample_condition as sample_e011_condition,
)
from spectral_diffusion_playground.e012_frequency_localization import (
    BASE_MODEL,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DONOR_MODEL,
    EXPECTED_RECORDS,
    EXPERIMENT_ID,
    SAMPLE_SEEDS,
    SMOKE_SEEDS,
    RadialBandCondition,
    condition_mode_count,
    factorial_contrast_vectors,
    frozen_conditions,
    mask_for_condition,
    overlap_with_full,
    paired_effect,
    radial_band_masks,
    ratio_summary,
    sample_condition,
    summarize_factorial_contrasts,
    validate_band_geometry,
    validate_condition_registry,
)

DEFAULT_CONFIG = REPO_ROOT / "configs/e012_frequency_localization.json"
PER_SAMPLE = "experiment_12_per_sample.csv"
FAILURES = "experiment_12_failures.csv"
FID_METRICS = "experiment_12_fid_metrics.json"


def parse_args() -> argparse.Namespace:
    """Parse the guarded E012 interface."""
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
    """Load one JSON object."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def dump_json(path: Path, payload: object) -> None:
    """Write deterministic strict JSON."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(
    path: Path, rows: Sequence[Mapping[str, object]], header: Sequence[str]
) -> None:
    """Write CSV rows with a fixed field order."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV records."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def git_commit() -> str:
    """Return the active repository commit."""
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def validate_execution_commit() -> str:
    """Require the exact clean checkout supplied by the launcher."""
    expected = os.environ.get("E012_REPO_COMMIT", "")
    if len(expected) != 40:
        raise RuntimeError("E012_REPO_COMMIT must contain the pinned commit")
    observed = git_commit()
    if observed != expected:
        raise RuntimeError(f"Execution commit mismatch: {observed} != {expected}")
    status = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain=v1"], text=True
    )
    if status:
        raise RuntimeError("E012 requires a clean repository checkout")
    return observed


def condition_from_record(record: Mapping[str, Any]) -> RadialBandCondition:
    """Construct one condition from its frozen manifest record."""
    return RadialBandCondition(
        condition_id=str(record["condition_id"]),
        bands=tuple(str(value) for value in record["bands"]),
    )


def validate_static_inputs(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate config, manifests, registry, bands, and seed contract."""
    config = load_json(config_path)
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("Unexpected E012 configuration")
    hashes: dict[str, str] = {"config_sha256": sha256_file(config_path)}
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
            raise RuntimeError(f"Frozen E012 {name} hash mismatch")
        hashes[f"{name}_sha256"] = digest
        manifests[name] = load_json(path)

    registered = tuple(
        condition_from_record(record)
        for record in manifests["conditions"]["conditions"]
    )
    validate_condition_registry(registered)
    if manifests["conditions"]["intervention_call_indices"] != list(range(6, 14)):
        raise RuntimeError("E012 intervention calls changed")
    seeds = manifests["seeds"]
    if (
        seeds["seed_start"] != SAMPLE_SEEDS[0]
        or seeds["seed_end"] != SAMPLE_SEEDS[-1]
        or seeds["expected_full_record_count"] != EXPECTED_RECORDS
    ):
        raise RuntimeError("E012 seed manifest changed")
    counts = validate_band_geometry()
    for name, count in counts.items():
        record = manifests["masks"]["bands"][name]
        if (
            record["stored_rfft2_coefficients"] != count["stored_coefficients"]
            or record["effective_full_spectrum_modes"]
            != count["effective_full_spectrum_modes"]
        ):
            raise RuntimeError(f"E012 mask manifest changed for {name}")
    return config, {"hashes": hashes, "manifests": manifests, "band_counts": counts}


def validate_external_inputs(
    config: Mapping[str, Any], frozen: Mapping[str, Any], edm_root: Path
) -> dict[str, Any]:
    """Hash checkpoints, reference inputs, and the official FID evaluator."""
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
            raise RuntimeError(f"Incompatible E012 model identity: {identity}")
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
    int_hash = hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()
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
    """Enable the deterministic settings validated by E011."""
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def load_networks(
    frozen: Mapping[str, Any], device: torch.device, edm_root: Path
) -> dict[str, torch.nn.Module]:
    """Load the exact E010/E011 EMA model pair."""
    if str(edm_root) not in sys.path:
        sys.path.insert(0, str(edm_root))
    networks: dict[str, torch.nn.Module] = {}
    pair = frozen["manifests"]["model_pair"]
    for role in ("memorizing_model", "generalizing_model"):
        record = pair[role]
        with Path(record["canonical_path"]).open("rb") as handle:
            network = pickle.load(handle)["ema"].to(device).eval()
        networks[record["model_id"]] = network
    return networks


def seeded_latent(seed: int, device: torch.device) -> torch.Tensor:
    """Create one isolated device-local latent from a frozen seed."""
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn((1, 3, 32, 32), generator=generator, device=device)


def load_reference(config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Load the frozen 1K CIFAR-10 bank as NCHW float64 in [-1,1]."""
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
    """Return the immutable E012 per-sample schema."""
    return [
        "experiment_id",
        "condition_id",
        "bands",
        "base_model",
        "donor_model",
        "base_checkpoint_sha256",
        "donor_checkpoint_sha256",
        "intervention_call_indices",
        "intervention_sigma_values",
        "swap_target",
        "shared_state",
        "fft_normalization",
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


def sample_to_uint8(sample: np.ndarray) -> np.ndarray:
    """Convert an unclamped NCHW sample to HWC uint8 for FID only."""
    image = np.transpose(np.asarray(sample, dtype=np.float64), (1, 2, 0))
    return np.clip(image * 127.5 + 128.0, 0.0, 255.0).astype(np.uint8)


def evaluate_condition(
    networks: Mapping[str, torch.nn.Module],
    condition: RadialBandCondition,
    seeds: Sequence[int],
    reference: np.ndarray,
    reference_indices: np.ndarray,
    frozen: Mapping[str, Any],
    device: torch.device,
    output_dir: Path,
) -> list[dict[str, object]]:
    """Generate, store, and score one condition."""
    pair = frozen["manifests"]["model_pair"]
    hashes = {
        pair["memorizing_model"]["model_id"]: pair["memorizing_model"]["sha256"],
        pair["generalizing_model"]["model_id"]: pair["generalizing_model"]["sha256"],
    }
    raw_dir = output_dir / "raw_samples" / condition.condition_id
    fid_dir = output_dir / "fid_images" / condition.condition_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    fid_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for seed in seeds:
        try:
            sample_tensor = sample_condition(
                networks, condition, seeded_latent(seed, device)
            )
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
                "condition_id": condition.condition_id,
                "bands": json.dumps(condition.bands, separators=(",", ":")),
                "base_model": BASE_MODEL,
                "donor_model": DONOR_MODEL if condition.is_hybrid else "",
                "base_checkpoint_sha256": hashes[BASE_MODEL],
                "donor_checkpoint_sha256": (
                    hashes[DONOR_MODEL] if condition.is_hybrid else ""
                ),
                "intervention_call_indices": json.dumps(
                    list(range(6, 14)), separators=(",", ":")
                ),
                "intervention_sigma_values": json.dumps(
                    [SIGMA_GRID[i] for i in range(6, 14)], separators=(",", ":")
                ),
                "swap_target": "denoiser_prediction",
                "shared_state": "true",
                "fft_normalization": "ortho",
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
                    "condition_id": condition.condition_id,
                    "bands": json.dumps(condition.bands, separators=(",", ":")),
                    "base_model": BASE_MODEL,
                    "donor_model": DONOR_MODEL if condition.is_hybrid else "",
                    "sample_seed": seed,
                    "memorized": 0,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        rows.append(row)
    return rows


def cached_url_artifact(url: str) -> dict[str, object] | None:
    """Describe one dnnlib-cached URL artifact when present."""
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
    """Return deterministic digest and size metadata for an artifact tree."""
    digest = hashlib.sha256()
    count = size = 0
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(sha256_file(item).encode())
        digest.update(b"\n")
        count += 1
        size += item.stat().st_size
    return {
        "path": str(path),
        "file_count": count,
        "size_bytes": size,
        "tree_sha256": digest.hexdigest(),
    }


def run_official_fid(
    output_dir: Path, config: Mapping[str, Any], edm_root: Path, mode: str
) -> dict[str, Any]:
    """Run the hash-pinned official EDM FID evaluator."""
    selected = frozen_conditions() if mode == "full" else (frozen_conditions()[0],)
    image_count = int(
        config["fid"]["full_image_count" if mode == "full" else "smoke_image_count"]
    )
    records = []
    log_dir = output_dir / "fid_logs"
    log_dir.mkdir(exist_ok=True)
    for condition in selected:
        image_dir = output_dir / "fid_images" / condition.condition_id
        if len(list(image_dir.glob("*.png"))) != image_count:
            raise RuntimeError(f"FID image count mismatch: {condition.condition_id}")
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
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
        log_path = log_dir / f"{condition.condition_id}.log"
        log_path.write_text(completed.stdout, encoding="utf-8")
        numbers = [
            line.strip()
            for line in completed.stdout.splitlines()
            if re.fullmatch(r"[-+0-9.eE]+", line.strip())
        ]
        fid_value = float(numbers[-1]) if numbers else None
        if completed.returncode != 0:
            raise RuntimeError(f"Official EDM FID failed: {log_path}")
        if mode == "full" and (fid_value is None or not np.isfinite(fid_value)):
            raise RuntimeError(f"Nonfinite FID for {condition.condition_id}")
        records.append(
            {
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
    """Build pre-execution provenance."""
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
        "condition_ids": [item.condition_id for item in frozen_conditions()],
        "condition_count": len(frozen_conditions()),
        "sample_seed_range": [seeds[0], seeds[-1]],
        "expected_records": len(seeds) * len(frozen_conditions()),
        "frequency_component_swap": True,
        "whole_denoiser_swap": False,
        "swap_target": "denoiser_prediction",
        "shared_state": True,
        "intervention_call_indices": list(range(6, 14)),
        "intervention_sigma_values": [SIGMA_GRID[i] for i in range(6, 14)],
        "training_executed": False,
        "device": str(device),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_file": torch.__file__,
        "numpy": np.__version__,
        "output_dir": str(output_dir),
        "started_at_unix": time.time(),
    }


def expected_keys(seeds: Sequence[int]) -> set[tuple[str, int]]:
    """Return exact expected condition/seed keys."""
    return {
        (condition.condition_id, seed)
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
    """Run the E012 smoke or explicitly authorized full evaluation."""
    if output_dir.exists() and not resume:
        raise RuntimeError(f"Refusing existing output directory: {output_dir}")
    if resume and mode != "full":
        raise RuntimeError("Resume is allowed only for the full run")
    output_dir.mkdir(parents=True, exist_ok=resume)
    if not torch.cuda.is_available():
        raise RuntimeError("E012 inference requires CUDA")
    device = torch.device("cuda")
    configure_determinism()
    manifest_path = output_dir / "experiment_12_manifest.json"
    if resume:
        manifest = load_json(manifest_path)
        if (
            manifest["repository_commit"] != commit
            or manifest["config_sha256"] != frozen["hashes"]["config_sha256"]
            or manifest["mode"] != "full"
        ):
            raise RuntimeError("Existing E012 output is incompatible with resume")
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
        (str(row["condition_id"]), int(row["sample_seed"]))
        for row in rows
        if row["status"] == "ok"
    }
    for condition in frozen_conditions():
        missing = [
            seed for seed in seeds if (condition.condition_id, seed) not in successful
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
                row["condition_id"] == condition.condition_id
                and int(row["sample_seed"]) in missing_set
            )
        ]
        rows.extend(new_rows)
        rows.sort(key=lambda row: (str(row["condition_id"]), int(row["sample_seed"])))
        write_csv(output_dir / PER_SAMPLE, rows, per_sample_header())
        write_csv(
            output_dir / FAILURES,
            [row for row in rows if row["status"] != "ok"],
            per_sample_header(),
        )
    keys = {(str(row["condition_id"]), int(row["sample_seed"])) for row in rows}
    failures = [row for row in rows if row["status"] != "ok"]
    if keys != expected_keys(seeds) or len(rows) != len(keys) or failures:
        raise RuntimeError("E012 generation keys or failure gate did not pass")

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
        name: directory_artifact(output_dir / name)
        for name in ("raw_samples", "fid_images")
    }
    manifest["outputs"] = {
        path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in sorted(output_dir.glob("experiment_12_*"))
        if path.name != manifest_path.name
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
    """Apply exact E012 smoke gates and write the review montage."""
    seed = SMOKE_SEEDS[0]
    row_lookup = {
        (str(row["condition_id"]), int(row["sample_seed"])): row for row in rows
    }
    repeats: dict[str, bool] = {}
    for condition in frozen_conditions():
        repeated = evaluate_condition(
            networks,
            condition,
            (seed,),
            reference,
            reference_indices,
            frozen,
            device,
            output_dir / "repeat_validation",
        )[0]
        repeats[condition.condition_id] = (
            dict(row_lookup[(condition.condition_id, seed)]) == repeated
        )

    latent = seeded_latent(seed, device)
    e011_g = FourierSwapCondition("try01_radial_r11", "disk", "G", BASE_MODEL, None)
    e011_full = FourierSwapCondition(
        "try01_radial_r11", "disk", "G_from_M", BASE_MODEL, DONOR_MODEL
    )
    g_hash = generated_sample_hash(
        sample_e011_condition(networks, e011_g, latent.clone())[0]
        .detach()
        .cpu()
        .numpy()
    )
    full_hash = generated_sample_hash(
        sample_e011_condition(networks, e011_full, latent.clone())[0]
        .detach()
        .cpu()
        .numpy()
    )
    equivalence = {
        "000_matches_e011_G": row_lookup[("000_none", seed)]["generated_sample_hash"]
        == g_hash,
        "111_matches_e011_radial_G_from_M": row_lookup[("111_B1_B2_B3", seed)][
            "generated_sample_hash"
        ]
        == full_hash,
        "111_mask_equals_e011_r11": bool(
            torch.equal(
                mask_for_condition(frozen_conditions()[7], 32, 32)[0, 0],
                disk_mask(32, 32, 11),
            )
        ),
    }
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for axis, condition in zip(axes.flat, frozen_conditions()):
        path = output_dir / "fid_images" / condition.condition_id / f"seed_{seed}.png"
        axis.imshow(PIL.Image.open(path))
        axis.set_title(condition.condition_id)
        axis.axis("off")
    fig.suptitle("E012 smoke: one shared seed across all radial-band conditions")
    fig.tight_layout()
    montage = figure_dir / "experiment_12_smoke_montage.png"
    fig.savefig(montage, dpi=180)
    plt.close(fig)
    finite = all(
        np.isfinite(float(row[field]))
        for row in rows
        for field in ("d1nn", "d2nn", "d1nn_over_d2nn")
    )
    status = (
        len(rows) == 18
        and not any(row["status"] != "ok" for row in rows)
        and finite
        and all(repeats.values())
        and all(equivalence.values())
        and bool(fid_payload["records"])
    )
    validation = {
        "status": "pass" if status else "fail",
        "record_count": len(rows),
        "failure_count": sum(row["status"] != "ok" for row in rows),
        "finite_outputs": finite,
        "exact_repeat_matches": repeats,
        "e011_equivalence": equivalence,
        "band_counts": validate_band_geometry(),
        "fid_invocation_count": len(fid_payload["records"]),
        "montage": str(montage),
    }
    dump_json(output_dir / "experiment_12_smoke_validation.json", validation)
    if not status:
        raise RuntimeError("E012 smoke validation failed")


def grouped(
    rows: Sequence[Mapping[str, str]], field: str, dtype: Any
) -> dict[str, np.ndarray]:
    """Return per-condition vectors aligned to frozen seed order."""
    result = {}
    for condition in frozen_conditions():
        selected = sorted(
            (row for row in rows if row["condition_id"] == condition.condition_id),
            key=lambda row: int(row["sample_seed"]),
        )
        result[condition.condition_id] = np.asarray(
            [row[field] for row in selected], dtype=dtype
        )
    return result


def summarize(output_dir: Path) -> None:
    """Validate and summarize a complete full run without inference."""
    rows = read_rows(output_dir / PER_SAMPLE)
    keys = {(row["condition_id"], int(row["sample_seed"])) for row in rows}
    failures = [row for row in rows if row["status"] != "ok"]
    finite = all(
        np.isfinite(float(row[field]))
        for row in rows
        for field in ("d1nn", "d2nn", "d1nn_over_d2nn")
    )
    if (
        len(rows) != EXPECTED_RECORDS
        or keys != expected_keys(SAMPLE_SEEDS)
        or failures
        or not finite
    ):
        raise RuntimeError("E012 full record validation failed")
    mem = grouped(rows, "memorized", np.int8)
    ratios = grouped(rows, "d1nn_over_d2nn", np.float64)
    fid_payload = load_json(output_dir / FID_METRICS)
    fids = {
        str(row["condition_id"]): float(row["fid_1k"]) for row in fid_payload["records"]
    }
    if set(fids) != {item.condition_id for item in frozen_conditions()} or not all(
        np.isfinite(value) for value in fids.values()
    ):
        raise RuntimeError("E012 requires one finite FID-1K per condition")
    baseline = mem["000_none"]
    summaries: list[dict[str, object]] = []
    for condition in frozen_conditions():
        effect = paired_effect(baseline, mem[condition.condition_id])
        summaries.append(
            {
                "condition_id": condition.condition_id,
                "bands": "+".join(condition.bands) or "none",
                "effective_modes": condition_mode_count(condition),
                "memorized_count": int(mem[condition.condition_id].sum()),
                "sample_count": 1000,
                "memorization_rate": float(mem[condition.condition_id].mean()),
                "delta_vs_G": effect["estimate"],
                "ci95_low": effect["ci95_low"],
                "ci95_high": effect["ci95_high"],
                "fid_1k": fids[condition.condition_id],
                "delta_fid_vs_G": fids[condition.condition_id] - fids["000_none"],
            }
        )
    contrasts = summarize_factorial_contrasts(mem)
    full = mem["111_B1_B2_B3"]
    overlap_rows = []
    for condition in frozen_conditions()[1:7]:
        overlap_rows.append(
            {
                "condition_id": condition.condition_id,
                **overlap_with_full(mem[condition.condition_id], full, SAMPLE_SEEDS),
            }
        )
    nn_rows = [
        {
            "condition_id": condition.condition_id,
            **ratio_summary(ratios[condition.condition_id]),
        }
        for condition in frozen_conditions()
    ]
    write_csv(
        output_dir / "experiment_12_condition_summary.csv",
        summaries,
        list(summaries[0]),
    )
    write_csv(
        output_dir / "experiment_12_factorial_contrasts.csv",
        contrasts,
        list(contrasts[0]),
    )
    write_csv(
        output_dir / "experiment_12_full_mask_overlap.csv",
        overlap_rows,
        list(overlap_rows[0]),
    )
    write_csv(
        output_dir / "experiment_12_nn_diagnostics.csv", nn_rows, list(nn_rows[0])
    )
    validation = {
        "status": "pass",
        "expected_records": EXPECTED_RECORDS,
        "observed_records": len(rows),
        "unique_keys": len(keys),
        "failure_count": 0,
        "finite_nearest_neighbor_records": finite,
        "band_counts": validate_band_geometry(),
        "seed_range": [SAMPLE_SEEDS[0], SAMPLE_SEEDS[-1]],
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "e011_reference": {
            "condition": "111_B1_B2_B3",
            "memorization_rate": 0.761,
            "fid_1k": 38.019,
        },
        "fid_1k_role": "descriptive_only",
        "causal_frequency_claim": False,
    }
    dump_json(output_dir / "experiment_12_validation.json", validation)
    generate_figures(output_dir, summaries, contrasts, overlap_rows)
    write_report(output_dir, summaries, contrasts, overlap_rows, nn_rows)


def generate_figures(
    output_dir: Path,
    summaries: Sequence[Mapping[str, object]],
    contrasts: Sequence[Mapping[str, object]],
    overlap_rows: Sequence[Mapping[str, object]],
) -> None:
    """Generate the five preregistered review figures."""
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    labels = [str(row["condition_id"]) for row in summaries]
    fig, axis = plt.subplots(figsize=(11, 5))
    axis.bar(
        labels, [float(row["memorization_rate"]) for row in summaries], color="#3d7188"
    )
    axis.set(
        ylabel="Memorization rate", ylim=(0, 1), title="E012 radial-band conditions"
    )
    axis.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(figure_dir / "experiment_12_memorization_rates.png", dpi=200)
    plt.close(fig)
    values = np.asarray([float(row["estimate"]) for row in contrasts])
    low = np.asarray([float(row["ci95_low"]) for row in contrasts])
    high = np.asarray([float(row["ci95_high"]) for row in contrasts])
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.errorbar(
        [str(row["contrast"]) for row in contrasts],
        values,
        yerr=np.vstack((values - low, high - values)),
        fmt="o",
        capsize=5,
        color="#a04e35",
    )
    axis.axhline(0, color="black", lw=1)
    axis.set(
        ylabel="Paired factorial contrast",
        title="Main and interaction contrasts (95% bootstrap CI)",
    )
    axis.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(figure_dir / "experiment_12_factorial_contrasts.png", dpi=200)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(
        [str(row["condition_id"]) for row in overlap_rows],
        [float(row["recovery_of_111"]) for row in overlap_rows],
        color="#708a55",
    )
    axis.set(
        ylabel="Fraction of 111 memorized seeds recovered",
        ylim=(0, 1),
        title="Memorized-seed overlap with full r<=11 mask",
    )
    axis.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(figure_dir / "experiment_12_full_mask_overlap.png", dpi=200)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(11, 5))
    axis.bar(labels, [float(row["fid_1k"]) for row in summaries], color="#c48a52")
    axis.set(ylabel="FID-1K (descriptive only)", title="E012 descriptive image quality")
    axis.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(figure_dir / "experiment_12_fid_1k.png", dpi=200)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(10, 2.8))
    colors = {"B1": "#315d77", "B2": "#4e8d76", "B3": "#d0944b", "B4": "#b85c4c"}
    intervals = {"B1": (0, 6), "B2": (6, 9), "B3": (9, 11), "B4": (11, 13)}
    for name, (left, right) in intervals.items():
        axis.barh(
            [0],
            [right - left],
            left=[left],
            color=colors[name],
            edgecolor="white",
            label=name,
        )
        axis.text(
            (left + right) / 2,
            0,
            name,
            ha="center",
            va="center",
            color="white",
            weight="bold",
        )
    axis.set(
        xlim=(0, 13),
        yticks=[],
        xlabel="Radial frequency r",
        title="Frozen E012 radial bands",
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "experiment_12_radial_band_schematic.png", dpi=200)
    plt.close(fig)


def write_report(
    output_dir: Path,
    summaries: Sequence[Mapping[str, object]],
    contrasts: Sequence[Mapping[str, object]],
    overlap_rows: Sequence[Mapping[str, object]],
    nn_rows: Sequence[Mapping[str, object]],
) -> None:
    """Write a compact result report without strengthening the evidence."""
    full = next(row for row in summaries if row["condition_id"] == "111_B1_B2_B3")
    lines = [
        "# E012 Radial Frequency Localization Results",
        "",
        "Generated from the frozen E012 full run. Interpretation must remain frequency-localization within calls `6..13`; no claim that middle frequencies cause memorization is supported.",
        "",
        "## E011 Positive-Control Comparison",
        "",
        "E011 radial `G <- M, r<=11` recorded `761/1000` memorized samples and FID-1K `38.0190`. "
        f"The exactly corresponding E012 `111` condition recorded `{full['memorized_count']}/1000` and FID-1K `{float(full['fid_1k']):.4f}`. "
        "Any discrepancy must be investigated rather than tuned away.",
        "",
        "## Condition Results",
        "",
        "| Condition | Bands | Modes | Mem count | Mem rate | Delta vs G | 95% CI | FID-1K |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['condition_id']} | {row['bands']} | {row['effective_modes']} | {row['memorized_count']} | {100*float(row['memorization_rate']):.2f}% | {100*float(row['delta_vs_G']):.2f} pp | [{100*float(row['ci95_low']):.2f}, {100*float(row['ci95_high']):.2f}] pp | {float(row['fid_1k']):.4f} |"
        )
    lines += [
        "",
        "## Factorial Contrasts",
        "",
        "| Contrast | Estimate | 95% CI |",
        "| --- | ---: | --- |",
    ]
    for row in contrasts:
        lines.append(
            f"| {row['contrast']} | {100*float(row['estimate']):.2f} pp | [{100*float(row['ci95_low']):.2f}, {100*float(row['ci95_high']):.2f}] pp |"
        )
    lines += [
        "",
        "## Full-Mask Overlap",
        "",
        "| Condition | Memorized | Intersection | Recovery of 111 | Jaccard |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in overlap_rows:
        lines.append(
            f"| {row['condition_id']} | {row['memorized_count']} | {row['intersection_with_111']} | {float(row['recovery_of_111']):.4f} | {float(row['jaccard']):.4f} |"
        )
    lines += [
        "",
        "## NN Ratio Diagnostics",
        "",
        "| Condition | Mean | Median | Q25 | Q75 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in nn_rows:
        lines.append(
            f"| {row['condition_id']} | {float(row['mean']):.6f} | {float(row['median']):.6f} | {float(row['q25']):.6f} | {float(row['q75']):.6f} |"
        )
    (output_dir / "experiment_12_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Validate frozen inputs and execute only the requested E012 mode."""
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ and args.mode not in ("summarize", "plot-only"):
        raise RuntimeError("E012 preflight and inference must run through Slurm")
    config_path = args.config.resolve()
    config, frozen = validate_static_inputs(config_path)
    edm_root = (args.edm_root or Path(config["execution"]["edm_root"])).resolve()
    if args.mode in ("summarize", "plot-only"):
        if args.output_dir is None:
            raise RuntimeError("E012 summarization requires --output-dir")
        summarize(args.output_dir.resolve())
        return
    commit = validate_execution_commit()
    external = validate_external_inputs(config, frozen, edm_root)
    if args.mode == "preflight":
        print(
            json.dumps(
                {
                    "status": "pass",
                    "commit": commit,
                    "config_sha256": frozen["hashes"]["config_sha256"],
                    "band_counts": frozen["band_counts"],
                    "external": external,
                },
                indent=2,
            )
        )
        return
    if args.output_dir is None:
        raise RuntimeError("E012 execution requires --output-dir")
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
