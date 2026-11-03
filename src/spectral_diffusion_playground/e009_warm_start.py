# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.
# Derived from NVlabs EDM training_loop.py under CC BY-NC-SA 4.0.

"""Deterministic warm-start training support for E009 Stage B.

The training loop is a narrowly scoped derivative of NVlabs EDM's training
loop at commit 008a4e5. It adds separate EMA loading and an extended state
schema; it does not alter the EDM loss, optimizer, network, or EMA update.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch


STATE_SCHEMA_VERSION = 2


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_digest(digest: Any, value: Any) -> None:
    """Update a digest with a deterministic representation of nested state."""
    if isinstance(value, torch.nn.Module):
        _update_digest(digest, value.state_dict())
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor")
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray")
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    elif isinstance(value, Mapping):
        digest.update(b"mapping")
        for key in sorted(value, key=lambda item: repr(item)):
            _update_digest(digest, key)
            _update_digest(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode())
        for item in value:
            _update_digest(digest, item)
    elif isinstance(value, (str, int, float, bool, type(None))):
        digest.update(type(value).__name__.encode())
        digest.update(repr(value).encode())
    else:
        raise TypeError(f"Unsupported digest value: {type(value)!r}")


def state_digest(value: Any) -> str:
    """Return a deterministic SHA-256 digest for nested tensor state."""
    digest = hashlib.sha256()
    _update_digest(digest, value)
    return digest.hexdigest()


def derive_rng_seeds(seed: int, rank: int, world_size: int) -> dict[str, int]:
    """Derive the Stage B NumPy and Torch seeds without changing global state."""
    numpy_seed = (seed * world_size + rank) % (1 << 31)
    random_state = np.random.RandomState(numpy_seed)
    torch_seed = int(random_state.randint(1 << 31))
    return {
        "stage_b_seed": seed,
        "numpy_seed": numpy_seed,
        "torch_seed": torch_seed,
        "sampler_seed": seed,
        "dataloader_generator_seed": seed,
    }


def initialize_rngs(
    seed: int, rank: int, world_size: int
) -> tuple[torch.Generator, dict[str, int]]:
    """Initialize all accessible Stage B RNG sources from the frozen seed."""
    seeds = derive_rng_seeds(seed, rank, world_size)
    np.random.seed(seeds["numpy_seed"])
    observed_torch_seed = int(np.random.randint(1 << 31))
    if observed_torch_seed != seeds["torch_seed"]:
        raise RuntimeError("Deterministic Torch seed derivation failed")
    torch.manual_seed(observed_torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(observed_torch_seed)
    generator = torch.Generator()
    generator.manual_seed(seeds["dataloader_generator_seed"])
    return generator, seeds


def seeded_rng_digest(generator: torch.Generator) -> str:
    """Fingerprint global and DataLoader RNG states immediately after seeding."""
    return state_digest(
        {
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
            "dataloader_generator": generator.get_state(),
        }
    )


class StatefulInfiniteSampler(torch.utils.data.Sampler[int]):
    """Match EDM's infinite sampler while exposing exact serializable state."""

    def __init__(
        self,
        dataset: Sequence[Any],
        *,
        rank: int = 0,
        num_replicas: int = 1,
        seed: int = 0,
        window_size: float = 0.5,
    ) -> None:
        if not len(dataset):
            raise ValueError("Sampler dataset must be nonempty")
        if not 0 <= rank < num_replicas:
            raise ValueError("Sampler rank is outside the replica range")
        if not 0 <= window_size <= 1:
            raise ValueError("Sampler window_size must be in [0,1]")
        super().__init__()
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.seed = seed
        self.window_size = window_size
        self.order = np.arange(len(dataset), dtype=np.int64)
        self.random_state = np.random.RandomState(seed)
        self.random_state.shuffle(self.order)
        self.window = int(np.rint(self.order.size * window_size))
        self.index = 0

    def __iter__(self) -> Iterator[int]:
        while True:
            position = self.index % self.order.size
            value = int(self.order[position])
            emit = self.index % self.num_replicas == self.rank
            if self.window >= 2:
                swap = (position - self.random_state.randint(self.window)) % len(
                    self.order
                )
                self.order[position], self.order[swap] = (
                    self.order[swap],
                    self.order[position],
                )
            self.index += 1
            if emit:
                yield value

    def state_dict(self) -> dict[str, Any]:
        """Return the exact sampler ordering, RNG state, and cursor."""
        return {
            "rank": self.rank,
            "num_replicas": self.num_replicas,
            "seed": self.seed,
            "window_size": self.window_size,
            "window": self.window,
            "index": self.index,
            "order": self.order.copy(),
            "random_state": self.random_state.get_state(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a state produced by :meth:`state_dict`."""
        for key in ("rank", "num_replicas", "seed", "window"):
            if int(state[key]) != int(getattr(self, key)):
                raise ValueError(f"Sampler {key} mismatch")
        if float(state["window_size"]) != self.window_size:
            raise ValueError("Sampler window_size mismatch")
        order = np.asarray(state["order"], dtype=np.int64)
        if order.shape != self.order.shape:
            raise ValueError("Sampler order shape mismatch")
        self.order = order.copy()
        self.index = int(state["index"])
        self.random_state.set_state(state["random_state"])


def capture_rng_state(
    sampler: StatefulInfiniteSampler, generator: torch.Generator
) -> dict[str, Any]:
    """Capture every RNG state accessible in the zero-worker Stage B loader."""
    return {
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "sampler": sampler.state_dict(),
        "dataloader_generator": generator.get_state(),
        "unavailable": ["dataloader_worker_rng:not_applicable_num_workers_0"],
    }


def restore_rng_state(
    state: Mapping[str, Any],
    sampler: StatefulInfiniteSampler,
    generator: torch.Generator,
) -> None:
    """Restore every serialized Stage B RNG source and verify exact equality."""
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_states = list(state["torch_cuda_all"])
    if cuda_states:
        if not torch.cuda.is_available():
            raise ValueError("Serialized CUDA RNG state requires CUDA")
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError("Serialized CUDA RNG device count changed")
        torch.cuda.set_rng_state_all(cuda_states)
    elif torch.cuda.is_available():
        raise ValueError("CUDA is available but serialized CUDA RNG state is empty")
    sampler.load_state_dict(state["sampler"])
    generator.set_state(state["dataloader_generator"])
    restored = capture_rng_state(sampler, generator)
    if state_digest(restored) != state_digest(state):
        raise RuntimeError("Restored Stage B RNG state differs from serialized state")


def validate_extended_state(state: Mapping[str, Any]) -> None:
    """Require all fields needed for an exact continuation within Stage B."""
    required = {
        "state_schema_version",
        "net",
        "optimizer_state",
        "ema",
        "progress",
        "rng_state",
        "warm_start",
    }
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"Extended state is missing fields: {missing}")
    if int(state["state_schema_version"]) != STATE_SCHEMA_VERSION:
        raise ValueError("Extended state schema version mismatch")
    for field in (
        "numpy",
        "torch_cpu",
        "torch_cuda_all",
        "sampler",
        "dataloader_generator",
        "unavailable",
    ):
        if field not in state["rng_state"]:
            raise ValueError(f"Extended state is missing RNG field: {field}")


def warm_start_training_loop(
    *,
    run_dir: str,
    dataset_kwargs: Mapping[str, Any],
    data_loader_kwargs: Mapping[str, Any],
    network_kwargs: Mapping[str, Any],
    loss_kwargs: Mapping[str, Any],
    optimizer_kwargs: Mapping[str, Any],
    parent_training_state: Path,
    parent_training_state_sha256: str,
    parent_ema_snapshot: Path,
    parent_ema_snapshot_sha256: str,
    resume_kind: str,
    execution_commit: str,
    start_kimg: int,
    seed: int,
    batch_size: int,
    batch_gpu: int,
    total_kimg: int,
    ema_halflife_kimg: float,
    kimg_per_tick: int,
    snapshot_ticks: int,
    state_dump_ticks: int,
    cudnn_benchmark: bool = True,
    augment_kwargs: Mapping[str, Any] | None = None,
    ema_rampup_ratio: float | None = 0.05,
    lr_rampup_kimg: float = 10000,
    loss_scaling: float = 1,
    device: torch.device = torch.device("cuda"),
) -> dict[str, Any]:
    """Train one deterministic warm-start lineage from separate parent states."""
    import dnnlib
    from torch_utils import distributed as dist
    from torch_utils import misc, training_stats

    start_time = time.time()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    loader_generator, derived_seeds = initialize_rngs(seed, rank, world_size)
    first_rng_digest = seeded_rng_digest(loader_generator)
    loader_generator, repeated_seeds = initialize_rngs(seed, rank, world_size)
    repeated_rng_digest = seeded_rng_digest(loader_generator)
    if repeated_seeds != derived_seeds or repeated_rng_digest != first_rng_digest:
        raise RuntimeError("Seed-1 RNG initialization is not reproducible")
    if derive_rng_seeds(seed, rank, world_size) != derived_seeds:
        raise RuntimeError("Seed-1 initialization is not reproducible")
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    if sha256_file(parent_training_state) != parent_training_state_sha256:
        raise ValueError("Parent training-state hash changed")
    if sha256_file(parent_ema_snapshot) != parent_ema_snapshot_sha256:
        raise ValueError("Parent EMA-snapshot hash changed")
    if total_kimg <= start_kimg:
        raise ValueError("Warm-start target must exceed its starting exposure")

    batch_gpu_total = batch_size // world_size
    batch_gpu = min(batch_gpu, batch_gpu_total)
    accumulation_rounds = batch_gpu_total // batch_gpu
    if batch_size != batch_gpu * accumulation_rounds * world_size:
        raise ValueError("Batch configuration is not divisible across ranks")

    dist.print0("Loading dataset...")
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    sampler = StatefulInfiniteSampler(
        dataset_obj, rank=rank, num_replicas=world_size, seed=seed
    )
    loader_options = dict(data_loader_kwargs)
    workers = int(loader_options.get("num_workers", 0))
    if workers != 0:
        raise ValueError("Stage B requires num_workers=0 for exact sampler state")
    loader_options.pop("prefetch_factor", None)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset_obj,
        sampler=sampler,
        batch_size=batch_gpu,
        generator=loader_generator,
        **loader_options,
    )

    dist.print0("Constructing network...")
    interface_kwargs = dict(
        img_resolution=dataset_obj.resolution,
        img_channels=dataset_obj.num_channels,
        label_dim=dataset_obj.label_dim,
    )
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)
    net.train().requires_grad_(True).to(device)
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    optimizer = dnnlib.util.construct_class_by_name(
        params=net.parameters(), **optimizer_kwargs
    )
    augment_pipe = (
        dnnlib.util.construct_class_by_name(**augment_kwargs)
        if augment_kwargs is not None
        else None
    )
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device])
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    dist.print0(f'Loading parent state from "{parent_training_state}"...')
    parent_state = torch.load(
        parent_training_state, map_location=torch.device("cpu"), weights_only=False
    )
    with parent_ema_snapshot.open("rb") as handle:
        snapshot = pickle.load(handle)
    if not isinstance(snapshot, dict) or "ema" not in snapshot:
        raise ValueError("Parent snapshot does not contain EMA")
    if resume_kind == "stage_a_warm_start":
        if set(parent_state) != {"net", "optimizer_state"}:
            raise ValueError("Parent Stage A state has an unexpected schema")
        parent_ema = snapshot["ema"]
        lineage_tick = 0
        initialization_filename = "warm_start_initialization.json"
        stats_filename = "stats.jsonl"
        exact_stage_b_resume = False
    elif resume_kind == "extended_stage_b_state":
        validate_extended_state(parent_state)
        progress = parent_state["progress"]
        if int(progress["cur_kimg"]) != start_kimg:
            raise ValueError("Extended-state progress does not match start_kimg")
        if int(progress["cur_nimg"]) != start_kimg * 1000:
            raise ValueError("Extended-state image counter is inconsistent")
        if int(parent_state["warm_start"]["seed"]) != seed:
            raise ValueError("Extended-state seed does not match Stage B seed")
        if state_digest(parent_state["ema"]) != state_digest(snapshot["ema"]):
            raise ValueError("Extended-state EMA differs from parent snapshot")
        parent_ema = parent_state["ema"]
        lineage_tick = int(progress["lineage_tick"])
        restore_rng_state(parent_state["rng_state"], sampler, loader_generator)
        first_rng_digest = state_digest(parent_state["rng_state"])
        initialization_filename = "stage_b_resume_initialization.json"
        stats_filename = "stats_stage_b_continuation.jsonl"
        exact_stage_b_resume = True
    else:
        raise ValueError(f"Unsupported Stage B resume kind: {resume_kind}")

    # DataLoader iterator construction consumes its generator. Construct it only
    # after restoring the serialized generator and sampler state.
    dataset_iterator = iter(data_loader)

    misc.copy_params_and_buffers(
        src_module=parent_state["net"], dst_module=net, require_all=True
    )
    optimizer.load_state_dict(parent_state["optimizer_state"])
    misc.copy_params_and_buffers(
        src_module=parent_ema, dst_module=ema, require_all=True
    )
    network_source_digest = state_digest(parent_state["net"])
    optimizer_source_digest = state_digest(parent_state["optimizer_state"])
    ema_source_digest = state_digest(parent_ema)
    if state_digest(net) != network_source_digest:
        raise RuntimeError("Loaded training network differs from parent state")
    if state_digest(optimizer.state_dict()) != optimizer_source_digest:
        raise RuntimeError("Loaded optimizer differs from parent state")
    if state_digest(ema) != ema_source_digest:
        raise RuntimeError("Loaded EMA differs from parent state")

    initialization = {
        "warm_start": True,
        "exact_stage_a_continuation": False,
        "exact_stage_b_resume": exact_stage_b_resume,
        "validated_parent_implementation": ("9e5782f09a3e024c298dc5ce8da1c0f44c9b4fbd"),
        "execution_commit": execution_commit,
        "resume_kind": resume_kind,
        "seed": seed,
        "derived_seeds": derived_seeds,
        "seed_derivation_reproducible": True,
        "restored_rng_state_digest": first_rng_digest,
        "start_kimg": start_kimg,
        "parent_training_state": str(parent_training_state.resolve()),
        "parent_training_state_sha256": parent_training_state_sha256,
        "parent_ema_snapshot": str(parent_ema_snapshot.resolve()),
        "parent_ema_snapshot_sha256": parent_ema_snapshot_sha256,
        "network_loaded_exactly": True,
        "network_state_digest": network_source_digest,
        "optimizer_loaded_exactly": True,
        "optimizer_state_digest": optimizer_source_digest,
        "ema_loaded_exactly": True,
        "ema_state_digest": ema_source_digest,
    }
    del parent_state, parent_ema, snapshot
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / initialization_filename).write_text(
        json.dumps(initialization, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    dist.print0(f"Warm-start training from {start_kimg} to {total_kimg} kimg...")
    cur_nimg = start_kimg * 1000
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(start_kimg, total_kimg)
    stats_jsonl = None
    last_loss_mean = float("nan")
    while True:
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        loss_count = 0
        for round_idx in range(accumulation_rounds):
            with misc.ddp_sync(ddp, round_idx == accumulation_rounds - 1):
                images, labels = next(dataset_iterator)
                images = images.to(device).to(torch.float32) / 127.5 - 1
                labels = labels.to(device)
                loss = loss_fn(
                    net=ddp,
                    images=images,
                    labels=labels,
                    augment_pipe=augment_pipe,
                )
                if not torch.isfinite(loss).all().item():
                    raise FloatingPointError("Nonfinite Stage B loss")
                training_stats.report("Loss/loss", loss)
                loss_sum += float(loss.detach().sum().cpu())
                loss_count += int(loss.numel())
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()

        for group in optimizer.param_groups:
            group["lr"] = optimizer_kwargs["lr"] * min(
                cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1
            )
        for name, parameter in net.named_parameters():
            if (
                parameter.grad is not None
                and not torch.isfinite(parameter.grad).all().item()
            ):
                raise FloatingPointError(f"Nonfinite Stage B gradient: {name}")
        optimizer.step()
        last_loss_mean = loss_sum / loss_count

        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for ema_parameter, net_parameter in zip(
            ema.parameters(), net.parameters(), strict=True
        ):
            ema_parameter.copy_(net_parameter.detach().lerp(ema_parameter, ema_beta))

        cur_nimg += batch_size
        done = cur_nimg >= total_kimg * 1000
        if not done and cur_nimg < tick_start_nimg + kimg_per_tick * 1000:
            continue

        tick_end_time = time.time()
        fields = [
            f"tick {training_stats.report0('Progress/tick', lineage_tick):<5d}",
            f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}",
            f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}",
            f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}",
            f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}",
            f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}",
            f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}",
            f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}",
            f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}",
        ]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(" ".join(fields))
        if not done and dist.should_stop():
            done = True
            dist.print0("Aborting...")

        if snapshot_ticks is not None and (done or lineage_tick % snapshot_ticks == 0):
            snapshot_data = {
                "ema": ema,
                "loss_fn": loss_fn,
                "augment_pipe": augment_pipe,
                "dataset_kwargs": dict(dataset_kwargs),
            }
            for key, value in snapshot_data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    snapshot_data[key] = value.cpu()
            if rank == 0:
                snapshot_path = run_path / (
                    f"network-snapshot-{cur_nimg // 1000:06d}.pkl"
                )
                with snapshot_path.open("wb") as handle:
                    pickle.dump(snapshot_data, handle)
            del snapshot_data

        if (
            state_dump_ticks is not None
            and (done or lineage_tick % state_dump_ticks == 0)
            and rank == 0
        ):
            extended_state = {
                "state_schema_version": STATE_SCHEMA_VERSION,
                "net": net,
                "optimizer_state": optimizer.state_dict(),
                "ema": ema,
                "progress": {
                    "start_kimg": start_kimg,
                    "cur_nimg": cur_nimg,
                    "cur_kimg": cur_nimg // 1000,
                    "lineage_tick": lineage_tick + 1,
                },
                "rng_state": capture_rng_state(sampler, loader_generator),
                "warm_start": initialization,
            }
            validate_extended_state(extended_state)
            state_path = run_path / f"training-state-{cur_nimg // 1000:06d}.pt"
            torch.save(extended_state, state_path)

        training_stats.default_collector.update()
        if rank == 0:
            if stats_jsonl is None:
                stats_jsonl = (run_path / stats_filename).open("a", encoding="utf-8")
            stats_jsonl.write(
                json.dumps(
                    {
                        **training_stats.default_collector.as_dict(),
                        "stage_b_loss_mean": last_loss_mean,
                        "timestamp": time.time(),
                    }
                )
                + "\n"
            )
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        lineage_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    if stats_jsonl is not None:
        stats_jsonl.close()
    dist.print0("Exiting...")
    return {
        "start_kimg": start_kimg,
        "final_kimg": cur_nimg // 1000,
        "lineage_ticks": lineage_tick,
        "last_loss_mean": last_loss_mean,
        "initialization": initialization,
    }
