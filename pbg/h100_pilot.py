"""Reproducible one-H100 pilot planning and model-only initialization."""

from __future__ import annotations

import json
import math
import os
import tempfile
import subprocess
import platform
from importlib.metadata import version
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml

from pbg.config import RootConfig, load_config_mapping
from pbg.core.persistence import CHECKPOINT_SCHEMA, model_state_sha256, sha256_file

INITIALIZER_SCHEMA = "pbg-h100-shared-initializer-v2"
PLAN_SCHEMA = "pbg-h100-replica-execution-plan-v2"
RESULT_SCHEMA = "pbg-h100-pilot-result-v1"
GIB = 1 << 30
H100_DECLARED_TOTAL_BYTES = 80_000_000_000
H100_MEMORY_GUARD_BYTES = 10 * GIB
H100_MEMORY_BUDGET_BYTES = H100_DECLARED_TOTAL_BYTES - H100_MEMORY_GUARD_BYTES
REPLICA_COUNTS = (1, 2, 4, 8, 16, 32)


class PilotError(RuntimeError):
    """The H100 pilot cannot preserve its declared experiment contract."""


@dataclass(frozen=True)
class Arm:
    name: str
    stage: str
    mode: str
    replicas: int
    roots: int
    completion_samples: int | None
    rollout_pack: int = 8
    completion_pack: int = 64
    learner_pack: int = 32
    warmup: int = 2
    repeats: int = 5
    seed: int = 104830

    def __post_init__(self) -> None:
        if self.mode not in ("normal", "rb"):
            raise ValueError("pilot mode must be normal or rb")
        if self.replicas not in REPLICA_COUNTS:
            raise ValueError("pilot replicas must be one of 1, 2, 4, 8, 16, 32")
        if (
            self.roots <= 0
            or min(self.rollout_pack, self.completion_pack, self.learner_pack) <= 0
        ):
            raise ValueError("pilot roots and execution packs must be positive")
        if self.mode == "rb" and (
            self.completion_samples is None or self.completion_samples <= 0
        ):
            raise ValueError("RB pilot arms require completion samples")
        if self.mode == "normal" and self.completion_samples is not None:
            raise ValueError("normal pilot arms cannot declare completion samples")
        if self.warmup < 0 or self.repeats <= 0:
            raise ValueError("pilot warmup/repeats are invalid")


def _atomic_torch_save(value: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _tensor_state(value: object, *, source: str) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise PilotError(f"{source} has no nonempty model tensor state")
    state = dict(value)
    if any(
        not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
        for name, tensor in state.items()
    ):
        raise PilotError(f"{source} model state contains a non-tensor entry")
    return {
        name: tensor.detach().to(device="cpu").contiguous().clone()
        for name, tensor in state.items()
    }


def export_model_initializer(
    checkpoint: str | Path,
    destination: str | Path,
) -> dict[str, object]:
    """Extract a portable, hash-bound model state from a trusted checkpoint."""

    source = Path(checkpoint)
    target = Path(destination)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise PilotError("initializer source is not a checkpoint mapping")
    if payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
        raise PilotError("initializer requires a shared-model checkpoint; split artifacts are unsupported")
    state = _tensor_state(payload.get("model"), source=str(source))
    if "lanes" in state:
        raise PilotError("replica checkpoint model state is not a single initializer")
    report = {
        "schema": INITIALIZER_SCHEMA,
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": sha256_file(source),
        "model_sha256": model_state_sha256(state),
        "parameter_tensors": len(state),
    }
    _atomic_torch_save({**report, "model": state}, target)
    return {**report, "path": str(target), "artifact_sha256": sha256_file(target)}


def load_model_initializer(
    path: str | Path,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Load and verify one model-only pilot initializer artifact."""

    source = Path(path)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema",
        "source_checkpoint",
        "source_checkpoint_sha256",
        "model_sha256",
        "parameter_tensors",
        "model",
    }:
        raise PilotError("initializer artifact has unknown or missing fields")
    if payload["schema"] != INITIALIZER_SCHEMA:
        raise PilotError("initializer artifact schema is unsupported")
    state = _tensor_state(payload["model"], source=str(source))
    if len(state) != payload["parameter_tensors"]:
        raise PilotError("initializer tensor count differs from its metadata")
    digest = model_state_sha256(state)
    if digest != payload["model_sha256"]:
        raise PilotError("initializer model hash differs from its metadata")
    metadata = {name: value for name, value in payload.items() if name != "model"}
    metadata["artifact_sha256"] = sha256_file(source)
    return state, metadata


def _scientific_mapping(
    *,
    mode: str,
    completion_samples: int | None,
    batch_size: int,
    gradient_steps: int,
) -> dict[str, object]:
    scientific: dict[str, object] = {
        "mode": {
            "training_regime": "self_play",
            "rollout_mode": "sequential",
            "private_estimator": ("sampled" if mode == "normal" else "rao_blackwell"),
        },
        "game": {"players": 2, "stack_bb": 32.0, "rank_count": 13},
        "model": {
            "d_model": 128,
            "n_heads": 4,
            "layers": 6,
            "raises": {"components": 2, "minimum_scale": 0.02},
            "class_embedding_dim": 32,
            "folder_beta_rank": 0,
            "folder_line_dim": 8,
            "epsilon": 1.0e-8,
        },
        "objective": {
            "policy_clip": 0.2,
            "value": {"kind": "clipped", "epsilon": 0.2, "coefficient": 0.125},
            "entropy": {
                "kind": "linear_decay",
                "coefficient": 0.025,
                "iterations": 500,
            },
            "reference_kl": {
                "coefficient": 0.05,
                "refresh": {
                    "effective_hands": (
                        900_000.0 if mode == "normal" else 172_800_000.0
                    )
                },
            },
        },
        "optimizer": {
            "kind": "adamw",
            "learning_rate": 5.0e-5,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1.0e-8,
            "weight_decay": 0.0,
            "grad_clip": 0.5,
        },
        "schedule": {"batch_size": batch_size, "steps": gradient_steps},
        "lambda": 0.95,
    }
    if mode == "rb":
        scientific["completion"] = {"samples": completion_samples}
    return scientific


def pilot_config_mapping(
    arm: Arm,
    *,
    output_dir: str | Path,
    device: str = "cuda:0",
    hardware: str = "h100",
) -> dict[str, object]:
    """Build one strict config for either the H100 pilot or local CUDA smoke."""

    if hardware not in ("h100", "local"):
        raise ValueError("hardware profile must be h100 or local")
    h100 = hardware == "h100"
    precision = "bf16" if h100 else "fp32"
    production = h100 and arm.stage != "smoke"
    batch_size = 256 if production else 8
    gradient_steps = 116 if production else 2 if h100 else 1
    rollout_pack = min(arm.rollout_pack, batch_size) if not h100 else arm.rollout_pack
    completion_pack = (
        min(arm.completion_pack, batch_size) if not h100 else arm.completion_pack
    )
    learner_pack = min(arm.learner_pack, batch_size)
    result: dict[str, object] = {
        "kind": "nlhe_run_v1",
        "scientific": _scientific_mapping(
            mode=arm.mode,
            completion_samples=arm.completion_samples,
            batch_size=batch_size,
            gradient_steps=gradient_steps,
        ),
        "run": {
            "seed": arm.seed,
            "reproducibility": "throughput",
            "utility": "stack_normalized",
            "workload": {
                "roots": arm.roots,
                **({"replicas": arm.replicas} if arm.replicas != 1 else {}),
            },
            "execution": {
                "device": device,
                "precision": precision,
                "attention": "sdpa",
                "reference_cache": "off",
                "tensor_program": {"mode": "captured" if h100 else "eager",
                                   "cache_entries": 8, "cache_bytes": 8 * GIB if h100 else 0},
                "memory": {
                    "reserve_bytes": (H100_MEMORY_GUARD_BYTES if h100 else 256 << 20)
                },
                "microbatch": {
                    "rollout_rows": rollout_pack,
                    "completion_rows": completion_pack,
                    "learner_rows": {"pre": learner_pack, "post_2": min(learner_pack, 8)},
                },
            },
            "iterations": 1,
            "checkpoint_every": 1,
            "output_dir": str(output_dir),
        },
    }
    return result


def write_arm_config(
    arm: Arm,
    destination: str | Path,
    *,
    output_dir: str | Path,
    hardware: str = "h100",
    resolve: bool = True,
) -> RootConfig | None:
    target = Path(destination)
    mapping = pilot_config_mapping(
        arm,
        output_dir=output_dir,
        hardware=hardware,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(mapping, sort_keys=False),
        encoding="utf-8",
    )
    return load_config_mapping(mapping) if resolve else None


def base_plan() -> dict[str, object]:
    """Small correctness gates, matched K=1/16/32 runs, and admitted K=8 fallback."""
    arms = []
    for mode in ("normal", "rb"):
        for replicas in (1, 2):
            arms.append(Arm(f"nlhe_hu_32bb_{mode}_h100_smoke_k{replicas}", "smoke", mode,
                            replicas, 4, 16 if mode == "rb" else None,
                            rollout_pack=2, completion_pack=4, learner_pack=2,
                            warmup=0, repeats=1))
        for replicas in (1, 16, 32, 8):
            arms.append(Arm(f"nlhe_hu_32bb_{mode}_h100_replica_k{replicas}",
                            "fallback" if replicas == 8 else "replica_scaling", mode,
                            replicas, 64, 256 if mode == "rb" else None))
    return {"schema": PLAN_SCHEMA, "deadline_minutes": 240,
            "shutdown_reserve_minutes": 10, "replica_gain_threshold": 0.10,
            "memory_ceiling_bytes": H100_MEMORY_BUDGET_BYTES,
            "fixed_batch_size": 256, "fixed_gradient_steps": 116,
            "production_precision": "bf16", "replica_counts": [1, 16, 32],
            "fallback_replicas": 8, "validation_seeds": [205991, 308177],
            "execution_packs": {"rollout_rows": 8, "completion_rows": 64,
                                "learner_rows": {"pre": 32, "post_2": 8}},
            "arms": [asdict(arm) for arm in arms]}


def select_replica_count(
    reports: Sequence[Mapping[str, object]],
    *,
    minimum_gain: float = 0.10,
    memory_ceiling_bytes: int = H100_MEMORY_BUDGET_BYTES,
) -> int:
    """Select the smallest K unless a larger K gains at least the threshold."""

    rows: dict[int, tuple[float, int]] = {}
    for report in reports:
        replicas = int(report["replicas"])
        rate = float(report["player_nodes_per_second"])
        peak = int(report["peak_reserved_bytes"])
        if rate > 0 and peak <= memory_ceiling_bytes:
            rows[replicas] = (rate, peak)
    if 1 not in rows:
        raise PilotError("replica selection has no successful K=1 baseline")
    selected = 1
    selected_rate = rows[1][0]
    for replicas in REPLICA_COUNTS[1:]:
        candidate = rows.get(replicas)
        if candidate is not None and candidate[0] >= selected_rate * (
            1.0 + minimum_gain
        ):
            selected = replicas
            selected_rate = candidate[0]
    return selected


def selected_gpu_snapshot(*, allow_current_process: bool = False) -> dict[str, object]:
    """Read resource state without initializing a CUDA context or altering jobs."""
    selector = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not selector or "," in selector:
        raise PilotError("set CUDA_VISIBLE_DEVICES to one idle GPU index or UUID")
    def smi(*arguments):
        return subprocess.check_output(["nvidia-smi", "-i", selector, *arguments], text=True).strip()
    raw = smi("--query-compute-apps=pid", "--format=csv,noheader,nounits")
    pids = [int(pid.strip()) for pid in raw.splitlines() if pid.strip().isdigit()]
    other = [pid for pid in pids if not allow_current_process or pid != os.getpid()]
    if other:
        raise PilotError(f"selected GPU has compute jobs: {other}; leave them untouched")
    return {"selector": selector,
            "used_memory_mib": int(smi("--query-gpu=memory.used", "--format=csv,noheader,nounits")),
            "name": smi("--query-gpu=name", "--format=csv,noheader,nounits")}


def h100_preflight() -> dict[str, object]:
    """Record hardware/software and execute real native BF16 fused backward."""
    idle = selected_gpu_snapshot(allow_current_process=True)
    expected = json.loads((Path(__file__).parents[1] / "configs/h100_environment.json").read_text())
    actual = {name: version(name) for name in expected["packages"]}
    drift = {name: {"expected": required, "actual": actual[name]}
             for name, required in expected["packages"].items() if actual[name] != required}
    if drift or platform.python_version() != expected["python"]:
        raise PilotError(f"rental environment differs from pinned reference: {drift}; Python {platform.python_version()}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise PilotError("pilot requires exactly one explicitly visible CUDA GPU")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    if ("H100" not in properties.name.upper() or properties.major < 9
        or properties.total_memory < 70 * GIB):
        raise PilotError("pilot requires a full H100 with native BF16 and at least 70 GiB reported memory")
    selector = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not selector or "," in selector:
        raise PilotError("set CUDA_VISIBLE_DEVICES to one idle GPU index or UUID")
    def smi(*arguments):
        return subprocess.check_output(["nvidia-smi", "-i", selector, *arguments], text=True).strip()
    processes = smi("--query-compute-apps=pid", "--format=csv,noheader,nounits")
    others = [int(pid.strip()) for pid in processes.splitlines() if pid.strip().isdigit() and int(pid) != os.getpid()]
    if others:
        raise PilotError(f"selected GPU has other compute jobs: {others}; no workload launched")
    hardware = smi("--query-gpu=uuid,name,driver_version,memory.total,mig.mode.current,clocks.sm,clocks.mem,power.limit",
                   "--format=csv,noheader,nounits")
    from pbg.core.precision import resolve_precision
    from pbg.games.poker.model.primitives import MHA
    from pbg.identity import accelerator_identity, code_identity
    policy = resolve_precision("bf16", device)
    module = MHA(128, 4).to(device)
    value = torch.randn(2, 64, 128, device=device, requires_grad=True)
    bias = torch.randn(2, 64, device=device, dtype=torch.float32)
    mask = torch.ones(2, 64, device=device, dtype=torch.bool)
    with torch.autocast("cuda", dtype=policy.compute_dtype):
        output = module(value, value, bias, mask)
        objective = output.float().square().mean()
    objective.backward()
    torch.cuda.synchronize(device)
    if output.dtype != torch.bfloat16 or not all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in module.parameters()
    ) or not bool(torch.isfinite(value.grad).all()):
        raise PilotError("native BF16 fused-attention forward/backward gate failed")
    identity = accelerator_identity(13)
    revision = identity["evaluator"].get("build_revision")
    if revision != expected["evaluator_revision"]:
        raise PilotError(f"evaluator revision differs from pin: {revision}")
    report = {"device": str(device), "idle_before": idle, "name": properties.name,
              "compute_capability": [properties.major, properties.minor],
              "total_memory_bytes": properties.total_memory,
              "memory_budget_bytes": properties.total_memory - H100_MEMORY_GUARD_BYTES,
              "bf16_supported": True, "fused_attention_backward": "passed",
              "nvidia_smi_fields": "uuid,name,driver,memory_MiB,MIG,SM_MHz,memory_MHz,power_W",
              "nvidia_smi": hardware, "torch": torch.__version__, "cuda": torch.version.cuda,
              "cudnn": torch.backends.cudnn.version(), "compiled_architectures": torch.cuda.get_arch_list(),
              "python": platform.python_version(), "packages": actual,
              "precision_policy": policy.manifest(), "identity": identity, "code": code_identity()}
    del module, value, output, objective, bias, mask
    torch.cuda.empty_cache()
    return report


def benchmark_summary(
    arm: Arm,
    benchmark: Mapping[str, object],
) -> dict[str, object]:
    median = benchmark.get("median")
    samples = benchmark.get("samples")
    scientific = benchmark.get("scientific_config")
    if (
        not isinstance(median, Mapping)
        or not isinstance(samples, Sequence)
        or len(samples) != arm.repeats
        or not isinstance(scientific, Mapping)
        or not isinstance(scientific.get("schedule"), Mapping)
    ):
        raise PilotError("benchmark report is incomplete")
    gradient_steps = int(scientific["schedule"]["steps"])
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise PilotError("benchmark sample is not an object")
        if arm.warmup > 0:
            program_before = sample.get("tensor_program_before", {})
            program_after = sample.get("tensor_program_after", {})
            if (program_before.get("entries") != program_after.get("entries") or
                program_before.get("denied_shapes") != program_after.get("denied_shapes")):
                raise PilotError("timed sample contains compile/capture admission; increase warmup")
        before = sample.get("logical_work_before")
        after = sample.get("logical_work_after")
        if (
            not isinstance(before, Sequence)
            or not isinstance(after, Sequence)
            or len(before) != arm.replicas
            or len(after) != arm.replicas
            or int(sample.get("player_nodes", 0)) <= 0
        ):
            raise PilotError("benchmark sample lost replica/work identity")
        for lane_before, lane_after in zip(before, after, strict=True):
            if (
                not isinstance(lane_before, Mapping)
                or not isinstance(lane_after, Mapping)
                or any(int(value) != 0 for value in lane_before.values())
                or int(lane_after["trajectories"]) != arm.roots
                or int(lane_after["optimizer_steps"]) != gradient_steps
                or (arm.mode == "rb" and int(lane_after["completion_samples"]) <= 0)
            ):
                raise PilotError(
                    "benchmark lane did not complete its declared fresh work"
                )
    for name in (
        "player_nodes_per_second",
        "seconds",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    ):
        if not math.isfinite(float(median[name])) or float(median[name]) < 0:
            raise PilotError(f"benchmark median {name} is invalid")
    return {
        "schema": RESULT_SCHEMA,
        "arm": arm.name,
        "stage": arm.stage,
        "mode": arm.mode,
        "replicas": arm.replicas,
        "roots_per_replica": arm.roots,
        "seed": arm.seed,
        "learner_presentations_per_second": float(median["learner_presentations_per_second"]),
        "learner_presentations": int(median["learner_presentations"]),
        "optimizer_steps": int(median["optimizer_steps"]),
        "completion_samples": arm.completion_samples,
        "rollout_pack": arm.rollout_pack,
        "completion_pack": arm.completion_pack,
        "learner_pack": arm.learner_pack,
        "player_nodes_per_second": float(median["player_nodes_per_second"]),
        "seconds": float(median["seconds"]),
        "peak_allocated_bytes": int(median["peak_allocated_bytes"]),
        "peak_reserved_bytes": int(median["peak_reserved_bytes"]),
        "initializer_model_sha256": benchmark["identity"]["initializer_model_sha256"],
    }


def write_json(value: object, destination: str | Path) -> None:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


__all__ = [
    "H100_MEMORY_BUDGET_BYTES",
    "INITIALIZER_SCHEMA",
    "PLAN_SCHEMA",
    "REPLICA_COUNTS",
    "Arm",
    "PilotError",
    "base_plan",
    "benchmark_summary",
    "export_model_initializer",
    "h100_preflight",
    "load_model_initializer",
    "pilot_config_mapping",
    "select_replica_count",
    "write_arm_config",
    "write_json",
]
