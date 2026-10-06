"""Explicit accelerator precision policy resolved outside tensor execution."""

from dataclasses import dataclass
import os

import torch


def configure_cuda_reproducibility(profile: str) -> dict[str, object]:
    """Apply the declared process-wide CUDA policy before model execution.

    Seeding alone does not make CUDA gradient reductions deterministic. Strict
    replay also requires deterministic operator dispatch and cuBLAS workspace.
    Native low-precision GEMMs retain FP32 accumulation in every profile.
    """
    if profile not in ("strict", "seeded", "throughput"):
        raise ValueError("invalid CUDA reproducibility profile")
    workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    strict = profile == "strict"
    if strict and workspace not in (":4096:8", ":16:8"):
        raise ValueError("strict CUDA reproducibility requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8")
    torch.use_deterministic_algorithms(strict)
    torch.backends.cudnn.deterministic = profile != "throughput"
    torch.backends.cudnn.benchmark = profile == "throughput"
    # FP32 incidence, folded summaries, and parity gates retain IEEE input
    # precision even when the major model projections use BF16 Tensor Cores.
    precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = precision
    torch.backends.cudnn.fp32_precision = precision
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    return {
        "profile": profile,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
        "cudnn_fp32_precision": torch.backends.cudnn.fp32_precision,
        "fp16_reduced_precision_reduction": False,
        "bf16_reduced_precision_reduction": False,
        "fp16_accumulation": False,
        "cublas_workspace_config": workspace,
    }


@dataclass(frozen=True)
class PrecisionPolicy:
    name: str
    compute_dtype: torch.dtype
    attention_backend: str
    device_type: str

    @property
    def scaling(self) -> bool:
        return self.name == "fp16"

    def manifest(self) -> dict[str, object]:
        return {
            "compute": str(self.compute_dtype).removeprefix("torch."),
            "activation": str(self.compute_dtype).removeprefix("torch."),
            "master": "float32", "gradient": "float32", "optimizer": "float32",
            "policy_loss_target_reduction": "float32",
            "incidence_accumulation": "float32",
            "dynamic_loss_scaling": self.scaling,
            "attention_backend": self.attention_backend,
            "device_type": self.device_type,
            "backward_autocast": "off",
        }


def resolve_precision(name: str, device: torch.device, *, attention_backend: str = "sdpa") -> PrecisionPolicy:
    if name not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision must be fp32, fp16, or bf16")
    if device.type not in ("cuda", "mps"):
        raise ValueError("precision policy requires explicit CUDA or MPS")
    if name == "bf16":
        if device.type != "cuda":
            raise ValueError("BF16 execution requires native CUDA support")
        # Native BF16 starts at SM80; never accept emulated BF16 on SM75.
        if torch.cuda.get_device_capability(device)[0] < 8:
            raise ValueError("selected CUDA device does not support native BF16")
    return PrecisionPolicy(name, {"fp32": torch.float32, "fp16": torch.float16,
                                  "bf16": torch.bfloat16}[name], attention_backend, device.type)
