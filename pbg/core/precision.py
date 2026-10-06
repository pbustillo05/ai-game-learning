"""Explicit accelerator precision policy resolved outside tensor execution."""

from dataclasses import dataclass

import torch


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
