"""Alias-aware CUDA accounting and schema-derived four-phase control."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any, Mapping

import torch
from torch import Tensor


def unique_tensor_bytes(
    value: Any,
    seen: set[tuple[str, int | None, int]] | None = None,
    *,
    device: torch.device | None = None,
    _seen_objects: set[int] | None = None,
) -> int:
    """Return unique tensor-storage bytes reachable from nested objects."""

    if seen is None:
        seen = set()
    if _seen_objects is None:
        _seen_objects = set()
    if isinstance(value, Tensor):
        if device is not None and value.device != device:
            return 0
        storage = value.untyped_storage()
        pointer = storage.data_ptr()
        device = storage.device
        identity = (device.type, device.index, pointer)
        # Empty storages may share the null pointer but consume no bytes.
        if not storage.nbytes() or identity in seen:
            return 0
        seen.add(identity)
        return int(storage.nbytes())
    identity = id(value)
    if identity in _seen_objects:
        return 0
    _seen_objects.add(identity)
    if isinstance(value, Mapping):
        return sum(
            unique_tensor_bytes(
                item, seen, device=device, _seen_objects=_seen_objects
            )
            for item in value.values()
        )
    if isinstance(value, (tuple, list)):
        return sum(
            unique_tensor_bytes(
                item, seen, device=device, _seen_objects=_seen_objects
            )
            for item in value
        )
    fields = getattr(value, "__dataclass_fields__", None)
    if fields is not None:
        return sum(
            unique_tensor_bytes(
                getattr(value, name),
                seen,
                device=device,
                _seen_objects=_seen_objects,
            )
            for name in fields
        )
    if hasattr(value, "__dict__"):
        return sum(
            unique_tensor_bytes(
                item, seen, device=device, _seen_objects=_seen_objects
            )
            for item in vars(value).values()
        )
    return 0


@dataclass
class MemoryLedger:
    """Take alias-aware, quiescent CUDA allocator measurements."""

    device: torch.device

    def snapshot(
        self, label: str, *, owned: Mapping[str, Any] | None = None
    ) -> dict[str, float]:
        if self.device.type != "cuda":
            return {}
        torch.cuda.synchronize(self.device)
        free, total = torch.cuda.mem_get_info(self.device)
        stats = torch.cuda.memory_stats(self.device)
        allocated = int(torch.cuda.memory_allocated(self.device))
        reserved = int(torch.cuda.memory_reserved(self.device))
        peak_allocated = int(torch.cuda.max_memory_allocated(self.device))
        peak_reserved = int(torch.cuda.max_memory_reserved(self.device))
        active = int(stats.get("active_bytes.all.current", allocated))
        inactive_split = int(stats.get("inactive_split_bytes.all.current", 0))
        reusable = max(0, reserved - active - inactive_split)
        prefix = f"memory_{label}_"
        result = {
            f"{prefix}cuda_total_bytes": float(total),
            f"{prefix}cuda_free_bytes": float(free),
            f"{prefix}device_used_bytes": float(total - free),
            f"{prefix}allocated_bytes": float(allocated),
            f"{prefix}reserved_bytes": float(reserved),
            f"{prefix}peak_allocated_bytes": float(peak_allocated),
            f"{prefix}peak_reserved_bytes": float(peak_reserved),
            f"{prefix}active_bytes": float(active),
            f"{prefix}reusable_bytes": float(reusable),
            f"{prefix}nonreleasable_bytes": float(inactive_split),
            f"{prefix}other_bytes": float(max(0, total - free - reserved)),
        }
        if owned:
            seen: set[tuple[str, int | None, int]] = set()
            for name, value in owned.items():
                result[f"{prefix}owned_{name}_bytes"] = float(
                    unique_tensor_bytes(value, seen, device=self.device)
                )
            named = sum(
                value
                for key, value in result.items()
                if key.startswith(f"{prefix}owned_")
            )
            result[f"{prefix}owned_total_bytes"] = float(named)
            result[f"{prefix}unattributed_active_bytes"] = float(
                max(0.0, active - named)
            )
        return result



# Sole sequential root-width controller: schema prior plus phase observations.
SEQUENTIAL_MEMORY_CONTROLLER_VERSION = "sequential-memory-v2"
MEMORY_PHASES = ("rollout", "seal", "targets", "optimization")


class MemoryAdmissionStop(RuntimeError):
    """A clean pre-draw or transactional stop; callers must not reroll."""


@dataclass(frozen=True)
class PhaseMemoryModel:
    """Monotone upper byte model for one physical iteration phase."""

    fixed_bytes: int
    upper_bytes_per_root: Mapping[str, float]
    residual_envelope_bytes: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.fixed_bytes, bool) or self.fixed_bytes < 0:
            raise ValueError("phase fixed bytes must be nonnegative")
        terms = {str(key): float(value) for key, value in self.upper_bytes_per_root.items()}
        if any(
            not key or not math.isfinite(value) or value < 0.0
            for key, value in terms.items()
        ):
            raise ValueError("phase per-root byte terms must be finite and nonnegative")
        if isinstance(self.residual_envelope_bytes, bool) or self.residual_envelope_bytes < 0:
            raise ValueError("phase residual envelope must be nonnegative")
        object.__setattr__(self, "upper_bytes_per_root", dict(terms))

    @property
    def bytes_per_root(self) -> float:
        return sum(self.upper_bytes_per_root.values())

    def predict(self, roots_per_gpu: int) -> int:
        if isinstance(roots_per_gpu, bool) or roots_per_gpu < 0:
            raise ValueError("roots_per_gpu must be nonnegative")
        return int(
            math.ceil(
                self.fixed_bytes
                + roots_per_gpu * self.bytes_per_root
                + self.residual_envelope_bytes
            )
        )


@dataclass(frozen=True)
class PhaseMemoryObservation:
    """Schema-validated completed observation for one rank and phase."""

    roots_per_gpu: int
    fixed_bytes: int
    variable_bytes: Mapping[str, int]
    predicted_peak_bytes: int
    actual_peak_bytes: int

    def __post_init__(self) -> None:
        if isinstance(self.roots_per_gpu, bool) or self.roots_per_gpu <= 0:
            raise ValueError("memory observations require positive per-GPU roots")
        integers = (self.fixed_bytes, self.predicted_peak_bytes, self.actual_peak_bytes)
        if any(isinstance(value, bool) or value < 0 for value in integers):
            raise ValueError("memory observation byte counts must be nonnegative")
        variables = {str(key): int(value) for key, value in self.variable_bytes.items()}
        if any(not key or isinstance(value, bool) or value < 0 for key, value in variables.items()):
            raise ValueError("memory observation variable bytes are invalid")
        accounted = self.fixed_bytes + sum(variables.values())
        if accounted > self.actual_peak_bytes:
            raise ValueError(
                "phase ledger ownership exceeds its measured actual peak: "
                f"accounted={accounted}, actual={self.actual_peak_bytes}"
            )
        object.__setattr__(self, "variable_bytes", variables)

    def state_dict(self) -> dict[str, Any]:
        return {
            "roots_per_gpu": self.roots_per_gpu,
            "fixed_bytes": self.fixed_bytes,
            "variable_bytes": dict(self.variable_bytes),
            "predicted_peak_bytes": self.predicted_peak_bytes,
            "actual_peak_bytes": self.actual_peak_bytes,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "PhaseMemoryObservation":
        return cls(
            roots_per_gpu=int(state["roots_per_gpu"]),
            fixed_bytes=int(state["fixed_bytes"]),
            variable_bytes={
                str(key): int(value) for key, value in state["variable_bytes"].items()
            },
            predicted_peak_bytes=int(state["predicted_peak_bytes"]),
            actual_peak_bytes=int(state["actual_peak_bytes"]),
        )


@dataclass
class PhaseMemoryForecaster:
    """Lagged one-sided phase forecasts from completed iterations only."""

    analytic_models: Mapping[str, PhaseMemoryModel]
    window: int = 32
    prediction_z: float = 3.0
    models: dict[str, PhaseMemoryModel] = field(default_factory=dict)
    observations: dict[str, deque[PhaseMemoryObservation]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if set(self.analytic_models) != set(MEMORY_PHASES):
            raise ValueError("analytic memory models must cover all four phases")
        if isinstance(self.window, bool) or self.window <= 0:
            raise ValueError("memory calibration window must be positive")
        if not math.isfinite(self.prediction_z) or self.prediction_z <= 0.0:
            raise ValueError("one-sided memory prediction level must be positive")
        self.analytic_models = dict(self.analytic_models)
        self.models = dict(self.models)
        self.observations = {
            phase: deque(values, maxlen=self.window)
            for phase, values in self.observations.items()
        }

    @property
    def calibrated(self) -> bool:
        return all(self.observations.get(phase) for phase in MEMORY_PHASES)

    def phase_model(self, phase: str) -> PhaseMemoryModel:
        if phase not in MEMORY_PHASES:
            raise KeyError(f"unknown memory phase {phase!r}")
        return self.models.get(phase, self.analytic_models[phase])

    def predict(self, roots_per_gpu: int) -> dict[str, int]:
        return {
            phase: self.phase_model(phase).predict(roots_per_gpu)
            for phase in MEMORY_PHASES
        }

    def limiting_phase(self, roots_per_gpu: int) -> tuple[str, int]:
        predictions = self.predict(roots_per_gpu)
        return max(predictions.items(), key=lambda item: (item[1], item[0]))

    def observe(self, phase: str, observation: PhaseMemoryObservation) -> None:
        """Widen immediately; upper evidence ages only after a full window."""

        if phase not in MEMORY_PHASES:
            raise KeyError(f"unknown memory phase {phase!r}")
        history = self.observations.setdefault(phase, deque(maxlen=self.window))
        history.append(observation)
        analytic = self.analytic_models[phase]
        keys = set(analytic.upper_bytes_per_root)
        for item in history:
            keys.update(item.variable_bytes)
        upper: dict[str, float] = {}
        for key in keys:
            analytic_value = analytic.upper_bytes_per_root.get(key, 0.0)
            samples = [
                item.variable_bytes.get(key, 0) / item.roots_per_gpu
                for item in history
            ]
            mean = sum(samples) / len(samples)
            variance = sum((value - mean) ** 2 for value in samples) / max(
                len(samples) - 1, 1
            )
            one_sided = mean + self.prediction_z * math.sqrt(variance)
            # The maximum provides immediate surprise widening. Its removal is
            # delayed until the bounded history has W later successes.
            upper[key] = max(analytic_value, max(samples), one_sided)
        fixed = max(analytic.fixed_bytes, *(item.fixed_bytes for item in history))
        raw_without_envelope = [
            fixed + item.roots_per_gpu * sum(upper.values()) for item in history
        ]
        residual = max(
            analytic.residual_envelope_bytes,
            0,
        )
        residual_samples = [
            max(0.0, item.actual_peak_bytes - predicted)
            for item, predicted in zip(history, raw_without_envelope, strict=True)
        ]
        if residual_samples:
            mean = sum(residual_samples) / len(residual_samples)
            variance = sum((value - mean) ** 2 for value in residual_samples) / max(
                len(residual_samples) - 1, 1
            )
            residual = max(
                residual,
                math.ceil(max(residual_samples)),
                math.ceil(mean + self.prediction_z * math.sqrt(variance)),
            )
        self.models[phase] = PhaseMemoryModel(fixed, upper, max(0, residual))

    def state_dict(self) -> dict[str, Any]:
        def model_state(model: PhaseMemoryModel) -> dict[str, Any]:
            return {
                "fixed_bytes": model.fixed_bytes,
                "upper_bytes_per_root": dict(model.upper_bytes_per_root),
                "residual_envelope_bytes": model.residual_envelope_bytes,
            }

        return {
            "window": self.window,
            "prediction_z": self.prediction_z,
            "analytic_models": {
                phase: model_state(model) for phase, model in self.analytic_models.items()
            },
            "models": {phase: model_state(model) for phase, model in self.models.items()},
            "observations": {
                phase: [item.state_dict() for item in history]
                for phase, history in self.observations.items()
            },
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "PhaseMemoryForecaster":
        def model(value: Mapping[str, Any]) -> PhaseMemoryModel:
            return PhaseMemoryModel(
                fixed_bytes=int(value["fixed_bytes"]),
                upper_bytes_per_root={
                    str(key): float(item)
                    for key, item in value["upper_bytes_per_root"].items()
                },
                residual_envelope_bytes=int(value.get("residual_envelope_bytes", 0)),
            )

        return cls(
            analytic_models={
                phase: model(value) for phase, value in state["analytic_models"].items()
            },
            window=int(state.get("window", 32)),
            prediction_z=float(state.get("prediction_z", 3.0)),
            models={phase: model(value) for phase, value in state.get("models", {}).items()},
            observations={
                phase: deque(
                    PhaseMemoryObservation.from_state_dict(item) for item in values
                )
                for phase, values in state.get("observations", {}).items()
            },
        )


@dataclass
class SequentialMemoryController:
    """Choose one common per-GPU root width solely from memory feasibility."""

    minimum_roots_per_gpu: int
    maximum_roots_per_gpu: int
    current_roots_per_gpu: int
    device_bytes: int
    guard_bytes: int
    forecaster: PhaseMemoryForecaster
    growth_fraction: float = 0.10
    growth_delay: int = 8
    successful_streak: int = 0
    limiting_phase: str | None = None
    limiting_field: str | None = None
    last_predicted_peaks: dict[str, int] = field(default_factory=dict)
    last_actual_peaks: dict[str, int] = field(default_factory=dict)
    terminal_failure: str | None = None
    version: str = SEQUENTIAL_MEMORY_CONTROLLER_VERSION

    def __post_init__(self) -> None:
        values = (
            self.minimum_roots_per_gpu,
            self.maximum_roots_per_gpu,
            self.current_roots_per_gpu,
        )
        if any(isinstance(value, bool) or value <= 0 for value in values):
            raise ValueError("sequential per-GPU root widths must be positive")
        if not (
            self.minimum_roots_per_gpu
            <= self.current_roots_per_gpu
            <= self.maximum_roots_per_gpu
        ):
            raise ValueError("current root width must lie within configured bounds")
        if self.device_bytes <= 0 or self.guard_bytes < 0 or self.guard_bytes >= self.device_bytes:
            raise ValueError("device/guard bytes do not define a positive hard ceiling")
        if not math.isfinite(self.growth_fraction) or not 0.0 < self.growth_fraction < 1.0:
            raise ValueError("growth_fraction must lie in (0,1)")
        if isinstance(self.growth_delay, bool) or self.growth_delay <= 0:
            raise ValueError("growth_delay must be positive")
        if self.version != SEQUENTIAL_MEMORY_CONTROLLER_VERSION:
            raise ValueError("unsupported sequential memory controller version")

    @property
    def hard_ceiling_bytes(self) -> int:
        return self.device_bytes - self.guard_bytes

    def local_safe_width(self) -> int:
        """Integer monotone-search maximum whose worst phase fits locally."""

        low = self.minimum_roots_per_gpu
        high = self.maximum_roots_per_gpu
        safe = 0
        while low <= high:
            middle = (low + high) // 2
            peak = max(self.forecaster.predict(middle).values())
            if peak <= self.hard_ceiling_bytes:
                safe = middle
                low = middle + 1
            else:
                high = middle - 1
        return safe

    def choose_root_width(self, *, rank_safe_widths: Tensor | None = None) -> int:
        """Freeze the next width before draws; optional tensor models all-reduce MIN."""

        if self.terminal_failure is not None:
            raise MemoryAdmissionStop(self.terminal_failure)
        local_safe = self.local_safe_width()
        safe = local_safe
        if rank_safe_widths is not None:
            values = rank_safe_widths.to(torch.int64)
            if values.ndim != 1 or values.numel() == 0 or bool((values < 0).any()):
                raise ValueError("rank safe widths must be a nonempty nonnegative vector")
            safe = min(safe, int(values.min()))
        elif torch.distributed.is_available() and torch.distributed.is_initialized():
            backend = torch.distributed.get_backend()
            if backend != "nccl":
                raise RuntimeError("distributed memory control is NCCL-only")
            reduction_device = torch.device("cuda", torch.cuda.current_device())
            value = torch.tensor(local_safe, dtype=torch.int64, device=reduction_device)
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MIN)
            safe = int(value)
        if safe < self.minimum_roots_per_gpu:
            self.terminal_failure = (
                "minimum sequential root width does not fit the four-phase memory ceiling"
            )
            raise MemoryAdmissionStop(self.terminal_failure)
        if safe < self.current_roots_per_gpu:
            selected = safe
            self.successful_streak = 0
        elif self.forecaster.calibrated and self.successful_streak >= self.growth_delay:
            cap = max(
                self.current_roots_per_gpu + 1,
                math.floor(self.current_roots_per_gpu * (1.0 + self.growth_fraction)),
            )
            selected = min(safe, self.maximum_roots_per_gpu, cap)
            self.successful_streak = 0
        else:
            selected = self.current_roots_per_gpu
        self.current_roots_per_gpu = selected
        self.last_predicted_peaks = self.forecaster.predict(selected)
        self.limiting_phase, _ = max(
            self.last_predicted_peaks.items(), key=lambda item: (item[1], item[0])
        )
        model = self.forecaster.phase_model(self.limiting_phase)
        self.limiting_field = max(
            model.upper_bytes_per_root.items(), key=lambda item: (item[1], item[0])
        )[0] if model.upper_bytes_per_root else "fixed"
        return selected

    def observe_success(
        self, observations: Mapping[str, PhaseMemoryObservation]
    ) -> dict[str, float]:
        """Commit a complete four-phase ledger for use by later iterations."""

        if set(observations) != set(MEMORY_PHASES):
            raise ValueError("a successful memory observation must cover all four phases")
        width = self.current_roots_per_gpu
        if any(item.roots_per_gpu != width for item in observations.values()):
            raise ValueError("phase observations must use the frozen current root width")
        covered = True
        for phase in MEMORY_PHASES:
            item = observations[phase]
            covered &= item.actual_peak_bytes <= item.predicted_peak_bytes
            self.forecaster.observe(phase, item)
        self.last_actual_peaks = {
            phase: observations[phase].actual_peak_bytes for phase in MEMORY_PHASES
        }
        self.successful_streak = self.successful_streak + 1 if covered else 0
        return {
            "sequential_roots_per_gpu": float(width),
            "memory_guard_bytes": float(self.guard_bytes),
            "memory_hard_ceiling_bytes": float(self.hard_ceiling_bytes),
            "memory_successful_streak": float(self.successful_streak),
            "memory_prediction_covered": float(covered),
            **{
                f"memory_{phase}_predicted_peak_bytes": float(
                    observations[phase].predicted_peak_bytes
                )
                for phase in MEMORY_PHASES
            },
            **{
                f"memory_{phase}_actual_peak_bytes": float(
                    observations[phase].actual_peak_bytes
                )
                for phase in MEMORY_PHASES
            },
        }

    def record_transaction_failure(self, phase: str, required_bytes: int) -> None:
        """Record conservative evidence and stop; never mutate width and retry."""

        if phase not in MEMORY_PHASES or required_bytes < 0:
            raise ValueError("invalid memory transaction failure evidence")
        self.terminal_failure = (
            f"{phase} transaction requires {required_bytes} bytes under a "
            f"{self.hard_ceiling_bytes}-byte hard ceiling; sampled iteration is not retried"
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "minimum_roots_per_gpu": self.minimum_roots_per_gpu,
            "maximum_roots_per_gpu": self.maximum_roots_per_gpu,
            "current_roots_per_gpu": self.current_roots_per_gpu,
            "device_bytes": self.device_bytes,
            "guard_bytes": self.guard_bytes,
            "growth_fraction": self.growth_fraction,
            "growth_delay": self.growth_delay,
            "successful_streak": self.successful_streak,
            "limiting_phase": self.limiting_phase,
            "limiting_field": self.limiting_field,
            "last_predicted_peaks": dict(self.last_predicted_peaks),
            "last_actual_peaks": dict(self.last_actual_peaks),
            "terminal_failure": self.terminal_failure,
            "forecaster": self.forecaster.state_dict(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "SequentialMemoryController":
        version = state.get("version")
        if version != SEQUENTIAL_MEMORY_CONTROLLER_VERSION:
            raise ValueError(
                "checkpoint sequential memory state requires an explicit v2 migration"
            )
        return cls(
            minimum_roots_per_gpu=int(state["minimum_roots_per_gpu"]),
            maximum_roots_per_gpu=int(state["maximum_roots_per_gpu"]),
            current_roots_per_gpu=int(state["current_roots_per_gpu"]),
            device_bytes=int(state["device_bytes"]),
            guard_bytes=int(state["guard_bytes"]),
            forecaster=PhaseMemoryForecaster.from_state_dict(state["forecaster"]),
            growth_fraction=float(state["growth_fraction"]),
            growth_delay=int(state["growth_delay"]),
            successful_streak=int(state.get("successful_streak", 0)),
            limiting_phase=state.get("limiting_phase"),
            limiting_field=state.get("limiting_field"),
            last_predicted_peaks={
                str(key): int(value)
                for key, value in state.get("last_predicted_peaks", {}).items()
            },
            last_actual_peaks={
                str(key): int(value)
                for key, value in state.get("last_actual_peaks", {}).items()
            },
            terminal_failure=state.get("terminal_failure"),
            version=str(version),
        )


ChooseSequentialRootWidth = SequentialMemoryController.choose_root_width
ObserveSequentialMemory = SequentialMemoryController.observe_success
