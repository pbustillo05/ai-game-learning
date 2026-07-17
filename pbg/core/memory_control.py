"""CUDA memory accounting and policy-conditioned online forecasting.

The analytical ledger and the empirical forecast deliberately remain
separate.  The ledger reports what CUDA and PyTorch own now; the forecast
learns how policy-dependent rollout geometry changes later phase peaks.
"""

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


@dataclass
class MemoryObservation:
    """One rank-max, iteration-level controller observation.

    ``end_reserved_bytes`` is retained as a checkpoint-compatible field name;
    new observations store the quiescent iteration-start reservation there.
    """

    starts: int
    peak_reserved_bytes: float
    end_reserved_bytes: float
    features_per_start: dict[str, float]
    exceeded: bool
    restarted: bool = False

    def state_dict(self) -> dict[str, Any]:
        return {
            "starts": self.starts,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "end_reserved_bytes": self.end_reserved_bytes,
            "features_per_start": dict(self.features_per_start),
            "exceeded": self.exceeded,
            "restarted": self.restarted,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "MemoryObservation":
        return cls(
            starts=int(state["starts"]),
            peak_reserved_bytes=float(state["peak_reserved_bytes"]),
            end_reserved_bytes=float(state["end_reserved_bytes"]),
            features_per_start={
                str(key): float(value)
                for key, value in state.get("features_per_start", {}).items()
            },
            exceeded=bool(state.get("exceeded", False)),
            restarted=bool(state.get("restarted", False)),
        )


@dataclass
class PolicyMemoryForecaster:
    """Rolling conservative forecast for nonstationary policy workloads.

    A small nonnegative linear fit supplies the center.  A one-sided rolling
    residual supplies the safety envelope.  The hard CUDA limit is never
    learned or weakened by this class.
    """

    window: int = 128
    half_life: float = 32.0
    minimum_observations: int = 40
    soft_exceedance_rate: float = 0.025
    observations: deque[MemoryObservation] = field(default_factory=deque)
    coefficients: dict[str, float] = field(default_factory=dict)
    intercept_bytes: float = 0.0
    residual_bound_bytes: float = 0.0
    calibration_epoch: int = 0
    consecutive_positive: int = 0
    clean_since_drift: int = 0
    freeze_upward: bool = False

    def __post_init__(self) -> None:
        if self.window <= 0 or self.minimum_observations <= 0:
            raise ValueError("memory forecast windows must be positive")
        if not math.isfinite(self.half_life) or self.half_life <= 0:
            raise ValueError("memory forecast half-life must be positive")
        if not 0.0 < self.soft_exceedance_rate < 1.0:
            raise ValueError("memory soft exceedance rate must lie in (0,1)")
        self.observations = deque(self.observations, maxlen=self.window)

    @staticmethod
    def feature_vector(metrics: Mapping[str, Any], starts: int) -> dict[str, float]:
        """Extract causal workload features, normalized per trajectory start."""

        if starts <= 0:
            raise ValueError("memory observations require positive starts")
        selected: dict[str, float] = {"starts": 1.0}
        direct = ("nodes", "leaves", "decisions", "retained_target_bytes")
        for key in direct:
            value = metrics.get(f"{key}_rank_max", metrics.get(key))
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                selected[key] = max(0.0, float(value)) / starts
        for key, raw in metrics.items():
            if not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
                continue
            if key.startswith("trajectory_length_"):
                # Quantiles/maxima describe the per-start shape distribution;
                # histogram bins are counts and therefore normalized by N.
                selected[key] = max(0.0, float(raw)) / (
                    starts if "_le_" in key else 1.0
                )
                continue
            if key.startswith("execution_"):
                if key.endswith("_rank_max"):
                    continue
                if not any(
                    marker in key
                    for marker in (
                        "_model_calls",
                        "_model_rows",
                        "_padded_tokens",
                        "_padded_token_pairs",
                        "queue_depth_max",
                        "queue_oldest_age_max",
                        "forced_drains",
                        "completion_batches",
                        "completion_rows",
                    )
                ):
                    continue
                # Maxima are shape/workspace features and must not be divided by N.
                selected[key] = max(0.0, float(raw)) / (
                    1.0 if key.endswith("_max") else starts
                )
        return selected

    def _weights(self) -> Tensor:
        count = len(self.observations)
        ages = torch.arange(count - 1, -1, -1, dtype=torch.float64)
        return torch.pow(torch.tensor(0.5, dtype=torch.float64), ages / self.half_life)

    def _fit(self) -> None:
        if not self.observations:
            return
        keys = sorted(
            {
                key
                for observation in self.observations
                for key in observation.features_per_start
                if key != "starts"
            }
        )
        # Fit the incremental peak above the quiescent iteration-start
        # reservation.  End-of-iteration reserved bytes usually equal the peak
        # because the caching allocator retains bins, so using the end value
        # here would erase the workload signal entirely.
        y = torch.tensor(
            [
                max(0.0, item.peak_reserved_bytes - item.end_reserved_bytes)
                for item in self.observations
            ],
            dtype=torch.float64,
        )
        if not keys:
            self.intercept_bytes = max(
                item.end_reserved_bytes for item in self.observations
            )
            self.residual_bound_bytes = max(float(value) for value in y)
            return
        rows = []
        for item in self.observations:
            rows.append(
                [
                    item.starts * item.features_per_start.get(key, 0.0)
                    for key in keys
                ]
            )
        x = torch.tensor(rows, dtype=torch.float64)
        scale = x.abs().amax(0).clamp_min(1.0)
        xs = x / scale
        weights = self._weights()
        root_w = weights.sqrt()[:, None]
        # Projected gradient gives a dependency-free NNLS fit for this small
        # online diagnostic model.
        beta = torch.zeros(len(keys), dtype=torch.float64)
        spectral = float(torch.linalg.matrix_norm(root_w * xs, ord=2))
        step = 1.0 / max(spectral * spectral, 1.0)
        for _ in range(256):
            residual = xs @ beta - y
            gradient = xs.T @ (weights * residual)
            updated = (beta - step * gradient).clamp_min_(0.0)
            if float((updated - beta).abs().max()) < 1e-7:
                beta = updated
                break
            beta = updated
        beta /= scale
        self.coefficients = {
            key: float(value) for key, value in zip(keys, beta, strict=True)
        }
        self.intercept_bytes = max(
            item.end_reserved_bytes for item in self.observations
        )
        fitted = x @ beta
        positive = (y - fitted).clamp_min(0.0).sort().values
        if positive.numel():
            if positive.numel() < self.minimum_observations:
                candidate = float(positive[-1])
            else:
                quantile = 1.0 - self.soft_exceedance_rate
                index = min(
                    positive.numel() - 1,
                    max(0, math.ceil((positive.numel() + 1) * quantile) - 1),
                )
                candidate = float(positive[index])
            # Positive surprises widen immediately.  Tightening waits for a
            # full clean calibration epoch.
            if candidate >= self.residual_bound_bytes or (
                self.clean_since_drift >= self.minimum_observations
            ):
                self.residual_bound_bytes = candidate

    def predict(
        self,
        starts: int,
        *,
        realized_metrics: Mapping[str, Any] | None = None,
        baseline_bytes: float | None = None,
    ) -> float | None:
        if starts <= 0:
            raise ValueError("predicted starts must be positive")
        if not self.observations:
            return None
        weights = self._weights()
        total_weight = float(weights.sum())
        feature_mean: dict[str, float] = {}
        for index, item in enumerate(self.observations):
            for key, value in item.features_per_start.items():
                if key == "starts":
                    continue
                feature_mean[key] = feature_mean.get(key, 0.0) + float(weights[index]) * value
        realized = (
            None
            if realized_metrics is None
            else self.feature_vector(realized_metrics, starts)
        )
        predicted_increment = 0.0
        for key, coefficient in self.coefficients.items():
            feature = (
                realized.get(key)
                if realized is not None and key in realized
                else feature_mean.get(key, 0.0) / max(total_weight, 1e-12)
            )
            predicted_increment += (
                starts * feature
            ) * coefficient
        if not self.coefficients:
            # Before a stable fit, scale the largest observed incremental peak.
            predicted_increment = max(
                max(0.0, item.peak_reserved_bytes - item.end_reserved_bytes)
                * starts
                / item.starts
                for item in self.observations
            )
        baseline = self.intercept_bytes if baseline_bytes is None else baseline_bytes
        return baseline + predicted_increment + self.residual_bound_bytes

    def largest_safe_starts(self, minimum: int, maximum: int, ceiling: int) -> int:
        if minimum <= 0 or maximum < minimum or ceiling <= 0:
            raise ValueError("invalid safe-start search bounds")
        if not self.observations:
            return maximum
        low, high = minimum, maximum
        safe = minimum
        while low <= high:
            middle = (low + high) // 2
            predicted = self.predict(middle)
            if predicted is not None and predicted <= ceiling:
                safe = middle
                low = middle + 1
            else:
                high = middle - 1
        return safe

    def observe(
        self,
        *,
        starts: int,
        peak_reserved_bytes: float,
        end_reserved_bytes: float,
        metrics: Mapping[str, Any],
        ceiling_bytes: int,
        restarted: bool = False,
    ) -> dict[str, float]:
        previous = self.predict(starts)
        exceeded = previous is not None and peak_reserved_bytes > previous
        signed_residual = 0.0 if previous is None else peak_reserved_bytes - previous
        self.consecutive_positive = (
            self.consecutive_positive + 1 if signed_residual > 0 else 0
        )
        self.clean_since_drift += int(not exceeded and not restarted)
        self.observations.append(
            MemoryObservation(
                starts=starts,
                peak_reserved_bytes=float(peak_reserved_bytes),
                end_reserved_bytes=float(end_reserved_bytes),
                features_per_start=self.feature_vector(metrics, starts),
                exceeded=exceeded,
                restarted=bool(restarted),
            )
        )
        recent = list(self.observations)[-100:]
        exceedance_rate = sum(item.exceeded for item in recent) / max(len(recent), 1)
        restart_rate = sum(item.restarted for item in recent) / max(len(recent), 1)
        rate_signal = (
            (exceeded or restarted)
            and len(recent) >= self.minimum_observations
            and (
                exceedance_rate > self.soft_exceedance_rate
                or restart_rate > self.soft_exceedance_rate
            )
        )
        drift = not self.freeze_upward and (
            rate_signal or self.consecutive_positive >= 8
        )
        if drift:
            self.calibration_epoch += 1
            self.freeze_upward = True
            self.clean_since_drift = 0
            self.consecutive_positive = 0
        elif self.freeze_upward and self.clean_since_drift >= self.minimum_observations:
            self.freeze_upward = False
        self._fit()
        predicted_next = self.predict(starts)
        return {
            "memory_forecast_previous_bytes": float(previous or 0.0),
            "memory_forecast_current_bytes": float(predicted_next or 0.0),
            "memory_forecast_signed_residual_bytes": float(signed_residual),
            "memory_forecast_residual_bound_bytes": float(self.residual_bound_bytes),
            "memory_forecast_exceeded": float(exceeded),
            "memory_forecast_exceedance_rate": float(exceedance_rate),
            "memory_forecast_restart_rate": float(restart_rate),
            "memory_forecast_calibration_epoch": float(self.calibration_epoch),
            "memory_forecast_freeze_upward": float(self.freeze_upward),
            "memory_forecast_observations": float(len(self.observations)),
            "memory_forecast_ceiling_bytes": float(ceiling_bytes),
            "memory_forecast_drift": float(drift),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "half_life": self.half_life,
            "minimum_observations": self.minimum_observations,
            "soft_exceedance_rate": self.soft_exceedance_rate,
            "observations": [item.state_dict() for item in self.observations],
            "coefficients": dict(self.coefficients),
            "intercept_bytes": self.intercept_bytes,
            "residual_bound_bytes": self.residual_bound_bytes,
            "calibration_epoch": self.calibration_epoch,
            "consecutive_positive": self.consecutive_positive,
            "clean_since_drift": self.clean_since_drift,
            "freeze_upward": self.freeze_upward,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "PolicyMemoryForecaster":
        result = cls(
            window=int(state.get("window", 128)),
            half_life=float(state.get("half_life", 32.0)),
            minimum_observations=int(state.get("minimum_observations", 40)),
            soft_exceedance_rate=float(state.get("soft_exceedance_rate", 0.025)),
            observations=deque(
                MemoryObservation.from_state_dict(item)
                for item in state.get("observations", [])
            ),
            coefficients={
                str(key): float(value)
                for key, value in state.get("coefficients", {}).items()
            },
            intercept_bytes=float(state.get("intercept_bytes", 0.0)),
            residual_bound_bytes=float(state.get("residual_bound_bytes", 0.0)),
            calibration_epoch=int(state.get("calibration_epoch", 0)),
            consecutive_positive=int(state.get("consecutive_positive", 0)),
            clean_since_drift=int(state.get("clean_since_drift", 0)),
            freeze_upward=bool(state.get("freeze_upward", False)),
        )
        return result


# ---------------------------------------------------------------------------
# Phase-specific memory-only sequential root-width control (controller v2).
# The legacy policy-conditioned forecaster above remains checkpoint-readable;
# production root selection uses only the schema/phase model below.

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
            reduction_device = (
                torch.device("cuda", torch.cuda.current_device())
                if backend == "nccl"
                else torch.device("cpu")
            )
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
