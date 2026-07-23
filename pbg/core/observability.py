"""Lossless execution counters shared by rollout and learner stages.

Production execution needs aggregate counters whose values do not depend on a
bounded diagnostic history.  This module deliberately stores only additive
totals and high-water marks; detailed traces belong in an explicitly enabled
profiler rather than in the correctness/benchmark accounting path.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
import re
from typing import Mapping


_ADDITIVE_COUNTERS = (
    "model_calls",
    "model_rows",
    "policy_pack_calls",
    "policy_pack_rows",
    "policy_pack_padded_tokens",
    "policy_pack_padded_token_pairs",
    "padded_tokens",
    "padded_token_pairs",
    "forced_drains",
    "forced_drain_rows",
    "evaluator_calls",
    "evaluator_rows",
    "evaluator_rejection_work",
    "host_synchronizations",
    "host_synchronization_seconds",
    "scheduler_control_synchronizations",
    "scheduler_control_synchronization_seconds",
    "scheduler_control_bytes",
    "undeclared_host_synchronizations",
    "h2d_transfers",
    "h2d_bytes",
    "d2h_transfers",
    "d2h_bytes",
    "target_bytes",
    "collective_calls",
    "collective_bytes",
    "collective_seconds",
    "collective_exposed_seconds",
    "storage_direct_accesses",
    "storage_contiguous_accesses",
    "storage_generic_accesses",
    "storage_segments_visited",
    "storage_rows_requested",
    "storage_rows_moved",
    "storage_candidate_rows",
    "storage_routing_bytes",
    "storage_routing_tensor_operations",
)

_MAXIMUM_COUNTERS = (
    "model_rows_max",
    "padded_tokens_max",
    "padded_token_pairs_max",
    "queue_depth_max",
    "queue_oldest_age_max",
    "allocated_bytes_max",
    "reserved_bytes_max",
    "target_bytes_max",
    "state_live_rows_high_water",
    "state_live_rows_at_seal",
    "history_live_rows_high_water",
    "history_live_rows_at_seal",
    "storage_segments_per_access_max",
    "storage_candidate_rows_per_access_max",
    "storage_routing_bytes_per_access_max",
)


def _metric_component(value: str) -> str:
    """Return a stable metric-safe spelling for a phase or family name."""

    component = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    if not component:
        raise ValueError("metric phase/family names must contain an alphanumeric character")
    return component


@dataclass
class ExecutionMetrics:
    """Exact aggregate execution counters with optional phase/family strata.

    Values are ordinary host numbers because updates occur at already-declared
    execution boundaries (model launch, queue mutation, completion launch, and
    collective completion).  No samples are retained or evicted, so a snapshot
    always describes the complete interval since :meth:`clear`.
    """

    _totals: dict[str, int | float] = field(default_factory=dict, init=False)
    _maxima: dict[str, int | float] = field(default_factory=dict, init=False)
    _strata: dict[str, dict[str, int | float]] = field(
        default_factory=lambda: defaultdict(dict), init=False
    )
    _batch_histogram: Counter[tuple[str, str, int]] = field(
        default_factory=Counter, init=False
    )
    _enabled: bool = field(default=True, init=False)

    def clear(self) -> None:
        """Start a new accounting interval without changing the schema."""

        self._totals.clear()
        self._maxima.clear()
        self._strata.clear()
        self._batch_histogram.clear()

    def add(
        self,
        name: str,
        value: int | float = 1,
        *,
        stratum: str | None = None,
    ) -> None:
        """Add a nonnegative finite amount to one exact counter."""

        if not self._enabled:
            return
        if name not in _ADDITIVE_COUNTERS:
            raise KeyError(f"unknown additive execution counter {name!r}")
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError("execution counter increments must be nonnegative numbers")
        destination = self._totals if stratum is None else self._strata[stratum]
        destination[name] = destination.get(name, 0) + value

    def maximum(
        self,
        name: str,
        value: int | float,
        *,
        stratum: str | None = None,
    ) -> None:
        """Observe one nonnegative high-water value."""

        if not self._enabled:
            return
        if name not in _MAXIMUM_COUNTERS:
            raise KeyError(f"unknown maximum execution counter {name!r}")
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError("execution maxima must be nonnegative numbers")
        destination = self._maxima if stratum is None else self._strata[stratum]
        destination[name] = max(destination.get(name, 0), value)

    def record_model_call(
        self,
        phase: str,
        family: str,
        rows: int,
        *,
        padded_tokens: int,
        padded_token_pairs: int,
    ) -> None:
        """Record one actual family-model invocation."""

        if not self._enabled:
            return
        values = (rows, padded_tokens, padded_token_pairs)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise ValueError("model-call work must use nonnegative integers")
        if rows == 0:
            raise ValueError("an actual model call must contain at least one row")
        phase_key = _metric_component(phase)
        family_key = _metric_component(family)
        stratum = f"{phase_key}_{family_key}"
        for destination in (None, stratum):
            self.add("model_calls", 1, stratum=destination)
            self.add("model_rows", rows, stratum=destination)
            self.add("padded_tokens", padded_tokens, stratum=destination)
            self.add("padded_token_pairs", padded_token_pairs, stratum=destination)
            self.maximum("model_rows_max", rows, stratum=destination)
            self.maximum("padded_tokens_max", padded_tokens, stratum=destination)
            self.maximum(
                "padded_token_pairs_max", padded_token_pairs, stratum=destination
            )
        self._batch_histogram[(phase_key, family_key, rows)] += 1

    @contextmanager
    def suspended(self):
        """Exclude diagnostic-only work from production throughput counters."""

        previous = self._enabled
        self._enabled = False
        try:
            yield
        finally:
            self._enabled = previous

    def record_policy_pack(
        self,
        phase: str,
        family: str,
        *,
        rows: int,
        padded_tokens: int,
        padded_token_pairs: int,
    ) -> None:
        """Record one calibrated physical pack and its exact padded work."""

        values = (rows, padded_tokens, padded_token_pairs)
        if any(isinstance(value, bool) or value < 0 for value in values):
            raise ValueError("policy-pack counts must be nonnegative integers")
        stratum = f"{_metric_component(phase)}_{_metric_component(family)}"
        for destination in (None, stratum):
            self.add("policy_pack_calls", 1, stratum=destination)
            self.add("policy_pack_rows", rows, stratum=destination)
            self.add(
                "policy_pack_padded_tokens", padded_tokens, stratum=destination
            )
            self.add(
                "policy_pack_padded_token_pairs",
                padded_token_pairs,
                stratum=destination,
            )

    def record_queue(self, family: str, *, depth: int, oldest_age: float) -> None:
        """Observe aggregate and per-family queue high-water state."""

        stratum = f"queue_{_metric_component(family)}"
        for destination in (None, stratum):
            self.maximum("queue_depth_max", depth, stratum=destination)
            self.maximum("queue_oldest_age_max", oldest_age, stratum=destination)

    def record_forced_drain(self, family: str, *, rows: int) -> None:
        """Record one partial family drain used to guarantee progress."""

        stratum = f"rollout_{_metric_component(family)}"
        for destination in (None, stratum):
            self.add("forced_drains", 1, stratum=destination)
            self.add("forced_drain_rows", rows, stratum=destination)

    def record_evaluator(self, *, rows: int, rejection_work: int = 0) -> None:
        self.add("evaluator_calls", 1)
        self.add("evaluator_rows", rows)
        self.add("evaluator_rejection_work", rejection_work)

    def record_host_synchronization(
        self, *, seconds: float = 0.0, source: str | None = None
    ) -> None:
        """Record an undeclared synchronization discovered in a hot path."""

        destinations = (None,) if source is None else (
            None,
            f"host_sync_{_metric_component(source)}",
        )
        for destination in destinations:
            self.add("host_synchronizations", 1, stratum=destination)
            self.add("host_synchronization_seconds", seconds, stratum=destination)
            self.add("undeclared_host_synchronizations", 1, stratum=destination)

    def record_scheduler_control_synchronization(
        self, *, seconds: float, nbytes: int
    ) -> None:
        """Record the one permitted fixed-size scheduler control transfer."""

        if seconds < 0 or isinstance(nbytes, bool) or nbytes < 0:
            raise ValueError("scheduler control synchronization values are invalid")
        for destination in (None, "scheduler_control"):
            self.add("host_synchronizations", 1, stratum=destination)
            self.add("host_synchronization_seconds", seconds, stratum=destination)
            self.add("scheduler_control_synchronizations", 1, stratum=destination)
            self.add(
                "scheduler_control_synchronization_seconds",
                seconds,
                stratum=destination,
            )
            self.add("scheduler_control_bytes", nbytes, stratum=destination)
        self.record_transfer("d2h", nbytes=nbytes)

    def record_transfer(self, direction: str, *, nbytes: int) -> None:
        """Record one declared host/device transfer and its exact payload."""

        if direction not in {"h2d", "d2h"}:
            raise ValueError("transfer direction must be 'h2d' or 'd2h'")
        if isinstance(nbytes, bool) or nbytes < 0:
            raise ValueError("transfer bytes must be nonnegative")
        self.add(f"{direction}_transfers", 1)
        self.add(f"{direction}_bytes", nbytes)

    def record_target_bytes(self, value: int) -> None:
        self.add("target_bytes", value)
        self.maximum("target_bytes_max", value)

    def record_storage_access(
        self,
        owner: str,
        segment_kind: str,
        operation: str,
        path: str,
        *,
        segments_visited: int,
        rows_requested: int,
        rows_moved: int,
        candidate_rows: int,
        routing_bytes: int,
        tensor_operations: int,
    ) -> None:
        """Record one page/chunk access without inspecting device values.

        The caller supplies host-known physical extents and the tensor
        operations issued by its routing implementation.  This accounting
        therefore adds no CUDA synchronization to the hot path.  Actual
        scheduler/host synchronization time remains in the dedicated
        synchronization counters.
        """

        if segment_kind not in {"page", "chunk"}:
            raise ValueError("storage segment kind must be 'page' or 'chunk'")
        if operation not in {"read", "write", "resolve"}:
            raise ValueError("storage operation must be read, write, or resolve")
        if path not in {"direct", "contiguous", "generic"}:
            raise ValueError("storage path must be direct, contiguous, or generic")
        values = (
            segments_visited,
            rows_requested,
            rows_moved,
            candidate_rows,
            routing_bytes,
            tensor_operations,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise ValueError("storage access values must be nonnegative integers")
        stratum = "storage_{}_{}_{}_{}".format(
            _metric_component(owner),
            segment_kind,
            operation,
            path,
        )
        for destination in (None, stratum):
            self.add(f"storage_{path}_accesses", 1, stratum=destination)
            self.add(
                "storage_segments_visited",
                segments_visited,
                stratum=destination,
            )
            self.add(
                "storage_rows_requested",
                rows_requested,
                stratum=destination,
            )
            self.add("storage_rows_moved", rows_moved, stratum=destination)
            self.add(
                "storage_candidate_rows",
                candidate_rows,
                stratum=destination,
            )
            self.add("storage_routing_bytes", routing_bytes, stratum=destination)
            self.add(
                "storage_routing_tensor_operations",
                tensor_operations,
                stratum=destination,
            )
            self.maximum(
                "storage_segments_per_access_max",
                segments_visited,
                stratum=destination,
            )
            self.maximum(
                "storage_candidate_rows_per_access_max",
                candidate_rows,
                stratum=destination,
            )
            self.maximum(
                "storage_routing_bytes_per_access_max",
                routing_bytes,
                stratum=destination,
            )

    def record_collective(
        self, *, nbytes: int, seconds: float, exposed_seconds: float = 0.0
    ) -> None:
        self.add("collective_calls", 1)
        self.add("collective_bytes", nbytes)
        self.add("collective_seconds", seconds)
        self.add("collective_exposed_seconds", exposed_seconds)

    def snapshot(self, *, prefix: str = "execution_") -> dict[str, float]:
        """Return a complete scalar snapshot, including zero-valued base fields."""

        result = {
            f"{prefix}{name}": float(self._totals.get(name, 0))
            for name in _ADDITIVE_COUNTERS
        }
        result.update(
            {
                f"{prefix}{name}": float(self._maxima.get(name, 0))
                for name in _MAXIMUM_COUNTERS
            }
        )
        for raw_stratum, values in sorted(self._strata.items()):
            stratum = _metric_component(raw_stratum)
            for name, value in sorted(values.items()):
                result[f"{prefix}{stratum}_{name}"] = float(value)
        for (phase, family, rows), calls in sorted(self._batch_histogram.items()):
            result[
                f"{prefix}{phase}_{family}_batch_size_{rows}_calls"
            ] = float(calls)
        return result

    def totals(self) -> Mapping[str, int | float]:
        """Expose a read-only copy for precise unit-test and debug assertions."""

        return dict(self._totals)
