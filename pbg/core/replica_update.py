"""Bounded fixed-rollout updates for independent replicas on one CUDA device."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from pbg.core.learner import (
    DistributedLearnerConfig,
    PhysicalPlanner,
    ReusableIndexBuffer,
    validate_learner_input,
)
from pbg.core.losses import _entropy_coefficient
from pbg.core.partition import PartitionSet
from pbg.core.replica_learner import (
    FunctionalReplicaLearner,
    ReplicaTransformUnsupported,
    StackedModuleReplicas,
    replica_forward,
    replica_grad_and_value,
)
from pbg.core.replica_optimizer import ReplicaOptimizerStepResult
from pbg.core.replica_replay import ReplicaModeLoss
from pbg.core.storage import TrainingPartition, TrainingStorage


@dataclass(frozen=True)
class ReplicaUpdateResult:
    """Per-lane outcomes of a complete replica optimizer call."""

    losses: Tensor
    steps: tuple[ReplicaOptimizerStepResult, ...]
    metrics: tuple[Mapping[str, float], ...]


def _stack_mappings(
    values: Sequence[Mapping[str, Tensor]], *, label: str
) -> dict[str, Tensor]:
    if not values:
        raise ValueError(f"{label} requires at least one replica mapping")
    names = set(values[0])
    if any(set(value) != names for value in values[1:]):
        raise ValueError(f"{label} fields differ across replicas")
    result: dict[str, Tensor] = {}
    for name in sorted(names):
        tensors = [value[name] for value in values]
        shape = tensors[0].shape
        if any(tensor.shape != shape for tensor in tensors[1:]):
            raise ValueError(
                f"{label} field {name!r} has incompatible replica shapes"
            )
        result[name] = torch.stack(tensors)
    return result


def _loss_pack(
    storages: Sequence[TrainingStorage],
    partitions: Sequence[TrainingPartition],
    rows: Sequence[Tensor],
    *,
    objective_scale: Tensor,
    entropy_coefficient: Tensor,
) -> dict[str, Tensor]:
    lane_packs: list[dict[str, Tensor]] = []
    for storage, partition, selected in zip(
        storages, partitions, rows, strict=True
    ):
        pack = {
            f"column.{name}": value[selected]
            for name, value in partition.columns.items()
        }
        if storage.mode == "sequential_rb":
            if storage.posterior_pool is None:
                raise ValueError("sequential RB replica omitted its posterior pool")
            posterior_ref = pack["column.posterior_ref"].to(torch.int64)
            pack["posterior"] = storage.posterior_pool.posterior[posterior_ref]
            pack["column.posterior_ref"] = torch.arange(
                selected.numel(),
                dtype=torch.int32,
                device=selected.device,
            )
        lane_packs.append(pack)
    stacked = _stack_mappings(lane_packs, label="replica loss pack")
    stacked["objective_scale"] = objective_scale
    stacked["entropy_coefficient"] = entropy_coefficient
    return stacked


class ReplicaFixedRolloutUpdate:
    """Consume $K$ sealed Normal/RB storages without crossing lane boundaries.

    Statistical indices are drawn from one generator and one reusable index
    buffer per lane. Matching stable subspans are then stacked only for the
    model/loss transform; targets, normalization, gradients, clipping, AdamW
    moments, finite gates, and counters retain their leading replica dimension.
    """

    def __init__(
        self,
        learner: FunctionalReplicaLearner,
        replica_partitions: Mapping[int, StackedModuleReplicas],
        partitions: PartitionSet,
        adapter: Any,
        loss_cfg: Any,
        config: DistributedLearnerConfig,
        *,
        reference_partitions: Mapping[int, StackedModuleReplicas] | None = None,
        statistical_counts_by_replica: Sequence[Mapping[int, int]] | None = None,
    ) -> None:
        self.learner = learner
        self.replica_partitions = dict(replica_partitions)
        self.partitions = partitions
        self.adapter = adapter
        self.loss_cfg = loss_cfg
        self.config = config
        self.reference_partitions = (
            {} if reference_partitions is None else dict(reference_partitions)
        )
        self.replicas = learner.replicas.replicas
        self.device = next(iter(learner.replicas.parameters.values())).device
        if self.device.type != "cuda":
            raise ValueError("replica fixed-rollout updates are CUDA-only")
        expected = {spec.partition_id for spec in partitions.specs}
        if set(self.replica_partitions) != expected:
            raise ValueError("replica model views must cover every partition")
        if float(getattr(loss_cfg, "c_kl", 0.0)) > 0.0:
            if set(self.reference_partitions) != expected:
                raise ValueError(
                    "positive reference KL requires every replica reference view"
                )
        for label, views in (
            ("learner", self.replica_partitions),
            ("reference", self.reference_partitions),
        ):
            for partition_id, view in views.items():
                if view.replicas != self.replicas:
                    raise ValueError(f"{label} partition {partition_id} has wrong K")
                if label == "learner":
                    for name, parameter in learner.replicas.parameters.items():
                        if view.parameters.get(name) is not parameter:
                            raise ValueError(
                                "learner partition views must share stacked parameters"
                            )
        self._planners = {
            partition_id: PhysicalPlanner(shapes)
            for partition_id, shapes in config.physical_shapes.items()
        }
        if set(self._planners) != expected:
            raise ValueError("physical shapes must cover every partition")
        if set(config.statistical_counts) != expected:
            raise ValueError("statistical counts must cover every partition")
        if statistical_counts_by_replica is None:
            self.statistical_counts = tuple(
                dict(config.statistical_counts) for _ in range(self.replicas)
            )
        else:
            if len(statistical_counts_by_replica) != self.replicas:
                raise ValueError("statistical count plans must have one entry per lane")
            self.statistical_counts = tuple(
                {int(key): int(value) for key, value in counts.items()}
                for counts in statistical_counts_by_replica
            )
            if any(set(counts) != expected for counts in self.statistical_counts):
                raise ValueError("every lane count plan must cover every partition")
            if any(
                value <= 0
                for counts in self.statistical_counts
                for value in counts.values()
            ):
                raise ValueError("replica statistical counts must be positive")
        capacity = max(
            value for counts in self.statistical_counts for value in counts.values()
        )
        self._index_buffers = [
            ReusableIndexBuffer(capacity, self.device)
            for _ in range(self.replicas)
        ]

    def _validate(
        self,
        storages: Sequence[TrainingStorage],
        generators: Sequence[torch.Generator],
        iterations: Sequence[int],
        optimizer_steps: int,
    ) -> str:
        if len(storages) != self.replicas:
            raise ValueError("replica update requires exactly K storages")
        if len(generators) != self.replicas:
            raise ValueError("replica update requires exactly K generators")
        if len(iterations) != self.replicas:
            raise ValueError("replica update requires exactly K logical iterations")
        if optimizer_steps <= 0:
            raise ValueError("optimizer_steps must be positive")
        modes = {storage.mode for storage in storages}
        if len(modes) != 1:
            raise ValueError("one physical replica update cannot mix training modes")
        mode = next(iter(modes))
        if mode not in {"normal", "sequential_rb"}:
            raise ValueError("replica update supports only Normal and sequential RB")
        for storage in storages:
            validate_learner_input(storage, self.partitions)
        for generator in generators:
            if generator.device.type != "cuda":
                raise ValueError("replica statistical generators must be CUDA generators")
        return mode

    @staticmethod
    def _alphas(
        storages: Sequence[TrainingStorage], partitions: PartitionSet
    ) -> Tensor:
        device = next(iter(storages[0].sampling.log_z.values())).device
        logs = torch.stack(
            [
                torch.stack(
                    [
                        storage.sampling.log_z.get(
                            spec.partition_id,
                            torch.tensor(
                                -torch.inf, dtype=torch.float64, device=device
                            ),
                        ).to(device=device, dtype=torch.float64)
                        for spec in partitions.specs
                    ]
                )
                for storage in storages
            ]
        )
        totals = torch.logsumexp(logs, dim=1)
        if not bool(torch.isfinite(totals).all()):
            raise ValueError("every replica fixed rollout must have positive mass")
        return torch.where(
            torch.isfinite(logs),
            torch.exp(logs - totals[:, None]),
            torch.zeros_like(logs),
        ).to(torch.float32)

    def __call__(
        self,
        storages: Sequence[TrainingStorage],
        generators: Sequence[torch.Generator],
        *,
        iterations: Sequence[int],
        optimizer_steps: int = 1,
    ) -> ReplicaUpdateResult:
        mode = self._validate(storages, generators, iterations, optimizer_steps)
        alphas = self._alphas(storages, self.partitions)
        entropy = torch.tensor(
            [
                _entropy_coefficient(self.loss_cfg, int(iteration))
                for iteration in iterations
            ],
            dtype=torch.float32,
            device=self.device,
        )
        lane_metrics = [
            {
                "statistical_rows": 0.0,
                "physical_subspans": 0.0,
                "model_rows": 0.0,
                "reference_rows": 0.0,
                "exact_objective_mass": 0.0,
            }
            for _ in range(self.replicas)
        ]
        step_losses: list[Tensor] = []
        outcomes: list[ReplicaOptimizerStepResult] = []
        c_kl = float(getattr(self.loss_cfg, "c_kl", 0.0))

        for _step_index in range(optimizer_steps):
            accumulated: dict[str, Tensor] | None = None
            losses = torch.zeros(
                self.replicas, dtype=torch.float32, device=self.device
            )
            for position, spec in enumerate(self.partitions.specs):
                partition_id = spec.partition_id
                counts = [
                    lane_counts[partition_id]
                    for lane_counts in self.statistical_counts
                ]
                actual_partitions = [
                    storage.partitions.get(partition_id) for storage in storages
                ]
                active = [
                    partition is not None and partition.inputs.rows > 0
                    for partition in actual_partitions
                ]
                if not any(active):
                    continue
                template_lane = active.index(True)
                template_partition = actual_partitions[template_lane]
                assert template_partition is not None
                template_storage = storages[template_lane]
                selected: list[Tensor] = []
                lane_partitions: list[TrainingPartition] = []
                work_storages: list[TrainingStorage] = []
                for lane, (storage, generator) in enumerate(
                    zip(storages, generators, strict=True)
                ):
                    partition = actual_partitions[lane]
                    if active[lane]:
                        assert partition is not None
                        indices = self._index_buffers[lane].draw_uniform(
                            partition.inputs.rows,
                            counts[lane],
                            generator,
                        ).clone()
                        lane_partitions.append(partition)
                        work_storages.append(storage)
                    else:
                        # This lane contributes zero objective mass. Reusing a
                        # valid peer pack is physical padding only: no index is
                        # drawn, no retained row is admitted, and scale[k]=0.
                        indices = torch.zeros(
                            0, dtype=torch.int32, device=self.device
                        )
                        lane_partitions.append(template_partition)
                        work_storages.append(template_storage)
                    selected.append(indices.clone())
                lane_spans = [
                    (
                        self._planners[partition_id].plan(
                            indices, self.config.available_transient_bytes
                        )
                        if indices.numel()
                        else ()
                    )
                    for indices in selected
                ]
                for lane, metrics in enumerate(lane_metrics):
                    if active[lane]:
                        metrics["statistical_rows"] += float(counts[lane])
                        metrics["physical_subspans"] += float(
                            len(lane_spans[lane])
                        )
                for span_index in range(max(map(len, lane_spans))):
                    natural_rows = [
                        (
                            lane_spans[lane][span_index]
                            .select(selected[lane])
                            .to(torch.int64)
                            if span_index < len(lane_spans[lane])
                            else torch.empty(
                                0, dtype=torch.int64, device=self.device
                            )
                        )
                        for lane in range(self.replicas)
                    ]
                    batch_rows = max(row.numel() for row in natural_rows)
                    template_row = next(
                        row[:1] for row in natural_rows if row.numel()
                    )
                    rows = []
                    row_weights = []
                    for lane, row in enumerate(natural_rows):
                        actual = row.numel()
                        if actual < batch_rows:
                            lane_padding = (
                                row[:1]
                                if actual
                                else selected[lane][:1]
                                if selected[lane].numel()
                                else template_row
                            )
                            padding = lane_padding.expand(batch_rows - actual)
                            row = torch.cat((row, padding))
                        rows.append(row)
                        row_weights.append(
                            torch.cat(
                                (
                                    torch.ones(
                                        actual,
                                        dtype=torch.float32,
                                        device=self.device,
                                    ),
                                    torch.zeros(
                                        batch_rows - actual,
                                        dtype=torch.float32,
                                        device=self.device,
                                    ),
                                )
                            )
                        )
                    model_inputs = _stack_mappings(
                        [
                            partition.gather_inputs(lane_rows)
                            for partition, lane_rows in zip(
                                lane_partitions, rows, strict=True
                            )
                        ],
                        label=f"partition {partition_id} model inputs",
                    )
                    scale = alphas[:, position] / torch.tensor(
                        counts,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    scale = scale * torch.tensor(
                        active, dtype=torch.float32, device=self.device
                    )
                    pack = _loss_pack(
                        work_storages,
                        lane_partitions,
                        rows,
                        objective_scale=scale,
                        entropy_coefficient=entropy,
                    )
                    pack["row_weight"] = torch.stack(row_weights)
                    if c_kl > 0.0:
                        with torch.autocast(
                            device_type="cuda",
                            enabled=self.config.amp_enabled,
                            dtype=self.config.amp_dtype,
                        ):
                            reference = replica_forward(
                                self.reference_partitions[partition_id],
                                (model_inputs,),
                            )
                        pack.update(
                            {
                                f"reference.{name}": value
                                for name, value in reference.items()
                            }
                        )
                    loss_fn = ReplicaModeLoss(
                        mode=mode,
                        spec=spec,
                        adapter=self.adapter,
                        loss_cfg=self.loss_cfg,
                        iteration=0,
                    )
                    with torch.autocast(
                        device_type="cuda",
                        enabled=self.config.amp_enabled,
                        dtype=self.config.amp_dtype,
                    ):
                        gradients, contribution = replica_grad_and_value(
                            self.replica_partitions[partition_id],
                            loss_fn,
                            (model_inputs,),
                            (pack,),
                        )
                    if accumulated is None:
                        accumulated = gradients
                    else:
                        for name, gradient in gradients.items():
                            accumulated[name].add_(gradient)
                    losses.add_(contribution.detach().to(losses))
                    for lane, metrics in enumerate(lane_metrics):
                        actual_rows = natural_rows[lane].numel()
                        if actual_rows:
                            metrics["model_rows"] += float(actual_rows)
                            if c_kl > 0.0:
                                metrics["reference_rows"] += float(actual_rows)
                    del gradients, contribution, pack, model_inputs, rows
            if accumulated is None:
                raise RuntimeError("replica update produced no learner work")
            outcomes.append(
                self.learner.optimizer.step(
                    accumulated, grad_clip=self.learner.grad_clip
                )
            )
            step_losses.append(losses)

        for lane, metrics in enumerate(lane_metrics):
            metrics["exact_objective_mass"] = float(
                optimizer_steps * alphas[lane].sum()
            )
            metrics["optimizer_steps_applied"] = float(
                sum(bool(outcome.optimizer_step_applied[lane]) for outcome in outcomes)
            )
            metrics["optimizer_steps_nonfinite"] = float(
                sum(not bool(outcome.gradients_finite[lane]) for outcome in outcomes)
            )
            metrics["grad_norm_pre_clip_last"] = float(
                outcomes[-1].grad_norm_pre_clip[lane]
            )
            metrics["grad_clip_scale_last"] = float(
                outcomes[-1].grad_clip_scale[lane]
            )
        return ReplicaUpdateResult(
            losses=torch.stack(step_losses),
            steps=tuple(outcomes),
            metrics=tuple(lane_metrics),
        )


__all__ = [
    "ReplicaFixedRolloutUpdate",
    "ReplicaTransformUnsupported",
    "ReplicaUpdateResult",
]
