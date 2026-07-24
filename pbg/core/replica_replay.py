"""Exact Normal/sequential-RB loss replay for replica-batched learner packs."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from pbg.core.learner import mode_loss
from pbg.core.partition import PartitionSpec
from pbg.games.poker.adapter import PokerPolicyAdapter
from pbg.games.poker.model.preflop import PolicyOutput


class _PrevalidatedPokerAdapter:
    """Expose ordinary adapter math after pack validation ran outside vmap."""

    def __init__(self, adapter: PokerPolicyAdapter) -> None:
        self.adapter = adapter

    def policy_terms(self, *arguments: Any):
        return self.adapter.policy_terms_prevalidated(*arguments)

    def log_prob(self, *arguments: Any):
        return self.adapter.log_prob_prevalidated(*arguments)

    def entropy(self, *arguments: Any):
        return self.adapter.entropy_prevalidated(*arguments)

    def kl(self, *arguments: Any):
        return self.adapter.kl_prevalidated(*arguments)


def _columns(pack: Mapping[str, Tensor]) -> dict[str, Tensor]:
    prefix = "column."
    return {
        name.removeprefix(prefix): value
        for name, value in pack.items()
        if name.startswith(prefix)
    }


def _reference_output(pack: Mapping[str, Tensor]) -> PolicyOutput | None:
    raw = {
        name.removeprefix("reference.raw."): value
        for name, value in pack.items()
        if name.startswith("reference.raw.")
    }
    value = pack.get("reference.v")
    if not raw and value is None:
        return None
    if not raw or value is None:
        raise ValueError("replica reference output pack is incomplete")
    return PolicyOutput(
        raw=raw,
        v=value,
        t=pack.get("reference.t"),
        c=pack.get("reference.c"),
    )


class ReplicaModeLoss:
    """Callable exact loss used by :func:`replica_grad_and_value`.

    The pack contains only the already gathered physical subspan. Its
    ``objective_scale`` is the existing alpha_m / B_m normalization, so
    padding never affects the objective.
    """

    def __init__(
        self,
        *,
        mode: str,
        spec: PartitionSpec,
        adapter: PokerPolicyAdapter,
        loss_cfg: Any,
        iteration: int,
    ) -> None:
        if mode not in {"normal", "sequential_rb"}:
            raise ValueError("replica replay currently supports Normal and sequential RB")
        self.mode = mode
        self.spec = spec
        self.adapter = _PrevalidatedPokerAdapter(adapter)
        self.loss_cfg = loss_cfg
        self.iteration = int(iteration)

    def prepare_replica_loss_arguments(
        self, loss_arguments: Sequence[Any]
    ) -> tuple[Any, ...]:
        if len(loss_arguments) != 1 or not isinstance(
            loss_arguments[0], Mapping
        ):
            raise TypeError("replica mode loss expects one tensor pack mapping")
        pack = loss_arguments[0]
        tensors = tuple(pack.values())
        if not tensors or any(not isinstance(value, Tensor) for value in tensors):
            raise TypeError("replica loss pack values must be tensors")
        replicas = tensors[0].shape[0]
        if replicas <= 0 or any(value.shape[0] != replicas for value in tensors):
            raise ValueError("replica loss pack requires one common leading K")

        legal_ids = pack["column.legal_event_ids"]
        legal_mask = pack["column.legal_event_mask"]
        event = pack["column.event"]
        if (
            legal_ids.ndim != 3
            or legal_mask.shape != legal_ids.shape
            or legal_mask.dtype != torch.bool
        ):
            raise ValueError("replica legal ids/mask must have shape [K,B,E]")
        if event.shape != legal_ids.shape[:2]:
            raise ValueError("replica selected events must have shape [K,B]")
        torch._assert_async(
            legal_mask.any(-1).all(),
            "every replica decision must supply a legal event",
        )
        torch._assert_async(
            (
                (
                    legal_ids
                    == event[..., None].to(legal_ids.dtype)
                )
                & legal_mask
            )
            .any(-1)
            .all(),
            "every replica event must select a supplied legal event",
        )
        if self.mode == "sequential_rb":
            posterior = pack.get("posterior")
            if posterior is None or posterior.ndim != 3:
                raise ValueError("sequential RB replica pack requires [K,Q,X] posterior")
            torch._assert_async(
                (
                    torch.isfinite(posterior)
                    & (posterior >= 0)
                ).all(),
                "replica posteriors must be finite and nonnegative",
            )
            torch._assert_async(
                (posterior.sum(-1) > 0).all(),
                "replica posterior rows must have positive mass",
            )
        has_reference = any(name.startswith("reference.") for name in pack)
        if float(getattr(self.loss_cfg, "c_kl", 0.0)) > 0.0 and not has_reference:
            raise ValueError("positive replica reference KL requires reference output")
        if has_reference:
            reference = _reference_output(pack)
            assert reference is not None
            expected = legal_ids.shape[:2]
            leaves = (*reference.raw.values(), reference.v)
            if reference.t is not None:
                leaves = (*leaves, reference.t)
            if reference.c is not None:
                leaves = (*leaves, reference.c)
            if any(value.shape[:2] != expected for value in leaves):
                raise ValueError(
                    "replica reference output must be aligned [K,B,...]"
                )
        return (pack,)

    def __call__(
        self,
        output: Any,
        _model_inputs: Mapping[str, Tensor],
        pack: Mapping[str, Tensor],
    ) -> Tensor:
        columns = _columns(pack)
        first = next(iter(columns.values()))
        rows = torch.arange(first.shape[0], device=first.device)
        partition = SimpleNamespace(
            columns=columns,
            action_columns=None,
            max_actions_per_decision=None,
        )
        posterior = pack.get("posterior")
        storage = SimpleNamespace(
            mode=self.mode,
            posterior_pool=(
                None
                if posterior is None
                else SimpleNamespace(posterior=posterior)
            ),
        )
        result = mode_loss(
            self.mode,
            self.spec,
            storage,
            partition,
            rows,
            output,
            self.adapter,
            self.loss_cfg,
            iteration=self.iteration,
            reference=_reference_output(pack),
            prevalidated=True,
            entropy_coefficient_override=pack.get("entropy_coefficient"),
        )
        scale = pack["objective_scale"].to(result.per_decision)
        row_weight = pack.get("row_weight")
        if row_weight is None:
            return scale * result.per_decision.sum()
        return scale * (
            row_weight.to(result.per_decision) * result.per_decision
        ).sum()


__all__ = ["ReplicaModeLoss"]
