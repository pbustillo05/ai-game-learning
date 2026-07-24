"""Prevalidated functional poker partitions for replica-batched learners."""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from pbg.games.poker.adapter import PokerModelBundle
from pbg.games.poker.cards import DeckGeometry
from pbg.games.poker.folded import PartitionSummaryBatch
from pbg.games.poker.model.incidence import CompactMap
from pbg.games.poker.model.postflop import PostflopMasks
from pbg.games.poker.model.preflop import (
    PackedPublicHistory,
    PokerPublicBatch,
    PreflopMasks,
)


_PREPARED = "__replica_prepared_"


def _public_from_inputs(inputs: Mapping[str, Tensor]) -> PokerPublicBatch:
    history = PackedPublicHistory(
        actor=inputs["history_actor"].to(torch.int64),
        street=inputs["history_street"].to(torch.int64),
        action=inputs["history_action"].to(torch.int64),
        amount=inputs["history_amount"].to(torch.float32),
        pot_before=inputs["history_pot_before"].to(torch.float32),
        call_amount=inputs["history_call_amount"].to(torch.float32),
        current_bet=inputs["history_current_bet"].to(torch.float32),
        min_raise_to=inputs["history_min_raise_to"].to(torch.float32),
        max_raise_to=inputs["history_max_raise_to"].to(torch.float32),
        mask=inputs["history_mask"].to(torch.bool),
    )
    return PokerPublicBatch(
        street=inputs["street"].to(torch.int64),
        board=inputs["board"].to(torch.int64),
        pot=inputs["pot"].to(torch.float32),
        stacks=inputs["stacks"].to(torch.float32),
        committed=inputs["committed"].to(torch.float32),
        to_act=inputs["to_act"].to(torch.int64),
        legality_summary=inputs["legality_summary"].to(torch.float32),
        n_ticks=inputs["n_ticks"].to(torch.int64),
        raise_grid_bounds=inputs["raise_grid_bounds"].to(torch.float32),
        status=inputs["status"].to(torch.int64),
        seat_mask=inputs["seat_mask"].to(torch.bool),
        button=inputs["button"].to(torch.int64),
        committed_street=inputs["committed_street"].to(torch.float32),
        call_amount=inputs["call_amount"].to(torch.float32),
        current_bet=inputs["current_bet"].to(torch.float32),
        min_raise_to=inputs["min_raise_to"].to(torch.float32),
        max_raise_to=inputs["max_raise_to"].to(torch.float32),
        last_full_raise=inputs["last_full_raise"].to(torch.float32),
        legal_action_mask=inputs["legal_action_mask"].to(torch.bool),
        raise_ticks=inputs["raise_ticks"].to(torch.float32),
        history=history,
    )


def _summary_from_inputs(
    inputs: Mapping[str, Tensor],
) -> PartitionSummaryBatch | None:
    fields = {
        name.removeprefix("folded_summary."): value
        for name, value in inputs.items()
        if name.startswith("folded_summary.")
    }
    if not fields:
        return None
    return PartitionSummaryBatch(
        fields=fields,
        metadata={"cache": "sealed_sparse_anchor"},
    )


def _lane(inputs: Mapping[str, Tensor], lane: int) -> dict[str, Tensor]:
    return {name: value[lane] for name, value in inputs.items()}


class PokerReplicaPartition(nn.Module):
    """One fixed semantic family over a shared poker model bundle.

    ``prepare_replica_arguments`` executes parameter-free validation outside
    ``vmap``. ``forward`` accepts only the resulting marked mapping and runs
    the same model tensor math without repeating side-effecting assertions.
    """

    def __init__(
        self,
        bundle: PokerModelBundle,
        *,
        partition_id: int,
        n_players: int,
        rank_count: int,
    ) -> None:
        super().__init__()
        if not 0 <= partition_id < n_players:
            raise ValueError("poker replica partition id is outside [0,P)")
        self.bundle = bundle
        self.partition_id = int(partition_id)
        self.n_players = int(n_players)
        geometry = DeckGeometry(rank_count)
        if bundle.geometry != geometry:
            raise ValueError("replica partition geometry differs from its model")
        self.n_hands = geometry.n_hands
        self.n_classes = geometry.n_classes
        self.n_cards = geometry.n_cards

    def with_partition(self, partition_id: int) -> PokerReplicaPartition:
        """Return a code-only family view with the identical bundle layout."""

        if not 0 <= partition_id < self.n_players:
            raise ValueError("poker replica partition id is outside [0,P)")
        result = copy.deepcopy(self)
        result.partition_id = int(partition_id)
        return result

    @staticmethod
    def _pre_metadata(inputs: Mapping[str, Tensor]) -> tuple[Tensor, ...]:
        return tuple(inputs[f"{_PREPARED}pre_{index}"] for index in range(6))

    @staticmethod
    def _post_metadata(
        inputs: Mapping[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, CompactMap]:
        mapping = CompactMap(
            compact_to_full=inputs[f"{_PREPARED}compact_to_full"],
            full_to_compact=inputs[f"{_PREPARED}full_to_compact"],
            valid=inputs[f"{_PREPARED}compact_valid"],
        )
        return (
            inputs[f"{_PREPARED}post_actor"],
            inputs[f"{_PREPARED}post_seat_mask"],
            inputs[f"{_PREPARED}post_live_ids"],
            inputs[f"{_PREPARED}post_live_valid"],
            inputs[f"{_PREPARED}post_status"],
            inputs[f"{_PREPARED}post_combo_mask"],
            mapping,
        )

    def forward(self, inputs: Mapping[str, Tensor]):
        if f"{_PREPARED}validated" not in inputs:
            raise ValueError(
                "replica poker forward requires prepare_replica_arguments"
            )
        public = _public_from_inputs(inputs)
        if self.partition_id == 0:
            natural = inputs["range_weights"].to(torch.float32)
            masks = PreflopMasks(
                seat_mask=public.seat_mask,
                status=public.status,
            )
            return self.bundle.preflop.forward_prevalidated(
                public,
                natural,
                self._pre_metadata(inputs),
                masks,
            )

        weights = inputs[f"{_PREPARED}post_weights"]
        masks = PostflopMasks(
            compatibility=inputs[f"{_PREPARED}compatibility"],
            seat_mask=public.seat_mask,
            combo_mask=inputs[f"{_PREPARED}post_combo_mask"],
            status=public.status,
            evaluator_summary=_summary_from_inputs(inputs),
            live_seat_ids=inputs[f"{_PREPARED}post_live_ids"],
            compact_size=math.comb(self.n_cards - 3, 2),
        )
        return self.bundle.postflop.forward_prevalidated(
            public,
            weights,
            masks,
            self._post_metadata(inputs),
        )

    def prepare_replica_arguments(
        self,
        parameters: Mapping[str, Tensor],
        buffers: Mapping[str, Tensor],
        model_arguments: Sequence[Any],
    ) -> tuple[Any, ...]:
        """Validate each lane and stack parameter-free forward metadata."""

        del parameters
        if len(model_arguments) != 1 or not isinstance(
            model_arguments[0], Mapping
        ):
            raise TypeError("poker replica partition expects one input mapping")
        stacked = model_arguments[0]
        tensors = tuple(stacked.values())
        if not tensors or any(not isinstance(value, Tensor) for value in tensors):
            raise TypeError("poker replica inputs must be tensors")
        replicas = tensors[0].shape[0]
        if replicas <= 0 or any(value.shape[0] != replicas for value in tensors):
            raise ValueError("poker replica inputs require one common leading K")

        prepared = dict(stacked)
        prepared[f"{_PREPARED}validated"] = torch.ones(
            replicas,
            dtype=torch.bool,
            device=tensors[0].device,
        )
        if self.partition_id == 0:
            lane_metadata = []
            for lane_index in range(replicas):
                values = _lane(stacked, lane_index)
                public = _public_from_inputs(values)
                natural = values["range_weights"]
                expected = (
                    natural.shape[0],
                    self.n_players,
                    self.n_classes,
                )
                if tuple(natural.shape) != expected:
                    raise ValueError(
                        f"preflop natural ranges must have shape {expected}"
                    )
                torch._assert_async(
                    (public.street == 0).all(),
                    "preflop partition contains a postflop state",
                )
                lane_metadata.append(
                    self.bundle.preflop.prepare_forward(
                        public,
                        natural.to(torch.float32),
                        public.to_act,
                        PreflopMasks(
                            seat_mask=public.seat_mask,
                            status=public.status,
                        ),
                    )
                )
            for index in range(6):
                prepared[f"{_PREPARED}pre_{index}"] = torch.stack(
                    [metadata[index] for metadata in lane_metadata]
                )
            return (prepared,)

        live = self.partition_id + 1
        combo_cards = buffers.get("bundle.postflop.combo_cards")
        if combo_cards is None:
            raise KeyError("stacked poker model omitted postflop combo geometry")
        combo_cards = combo_cards[0]
        lane_metadata = []
        lane_weights = []
        lane_compatibility = []
        for lane_index in range(replicas):
            values = _lane(stacked, lane_index)
            public = _public_from_inputs(values)
            natural = values["range_weights"]
            expected = (natural.shape[0], live, self.n_hands)
            if tuple(natural.shape) != expected:
                raise ValueError(
                    f"postflop natural ranges must have shape {expected}"
                )
            status = public.status
            if status is None:
                raise ValueError("postflop sealed inputs require status")
            actual_live = (status != 1).sum(1)
            torch._assert_async(
                ((public.street != 0) & (actual_live == live)).all(),
                "postflop partition router/domain mismatch",
            )
            live_ids = torch.where(
                status != 1,
                torch.arange(self.n_players, device=natural.device)[None],
                self.n_players,
            ).sort(1).values[:, :live]
            weights = torch.ones(
                (natural.shape[0], self.n_players, self.n_hands),
                dtype=torch.float32,
                device=natural.device,
            )
            batch = torch.arange(natural.shape[0], device=natural.device)[:, None]
            weights[batch, live_ids] = natural.to(torch.float32)
            real = public.board >= 0
            conflict = (
                (
                    combo_cards[None, :, :, None]
                    == public.board[:, None, None, :]
                )
                & real[:, None, None, :]
            ).any(3).any(2)
            compatibility = ~conflict
            summary = _summary_from_inputs(values)
            if (
                self.bundle.postflop.requires_partition_summary
                and summary is None
                and live < self.n_players
            ):
                raise ValueError(
                    "postflop immutable input omitted folded summary rows"
                )
            masks = PostflopMasks(
                compatibility=compatibility,
                seat_mask=public.seat_mask,
                combo_mask=weights > 0,
                status=public.status,
                evaluator_summary=summary,
                live_seat_ids=live_ids,
                compact_size=math.comb(self.n_cards - 3, 2),
            )
            lane_metadata.append(
                self.bundle.postflop.prepare_forward(
                    public,
                    weights,
                    public.to_act,
                    masks,
                )
            )
            lane_weights.append(weights)
            lane_compatibility.append(compatibility)

        prepared[f"{_PREPARED}post_weights"] = torch.stack(lane_weights)
        prepared[f"{_PREPARED}compatibility"] = torch.stack(
            lane_compatibility
        )
        names = (
            "post_actor",
            "post_seat_mask",
            "post_live_ids",
            "post_live_valid",
            "post_status",
            "post_combo_mask",
        )
        for index, name in enumerate(names):
            prepared[f"{_PREPARED}{name}"] = torch.stack(
                [metadata[index] for metadata in lane_metadata]
            )
        mappings = [metadata[6] for metadata in lane_metadata]
        prepared[f"{_PREPARED}compact_to_full"] = torch.stack(
            [mapping.compact_to_full for mapping in mappings]
        )
        prepared[f"{_PREPARED}full_to_compact"] = torch.stack(
            [mapping.full_to_compact for mapping in mappings]
        )
        prepared[f"{_PREPARED}compact_valid"] = torch.stack(
            [mapping.valid for mapping in mappings]
        )
        return (prepared,)


__all__ = ["PokerReplicaPartition"]
