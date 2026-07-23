"""Natural-width poker implementation of the game-neutral partition contract."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import torch
from torch import Tensor

from pbg.core.partition import (
    ByteWorkContract,
    CompletionSpec,
    PhysicalPack,
    PartitionSet,
    PartitionSpec,
)
from pbg.core.storage import TensorFieldSchema, TensorSchema
from pbg.games.poker.cards import DeckGeometry


def poker_partition_specs(
    *,
    rank_count: int,
    n_players: int,
    router: Callable[[Any], int | Tensor],
    forwards: Mapping[int, Callable[[Mapping[str, Tensor]], Any]],
    max_history_tokens: int = 1,
    n_ticks: int = 1,
    e_max: int = 3,
    public_input_bytes: Mapping[int, int] | None = None,
    pack_planners: Mapping[int, Callable[[Any], PhysicalPack]] | None = None,
    completion: CompletionSpec | None = None,
) -> PartitionSet:
    """Build ``pre`` plus fixed-width ``post_A`` partitions.

    Partition ids are stable: preflop is zero and ``post_A`` is ``A-1``.
    Names remain diagnostics only. Every public/history field required by a
    model forward is immutable row-owned data; no handle into a mutable rollout
    state store crosses the seal boundary. Range rows are stored only on the
    model's natural domain. Postflop folded summaries are referenced through an
    exact immutable pool and are never duplicated per decision.
    """

    if not 2 <= n_players <= 6:
        raise ValueError("poker n_players must lie in [2,6]")
    geometry = DeckGeometry(rank_count)
    exact_width = geometry.n_hands
    class_width = geometry.n_classes
    widths = {} if public_input_bytes is None else dict(public_input_bytes)
    planners = {} if pack_planners is None else dict(pack_planners)
    if max_history_tokens <= 0 or n_ticks <= 0 or e_max < 3:
        raise ValueError("poker input geometry must be positive")

    def input_schema(partition_id: int, live: int, model_width: int) -> TensorSchema:
        fields = [
            TensorFieldSchema("street", torch.int32, alignment=4),
            TensorFieldSchema("board", torch.int16, (5,), fill=-1, alignment=2),
            TensorFieldSchema("pot", torch.float32, alignment=4),
            TensorFieldSchema("stacks", torch.float32, (n_players,), alignment=16),
            TensorFieldSchema("committed", torch.float32, (n_players,), alignment=16),
            TensorFieldSchema("to_act", torch.int32, alignment=4),
            TensorFieldSchema("legality_summary", torch.float32, (3,), alignment=4),
            TensorFieldSchema("n_ticks", torch.int32, alignment=4),
            TensorFieldSchema("raise_grid_bounds", torch.float32, (2,), alignment=4),
            TensorFieldSchema("status", torch.int8, (n_players,)),
            TensorFieldSchema("seat_mask", torch.bool, (n_players,)),
            TensorFieldSchema("button", torch.int32, alignment=4),
            TensorFieldSchema("committed_street", torch.float32, (n_players,), alignment=16),
            TensorFieldSchema("call_amount", torch.float32, alignment=4),
            TensorFieldSchema("current_bet", torch.float32, alignment=4),
            TensorFieldSchema("min_raise_to", torch.float32, alignment=4),
            TensorFieldSchema("max_raise_to", torch.float32, alignment=4),
            TensorFieldSchema("last_full_raise", torch.float32, alignment=4),
            TensorFieldSchema("legal_action_mask", torch.bool, (e_max,)),
            TensorFieldSchema("raise_ticks", torch.float32, (n_ticks,), alignment=4),
            TensorFieldSchema("history_actor", torch.int8, (max_history_tokens,), fill=-1),
            TensorFieldSchema("history_street", torch.int8, (max_history_tokens,), fill=-1),
            TensorFieldSchema("history_action", torch.int8, (max_history_tokens,), fill=-1),
            TensorFieldSchema("history_mask", torch.bool, (max_history_tokens,)),
            TensorFieldSchema(
                "range_weights", torch.float32, (live, model_width), alignment=16
            ),
        ]
        for name in (
            "amount",
            "pot_before",
            "call_amount",
            "current_bet",
            "min_raise_to",
            "max_raise_to",
        ):
            fields.append(
                TensorFieldSchema(
                    f"history_{name}",
                    torch.float32,
                    (max_history_tokens,),
                    alignment=16,
                )
            )
        if partition_id and live < n_players:
            fields.append(TensorFieldSchema("folded_summary_ref", torch.int32, fill=-1))
        return TensorSchema(f"poker_partition_{partition_id}_input", tuple(fields))

    specs = []
    for partition_id, live, model_width, mapping, name, preorder in (
        (
            0,
            n_players,
            class_width,
            tuple(geometry.class_of_combo),
            "pre",
            0,
        ),
        *(
            (
                live - 1,
                live,
                exact_width,
                tuple(range(exact_width)),
                f"post_{live}",
                1,
            )
            for live in range(2, n_players + 1)
        ),
    ):
        schema = input_schema(partition_id, live, model_width)
        try:
            forward = forwards[partition_id]
        except KeyError as exc:
            raise KeyError(f"missing poker forward for partition {partition_id}") from exc
        specs.append(
            PartitionSpec(
                partition_id=partition_id,
                name=name,
                router=router,
                exact_width=exact_width,
                model_width=model_width,
                exact_to_model=mapping,
                forward=forward,
                byte_work=ByteWorkContract(
                    retained_per_row={"natural_input": schema.row_payload_bytes},
                    transient_per_pack={"singleton_output": model_width * 4},
                    max_rows=2**31 - 1,
                    pack=planners.get(partition_id),
                ),
                input_schema=schema,
                preorder=preorder,
                completion=completion,
            )
        )
    return PartitionSet(tuple(specs))
