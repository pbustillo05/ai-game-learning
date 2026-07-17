"""Natural-width poker implementation of the game-neutral partition contract."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import torch
from torch import Tensor

from pbg.core.partition import (
    ByteWorkContract,
    CompletionSpec,
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
    public_input_bytes: Mapping[int, int] | None = None,
    completion: CompletionSpec | None = None,
) -> PartitionSet:
    """Build ``pre`` plus fixed-width ``post_A`` partitions.

    Partition ids are stable: preflop is zero and ``post_A`` is ``A-1``.
    Names remain diagnostics only.  Public inputs are represented as opaque
    immutable byte rows here; poker's model-input owner supplies their schema-
    conforming encoding.
    """

    if not 2 <= n_players <= 6:
        raise ValueError("poker n_players must lie in [2,6]")
    geometry = DeckGeometry(rank_count)
    exact_width = geometry.n_hands
    class_width = geometry.n_classes
    widths = {} if public_input_bytes is None else dict(public_input_bytes)

    def input_schema(partition_id: int, live: int, model_width: int) -> TensorSchema:
        fields = [
            TensorFieldSchema(
                "public", torch.uint8, (max(1, int(widths.get(partition_id, 1))),)
            ),
            TensorFieldSchema(
                "range_weights", torch.float32, (live, model_width), alignment=16
            ),
        ]
        if partition_id and live < n_players:
            fields.append(TensorFieldSchema("summary_anchor", torch.int32, fill=-1))
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
                ),
                input_schema=schema,
                preorder=preorder,
                completion=completion,
            )
        )
    return PartitionSet(tuple(specs))


def assert_poker_storage_widths(
    specs: PartitionSet,
    *,
    partition_id: int,
    model_rows: Tensor | None = None,
    exact_rows: Tensor | None = None,
) -> None:
    """Hard store/cache-boundary assertions used outside model internals."""

    spec = specs[partition_id]
    if model_rows is not None:
        spec.assert_model_width(model_rows)
    if exact_rows is not None:
        spec.assert_exact_width(exact_rows)
