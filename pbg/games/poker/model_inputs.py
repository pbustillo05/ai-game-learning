"""Device-resident construction of poker model inputs."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from pbg.games.poker.model.preflop import PackedPublicHistory


class DeviceHistoryMaterializer:
    """Gather lossless public histories through a binary-lift ancestor table.

    CUDA batches use the configured bounded width so tensor shape construction
    never reads a device scalar. CPU batches trim trailing padding because the
    CPU tensor itself is already host-resident. Ancestor traversal is logarithmic
    in the configured bound and batched across every row and history position.
    """

    _FLOAT_FIELDS = (
        ("amount", "history_amount"),
        ("pot_before", "history_pot_before"),
        ("call_amount", "history_call_amount"),
        ("current_bet", "history_current_bet"),
        ("min_raise_to", "history_min_raise_to"),
        ("max_raise_to", "history_max_raise_to"),
    )

    def __init__(self, store: Any) -> None:
        self.store = store
        self.device = store.device
        self.max_history_tokens = int(store.max_rollout_steps)

    def _width(self, lengths: Tensor, batch_size: int) -> int:
        if batch_size == 0:
            return 0
        if self.device.type == "cuda":
            return self.max_history_tokens
        return int(lengths.max())

    def pack(self, states: Tensor) -> PackedPublicHistory:
        """Return chronological padded histories for state handles ``[B]``."""

        rows = states.to(device=self.device, dtype=torch.int64)
        lengths = self.store.history_length[rows].to(torch.int64)
        width = self._width(lengths, states.numel())
        positions = torch.arange(width, device=self.device)[None, :]
        mask = positions < lengths[:, None]
        distances = lengths[:, None] - 1 - positions
        ancestors = self.store.history_ref[rows].to(torch.int32)[:, None].expand(
            -1, width
        )

        # Follow every requested ancestor in parallel. The loop count depends
        # only on the configured bound (at most eleven for the active domain),
        # never on a realized trajectory length.
        for level in range(self.store.history_jump.shape[1]):
            follow = mask & (((distances.clamp_min(0) >> level) & 1) != 0)
            safe = ancestors.clamp_min(0).to(torch.int64)
            parent = self.store.history_jump[safe, level]
            ancestors = torch.where(follow, parent, ancestors)

        safe = ancestors.clamp_min(0).to(torch.int64)

        def integer_field(name: str) -> Tensor:
            value = getattr(self.store, name)[safe].to(torch.int64)
            return value.masked_fill(~mask, -1)

        floats = {
            target: getattr(self.store, source)[safe].masked_fill(~mask, 0.0)
            for target, source in self._FLOAT_FIELDS
        }
        return PackedPublicHistory(
            actor=integer_field("history_actor"),
            street=integer_field("history_street"),
            action=integer_field("history_action"),
            mask=mask,
            **floats,
        )
