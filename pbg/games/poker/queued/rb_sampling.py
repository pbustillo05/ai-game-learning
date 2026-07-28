"""Repository-owned collision-conditioned sampling for branched RB poker."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from pbg.games.poker.queued.hooks import NLHEGeometry, PAD_CARD
from pbg.core.rollout import DrawPurpose, logical_rng_keys, logical_uniform


class PosteriorSamplingError(RuntimeError):
    """A posterior-profile request violated the bounded CUDA contract."""


@dataclass(frozen=True)
class RejectionObservation:
    """Device-resident rejection-tail telemetry for one sampler call."""

    attempt_count: Tensor
    exhausted: Tensor
    minimum_failing_address: Tensor


def _fixed_upper_bound(
    cdf: Tensor,
    target: Tensor,
    source: Tensor,
    steps: int,
) -> Tensor:
    """First CDF entry strictly above target using a fixed search schedule."""

    if cdf.ndim != 3 or target.ndim != 3:
        raise PosteriorSamplingError(
            "CDF search requires [G,P,H] and [G,A,P]"
        )
    parents, players, width = cdf.shape
    rows = target.shape[0]
    if target.shape[2] != players or source.shape != (rows,):
        raise PosteriorSamplingError("CDF targets are not row aligned")
    lo = torch.zeros_like(target, dtype=torch.int64)
    hi = torch.full_like(lo, width)
    parent = source[:, None, None]
    seat = torch.arange(
        players, dtype=torch.int64, device=cdf.device
    )[None, None, :]
    for _ in range(steps):
        mid = (lo + hi) >> 1
        safe = mid.clamp_max(width - 1)
        below = (mid < width) & cdf[parent, seat, safe].le(target)
        lo = torch.where(below, mid + 1, lo)
        hi = torch.where(below, hi, mid)
    return lo.clamp_max(width - 1)


class CollisionConditionedProfileSampler:
    """Exact all-or-reject posterior sampler with logical CUDA addresses.

    Proposals are independent seat-wise categorical draws.  Accepting exactly
    the collision-free proposals yields the joint collision-conditioned law.
    The complete bounded attempt envelope is evaluated on CUDA, so subdivision
    and physical scheduling cannot change the selected first success.
    """

    def __init__(
        self,
        geometry: NLHEGeometry,
        *,
        max_attempts: int = 256,
    ) -> None:
        if (
            isinstance(max_attempts, bool)
            or not isinstance(max_attempts, int)
            or max_attempts <= 0
        ):
            raise ValueError("max_attempts must be a positive integer")
        self.geometry = geometry
        self.max_attempts = max_attempts
        self.last_observation: RejectionObservation | None = None

    def sample_children(
        self,
        *,
        range_weights: Tensor,
        board_cards: Tensor,
        inherited_profile: Tensor,
        source: Tensor,
        ordinal: Tensor,
        active: Tensor,
        root: Tensor,
        path: Tensor,
        seed: int,
        iteration: int,
        retry_epoch: Tensor | int,
    ) -> Tensor:
        """Return one generating profile for every fixed-envelope child row.

        ``ordinal == 0`` is the inherited child and consumes no posterior draw.
        Inactive envelope rows also consume no address and retain their source
        profile only as non-published scratch.
        """

        device = range_weights.device
        players = self.geometry.players
        hands = self.geometry.private_hands
        if device.type != "cuda":
            raise PosteriorSamplingError("branched RB sampling requires CUDA")
        if (
            range_weights.dtype != torch.float32
            or range_weights.ndim != 3
            or range_weights.shape[1:] != (players, hands)
            or board_cards.shape != (range_weights.shape[0], 5)
            or board_cards.dtype != torch.uint8
            or inherited_profile.shape != (range_weights.shape[0], players)
            or inherited_profile.dtype != torch.int32
        ):
            raise PosteriorSamplingError(
                "posterior parents require FP32 [B,P,H], uint8 [B,5], "
                "and int32 [B,P]"
            )
        edges = source.numel()
        if any(
            value.device != device or value.shape != (edges,)
            for value in (source, ordinal, active, root, path)
        ):
            raise PosteriorSamplingError(
                "posterior child descriptors must be device-local [G]"
            )
        if active.dtype != torch.bool:
            raise PosteriorSamplingError("posterior active mask must be bool")
        torch._assert_async(
            (
                source.ge(0)
                & source.lt(range_weights.shape[0])
                & ordinal.ge(0)
            ).all(),
            "posterior source or child ordinal is invalid",
        )
        source64 = source.to(torch.int64)
        inherited = inherited_profile.index_select(0, source64)
        fresh = active & ordinal.gt(0)

        weights = range_weights.float()
        board = board_cards
        combo_ids = torch.arange(
            hands, dtype=torch.int32, device=device
        )
        combo_cards = self._combo_cards(combo_ids)
        board_conflict = (
            (
                board.ne(PAD_CARD)[..., None, None]
                & board.long()[..., None, None].eq(combo_cards[None])
            )
            .any(1)
            .any(-1)
        )
        weights = weights.masked_fill(board_conflict[:, None], 0.0)
        mass = weights.sum(-1)
        torch._assert_async(
            (
                ~fresh[:, None]
                | mass.index_select(0, source64).gt(0)
            ).all(),
            "posterior seat has no board-legal mass",
        )
        cdf = weights.cumsum(-1)

        attempts = torch.arange(
            self.max_attempts, dtype=torch.int64, device=device
        )
        seats = torch.arange(players, dtype=torch.int64, device=device)
        edge_grid, attempt_grid, seat_grid = torch.meshgrid(
            torch.arange(edges, dtype=torch.int64, device=device),
            attempts,
            seats,
            indexing="ij",
        )
        proposal_active = fresh[:, None, None].expand_as(edge_grid)
        keys = logical_rng_keys(
            seed=seed,
            iteration=iteration,
            root=root[:, None, None].expand_as(edge_grid),
            path=path[:, None, None].expand_as(edge_grid),
            draw=attempt_grid,
            purpose=DrawPurpose.POSTERIOR,
            lane=seat_grid,
            retry_epoch=retry_epoch,
            active=proposal_active,
        )
        totals = (
            mass.index_select(0, source64)[:, None]
            .expand(edges, self.max_attempts, players)
        )
        target = torch.minimum(
            logical_uniform(keys) * totals,
            torch.nextafter(totals, torch.zeros_like(totals)),
        )
        selected = _fixed_upper_bound(
            cdf,
            target,
            source64,
            11,
        )
        selected = selected.to(torch.int32)

        cards = combo_cards[selected.to(torch.int64)]
        # A profile has at most twelve private cards (six two-card seats).
        # Pairwise hand comparisons are therefore both exact and materially
        # smaller than an [edge, attempt, deck-card] collision histogram.
        collision_free = torch.ones(
            (edges, self.max_attempts),
            dtype=torch.bool,
            device=device,
        )
        for first_seat in range(players):
            first_cards = cards[:, :, first_seat]
            for second_seat in range(first_seat + 1, players):
                second_cards = cards[:, :, second_seat]
                collision_free &= ~(
                    first_cards[:, :, 0, None]
                    .eq(second_cards)
                    .any(-1)
                    | first_cards[:, :, 1, None]
                    .eq(second_cards)
                    .any(-1)
                )
        accepted = collision_free & fresh[:, None]
        has_success = accepted.any(1)
        first = accepted.to(torch.int64).argmax(1)
        row = torch.arange(edges, dtype=torch.int64, device=device)
        sampled = selected[row, first]
        result = torch.where(fresh[:, None], sampled, inherited)

        attempt_count = torch.where(
            fresh,
            first + 1,
            torch.zeros_like(first),
        ).to(torch.int32)
        exhausted = fresh & ~has_success
        attempt_count = torch.where(
            exhausted,
            torch.full_like(
                attempt_count, self.max_attempts
            ),
            attempt_count,
        )
        result = torch.where(
            exhausted[:, None], inherited, result
        )
        max_address = torch.full_like(root, torch.iinfo(torch.int64).max)
        minimum_failing = torch.where(exhausted, path, max_address).amin()
        self.last_observation = RejectionObservation(
            attempt_count,
            exhausted,
            minimum_failing,
        )
        return result.contiguous()

    def _combo_cards(self, combo_ids: Tensor) -> Tensor:
        deck = self.geometry.deck_cards
        ids = combo_ids.to(torch.int64)
        first = torch.zeros_like(ids)
        remaining = ids.clone()
        unresolved = torch.ones_like(ids, dtype=torch.bool)
        for card in range(deck - 1):
            count = deck - card - 1
            advance = unresolved & remaining.ge(count)
            first = torch.where(advance, first + 1, first)
            remaining = torch.where(
                advance, remaining - count, remaining
            )
            unresolved &= advance
        second = first + remaining + 1
        return torch.stack((first, second), -1).add(
            self.geometry.first_card
        ).to(torch.int64)


__all__ = [
    "CollisionConditionedProfileSampler",
    "PosteriorSamplingError",
    "RejectionObservation",
]
