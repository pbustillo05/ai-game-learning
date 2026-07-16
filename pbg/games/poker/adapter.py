"""Poker public navigation and factorized policy adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from pbg.games.poker.cards import sample_uniform_complement
from pbg.games.poker.engine import (
    ALL_IN,
    CHECK_CALL,
    FOLD,
    FOLDED,
    LIVE,
    RAISE_OFFSET,
    NLHEEngine,
)
from pbg.games.poker.folded import PokerGPUPartitionAdapter
from pbg.games.poker.mc import PokerGPURejectionSampler, PokerGPUShowdownEvaluator
from pbg.games.poker.model.heads import dlogistic_tick
from pbg.games.poker.model.postflop import PostflopMasks, PostflopNet
from pbg.games.poker.model.preflop import (
    PackedPublicHistory,
    PokerPublicBatch,
    PolicyOutput,
    PreflopMasks,
    PreflopNet,
)
from pbg.games.poker.ranges import RangeState


def _raw_mapping(output: PolicyOutput | Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    return output.raw if isinstance(output, PolicyOutput) else output


def _require_device(condition: Tensor, message: str) -> None:
    """Validate a scalar condition without synchronizing CUDA to the host."""

    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


class PokerPolicyAdapter:
    """Compose contextual action types and conditional raise ticks."""

    def __init__(self, n_ticks: int | None = None, eps: float = 1e-12) -> None:
        if n_ticks is not None and n_ticks <= 0:
            raise ValueError("n_ticks must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.n_ticks = n_ticks
        self.eps = float(eps)

    @staticmethod
    def select_actor_row(
        output: PolicyOutput | Mapping[str, Tensor], profiles: Tensor, actor: Tensor
    ) -> dict[str, Tensor]:
        """Select exactly the actor combo stipulated by each fresh profile."""

        raw = _raw_mapping(output)
        if profiles.ndim == 3:
            if profiles.shape[1] != 1:
                raise ValueError("profiles must contain one fresh profile per spot")
            profiles = profiles[:, 0]
        if profiles.ndim != 2 or actor.shape != (profiles.shape[0],):
            raise ValueError("profiles and actor must have shapes [B,P] and [B]")
        actor = actor.to(device=profiles.device, dtype=torch.long)
        combo = profiles.gather(1, actor[:, None]).squeeze(1).long()
        raw_widths = {value.shape[1] for value in raw.values() if value.ndim >= 2}
        if len(raw_widths) != 1:
            raise ValueError("raw policy fields must share one compact hand width")
        n_hands = next(iter(raw_widths))
        _require_device(
            ((combo >= 0) & (combo < n_hands)).all(),
            "the stipulated actor combo is invalid",
        )
        batch_ids = torch.arange(profiles.shape[0], device=profiles.device)
        selected: dict[str, Tensor] = {}
        for name, value in raw.items():
            if value.shape[:2] != (profiles.shape[0], n_hands):
                raise ValueError(
                    f"raw field {name} must begin [B,{n_hands}]"
                )
            selected[name] = value[batch_ids.to(value.device), combo.to(value.device)]
        return selected

    def probabilities(
        self,
        raw: PolicyOutput | Mapping[str, Tensor],
        legal_event_ids: Tensor,
        legal_mask: Tensor,
    ) -> Tensor:
        """Return normalized probabilities over the supplied event columns."""

        fields = _raw_mapping(raw)
        required = {"type_logits", "omega", "mu", "s", "iota"}
        if not required.issubset(fields):
            raise ValueError("raw policy output is missing factorized head fields")
        type_logits = fields["type_logits"]
        if type_logits.ndim < 2 or type_logits.shape[-1] != 3:
            raise ValueError("type_logits must have shape [B,...,3]")
        batch = type_logits.shape[0]
        if legal_event_ids.shape != legal_mask.shape or legal_event_ids.ndim != 2:
            raise ValueError("legal event ids and masks must have shape [B,E]")
        if legal_event_ids.shape[0] != batch or legal_mask.dtype != torch.bool:
            raise TypeError("legal mask must be bool and batch-aligned")
        # Poker backends expose the complete contiguous
        # fold/check-call/raise-tick row, so its static width determines the
        # number of raise ticks without synchronizing a device scalar.
        inferred = legal_event_ids.shape[1] - RAISE_OFFSET
        n_ticks = self.n_ticks if self.n_ticks is not None else inferred
        if n_ticks <= 0:
            raise ValueError("the legal event set has no raise-tick width")
        type_logits = type_logits.float()
        _require_device(
            torch.stack(
                [torch.isfinite(fields[name]).all() for name in sorted(required)]
            ).all(),
            "raw policy fields must be finite",
        )
        type_probability = torch.softmax(type_logits, dim=-1)
        prefix = type_logits.shape[:-1]
        omega = fields["omega"].float()
        mu = fields["mu"].float()
        scale = fields["s"].float()
        iota = fields["iota"].float()
        logistic = dlogistic_tick(n_ticks, mu, scale)
        mixture = (omega[..., None] * logistic).sum(dim=-2)
        raise_mask = legal_mask[:, RAISE_OFFSET:].to(type_logits.device)
        mask_shape = (batch,) + (1,) * (mixture.ndim - 2) + (n_ticks,)
        expanded_raise_mask = raise_mask.reshape(mask_shape).expand_as(mixture)
        mixture = mixture.masked_fill(~expanded_raise_mask, 0.0)
        mixture = mixture / mixture.sum(-1, keepdim=True).clamp_min(self.eps)
        tick_index = torch.arange(n_ticks, device=type_logits.device)[None]
        allin_index = torch.where(
            raise_mask,
            tick_index,
            torch.full_like(tick_index, -1),
        ).max(1).values
        allin = torch.zeros((batch, n_ticks), dtype=torch.float32, device=type_logits.device)
        has_raise = allin_index >= 0
        allin.scatter_(
            1,
            allin_index.clamp_min(0)[:, None],
            has_raise.to(torch.float32)[:, None],
        )
        allin = allin.reshape(mask_shape).expand_as(mixture)
        ticks = (1.0 - iota[..., None]) * mixture + iota[..., None] * allin
        ticks = ticks.masked_fill(~expanded_raise_mask, 0.0)

        # Preserve the ordinary probability-space path exactly.  This keeps
        # seeded rollout behavior unchanged except on the zero-mass corner
        # which previously raised a device-side assertion.
        full = torch.zeros(
            prefix + (RAISE_OFFSET + n_ticks,),
            dtype=torch.float32,
            device=type_logits.device,
        )
        facing_bet = ((legal_event_ids == FOLD) & legal_mask).any(dim=1)
        facing = facing_bet.reshape((batch,) + (1,) * (type_logits.ndim - 2))
        full[..., FOLD] = torch.where(facing, type_probability[..., 0], 0.0)
        full[..., CHECK_CALL] = torch.where(
            facing, type_probability[..., 1], type_probability[..., 0]
        )
        full[..., RAISE_OFFSET:] = type_probability[..., 2, None] * ticks

        # Condition on the currently legal raise grid before taking logs.  In
        # exact arithmetic the mixture already has positive mass on every
        # interval, but extreme learned locations/scales can underflow an
        # entire legal tail in fp32.  A uniform conditional fallback is used
        # only for that zero-mass numerical corner.
        tick_mass = ticks.sum(-1, keepdim=True)
        legal_raise_count = expanded_raise_mask.sum(-1, keepdim=True)
        uniform_legal_raise = expanded_raise_mask.to(ticks) / legal_raise_count.clamp_min(1)
        stable_ticks = torch.where(
            (tick_mass > 0) | (legal_raise_count == 0),
            ticks / tick_mass.clamp_min(self.eps),
            uniform_legal_raise,
        )

        # Build unnormalized event *logits* and softmax only over supplied
        # legal events.  The previous probability-space implementation first
        # softmaxed all three action types, then discarded illegal types.  If
        # an illegal type exceeded every legal type by roughly 100 logits,
        # every supplied legal probability underflowed to exactly zero.
        full_logits = torch.full(
            prefix + (RAISE_OFFSET + n_ticks,),
            -torch.inf,
            dtype=torch.float32,
            device=type_logits.device,
        )
        full_logits[..., FOLD] = type_logits[..., 0]
        full_logits[..., CHECK_CALL] = torch.where(
            facing, type_logits[..., 1], type_logits[..., 0]
        )
        tick_logits = torch.where(
            stable_ticks > 0,
            torch.log(stable_ticks.clamp_min(self.eps)),
            torch.full_like(stable_ticks, -torch.inf),
        )
        full_logits[..., RAISE_OFFSET:] = type_logits[..., 2, None] + tick_logits
        ids = legal_event_ids.to(device=full_logits.device, dtype=torch.long)
        _require_device(
            ((ids >= 0) & (ids < full_logits.shape[-1])).all(),
            "legal event id is outside the factorized event space",
        )
        event_shape = (batch,) + (1,) * (type_logits.ndim - 2) + (ids.shape[1],)
        ids = ids.reshape(event_shape).expand(prefix + (ids.shape[1],))
        mask = legal_mask.to(full_logits.device).reshape(event_shape).expand_as(ids)
        _require_device(
            mask.any(-1).all(),
            "each policy row must supply at least one legal event",
        )
        probability = full.gather(-1, ids).masked_fill(~mask, 0.0)
        normalizer = probability.sum(dim=-1, keepdim=True)
        ordinary = probability / normalizer.clamp_min(self.eps)
        legal_logits = full_logits.gather(-1, ids).masked_fill(~mask, -torch.inf)
        fallback = torch.softmax(legal_logits, dim=-1).masked_fill(~mask, 0.0)
        probability = torch.where(normalizer > 0, ordinary, fallback)
        _require_device(
            torch.isfinite(probability).all(),
            "legal-event probabilities must be finite",
        )
        return probability

    compose = probabilities

    def log_prob(
        self,
        raw: PolicyOutput | Mapping[str, Tensor],
        selected_event_ids: Tensor,
        legal_event_ids: Tensor,
        legal_mask: Tensor,
    ) -> Tensor:
        """Return selected legal-event log probabilities ``[B,E_selected,X]``."""

        probability = self.probabilities(raw, legal_event_ids, legal_mask)
        event_shape = (probability.shape[0],) + (1,) * (probability.ndim - 2) + (probability.shape[-1],)
        mask = legal_mask.to(probability.device).reshape(event_shape).expand_as(probability)
        log_rows = torch.where(
            mask, torch.log(probability.clamp_min(self.eps)),
            torch.full_like(probability, -torch.inf),
        )
        if selected_event_ids.ndim == 1:
            selected_event_ids = selected_event_ids[:, None]
        matches = (
            legal_event_ids[:, None, :].to(selected_event_ids.device)
            == selected_event_ids[:, :, None]
        ) & legal_mask[:, None, :].to(selected_event_ids.device)
        _require_device(
            matches.any(-1).all(),
            "selected_event_ids must select supplied legal events",
        )
        columns = matches.to(torch.int64).argmax(-1).to(log_rows.device)  # [B,E_selected]
        if log_rows.ndim != 3:
            raise ValueError("poker policy log rows must have shape [B,X,E_legal]")
        gathered = log_rows.gather(
            -1, columns[:, None, :].expand(log_rows.shape[0], log_rows.shape[1], -1)
        )  # [B,X,E_selected]
        return gathered.transpose(1, 2)  # [B,E_selected,X]

    def entropy(self, raw: PolicyOutput | Mapping[str, Tensor], legal_event_ids: Tensor, legal_mask: Tensor) -> Tensor:
        probability = self.probabilities(raw, legal_event_ids, legal_mask)
        return -(probability * torch.log(probability.clamp_min(self.eps))).sum(dim=-1)

    def kl(
        self, raw: PolicyOutput | Mapping[str, Tensor], reference: PolicyOutput | Mapping[str, Tensor],
        legal_event_ids: Tensor, legal_mask: Tensor,
    ) -> Tensor:
        probability = self.probabilities(raw, legal_event_ids, legal_mask)
        reference_probability = self.probabilities(reference, legal_event_ids, legal_mask)
        return (probability * (torch.log(probability.clamp_min(self.eps)) -
                               torch.log(reference_probability.clamp_min(self.eps)))).sum(dim=-1)

    def sample(
        self, raw: PolicyOutput | Mapping[str, Tensor], legal_event_ids: Tensor,
        legal_mask: Tensor, gen: torch.Generator,
    ) -> Tensor:
        probability = self.probabilities(raw, legal_event_ids, legal_mask)
        if probability.ndim != 2:
            raise ValueError("sample requires one already-selected private row per spot")
        column = torch.multinomial(probability, 1, generator=gen).squeeze(1)
        return legal_event_ids.gather(1, column[:, None].to(legal_event_ids.device)).squeeze(1).to(torch.int32)

    def draw(
        self,
        output: PolicyOutput | Mapping[str, Tensor],
        profiles: Tensor,
        actor: Tensor,
        legal_event_ids: Tensor,
        legal_mask: Tensor,
        gen: torch.Generator,
    ) -> Tensor:
        """Select stipulated actor rows and draw one legal public event."""

        selected = self.select_actor_row(output, profiles, actor)
        return self.sample(selected, legal_event_ids, legal_mask, gen)


class PokerModelBundle(nn.Module):
    """Own and route the two street-specific poker models."""

    def __init__(self, preflop: PreflopNet, postflop: PostflopNet) -> None:
        super().__init__()
        if preflop.geometry != postflop.geometry:
            raise ValueError("preflop and postflop models must share deck geometry")
        self.preflop = preflop
        self.postflop = postflop
        self.geometry = preflop.geometry
        self.n_hands = self.geometry.n_hands
        self.n_classes = self.geometry.n_classes
        combo_class = torch.tensor(
            self.geometry.class_of_combo, dtype=torch.long
        )
        self.register_buffer("combo_class", combo_class, persistent=True)

    @staticmethod
    def _slice_public(pub: PokerPublicBatch, rows: Tensor) -> PokerPublicBatch:
        values = {}
        for name in pub.__dataclass_fields__:
            value = getattr(pub, name)
            if isinstance(value, PackedPublicHistory):
                values[name] = PackedPublicHistory(
                    **{
                        field: getattr(value, field)[rows]
                        for field in value.__dataclass_fields__
                    }
                )
            else:
                values[name] = None if value is None else value[rows]
        return PokerPublicBatch(**values)

    @staticmethod
    def _slice_post_masks(masks: PostflopMasks, rows: Tensor, batch: int) -> PostflopMasks:
        values: dict[str, Any] = {}
        for name in masks.__dataclass_fields__:
            value = getattr(masks, name)
            if isinstance(value, Tensor):
                values[name] = value[rows] if value.ndim and value.shape[0] == batch else value
            elif isinstance(value, Mapping):
                values[name] = {key: item[rows] for key, item in value.items()}
            elif hasattr(value, "fields"):
                values[name] = type(value)({key: item[rows] for key, item in value.fields.items()}, value.metadata)
            else:
                values[name] = value
        return PostflopMasks(**values)

    @staticmethod
    def _slice_pre_masks(masks: PreflopMasks | None, rows: Tensor, batch: int) -> PreflopMasks | None:
        if masks is None:
            return None
        values: dict[str, Tensor | None] = {}
        for name in masks.__dataclass_fields__:
            value = getattr(masks, name)
            values[name] = value[rows] if isinstance(value, Tensor) and value.ndim and value.shape[0] == batch else value
        return PreflopMasks(**values)

    def forward_preflop(
        self, pub: PokerPublicBatch, class_ranges: Tensor, actor_seat: Tensor | None = None,
        masks: PreflopMasks | None = None,
    ) -> PolicyOutput:
        """Run preflop and lift its compact class rows to compact combo rows."""

        output = self.preflop(pub, class_ranges, actor_seat, masks)
        return PolicyOutput(
            {name: value[:, self.combo_class] for name, value in output.raw.items()},
            output.v[:, self.combo_class], output.vvec, output.u,
        )

    def forward_postflop(
        self, pub: PokerPublicBatch, combo_ranges: Tensor, actor_seat: Tensor | None = None,
        masks: PostflopMasks | None = None,
    ) -> PolicyOutput:
        """Run the compact postflop network without changing row semantics."""

        return self.postflop(pub, combo_ranges, actor_seat, masks)

    def forward(
        self, pub: PokerPublicBatch, ranges: Tensor, actor_seat: Tensor | None = None,
        *, preflop_masks: PreflopMasks | None = None, postflop_masks: PostflopMasks | None = None,
    ) -> PolicyOutput:
        """Route a combo-range batch and return uniformly combo-indexed rows."""

        if ranges.ndim != 3 or ranges.shape[-1] != self.n_hands:
            raise ValueError(
                f"bundle ranges must have shape [B,P,{self.n_hands}]"
            )
        batch = ranges.shape[0]
        actor = pub.to_act if actor_seat is None else actor_seat
        pre_rows = (pub.street == 0).nonzero(as_tuple=False).squeeze(1)
        post_rows = (pub.street != 0).nonzero(as_tuple=False).squeeze(1)
        outputs: dict[int, PolicyOutput] = {}
        if pre_rows.numel():
            selected = ranges[pre_rows]
            class_ranges = torch.zeros(
                selected.shape[:2] + (self.n_classes,),
                dtype=selected.dtype,
                device=selected.device,
            )
            ids = self.combo_class.view(1, 1, -1).expand_as(selected)
            class_ranges.scatter_add_(2, ids, selected)
            lifted = self.forward_preflop(
                self._slice_public(pub, pre_rows), class_ranges, actor[pre_rows],
                self._slice_pre_masks(preflop_masks, pre_rows, batch),
            )
            for local, global_row in enumerate(pre_rows.tolist()):
                outputs[global_row] = PolicyOutput({k: v[local:local+1] for k, v in lifted.raw.items()},
                                                   lifted.v[local:local+1],
                                                   None if lifted.vvec is None else lifted.vvec[local:local+1],
                                                   None if lifted.u is None else lifted.u[local:local+1])
        if post_rows.numel():
            if postflop_masks is None:
                raise ValueError("postflop rows require postflop_masks")
            post = self.postflop(self._slice_public(pub, post_rows), ranges[post_rows], actor[post_rows],
                                 self._slice_post_masks(postflop_masks, post_rows, batch))
            for local, global_row in enumerate(post_rows.tolist()):
                outputs[global_row] = PolicyOutput({k: v[local:local+1] for k, v in post.raw.items()},
                                                   post.v[local:local+1],
                                                   None if post.vvec is None else post.vvec[local:local+1],
                                                   None if post.u is None else post.u[local:local+1])
        raw = {key: torch.cat([outputs[row].raw[key] for row in range(batch)]) for key in next(iter(outputs.values())).raw}
        value = torch.cat([outputs[row].v for row in range(batch)])
        vvec = None if any(outputs[row].vvec is None for row in range(batch)) else torch.cat([outputs[row].vvec for row in range(batch)])
        u = None if any(outputs[row].u is None for row in range(batch)) else torch.cat([outputs[row].u for row in range(batch)])
        return PolicyOutput(raw, value, vvec, u)


@dataclass
class ChanceDraw:
    state_refs: Tensor
    cards: Tensor


class PokerGame:
    """Public NLHE game facade with evaluator-owned private operations."""

    def __init__(
        self,
        engine: NLHEEngine,
        *,
        profile_sampler: PokerGPURejectionSampler | Any | None = None,
        showdown_evaluator: PokerGPUShowdownEvaluator | Any | None = None,
        partition_adapter: PokerGPUPartitionAdapter | Any | None = None,
    ) -> None:
        self.engine = engine
        self.n_players = engine.n_players
        self.geometry = engine.cfg.geometry
        self.n_hands = self.geometry.n_hands
        self.x_sizes = (self.n_hands,) * self.n_players
        self.profile_sampler = profile_sampler or PokerGPURejectionSampler(
            rank_count=self.geometry.rank_count
        )
        self.showdown_evaluator = showdown_evaluator or PokerGPUShowdownEvaluator(
            rank_count=self.geometry.rank_count
        )
        self.partition_adapter = partition_adapter or PokerGPUPartitionAdapter(
            rank_count=self.geometry.rank_count
        )
        self.summary_cache: dict[object, Any] = {}
        self.ranges = RangeState(
            0, self.n_players, rank_count=self.geometry.rank_count
        )
        self.range_state = self.ranges
        self._combo_cards = torch.tensor(
            self.geometry.combo_cards, dtype=torch.int16
        )

    def initial_states(self, g: int) -> Tensor:
        return self.engine.initial_states(g)

    def reset_ranges(self, n_slots: int, device: torch.device | str | None = None) -> RangeState:
        """Allocate the action-likelihood slots owned by this public game."""

        self.ranges = RangeState(
            n_slots,
            self.n_players,
            device,
            rank_count=self.geometry.rank_count,
        )
        self.range_state = self.ranges
        return self.ranges

    def owner_of(self, state_refs: Tensor) -> Tensor:
        return self.engine.owner_of(state_refs)

    def legal_events(self, state_refs: Tensor) -> tuple[Tensor, Tensor]:
        return self.engine.legal_events(state_refs)

    def chance_probs(self, state_refs: Tensor, xprofile: Tensor) -> Tensor:
        """Poker chance is sampled directly from a profile-conditioned complement."""

        raise NotImplementedError("use sample_chance_cards for dynamic poker chance events")

    def step(self, state_refs: Tensor, event_ids: Tensor) -> Tensor:
        return self.engine.step(state_refs, event_ids)

    def is_terminal(self, state_refs: Tensor) -> Tensor:
        return self.engine.is_terminal(state_refs)

    def is_pre_terminal(self, state_refs: Tensor) -> Tensor:
        return self.engine.is_pre_terminal(state_refs)

    def _state_tensors(self, state_refs: Tensor) -> tuple[Tensor, Tensor]:
        refs = [int(value) for value in state_refs.detach().cpu().tolist()]
        board = torch.tensor([self.engine.states.board[ref] for ref in refs], dtype=torch.int16, device=state_refs.device)
        status = torch.tensor([self.engine.states.status[ref] for ref in refs], dtype=torch.long, device=state_refs.device)
        return board, status

    def sample_profiles(self, state_refs: Tensor, reach_rows: Tensor, n: int, gen: torch.Generator) -> Tensor:
        board, _ = self._state_tensors(state_refs)
        dealt = torch.ones(reach_rows.shape[:2], dtype=torch.bool, device=reach_rows.device)
        return self.profile_sampler.sample(reach_rows, board, dealt, n, gen)

    def sample_mu(self, g: int, gen: torch.Generator) -> Tensor:
        refs = self.initial_states(g)
        reach = torch.zeros(
            (g, self.n_players, self.n_hands),
            dtype=torch.float32,
            device=refs.device,
        )
        return self.sample_profiles(refs, reach, 1, gen)[:, 0]

    def sample_chance_cards(self, state_refs: Tensor, profiles: Tensor, gen: torch.Generator) -> Tensor:
        if profiles.ndim != 2 or profiles.shape != (state_refs.numel(), self.n_players):
            raise ValueError("profiles must have shape [B,P]")
        board, _ = self._state_tensors(state_refs)
        table = self._combo_cards.to(profiles.device)
        valid = profiles >= 0
        safe = profiles.clamp_min(0).long()
        private_cards = table[safe].reshape(profiles.shape[0], -1)
        private_valid = valid[..., None].expand(-1, -1, 2).reshape(profiles.shape[0], -1)
        private_cards = private_cards.masked_fill(~private_valid, -1)
        unavailable = torch.cat((board, private_cards), dim=1).to(torch.int16)
        refs = [int(value) for value in state_refs.detach().cpu().tolist()]
        counts = torch.tensor(
            [3 if self.engine.states.street[ref] == 0 else 1 for ref in refs],
            dtype=torch.int32, device=state_refs.device,
        )
        return sample_uniform_complement(
            unavailable,
            counts,
            gen,
            rank_count=self.geometry.rank_count,
        )

    def draw_chance(self, state_refs: Tensor, profiles: Tensor, gen: torch.Generator) -> ChanceDraw:
        cards = self.sample_chance_cards(state_refs, profiles, gen)
        return ChanceDraw(self.engine.deal(state_refs, cards), cards)

    def reward(self, xprofile: Tensor, state_refs: Tensor) -> Tensor:
        preterminal = self.engine.is_pre_terminal(state_refs)
        payoff = torch.empty((state_refs.numel(), self.n_players), dtype=torch.float32, device=state_refs.device)
        terminal_rows = (~preterminal).nonzero(as_tuple=False).squeeze(1)
        if terminal_rows.numel():
            payoff[terminal_rows] = self.engine.reward(xprofile[terminal_rows], state_refs[terminal_rows])
        showdown_rows = preterminal.nonzero(as_tuple=False).squeeze(1)
        if showdown_rows.numel():
            showdown_refs = state_refs[showdown_rows]
            board, status = self._state_tensors(showdown_refs)
            strata = self.engine.pot_strata(showdown_refs)
            result = self.showdown_evaluator.evaluate(
                board_cards=board,
                combo_ids=xprofile[showdown_rows].to(device=board.device, dtype=torch.int32),
                active_mask=status != FOLDED,
                stratum_players=strata.stratum_players,
            )
            payoff[showdown_rows] = self.engine.assemble_showdown_payoff(strata, result.showdown_share)
        return payoff

    def encode_public(self, state_refs: Tensor) -> PokerPublicBatch:
        refs = [int(value) for value in state_refs.detach().cpu().tolist()]
        device = state_refs.device
        board, status = self._state_tensors(state_refs)
        stacks = torch.tensor([self.engine.states.stacks[ref] for ref in refs], dtype=torch.float32, device=device)
        committed = torch.tensor([self.engine.states.committed_total[ref] for ref in refs], dtype=torch.float32, device=device)
        pot = torch.tensor([self.engine.states.pot[ref] for ref in refs], dtype=torch.float32, device=device)
        actor = torch.tensor([self.engine.states.actor[ref] for ref in refs], dtype=torch.long, device=device)
        street = torch.tensor([self.engine.states.street[ref] for ref in refs], dtype=torch.long, device=device)
        button = torch.tensor([self.engine.states.button[ref] for ref in refs], dtype=torch.long, device=device)
        ids, legal = self.engine.legal_events(state_refs)
        fold_legal = legal[:, FOLD].float()
        raise_fraction = legal[:, RAISE_OFFSET:].float().mean(1)
        legality = torch.stack((fold_legal, legal[:, CHECK_CALL].float(), raise_fraction), dim=1)
        targets = self.engine.raise_targets(state_refs)
        raise_legal = legal[:, RAISE_OFFSET:]
        lower = targets.masked_fill(~raise_legal, torch.inf).min(1).values
        upper = targets.masked_fill(~raise_legal, -torch.inf).max(1).values
        has_raise = raise_legal.any(1)
        lower = torch.where(has_raise, lower, torch.zeros_like(lower))
        upper = torch.where(has_raise, upper, torch.zeros_like(upper))
        bounds = torch.stack((lower, upper), dim=1)
        committed_street = torch.tensor(
            [self.engine.states.committed_street[ref] for ref in refs],
            dtype=torch.float32,
            device=device,
        )
        current_bet = torch.tensor(
            [self.engine.states.current_bet[ref] for ref in refs],
            dtype=torch.float32,
            device=device,
        )
        last_full_raise = torch.tensor(
            [self.engine.states.last_full_raise[ref] for ref in refs],
            dtype=torch.float32,
            device=device,
        )
        call_amount = (
            current_bet - committed_street.gather(1, actor[:, None]).squeeze(1)
        ).clamp_min(0.0)
        histories = [self.engine.states.public_history[ref] for ref in refs]
        width = max((len(row) for row in histories), default=0)
        history_shape = (len(refs), width)
        history_actor = torch.full(history_shape, -1, dtype=torch.long, device=device)
        history_street = torch.full(history_shape, -1, dtype=torch.long, device=device)
        history_action = torch.full(history_shape, -1, dtype=torch.long, device=device)
        history_mask = torch.zeros(history_shape, dtype=torch.bool, device=device)
        history_float = {
            name: torch.zeros(history_shape, dtype=torch.float32, device=device)
            for name in (
                "amount", "pot_before", "call_amount", "current_bet",
                "min_raise_to", "max_raise_to",
            )
        }
        for row_index, row in enumerate(histories):
            for column, event in enumerate(row):
                history_actor[row_index, column] = event.actor
                history_street[row_index, column] = event.street
                history_action[row_index, column] = event.action
                history_mask[row_index, column] = True
                for name in history_float:
                    history_float[name][row_index, column] = getattr(event, name)
        return PokerPublicBatch(
            street=street, board=board.long(), pot=pot, stacks=stacks, committed=committed,
            to_act=actor, legality_summary=legality,
            n_ticks=torch.full((len(refs),), self.engine.n_ticks, dtype=torch.long, device=device),
            raise_grid_bounds=bounds,
            status=status,
            seat_mask=torch.ones_like(status, dtype=torch.bool),
            button=button,
            committed_street=committed_street,
            call_amount=call_amount,
            current_bet=current_bet,
            min_raise_to=lower,
            max_raise_to=upper,
            last_full_raise=last_full_raise,
            legal_action_mask=legal,
            raise_ticks=targets,
            history=PackedPublicHistory(
                actor=history_actor,
                street=history_street,
                action=history_action,
                mask=history_mask,
                **history_float,
            ),
        )

    def reach_update(self, log_r_rows: Tensor, state_refs: Tensor, event_ids: Tensor, policy_rows: Tensor) -> Tensor:
        result = log_r_rows.clone()
        owners = self.engine.owner_of(state_refs).long()
        player = owners >= 0
        if bool(player.any()):
            batch = torch.arange(state_refs.numel(), device=log_r_rows.device)[player]
            event = event_ids[player].long()
            result[batch, owners[player]] += policy_rows[batch, :, event].detach().to(result.dtype)
        return result
