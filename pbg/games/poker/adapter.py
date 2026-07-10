"""Poker public navigation and factorized policy adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from pbg.games.poker.cards import N_CLASSES, N_HANDS, class_of_hand, sample_uniform_complement
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
from pbg.games.poker.model.heads import raise_tick_probs
from pbg.games.poker.model.postflop import PostflopMasks, PostflopNet
from pbg.games.poker.model.preflop import PokerPublicBatch, PolicyOutput, PreflopMasks, PreflopNet
from pbg.games.poker.ranges import RangeState


def _raw_mapping(output: PolicyOutput | Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    return output.raw if isinstance(output, PolicyOutput) else output


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
        if bool(((combo < 0) | (combo >= N_HANDS)).any()):
            raise ValueError("the stipulated actor combo is invalid")
        batch_ids = torch.arange(profiles.shape[0], device=profiles.device)
        selected: dict[str, Tensor] = {}
        for name, value in raw.items():
            if value.shape[:2] != (profiles.shape[0], N_HANDS):
                raise ValueError(f"raw field {name} must begin [B,1326]")
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
        inferred = max(
            legal_event_ids.shape[1] - RAISE_OFFSET,
            int(legal_event_ids.max().item()) - RAISE_OFFSET + 1 if legal_event_ids.numel() else 0,
        )
        n_ticks = self.n_ticks if self.n_ticks is not None else inferred
        if n_ticks <= 0:
            raise ValueError("the legal event set has no raise-tick width")
        type_probability = torch.softmax(type_logits.float(), dim=-1)
        ticks = raise_tick_probs(
            fields["omega"].float(), fields["mu"].float(), fields["s"].float(),
            fields["iota"].float(), n_ticks,
        )
        prefix = type_logits.shape[:-1]
        full = torch.zeros(prefix + (RAISE_OFFSET + n_ticks,), dtype=torch.float32, device=type_logits.device)
        facing_bet = ((legal_event_ids == FOLD) & legal_mask).any(dim=1)
        facing = facing_bet.reshape((batch,) + (1,) * (type_logits.ndim - 2))
        full[..., FOLD] = torch.where(facing, type_probability[..., 0], 0.0)
        full[..., CHECK_CALL] = torch.where(facing, type_probability[..., 1], type_probability[..., 0])
        full[..., RAISE_OFFSET:] = type_probability[..., 2, None] * ticks
        ids = legal_event_ids.to(device=full.device, dtype=torch.long)
        if bool(((ids < 0) | (ids >= full.shape[-1])).any()):
            raise ValueError("legal event id is outside the factorized event space")
        event_shape = (batch,) + (1,) * (type_logits.ndim - 2) + (ids.shape[1],)
        ids = ids.reshape(event_shape).expand(prefix + (ids.shape[1],))
        mask = legal_mask.to(full.device).reshape(event_shape).expand_as(ids)
        probability = full.gather(-1, ids).masked_fill(~mask, 0.0)
        normalizer = probability.sum(dim=-1, keepdim=True)
        if bool((normalizer <= 0).any()):
            raise ValueError("supplied legal events have zero policy mass")
        return probability / normalizer

    compose = probabilities

    def log_prob(
        self, raw: PolicyOutput | Mapping[str, Tensor], legal_event_ids: Tensor,
        legal_mask: Tensor, event_ids: Tensor | None = None,
    ) -> Tensor:
        probability = self.probabilities(raw, legal_event_ids, legal_mask)
        event_shape = (probability.shape[0],) + (1,) * (probability.ndim - 2) + (probability.shape[-1],)
        mask = legal_mask.to(probability.device).reshape(event_shape).expand_as(probability)
        log_rows = torch.where(
            mask, torch.log(probability.clamp_min(self.eps)),
            torch.full_like(probability, -torch.inf),
        )
        if event_ids is None:
            return log_rows
        columns = (legal_event_ids.to(event_ids.device) == event_ids[:, None]).to(torch.int64).argmax(1)
        present = ((legal_event_ids.to(event_ids.device) == event_ids[:, None]) & legal_mask.to(event_ids.device)).any(1)
        if not bool(present.all()):
            raise ValueError("event_ids must select supplied legal events")
        column_shape = (log_rows.shape[0],) + (1,) * (log_rows.ndim - 2) + (1,)
        columns = columns.to(log_rows.device).reshape(column_shape).expand(log_rows.shape[:-1] + (1,))
        return log_rows.gather(-1, columns).squeeze(-1)

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
        self.preflop = preflop
        self.postflop = postflop
        combo_class = torch.tensor([class_of_hand(hand) for hand in range(N_HANDS)], dtype=torch.long)
        self.register_buffer("combo_class", combo_class, persistent=True)

    @staticmethod
    def _slice_public(pub: PokerPublicBatch, rows: Tensor) -> PokerPublicBatch:
        values = {}
        for name in pub.__dataclass_fields__:
            value = getattr(pub, name)
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
        """Run preflop and lift its 169 class rows to 1326 combo rows."""

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

        if ranges.ndim != 3 or ranges.shape[-1] != N_HANDS:
            raise ValueError("bundle ranges must have shape [B,P,1326]")
        batch = ranges.shape[0]
        actor = pub.to_act if actor_seat is None else actor_seat
        pre_rows = (pub.street == 0).nonzero(as_tuple=False).squeeze(1)
        post_rows = (pub.street != 0).nonzero(as_tuple=False).squeeze(1)
        outputs: dict[int, PolicyOutput] = {}
        if pre_rows.numel():
            selected = ranges[pre_rows]
            class_ranges = torch.zeros(selected.shape[:2] + (N_CLASSES,), dtype=selected.dtype, device=selected.device)
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
        self.x_sizes = (N_HANDS,) * self.n_players
        self.profile_sampler = profile_sampler or PokerGPURejectionSampler()
        self.showdown_evaluator = showdown_evaluator or PokerGPUShowdownEvaluator()
        self.partition_adapter = partition_adapter or PokerGPUPartitionAdapter()
        self.summary_cache: dict[object, Any] = {}
        self.ranges = RangeState(0, self.n_players)
        self.range_state = self.ranges
        self._combo_cards = torch.triu_indices(52, 52, offset=1).T.to(torch.int16)

    def root_states(self, g: int) -> Tensor:
        return self.engine.root_states(g)

    def reset_ranges(self, n_slots: int, device: torch.device | str | None = None) -> RangeState:
        """Allocate the action-likelihood slots owned by this public game."""

        self.ranges = RangeState(n_slots, self.n_players, device)
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
        refs = self.root_states(g)
        reach = torch.zeros((g, self.n_players, N_HANDS), dtype=torch.float32, device=refs.device)
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
        return sample_uniform_complement(unavailable, counts, gen)

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
        bounds = torch.stack((targets[:, 0], targets[:, -1]), dim=1)
        return PokerPublicBatch(
            street=street, board=board.long(), pot=pot, stacks=stacks, committed=committed,
            to_act=actor, legality_summary=legality,
            n_ticks=torch.full((len(refs),), self.engine.n_ticks, dtype=torch.long, device=device),
            raise_grid_bounds=bounds, status=status, seat_mask=torch.ones_like(status, dtype=torch.bool), button=button,
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
