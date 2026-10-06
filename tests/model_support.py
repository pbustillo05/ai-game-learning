"""Test-only input adapters for the production shared model and fixed outputs."""

import torch
from torch import nn

from pbg.core.learner import ModelEvaluation
from pbg.games.poker.cards import DeckGeometry
from pbg.games.poker.model.preflop import PreflopEncoder
from pbg.games.poker.model.postflop import PostflopEncoder
from pbg.games.poker.model.shared import SharedPokerParameters
from pbg.games.poker.queued.model import QueuedPokerModel, PokerModelError


class FamilyModelFixture(nn.Module):
    """Expose direct family inputs to the same production shared tensor program."""

    def __init__(self, cfg, family):
        super().__init__()
        self.cfg = cfg
        self.shared = SharedPokerParameters(cfg, 52)
        self.encoder = (PreflopEncoder if family == "pre" else PostflopEncoder)(cfg)

    def forward(self, *args, **kwargs):
        return self.encoder(self.shared, *args, **kwargs)


class FixedOutputPokerModel(QueuedPokerModel):
    """Inject small deterministic heads to test queue/RNG/target invariants."""

    def __init__(self, geometry, preflop, postflop):
        nn.Module.__init__(self)
        self.geometry = geometry
        self.preflop = preflop
        self.postflop = postflop
        self.register_buffer("combo_class", torch.tensor(DeckGeometry(geometry.rank_count).class_of_combo))

    @property
    def replica_shared_buffers(self):
        return ("combo_class",)

    @staticmethod
    def _select(result, selected_private):
        if selected_private is None:
            return result
        rows = torch.arange(selected_private.numel(), device=selected_private.device)
        return ModelEvaluation(
            {name: value[rows, selected_private].unsqueeze(1) for name, value in result.policy_raw.items()},
            result.value[rows, selected_private].unsqueeze(1), result.public_heads,
        )

    def _call_preflop(self, model_input, selected_private=None):
        return self._select(self.preflop(model_input), selected_private)

    def _call_postflop(self, input_type, model_input, selected_private=None):
        if self.geometry.folded_summary_width:
            state = self._state(input_type, model_input)
            if not bool((~state.status.eq(1).any(-1) | model_input[:, -1].gt(0.5)).all()):
                raise PokerModelError("postflop folded seats require sealed folded summaries")
        return self._select(self.postflop(model_input), selected_private)
