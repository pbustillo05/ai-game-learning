"""Float64 exact analysis for feasible synthetic public-action games."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import torch
from torch import Tensor

from pbg.core.completion import BatchedCompletionStats
from pbg.core.game import CHANCE, TERMINAL
from pbg.games.synthetic.family import SyntheticPBG

MAX_EXACT_NODES = 2_000_000
PolicyTables: TypeAlias = dict[int, np.ndarray]


class FeasibilityError(RuntimeError):
    """Raised before an exact calculation would exceed its hard guard."""


@dataclass(frozen=True)
class PublicTreeIndex:
    """Indexed public tree with state refs, owners, parents, and event edges."""

    nodes: tuple[int, ...]
    node_keys: np.ndarray  # int64 [N]
    owners: np.ndarray  # int32 [N]
    children: dict[int, tuple[tuple[int, int], ...]]
    parent: np.ndarray  # int64 [N], -1 at root
    parent_event: np.ndarray  # int32 [N], -1 at root

    def index_of_key(self, node_key: int) -> int:
        """Return the tree index for one perfect public-history key."""

        matches = np.flatnonzero(self.node_keys == node_key)
        if matches.size != 1:
            raise KeyError(f"unknown or ambiguous public node key {node_key}")
        return int(matches[0])


def enumerate_public_tree(game: SyntheticPBG) -> PublicTreeIndex:
    """Enumerate the complete public tree, failing beyond ``MAX_EXACT_NODES``."""

    root = int(game.root_states(1)[0])
    nodes = [root]
    parents = [-1]
    parent_events = [-1]
    children: dict[int, tuple[tuple[int, int], ...]] = {}
    cursor = 0
    while cursor < len(nodes):
        ref = nodes[cursor]
        event_ids, event_mask = game.legal_events(torch.tensor([ref], dtype=torch.int32))
        events = event_ids[0, event_mask[0]].tolist()
        edges: list[tuple[int, int]] = []
        for event in events:
            successor = int(
                game.step(
                    torch.tensor([ref], dtype=torch.int32),
                    torch.tensor([event], dtype=torch.int32),
                )[0]
            )
            nodes.append(successor)
            parents.append(cursor)
            parent_events.append(int(event))
            child_index = len(nodes) - 1
            edges.append((int(event), child_index))
            if len(nodes) > MAX_EXACT_NODES:
                raise FeasibilityError(
                    f"public tree exceeds MAX_EXACT_NODES={MAX_EXACT_NODES}"
                )
        children[cursor] = tuple(edges)
        cursor += 1
    refs = torch.tensor(nodes, dtype=torch.int32)
    return PublicTreeIndex(
        nodes=tuple(nodes),
        node_keys=game.encode_public(refs).cpu().numpy().astype(np.int64, copy=False),
        owners=game.owner_of(refs).cpu().numpy().astype(np.int32, copy=False),
        children=children,
        parent=np.asarray(parents, dtype=np.int64),
        parent_event=np.asarray(parent_events, dtype=np.int32),
    )


def _profiles(game: SyntheticPBG) -> np.ndarray:
    raw_count = math.prod(game.x_sizes)
    if raw_count > MAX_EXACT_NODES:
        raise FeasibilityError(
            f"private profile product {raw_count} exceeds {MAX_EXACT_NODES}"
        )
    feasible = [
        profile
        for profile in itertools.product(range(game.n_types), repeat=game.n_players)
        if game._profile_feasible(profile, set())
    ]
    if not feasible:
        raise FeasibilityError("game has no feasible private profile")
    return np.asarray(feasible, dtype=np.int64)  # [Z, P]


def exact_completion_stats(
    game: SyntheticPBG,
    state_refs: Tensor,
    reach_rows: Tensor,
) -> BatchedCompletionStats:
    """Enumerate exact posterior/value statistics at synthetic terminal cuts.

    The joint posterior is proportional to the product of the per-player reach
    factors, restricted to collision-free private profiles compatible with the
    public board.  This is the same finite law sampled by
    :meth:`SyntheticPBG.sample_profiles`, but contains no Monte Carlo error.
    """

    refs, device = game._refs(state_refs)
    if reach_rows.shape != (len(refs), game.n_players, game.n_types):
        raise ValueError("reach_rows must have shape [B, P, X]")
    raw_profiles = math.prod(game.x_sizes)
    if raw_profiles > MAX_EXACT_NODES:
        raise FeasibilityError(
            f"private profile product {raw_profiles} exceeds {MAX_EXACT_NODES}"
        )

    batch = len(refs)
    x_max = max(game.x_sizes)
    p_hat = torch.zeros(
        (batch, game.n_players, x_max), dtype=torch.float64, device=device
    )
    v_hat = torch.zeros_like(p_hat)
    vvec_hat = torch.zeros(
        (batch, game.n_players), dtype=torch.float64, device=device
    )
    u_hat = torch.zeros(batch, dtype=torch.float64, device=device)
    n_eff = torch.zeros(batch, dtype=torch.float64, device=device)
    sample_count = torch.zeros(batch, dtype=torch.int64, device=device)
    profile_counts: list[int] = []

    for row, ref in enumerate(refs):
        board = {card for card in game.states.board[ref] if card >= 0}
        profiles = np.asarray(
            [
                profile
                for profile in itertools.product(
                    range(game.n_types), repeat=game.n_players
                )
                if game._profile_feasible(profile, board)
            ],
            dtype=np.int64,
        )
        if profiles.size == 0:
            raise FeasibilityError("public board has no feasible private profile")
        profile_counts.append(len(profiles))
        profile_tensor = torch.from_numpy(profiles).to(device=device, dtype=torch.int64)
        log_weight = torch.zeros(len(profiles), dtype=torch.float64, device=device)
        for player in range(game.n_players):
            log_weight += reach_rows[row, player, profile_tensor[:, player]].to(
                torch.float64
            )
        weight = game._normalized_weights(log_weight)
        n_eff[row] = weight.square().sum().reciprocal()
        sample_count[row] = len(profiles)
        rewards = game.reward(
            profile_tensor.to(torch.int32),
            torch.full((len(profiles),), ref, dtype=torch.int32, device=device),
        ).to(torch.float64)
        weighted_reward = weight[:, None] * rewards
        vvec_hat[row] = weighted_reward.sum(0)
        u_hat[row] = (weight * rewards.square().sum(1)).sum()
        for player, width in enumerate(game.x_sizes):
            types = profile_tensor[:, player]
            mass = torch.zeros(width, dtype=torch.float64, device=device)
            moment = torch.zeros_like(mass)
            mass.index_add_(0, types, weight)
            moment.index_add_(0, types, weighted_reward[:, player])
            p_hat[row, player, :width] = mass
            supported = mass > 0
            v_hat[row, player, :width] = torch.where(
                supported,
                moment / mass.clamp_min(torch.finfo(mass.dtype).tiny),
                torch.zeros_like(moment),
            )

    output_valid = p_hat > 0
    return BatchedCompletionStats(
        p_hat=p_hat.to(torch.float32),
        v_hat=v_hat.to(torch.float32),
        vvec_hat=vvec_hat.to(torch.float32),
        u_hat=u_hat.to(torch.float32),
        n_eff=n_eff.to(torch.float32),
        output_valid=output_valid,
        sample_count=sample_count,
        estimator_stddev=torch.zeros_like(p_hat, dtype=torch.float32),
        metadata={
            "backend": "exact",
            "exact_profile_count_min": min(profile_counts, default=0),
            "exact_profile_count_max": max(profile_counts, default=0),
        },
    )


def _resolve_node(tree_index: PublicTreeIndex, node: int) -> int:
    if 0 <= int(node) < len(tree_index.nodes):
        return int(node)
    return tree_index.index_of_key(int(node))


def _path(tree_index: PublicTreeIndex, node_index: int) -> list[tuple[int, int]]:
    reversed_edges: list[tuple[int, int]] = []
    cursor = node_index
    while tree_index.parent[cursor] >= 0:
        parent = int(tree_index.parent[cursor])
        reversed_edges.append((parent, int(tree_index.parent_event[cursor])))
        cursor = parent
    return list(reversed(reversed_edges))


def exact_posterior(
    game: SyntheticPBG, tree_index: PublicTreeIndex, node: int
) -> np.ndarray:
    """Return uniform-policy type marginals at ``node`` as float64 ``[P, X]``."""

    profiles = _profiles(game)
    weights = np.full(len(profiles), 1.0 / len(profiles), dtype=np.float64)
    for parent_index, event in _path(tree_index, _resolve_node(tree_index, node)):
        ref = tree_index.nodes[parent_index]
        owner = int(tree_index.owners[parent_index])
        event_ids, mask = game.legal_events(torch.tensor([ref], dtype=torch.int32))
        legal = event_ids[0, mask[0]].tolist()
        column = legal.index(event)
        if owner == CHANCE:
            refs = torch.full((len(profiles),), ref, dtype=torch.int32)
            chance = game.chance_probs(refs, torch.from_numpy(profiles).to(torch.int32))
            weights *= chance[:, column].cpu().numpy().astype(np.float64, copy=False)
        elif owner != TERMINAL:
            weights /= len(legal)
    total = weights.sum()
    if not total > 0.0:
        raise FeasibilityError("observed public node has zero exact reach")
    weights /= total
    posterior = np.zeros((game.n_players, game.n_types), dtype=np.float64)
    for player in range(game.n_players):
        posterior[player] = np.bincount(
            profiles[:, player], weights=weights, minlength=game.n_types
        )
    return posterior  # [P, X]


def uniform_policy_tables(game: SyntheticPBG) -> PolicyTables:
    """Return type-independent uniform policies at every player decision node."""

    tree = enumerate_public_tree(game)
    tables: PolicyTables = {}
    for index, owner in enumerate(tree.owners):
        if not 0 <= int(owner) < game.n_players:
            continue
        ref = tree.nodes[index]
        event_ids, mask = game.legal_events(torch.tensor([ref], dtype=torch.int32))
        legal = [int(event) for event in event_ids[0, mask[0]].tolist()]
        table = np.zeros((game.n_types, game.n_player_events), dtype=np.float64)
        table[:, legal] = 1.0 / len(legal)
        tables[int(tree.node_keys[index])] = table
    return tables


def _table_for(
    game: SyntheticPBG, policy_tables: PolicyTables, ref: int, legal: list[int]
) -> np.ndarray:
    key = int(game.encode_public(torch.tensor([ref], dtype=torch.int32))[0])
    if key not in policy_tables:
        raise KeyError(f"policy table missing public node key {key}")
    table = np.asarray(policy_tables[key], dtype=np.float64)
    if table.shape != (game.n_types, game.n_player_events):
        raise ValueError(
            f"policy table at key {key} must have shape "
            f"({game.n_types}, {game.n_player_events})"
        )
    if not np.isfinite(table).all() or (table < 0.0).any():
        raise ValueError(f"policy table at key {key} contains invalid probabilities")
    if not np.allclose(table[:, legal].sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError(f"policy table at key {key} is not normalized on legal events")
    invalid = [event for event in range(game.n_player_events) if event not in legal]
    if invalid and not np.allclose(table[:, invalid], 0.0, atol=1e-12, rtol=0.0):
        raise ValueError(f"policy table at key {key} has mass on illegal events")
    return table


def _terminal_rewards(
    game: SyntheticPBG, ref: int, profiles: np.ndarray
) -> np.ndarray:
    refs = torch.full((len(profiles),), ref, dtype=torch.int32)
    values = game.reward(torch.from_numpy(profiles).to(torch.int32), refs)
    return values.cpu().numpy().astype(np.float64, copy=False)  # [Z, P]


def _policy_recurse(
    game: SyntheticPBG,
    policy_tables: PolicyTables,
    ref: int,
    profiles: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    owner = int(game.owner_of(torch.tensor([ref], dtype=torch.int32))[0])
    if owner == TERMINAL:
        return (weights[:, None] * _terminal_rewards(game, ref, profiles)).sum(axis=0)
    event_ids, mask = game.legal_events(torch.tensor([ref], dtype=torch.int32))
    legal = [int(event) for event in event_ids[0, mask[0]].tolist()]
    value = np.zeros(game.n_players, dtype=np.float64)
    if owner == CHANCE:
        refs = torch.full((len(profiles),), ref, dtype=torch.int32)
        chance = game.chance_probs(refs, torch.from_numpy(profiles).to(torch.int32))
        chance_np = chance.cpu().numpy().astype(np.float64, copy=False)
        for column, event in enumerate(legal):
            child = int(
                game.step(
                    torch.tensor([ref], dtype=torch.int32),
                    torch.tensor([event], dtype=torch.int32),
                )[0]
            )
            value += _policy_recurse(
                game, policy_tables, child, profiles, weights * chance_np[:, column]
            )
        return value
    table = _table_for(game, policy_tables, ref, legal)
    for event in legal:
        child = int(
            game.step(
                torch.tensor([ref], dtype=torch.int32),
                torch.tensor([event], dtype=torch.int32),
            )[0]
        )
        action_probability = table[profiles[:, owner], event]
        value += _policy_recurse(
            game, policy_tables, child, profiles, weights * action_probability
        )
    return value


def _br_recurse(
    game: SyntheticPBG,
    policy_tables: PolicyTables,
    br_player: int,
    ref: int,
    profiles: np.ndarray,
    weights: np.ndarray,
) -> float:
    owner = int(game.owner_of(torch.tensor([ref], dtype=torch.int32))[0])
    if owner == TERMINAL:
        rewards = _terminal_rewards(game, ref, profiles)[:, br_player]
        return float(np.dot(weights, rewards))
    event_ids, mask = game.legal_events(torch.tensor([ref], dtype=torch.int32))
    legal = [int(event) for event in event_ids[0, mask[0]].tolist()]
    if owner == CHANCE:
        refs = torch.full((len(profiles),), ref, dtype=torch.int32)
        chance = game.chance_probs(refs, torch.from_numpy(profiles).to(torch.int32))
        chance_np = chance.cpu().numpy().astype(np.float64, copy=False)
        total = 0.0
        for column, event in enumerate(legal):
            child = int(
                game.step(
                    torch.tensor([ref], dtype=torch.int32),
                    torch.tensor([event], dtype=torch.int32),
                )[0]
            )
            total += _br_recurse(
                game,
                policy_tables,
                br_player,
                child,
                profiles,
                weights * chance_np[:, column],
            )
        return total
    if owner == br_player:
        action_values = []
        for event in legal:
            child = int(
                game.step(
                    torch.tensor([ref], dtype=torch.int32),
                    torch.tensor([event], dtype=torch.int32),
                )[0]
            )
            action_values.append(
                _br_recurse(game, policy_tables, br_player, child, profiles, weights)
            )
        return max(action_values)
    table = _table_for(game, policy_tables, ref, legal)
    total = 0.0
    for event in legal:
        child = int(
            game.step(
                torch.tensor([ref], dtype=torch.int32),
                torch.tensor([event], dtype=torch.int32),
            )[0]
        )
        probabilities = table[profiles[:, owner], event]
        total += _br_recurse(
            game,
            policy_tables,
            br_player,
            child,
            profiles,
            weights * probabilities,
        )
    return total


def best_response_value(game: SyntheticPBG, policy_tables: PolicyTables) -> np.ndarray:
    """Return each player's exact best-response value as float64 ``[P]``."""

    enumerate_public_tree(game)
    profiles = _profiles(game)
    initial_weight = 1.0 / len(profiles)
    root = int(game.root_states(1)[0])
    values = np.zeros(game.n_players, dtype=np.float64)
    for player in range(game.n_players):
        for xtype in range(game.n_types):
            selected = profiles[:, player] == xtype
            if not selected.any():
                continue
            player_profiles = profiles[selected]
            weights = np.full(len(player_profiles), initial_weight, dtype=np.float64)
            values[player] += _br_recurse(
                game, policy_tables, player, root, player_profiles, weights
            )
    return values  # [P]


def exploitability(game: SyntheticPBG, policy_tables: PolicyTables) -> float:
    """Return ``sum_p(BR_p - J_p(policy))`` in exact chip units."""

    enumerate_public_tree(game)
    profiles = _profiles(game)
    weights = np.full(len(profiles), 1.0 / len(profiles), dtype=np.float64)
    root = int(game.root_states(1)[0])
    on_policy = _policy_recurse(game, policy_tables, root, profiles, weights)
    best_response = best_response_value(game, policy_tables)
    return float(np.sum(best_response - on_policy))
