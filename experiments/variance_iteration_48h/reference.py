"""Direct completion-world reference objective for the 48-hour study."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise

import torch
from torch import Tensor

from pbg.games.poker.queued.evaluator import bound_evaluator


class DirectReferenceError(RuntimeError):
    """The held-out topology cannot support the declared direct reference."""


def direct_reference_moments(
    owners: object,
    state: object,
    operations: object,
    *,
    completions: int,
    board_seed: int,
    lambda_: float,
) -> tuple[dict[str, dict[str, Tensor]], dict[str, object]]:
    """Sample exact completion worlds and cache signed advantages.

    The evaluator's regular completion reduction exposes conditional first
    moments, which are insufficient after PPO's sign-dependent clipping.  This
    routine samples collision-conditioned joint profiles directly, samples the
    missing public cards uniformly on CUDA, settles every world with the
    evaluator, and replays the sequential lambda recurrence.  It retains sparse
    hand/advantage samples to verify the signed cache against direct averaging.
    """

    plan = state.topology.plan
    if owners.plan.manifest.n_replica != 1:
        raise DirectReferenceError("held-out reference requires one replica")
    if plan.branched or not bool(plan.target_degree.eq(1).all()):
        raise DirectReferenceError(
            "direct signed reference requires sequential unit targets"
        )
    if not bool(plan.child_count.eq(1).all()):
        raise DirectReferenceError(
            "direct signed reference requires one executed child"
        )
    leaf_count = int(state.topology.leaf_counts[0])
    if leaf_count <= 0 or leaf_count != int(plan.leaf_rows[0]):
        raise DirectReferenceError("held-out leaf topology is not dense")
    if completions <= 0 or not 0.0 <= lambda_ <= 1.0:
        raise DirectReferenceError("invalid direct-reference configuration")

    device = owners.plan.device
    geometry = operations.geometry
    evaluator = bound_evaluator(geometry.rank_count)
    leaf = owners.slab("Leaf")
    fields = {name: value[0, :leaf_count] for name, value in leaf.columns.items()}
    profiles = evaluator.sample_joint_profiles(
        board_cards=fields["leaf_public_state"].contiguous(),
        range_weights=fields["range_weights"].contiguous(),
        n_samples=int(completions),
        seed=fields["seed"].contiguous(),
    )
    if profiles.shape != (leaf_count, completions, geometry.players):
        raise DirectReferenceError("joint-profile sampler returned an unexpected shape")

    algebra = operations.target_algebra(owners.plan)
    output_names = tuple(plan.output_names)
    post = output_names.index("post")
    post_width = int(plan.output_widths[post])
    hand_maps: list[Tensor] = []
    for output, width in enumerate(plan.output_widths):
        if output == post:
            mapping = torch.arange(post_width, dtype=torch.int64, device=device)
        else:
            try:
                mask = algebra._cpu_masks[(post, output)]
            except KeyError as exc:
                raise DirectReferenceError(
                    f"missing exact-hand conversion for output {output_names[output]}"
                ) from exc
            if mask.shape != (width, post_width) or not bool(mask.sum(0).eq(1).all()):
                raise DirectReferenceError(
                    "private-hand conversion is not one-to-one by column"
                )
            mapping = mask.argmax(0).to(device=device, dtype=torch.int64)
        hand_maps.append(mapping)

    input_names = tuple(plan.input_names)
    cache: dict[str, dict[str, Tensor]] = {}
    for input_index, name in enumerate(input_names):
        rows = int(plan.action_rows[0][input_index])
        output_name = algebra.partitions.output_for(name).name
        width = int(plan.output_widths[output_names.index(output_name)])
        cache[name] = {
            "count": torch.zeros((rows, width), dtype=torch.float32, device=device),
            "positive_sum": torch.zeros(
                (rows, width), dtype=torch.float32, device=device
            ),
            "negative_sum": torch.zeros(
                (rows, width), dtype=torch.float32, device=device
            ),
            "world_hand": torch.empty(
                (rows, completions), dtype=torch.int16, device=device
            ),
            "world_advantage": torch.empty(
                (rows, completions), dtype=torch.float32, device=device
            ),
        }

    combo_cards = evaluator.combo_cards.to(device=device)
    active_cards = torch.arange(
        52 - 4 * geometry.rank_count,
        52,
        dtype=torch.int64,
        device=device,
    )
    board_generator = torch.Generator(device=device).manual_seed(int(board_seed))
    node_columns = {
        name: getattr(plan, name).to(device=device, dtype=torch.int64)
        for name in (
            "child_begin",
            "destination_work_row",
            "input_type",
            "output_type",
            "owner_slot",
            "sigma_head_row",
            "action_begin",
            "child_source_work_row",
            "child_source_output_type",
            "movement_source_leaf_row",
            "movement_destination_work_row",
            "movement_output_type",
        )
    }
    utility_scale = (
        1.0 if operations.utility == "raw_bb" else 1.0 / float(geometry.stack_bb)
    )
    world_chunk = 64
    for world_begin in range(0, completions, world_chunk):
        world_end = min(world_begin + world_chunk, completions)
        chunk = world_end - world_begin
        chunk_profiles = profiles[:, world_begin:world_end].to(torch.int64)
        private_cards = combo_cards[chunk_profiles.clamp_min(0)]
        base_board = fields["leaf_public_state"][:, None].expand(-1, chunk, -1)
        forbidden = (
            active_cards[None, None, :, None, None]
            == private_cards[:, :, None, :, :].to(torch.int64)
        ).any(dim=(-1, -2))
        forbidden |= (
            active_cards[None, None, :, None]
            == base_board[:, :, None, :].to(torch.int64)
        ).any(-1)
        scores = torch.rand(
            (leaf_count, chunk, active_cards.numel()),
            dtype=torch.float32,
            device=device,
            generator=board_generator,
        ).masked_fill(forbidden, 2.0)
        runout = scores.topk(5, dim=-1, largest=False).indices
        runout = active_cards.index_select(0, runout.reshape(-1)).reshape_as(runout)
        pad = base_board.eq(255)
        pad_ordinal = pad.cumsum(-1).sub(1).clamp_min(0).to(torch.int64)
        completed_board = torch.where(
            pad,
            runout.gather(-1, pad_ordinal).to(torch.uint8),
            base_board,
        ).contiguous()
        flat = leaf_count * chunk
        showdown_profile = chunk_profiles.masked_fill(
            fields["stratum_level"][:, None].lt(0), -1
        )
        payoff = evaluator.evaluate_showdowns(
            board_cards=completed_board.reshape(flat, 5),
            combo_ids=showdown_profile.reshape(flat, geometry.players).to(torch.int32),
            stratum_level=fields["stratum_level"][:, None]
            .expand(-1, chunk, -1)
            .reshape(flat, geometry.players)
            .contiguous(),
            payoff_baseline=fields["payoff_baseline"][:, None]
            .expand(-1, chunk, -1)
            .reshape(flat, geometry.players)
            .contiguous(),
            stratum_amount=fields["stratum_amount"][:, None]
            .expand(-1, chunk, -1)
            .reshape(flat, -1)
            .contiguous(),
        ).reshape(leaf_count, chunk, geometry.players)
        payoff = payoff.permute(1, 0, 2).float().mul_(utility_scale)
        exact_combo = chunk_profiles.permute(1, 0, 2).contiguous()

        work_value = [
            [
                torch.empty(
                    (
                        chunk,
                        int(plan.arena_rows[0][output]),
                        geometry.players,
                    ),
                    dtype=torch.float32,
                    device=device,
                )
                for _ in range(2)
            ]
            for output in range(len(output_names))
        ]
        work_combo = [
            [
                torch.empty(
                    (
                        chunk,
                        int(plan.arena_rows[0][output]),
                        geometry.players,
                    ),
                    dtype=torch.int64,
                    device=device,
                )
                for _ in range(2)
            ]
            for output in range(len(output_names))
        ]
        movement_source = node_columns["movement_source_leaf_row"]
        movement_destination = node_columns["movement_destination_work_row"]
        movement_output = node_columns["movement_output_type"]
        for output in range(len(output_names)):
            selected = torch.nonzero(
                movement_output.eq(output), as_tuple=False
            ).flatten()
            if selected.numel() == 0:
                continue
            source = movement_source.index_select(0, selected)
            destination = movement_destination.index_select(0, selected)
            work_value[output][0][:, destination] = payoff.index_select(1, source)
            work_combo[output][0][:, destination] = exact_combo.index_select(1, source)

        layer_ptr = plan.layer_ptr.tolist()
        for layer, (first, end) in enumerate(pairwise(layer_ptr)):
            if first == end:
                continue
            nodes = torch.arange(first, end, dtype=torch.int64, device=device)
            edges = node_columns["child_begin"].index_select(0, nodes)
            encoded = node_columns["child_source_work_row"].index_select(0, edges)
            source_output = node_columns["child_source_output_type"].index_select(
                0, edges
            )
            child_value = torch.empty(
                (chunk, nodes.numel(), geometry.players),
                dtype=torch.float32,
                device=device,
            )
            child_combo = torch.empty(
                (chunk, nodes.numel(), geometry.players),
                dtype=torch.int64,
                device=device,
            )
            for output in range(len(output_names)):
                for arena in range(2):
                    arena_mask = encoded.lt(0) if arena else encoded.ge(0)
                    selected = torch.nonzero(
                        source_output.eq(output) & arena_mask,
                        as_tuple=False,
                    ).flatten()
                    if selected.numel() == 0:
                        continue
                    source_row = encoded.index_select(0, selected)
                    if arena:
                        source_row = -source_row - 1
                    child_value[:, selected] = work_value[output][arena].index_select(
                        1, source_row
                    )
                    child_combo[:, selected] = work_combo[output][arena].index_select(
                        1, source_row
                    )

            parent_value = child_value.clone()
            owner = node_columns["owner_slot"].index_select(0, nodes)
            output_type = node_columns["output_type"].index_select(0, nodes)
            input_type = node_columns["input_type"].index_select(0, nodes)
            destination = node_columns["destination_work_row"].index_select(0, nodes)
            action_row = node_columns["action_begin"].index_select(0, nodes)
            sigma_row = node_columns["sigma_head_row"].index_select(0, nodes)
            trained = torch.nonzero(owner.ge(0), as_tuple=False).flatten()
            for input_index, name in enumerate(input_names):
                within = torch.nonzero(
                    input_type.index_select(0, trained).eq(input_index),
                    as_tuple=False,
                ).flatten()
                selected = trained.index_select(0, within)
                if selected.numel() == 0:
                    continue
                selected_owner = owner.index_select(0, selected)
                selected_output = output_type.index_select(0, selected)
                if not bool(selected_output.eq(selected_output[0]).all()):
                    raise DirectReferenceError(
                        "one input partition mapped to mixed outputs"
                    )
                output = int(selected_output[0])
                owner_index = selected_owner[None, :, None].expand(chunk, -1, 1)
                owner_combo = child_combo[:, selected].gather(2, owner_index).squeeze(2)
                hand = (
                    hand_maps[output]
                    .index_select(0, owner_combo.reshape(-1))
                    .reshape(chunk, -1)
                )
                old_table = owners.slab("SigmaHead", name).columns["old_value"][0]
                selected_sigma = sigma_row.index_select(0, selected)
                old_value = old_table[
                    selected_sigma[None].expand(chunk, -1), hand
                ].float()
                child_owner = child_value[:, selected].gather(2, owner_index).squeeze(2)
                advantage = child_owner - old_value
                selected_action = action_row.index_select(0, selected)
                width = cache[name]["count"].shape[1]
                dense_index = selected_action[None] * width + hand
                cache[name]["count"].view(-1).scatter_add_(
                    0,
                    dense_index.reshape(-1),
                    torch.ones_like(advantage).reshape(-1),
                )
                cache[name]["positive_sum"].view(-1).scatter_add_(
                    0,
                    dense_index.reshape(-1),
                    advantage.clamp_min(0).reshape(-1),
                )
                cache[name]["negative_sum"].view(-1).scatter_add_(
                    0,
                    dense_index.reshape(-1),
                    advantage.clamp_max(0).reshape(-1),
                )
                cache[name]["world_hand"][selected_action, world_begin:world_end] = (
                    hand.t().to(torch.int16)
                )
                cache[name]["world_advantage"][
                    selected_action, world_begin:world_end
                ] = advantage.t()
                updated = lambda_ * child_owner + (1.0 - lambda_) * old_value
                parent_value[:, selected].scatter_(2, owner_index, updated[:, :, None])

            destination_arena = (layer + 1) & 1
            for output in range(len(output_names)):
                selected = torch.nonzero(
                    output_type.eq(output), as_tuple=False
                ).flatten()
                if selected.numel() == 0:
                    continue
                rows = destination.index_select(0, selected)
                work_value[output][destination_arena][:, rows] = parent_value[
                    :, selected
                ]
                work_combo[output][destination_arena][:, rows] = child_combo[
                    :, selected
                ]

    for name, values in cache.items():
        count = values.pop("count")
        positive = values.pop("positive_sum")
        negative = values.pop("negative_sum")
        if not bool(count.sum(1).eq(float(completions)).all()):
            raise DirectReferenceError(f"signed reference omitted worlds in {name}")
        denominator = count.clamp_min(1.0)
        values["posterior"] = count / float(completions)
        values["positive"] = positive / denominator
        values["negative"] = negative / denominator
    return cache, {
        "sampler": (
            "exact collision-conditioned joint profiles plus uniform CUDA runouts"
        ),
        "worlds_per_public_trajectory": completions,
        "leaf_count": leaf_count,
        "board_seed": board_seed,
        "world_chunk": world_chunk,
    }


def signed_batch_losses(
    ratio: Tensor,
    moments: Mapping[str, Tensor],
    *,
    clip_epsilon: float,
) -> tuple[Tensor, Tensor]:
    """Return cached and direct world-average losses for aligned rows."""

    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    positive = moments["positive"]
    negative = moments["negative"]
    posterior = moments["posterior"]
    cached = -(
        posterior
        * (
            torch.minimum(ratio, clipped) * positive
            + torch.maximum(ratio, clipped) * negative
        )
    ).sum(-1)
    world_hand = moments["world_hand"].to(torch.int64)
    world_advantage = moments["world_advantage"]
    world_ratio = ratio.gather(1, world_hand)
    world_clipped = world_ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    direct = -torch.minimum(
        world_ratio * world_advantage,
        world_clipped * world_advantage,
    ).mean(1)
    return cached, direct


__all__ = [
    "DirectReferenceError",
    "direct_reference_moments",
    "signed_batch_losses",
]
