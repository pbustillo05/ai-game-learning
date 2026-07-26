"""Generate small CUDA-kernel range databases from historical checkpoints.

This script is intentionally executed with the historical repository revision
that wrote a checkpoint.  It samples 1,000 sequential policy trajectories,
uniformly chooses 100 distinct decision nodes, and saves the complete
gauge-shifted range weights for every seat, including seats that have folded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import torch


T = TypeVar("T")
N_HANDS = 1326
FOLDED = 1


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataclass_from_mapping(cls: type[T], raw: dict[str, Any]) -> T:
    """Construct one historical config dataclass from checkpoint-ready values."""

    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    accepted = {item.name for item in fields(cls)}
    return cls(**{key: value for key, value in raw.items() if key in accepted})


def _legacy_config(payload: dict[str, Any]) -> Any:
    from pbg.core.config import (
        CompletionConfig,
        GameConfig,
        HardwareProfileConfig,
        LossConfig,
        RolloutConfig,
        RootConfig,
        TrainingConfig,
    )

    raw = payload["config"]
    game = dict(raw["game"]["poker"])
    game.pop("sb", None)
    game.pop("bb", None)
    game.pop("chip_unit_bb", None)
    # The pre-compact-deck checkpoint predates rank_count and used all ranks.
    rollout = dict(raw["rollout"])
    # Four consecutive batches avoid the historical half-free-VRAM preflight
    # while still sampling exactly 1,000 trajectories from one generator stream.
    rollout["n_trees"] = 250
    rollout["node_budget"] = 69
    rollout["reach_storage"] = "decisions_only"
    training = dict(raw["training"])
    training["mode"] = "rao_blackwell"
    training.setdefault("checkpoint_every", 1)
    return RootConfig(
        game=GameConfig(name="poker", poker=game),
        rollout=_dataclass_from_mapping(RolloutConfig, rollout),
        completion=_dataclass_from_mapping(CompletionConfig, raw["completion"]),
        model=dict(raw["model"]),
        loss=_dataclass_from_mapping(LossConfig, raw["loss"]),
        training=_dataclass_from_mapping(TrainingConfig, training),
        hardware=_dataclass_from_mapping(
            HardwareProfileConfig, payload["hardware_profile"]
        ),
    )


def _v3_config(payload: dict[str, Any]) -> Any:
    from pbg.core.config import (
        CompletionConfig,
        GameConfig,
        HardwareProfileConfig,
        LossConfig,
        ProbeConfig,
        RolloutConfig,
        RootConfig,
        TrainingConfig,
    )

    raw = payload["config"]
    rollout = dict(raw["rollout"])
    rollout["sequential_roots_per_gpu_initial"] = 1000
    rollout["sequential_roots_per_gpu_minimum"] = 1000
    rollout["sequential_roots_per_gpu_maximum"] = 1000
    training = dict(raw["training"])
    training.setdefault("checkpoint_every", 1)
    probe = raw.get("probe")
    return RootConfig(
        game=_dataclass_from_mapping(GameConfig, raw["game"]),
        rollout=_dataclass_from_mapping(RolloutConfig, rollout),
        completion=_dataclass_from_mapping(CompletionConfig, raw["completion"]),
        model=dict(raw["model"]),
        loss=_dataclass_from_mapping(LossConfig, raw["loss"]),
        training=_dataclass_from_mapping(TrainingConfig, training),
        probe=None if probe is None else _dataclass_from_mapping(ProbeConfig, probe),
        hardware=_dataclass_from_mapping(
            HardwareProfileConfig, payload["hardware_profile"]
        ),
    )


def _decision_rows(runtime: str, rollout: Any, backend: Any) -> tuple[Any, torch.Tensor]:
    if runtime == "legacy":
        tree, reach_cache, policy = rollout
        rows = reach_cache.rows_for(tree, policy.node_ids, backend, policy)
        return tree, rows
    tree, _leaf_reach, policy = rollout
    decision = policy.decision_reach
    rows = decision.gather(
        torch.arange(decision.node_ids.numel(), device=decision.node_ids.device),
        backend,
    )
    return tree, rows


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", choices=("legacy", "v3"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("range database generation requires CUDA")
    device = torch.device("cuda:0")
    checkpoint = args.checkpoint.resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = _legacy_config(payload) if args.runtime == "legacy" else _v3_config(payload)

    import pbg.core.rollout as rollout_module
    from pbg.core.rollout import sample_trees
    legacy_policy_fallback = {"count": 0}
    if args.runtime == "v3":
        from pbg.core.rollout import sample_trees_sequential
    else:
        from pbg.games.poker.adapter import PokerPolicyAdapter
        from pbg.games.poker.engine import CHECK_CALL, FOLD, RAISE_OFFSET
        from pbg.games.poker.model.heads import dlogistic_tick

        # This pre-sealed-storage runtime cached log probabilities for all 1,326
        # private hands after sampling an action. Some board-conflicting rows
        # intentionally have zero policy mass, and newer Torch versions enforce
        # the obsolete all-row assertion asynchronously. Reach generation does
        # not consume this cache, so retain its required shape without invoking
        # the irrelevant full-private-row log-probability calculation.
        def unused_policy_logp(
            adapter: Any,
            output: Any,
            selected: torch.Tensor,
            legal_ids: torch.Tensor,
            legal_mask: torch.Tensor,
        ) -> torch.Tensor:
            del adapter, legal_ids, legal_mask
            return torch.zeros(
                (*selected.shape, output.v.shape[-1]),
                dtype=torch.float32,
                device=selected.device,
            )

        rollout_module._selected_policy_logp = unused_policy_logp

        def robust_selected_probabilities(
            self: Any,
            raw: Any,
            legal_event_ids: torch.Tensor,
            legal_mask: torch.Tensor,
        ) -> torch.Tensor:
            """Historical selected-row policy with an explicit degenerate fallback."""

            fields = raw.raw if hasattr(raw, "raw") else raw
            type_logits = fields["type_logits"]
            if type_logits.ndim < 2 or type_logits.shape[-1] != 3:
                raise ValueError("legacy policy type logits must end in width three")
            batch = type_logits.shape[0]
            n_ticks = legal_event_ids.shape[1] - RAISE_OFFSET
            type_probability = torch.softmax(type_logits.float(), dim=-1)
            prefix = type_logits.shape[:-1]
            logistic = dlogistic_tick(
                n_ticks,
                fields["mu"].float(),
                fields["s"].float(),
            )
            mixture = (fields["omega"].float()[..., None] * logistic).sum(dim=-2)
            raise_mask = legal_mask[:, RAISE_OFFSET:].to(type_logits.device)
            mask_shape = (batch,) + (1,) * (mixture.ndim - 2) + (n_ticks,)
            expanded_raise_mask = raise_mask.reshape(mask_shape).expand_as(mixture)
            mixture = mixture.masked_fill(~expanded_raise_mask, 0.0)
            mixture = mixture / mixture.sum(-1, keepdim=True).clamp_min(self.eps)
            tick_index = torch.arange(n_ticks, device=type_logits.device)[None]
            allin_index = torch.where(
                raise_mask, tick_index, torch.full_like(tick_index, -1)
            ).max(1).values
            allin = torch.zeros(
                (batch, n_ticks),
                dtype=torch.float32,
                device=type_logits.device,
            )
            has_raise = allin_index >= 0
            allin.scatter_(
                1,
                allin_index.clamp_min(0)[:, None],
                has_raise.to(torch.float32)[:, None],
            )
            allin = allin.reshape(mask_shape).expand_as(mixture)
            ticks = (
                (1.0 - fields["iota"].float()[..., None]) * mixture
                + fields["iota"].float()[..., None] * allin
            ).masked_fill(~expanded_raise_mask, 0.0)
            full = torch.zeros(
                prefix + (RAISE_OFFSET + n_ticks,),
                dtype=torch.float32,
                device=type_logits.device,
            )
            facing = ((legal_event_ids == FOLD) & legal_mask).any(dim=1)
            facing = facing.reshape(
                (batch,) + (1,) * (type_logits.ndim - 2)
            )
            full[..., FOLD] = torch.where(
                facing, type_probability[..., 0], 0.0
            )
            full[..., CHECK_CALL] = torch.where(
                facing, type_probability[..., 1], type_probability[..., 0]
            )
            full[..., RAISE_OFFSET:] = type_probability[..., 2, None] * ticks
            event_shape = (
                (batch,)
                + (1,) * (type_logits.ndim - 2)
                + (legal_event_ids.shape[1],)
            )
            ids = legal_event_ids.to(
                device=full.device, dtype=torch.long
            ).reshape(event_shape).expand(prefix + (legal_event_ids.shape[1],))
            expanded_legal = legal_mask.to(full.device).reshape(
                event_shape
            ).expand_as(ids)
            probability = full.gather(-1, ids).masked_fill(~expanded_legal, 0.0)
            normalizer = probability.sum(dim=-1, keepdim=True)
            bad = (~torch.isfinite(normalizer)) | (normalizer <= 0)
            count = int(bad.sum().item()) if type_logits.ndim == 2 else 0
            if count:
                legacy_policy_fallback["count"] += count
            if bool(bad.any()):
                fallback = expanded_legal.to(probability.dtype)
                fallback = fallback / fallback.sum(dim=-1, keepdim=True)
                probability = torch.where(bad, fallback, probability)
                normalizer = probability.sum(dim=-1, keepdim=True)
            return probability / normalizer

        PokerPolicyAdapter.probabilities = robust_selected_probabilities
        PokerPolicyAdapter.compose = robust_selected_probabilities
    from pbg.games.poker.bundle import build_bundle

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    bundle = build_bundle(cfg, device)
    bundle.model.load_state_dict(payload["model"], strict=True)
    bundle.model.eval()
    # The v3 runtime normally compacts away folded decision rows after their
    # last production consumer. Kernel fixtures require the complete P x H row.
    bundle.rollout_backend.requires_full_decision_reach = True

    cpu_generator = torch.Generator().manual_seed(args.seed ^ 0x5EED5EED)

    def materialize(
        tree: Any,
        log_ranges: torch.Tensor,
        decision_ids: torch.Tensor,
        selected_rows_cpu: torch.Tensor,
        *,
        trajectory_offset: int,
        rollout_batch: int,
    ) -> dict[str, torch.Tensor]:
        selected_rows = selected_rows_cpu.to(device)
        selected_nodes = decision_ids[selected_rows].to(torch.int64)
        selected_states = tree.state_ref[selected_nodes].to(torch.int64)
        selected_log = log_ranges[selected_rows]
        finite = torch.isfinite(selected_log)
        gauge = selected_log.masked_fill(~finite, -torch.inf).amax(
            dim=2, keepdim=True
        )
        selected_ranges = torch.exp(selected_log - gauge).masked_fill(~finite, 0.0)
        state_store = bundle.rollout_backend.states
        return {
            "ranges": selected_ranges.cpu(),
            "is_folded": (state_store.status[selected_states] == FOLDED).cpu(),
            "trajectory_id": (
                tree.tree_id[selected_nodes].to(torch.int64) + trajectory_offset
            ).cpu(),
            "node_id": selected_nodes.cpu(),
            "decision_row": selected_rows_cpu,
            "street": state_store.street[selected_states].cpu(),
            "actor": state_store.actor[selected_states].cpu(),
            "rollout_batch": torch.full(
                (selected_rows_cpu.numel(),), rollout_batch, dtype=torch.int32
            ),
        }

    with torch.inference_mode():
        if args.runtime == "legacy":
            reservoir: dict[str, torch.Tensor] = {}
            reservoir_keys = torch.empty(0)
            decision_count = 0
            for batch in range(4):
                rollout = sample_trees(bundle, cfg, generator)
                tree, log_ranges = _decision_rows(
                    args.runtime, rollout, bundle.rollout_backend
                )
                decision_ids = rollout[2].node_ids
                batch_count = int(decision_ids.numel())
                if batch_count == 0:
                    continue
                keys = torch.rand(batch_count, generator=cpu_generator)
                batch_keep = min(100, batch_count)
                batch_keys, selected_rows_cpu = torch.topk(
                    keys, batch_keep, largest=False, sorted=False
                )
                candidate = materialize(
                    tree,
                    log_ranges,
                    decision_ids,
                    selected_rows_cpu,
                    trajectory_offset=batch * 250,
                    rollout_batch=batch,
                )
                combined_keys = torch.cat((reservoir_keys, batch_keys))
                combined = {
                    name: torch.cat((reservoir[name], value))
                    if name in reservoir
                    else value
                    for name, value in candidate.items()
                }
                keep = min(100, combined_keys.numel())
                reservoir_keys, selected = torch.topk(
                    combined_keys, keep, largest=False, sorted=False
                )
                reservoir = {name: value[selected] for name, value in combined.items()}
                decision_count += batch_count
            if decision_count < 100:
                raise RuntimeError(
                    f"only {decision_count} decision nodes were generated"
                )
            selected_data = reservoir
        else:
            rollout = sample_trees_sequential(
                bundle, cfg, generator, iteration=int(payload["iteration"])
            )
            tree, log_ranges = _decision_rows(
                args.runtime, rollout, bundle.rollout_backend
            )
            decision_ids = rollout[2].node_ids
            decision_count = int(decision_ids.numel())
            if decision_count < 100:
                raise RuntimeError(
                    f"only {decision_count} decision nodes were generated"
                )
            selected_rows_cpu = torch.randperm(
                decision_count, generator=cpu_generator
            )[:100]
            selected_data = materialize(
                tree,
                log_ranges,
                decision_ids,
                selected_rows_cpu,
                trajectory_offset=0,
                rollout_batch=0,
            )

    ranges = selected_data["ranges"]
    is_folded = selected_data["is_folded"]

    expected_players = (
        int(cfg.game.poker["n_players"])
        if args.runtime == "legacy"
        else int(cfg.game.n_players)
    )
    if ranges.shape != (100, expected_players, N_HANDS):
        raise RuntimeError(f"unexpected range database shape {tuple(ranges.shape)}")
    if is_folded.shape != ranges.shape[:2]:
        raise RuntimeError("folded mask does not align with ranges")
    if not bool(torch.isfinite(ranges).all()) or bool((ranges < 0).any()):
        raise RuntimeError("saved ranges must be finite and nonnegative")

    players = int(ranges.shape[1])
    database = {
        "schema_version": "pbg-range-kernel-db-v1",
        "ranges": ranges.contiguous(),
        "is_folded": is_folded.contiguous(),
        "trajectory_id": selected_data["trajectory_id"].to(torch.int32).contiguous(),
        "node_id": selected_data["node_id"].to(torch.int32).contiguous(),
        "decision_row": selected_data["decision_row"].to(torch.int32).contiguous(),
        "street": selected_data["street"].to(torch.int8).contiguous(),
        "actor": selected_data["actor"].to(torch.int32).contiguous(),
        "rollout_batch": selected_data["rollout_batch"].contiguous(),
        "metadata": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _checkpoint_sha256(checkpoint),
            "checkpoint_iteration": int(payload["iteration"]),
            "checkpoint_schema_version": payload.get("checkpoint_schema_version"),
            "historical_source_revision": args.source_revision,
            "runtime": args.runtime,
            "seed": args.seed,
            "trajectory_count": 1000,
            "decision_node_count": decision_count,
            "sample_count": 100,
            "players": players,
            "hands": N_HANDS,
            "range_semantics": (
                "exp(log_reach - per-seat finite maximum); folded seats retained"
            ),
            "sampling": "100 distinct decision rows sampled uniformly without replacement",
            "legacy_uniform_legal_action_fallbacks": legacy_policy_fallback["count"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(database, args.output)
    summary = {
        **database["metadata"],
        "output": str(args.output.resolve()),
        "shape": list(database["ranges"].shape),
        "dtype": str(database["ranges"].dtype),
        "folded_rows": int(database["is_folded"].sum()),
        "streets": torch.bincount(
            database["street"].to(torch.int64), minlength=4
        ).tolist(),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    _main()
