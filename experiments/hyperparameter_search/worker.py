"""CUDA-only, resumable worker for shared Normal/RB parameter experiments.

Use campaign.py for the registered search, or --profile for standalone training.
The external exposure clock is part of the checkpoint and is authoritative;
these profiles must not be launched through the ordinary production CLI.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import signal
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from experiments.hyperparameter_search.exposure import (
    install_posterior_exposure_observer,
)
from experiments.hyperparameter_search.protocol import (
    ExposureClock,
    atomic_json,
    canonical_sha256,
    resolve_spec,
    sha256_file,
)
from experiments.training_diagnosis.restoration_smoke import historical_adapter
from pbg.config import load_config_mapping
from pbg.core.learner import EntropyLossConfig
from pbg.core.optimization import LossScaler
from pbg.core.persistence import (
    load_checkpoint,
    load_distributed_checkpoint_manifest,
    model_state_sha256,
)
from pbg.games.poker.queued.model import build_model_from_config
from pbg.runtime import TrainingSession


def install_adapters(session):
    for owner in (
        session.modules,
        session.reference_modules,
        getattr(getattr(session, "frozen_inference", None), "modules", None),
    ):
        if owner is not None:
            historical_adapter(owner.lane_module(0), session.geometry)
            historical_adapter(owner._code, session.geometry)


def install_global_mean(session, world):
    if session.distributed is not None:
        original = session.distributed.sum_gradients

        def mean_gradients(gradients, active):
            original(gradients, active)
            gradients.div_(world)

        session.distributed.sum_gradients = mean_gradients


def install_entropy_clock(session, clock, control):
    original = session._iteration_engine

    def factory():
        engine = original()
        entropy = EntropyLossConfig(
            kind="constant", coefficient=clock.entropy_coefficient(control)
        )
        engine.config = replace(
            engine.config,
            lanes=tuple(
                replace(lane, loss=replace(lane.loss, entropy=entropy))
                for lane in engine.config.lanes
            ),
        )
        return engine

    session._iteration_engine = factory


def response_probe(source, spec, device):
    """Supply the production fixed-target cutoff protocol with verified hashes."""
    if spec["initialization"] == "target":
        payload = torch.load(
            spec["target_checkpoint"], map_location="cpu", weights_only=False, mmap=True
        )
        initial_hash = model_state_sha256(payload["model"])
        del payload
    else:
        cfg = load_config_mapping(source)
        torch.manual_seed(source["run"]["seed"])
        model = build_model_from_config(cfg.scientific_mapping(), device)
        initial_hash = model_state_sha256(model)
        del model
        torch.cuda.empty_cache()
    source["scientific"]["probe"] = {
        "target_sha256": source["run"]["frozen"]["sha256"],
        "initial_model_sha256": initial_hash,
    }


def restore_bundle(session, path, world):
    """Validate the rank-zero manifest, then each rank's own hardware manifest.

    The server's nominally identical GPUs expose slightly different total VRAM.
    Production's distributed loader assumes one identical execution manifest;
    retaining strict per-rank checks avoids silently relaxing that invariant.
    """
    execution = session.config.execution.mapping()
    scientific = session.config.scientific_mapping()
    if world > 1:
        rank_zero = [execution if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(rank_zero, src=0)
        paths = load_distributed_checkpoint_manifest(
            path,
            expected_scientific_config=scientific,
            expected_execution_config=rank_zero[0],
            world_size=world,
        )
        path = paths[dist.get_rank()]
    optimizer = session.optimizers.optimizers[0]
    scaler, reference, state, _ = load_checkpoint(
        path,
        model=session.modules.lane_module(0),
        optimizer=optimizer,
        reference_model=None
        if session.reference_modules is None
        else session.reference_modules.lane_module(0),
        expected_scientific_config=scientific,
        expected_execution_config=execution,
        generator=session.generators[0],
    )
    optimizer.scaler = scaler
    session.reference_states = None if reference is None else (reference,)
    session.run_states = (state,)


def save_snapshot(session, path, spec, checkpoint_hash):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    # Host transfers here are serialization only; training remains CUDA-only.
    payload = {
        "model": {
            key: value.detach().cpu()
            for key, value in session.modules.lane_module(0).state_dict().items()
        },
        "scientific_config": session.config.scientific_mapping(),
        "iteration": session.run_states[0].iteration,
        "legacy_preflop_inputs": True,
        "input_encoding": "july_exact_hand_preflop_v1",
        "utility": "stack_normalized",
        "source_checkpoint_sha256": checkpoint_hash,
        "policy_role": "trained_best_response"
        if spec["kind"] == "best_response"
        else "self_play_target",
        "initialization_kind": spec["initialization"],
        "initializer_checkpoint": spec["profile"]["initialization"]["checkpoint"]
        if spec["kind"] == "self_play"
        else spec["target_checkpoint"],
        "metadata": {
            "job_sha256": canonical_sha256(spec),
            "private_estimator": session.config.scientific_mapping()["mode"][
                "private_estimator"
            ],
        },
    }
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run(spec, *, interrupt_after_iterations=None):
    torch.set_num_threads(2)
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if not visible[0] or len(visible) != world:
        raise RuntimeError(
            "Explicit CUDA_VISIBLE_DEVICES must match the single-node worker world size"
        )
    if int(os.environ.get("LOCAL_WORLD_SIZE", str(world))) != world:
        raise RuntimeError("Only single-node CUDA execution is supported")
    # Check before creating a CUDA context, independently on each selected GPU.
    busy = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            visible[rank],
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    if busy:
        raise RuntimeError(f"Refusing occupied GPU {visible[rank]}: {busy}")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    source, control = resolve_spec(spec, rank, world)
    output = Path(spec["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec_hash = canonical_sha256(spec)
    if spec["kind"] == "best_response":
        response_probe(source, spec, device)
    cfg = load_config_mapping(source)
    resolved = {
        "config": source,
        "control": control,
        "spec_sha256": spec_hash,
        "world_size": world,
    }
    config_path = output / f"resolved.rank_{rank:04d}.json"
    if config_path.exists() and json.loads(config_path.read_text()) != resolved:
        raise RuntimeError("Resume configuration/topology differs from original worker")
    atomic_json(config_path, resolved)
    latest_path = output / "latest.json"
    previous = json.loads(latest_path.read_text()) if latest_path.exists() else None
    if previous and previous["spec_sha256"] != spec_hash:
        raise RuntimeError("Resume job specification differs from saved checkpoint")
    if (output / "result.json").exists():
        completed = json.loads((output / "result.json").read_text())
        if completed.get("complete"):
            return 0
    clock = ExposureClock.from_dict(previous["clock"]) if previous else ExposureClock()
    resume_from = previous["checkpoint_manifest"] if previous else None
    initialization = spec["profile"]["initialization"]
    initial_state = None
    if spec["kind"] == "self_play" and previous is None:
        path = Path(initialization["checkpoint"])
        if sha256_file(path) != initialization["sha256"]:
            raise RuntimeError("Initializer checkpoint content changed")
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        initial_state = payload[spec["initialization"]]
    session = TrainingSession(
        cfg, initial_model_state=initial_state, benchmark_telemetry=True
    )
    del initial_state
    if spec["kind"] == "self_play" and previous is None:
        del payload
    monitor = None
    telemetry_stream = None
    old_handlers = {}
    stopping = False

    def on_signal(signum, frame):
        nonlocal stopping
        stopping = True

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            old_handlers[signum] = signal.signal(signum, on_signal)
        if resume_from is not None:
            restore_bundle(session, resume_from, world)
        if len(session.run_states) != 1:
            raise RuntimeError(
                "Campaign worker requires exactly one policy replica per rank"
            )
        install_adapters(session)
        install_global_mean(session, world)
        model = session.modules.lane_module(0)
        if previous is None:
            if spec["kind"] == "self_play":
                names = []
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        if name.endswith(
                            ("value_head.affine.weight", "value_head.affine.bias")
                        ):
                            parameter.div_(float(initialization["value_scale"]))
                            names.append(name)
                if len(names) != 4:
                    raise RuntimeError(
                        f"Unexpected normalized value-head migration: {names}"
                    )
                if session.reference_modules is not None:
                    session.reference_modules.lane_module(0).load_state_dict(
                        model.state_dict(), strict=True
                    )
            session.optimizers.optimizers[0].scaler = LossScaler(
                enabled=True,
                scale=float(control["initial_loss_scale"]),
                growth_interval=2000,
            )
        session.initial_model_sha256 = (
            (model_state_sha256(model),)
            if previous is None
            else tuple(previous["initial_model_sha256"])
        )
        install_entropy_clock(session, clock, control)
        exposure_observations = []
        install_posterior_exposure_observer(session, exposure_observations)
        telemetry_stream = (output / f"gpu.rank_{rank:04d}.csv").open("a")
        monitor = subprocess.Popen(
            [
                "nvidia-smi",
                "-i",
                visible[rank],
                "--query-gpu=timestamp,uuid,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown",
                "--format=csv,noheader,nounits",
                "-l",
                "5",
            ],
            stdout=telemetry_stream,
            stderr=subprocess.DEVNULL,
        )
        budget_path = output / "budget.json"
        charged = float(previous["training_seconds"]) if previous else 0.0
        if budget_path.exists():
            ledger = json.loads(budget_path.read_text())
            if ledger["spec_sha256"] != spec_hash:
                raise RuntimeError("Budget ledger belongs to another job")
            charged = max(charged, float(ledger["training_seconds"]))
        started = time.monotonic()
        last_checkpoint = started
        last_snapshot = started
        last_native = Path(previous["checkpoint"]) if previous else None
        interval = (
            control["reference"]["interval_decisions"]
            if session.reference_modules is not None
            else None
        )
        global_roots = int(previous.get("global_roots", 0)) if previous else 0
        reference_copies = int(previous.get("reference_copies", 0)) if previous else 0
        first_iteration = session.run_states[0].iteration
        checkpoint_sequence = (
            int(previous.get("checkpoint_sequence", 0)) if previous else 0
        )
        metric_path = output / f"metrics.rank_{rank:04d}.jsonl"
        if previous and metric_path.exists():
            lines = metric_path.read_text().splitlines()
            kept = [
                line
                for line in lines
                if json.loads(line)["iteration"] <= first_iteration
            ]
            if len(kept) != len(lines):
                shutil.copy2(
                    metric_path,
                    metric_path.with_suffix(f".before_resume_{first_iteration}.jsonl"),
                )
                metric_path.write_text("".join(line + "\n" for line in kept))
        metrics = previous.get("last_metrics", {}) if previous else {}

        def shared_elapsed():
            elapsed = torch.tensor(
                charged + time.monotonic() - started, dtype=torch.float64, device=device
            )
            if world > 1:
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            return float(elapsed)

        def barrier():
            if world > 1:
                dist.barrier()

        def checkpoint(elapsed, snapshot=False):
            nonlocal last_native, last_checkpoint, last_snapshot, checkpoint_sequence
            iteration = session.run_states[0].iteration
            checkpoint_sequence += 1
            bundle = (
                output
                / "checkpoints"
                / f"iteration_{iteration:08d}_save_{checkpoint_sequence:08d}"
            )
            bundle.mkdir(parents=True, exist_ok=True)
            destination = bundle / "state.pt"
            session.checkpoint(destination)
            last_native = (
                destination.with_name("state.rank_0000.pt")
                if world > 1
                else destination
            )
            if rank == 0:
                checkpoint_hash = sha256_file(last_native)
                clock_payload = {
                    "spec_sha256": spec_hash,
                    "clock": clock.to_dict(),
                    "training_seconds": elapsed,
                    "global_roots": global_roots,
                    "reference_copies": reference_copies,
                    "checkpoint_sequence": checkpoint_sequence,
                    "checkpoint_manifest": str(destination),
                    "checkpoint": str(last_native),
                    "checkpoint_sha256": checkpoint_hash,
                    "initial_model_sha256": list(session.initial_model_sha256),
                    "last_metrics": metrics,
                    "iteration": iteration,
                }
                atomic_json(bundle / "clock.json", clock_payload)
                atomic_json(latest_path, clock_payload)
                if snapshot:
                    save_snapshot(
                        session,
                        output / "snapshots" / f"iteration_{iteration:08d}.pt",
                        spec,
                        checkpoint_hash,
                    )
                bundles = sorted((output / "checkpoints").glob("iteration_*"))
                for obsolete in bundles[:-2]:
                    shutil.rmtree(obsolete)
            barrier()
            last_checkpoint = time.monotonic()
            if snapshot:
                last_snapshot = last_checkpoint

        if previous is None:
            checkpoint(shared_elapsed(), snapshot=True)
        status = "complete"
        while True:
            elapsed = shared_elapsed()
            interrupted = stopping or (
                interrupt_after_iterations is not None
                and session.run_states[0].iteration - first_iteration
                >= interrupt_after_iterations
            )
            if session._agree_flag(interrupted):
                status = "interrupted"
                break
            if elapsed >= float(spec["budget_seconds"]) or (
                spec.get("max_iterations") is not None
                and session.run_states[0].iteration >= spec["max_iterations"]
            ):
                break
            torch.cuda.reset_peak_memory_stats(device)
            coefficient = clock.entropy_coefficient(control)
            iteration_start = time.monotonic()
            exposure_observations.clear()
            session.step()
            torch.cuda.synchronize(device)
            metrics = copy.deepcopy(session.run_states[0].metrics_prefix[-1])
            gradient = metrics.get("gradient", {})
            expected_steps = source["scientific"]["schedule"]["steps"]
            failed = (
                not gradient.get("finite", False)
                or gradient.get("applied_steps") != expected_steps
            )
            if session._agree_flag(failed):
                raise RuntimeError(
                    "Nonfinite gradients or skipped optimizer steps: trial invalid; inspect metrics/log"
                )
            if len(exposure_observations) != 1:
                raise RuntimeError(
                    "Expected one fresh-posterior observation per iteration"
                )
            posterior = exposure_observations[0]
            totals = torch.tensor(
                [
                    posterior[key]
                    for key in (
                        "fresh_player_decisions",
                        "sum_posterior_exp_entropy",
                        "sum_sqrt_posterior_exp_entropy",
                    )
                ],
                dtype=torch.float64,
                device=device,
            )
            if world > 1:
                dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            decisions, raw_breadth, sqrt_breadth = totals.tolist()
            decisions = int(decisions)
            if spec["kind"] == "self_play" and decisions != int(
                metrics["global_player_nodes"]
            ):
                raise RuntimeError(
                    f"Posterior observer decision count {decisions} differs from fresh rollout count {metrics['global_player_nodes']}"
                )
            exponent = control.get("posterior_clock", {}).get("exponent", 1.0)
            effective = raw_breadth if exponent == 1.0 else sqrt_breadth
            presentations = (
                world * source["scientific"]["schedule"]["batch_size"] * expected_steps
            )
            due = clock.observe(
                decisions,
                presentations,
                expected_steps,
                interval,
                effective_hands=effective,
                schedule_clock=control.get("reference_clock", control["clock"]),
            )
            global_roots += world * source["run"]["workload"]["roots"]
            if due:
                session.reference_modules.lane_module(0).load_state_dict(
                    model.state_dict(), strict=True
                )
                session.reference_states[0].reference_iteration = session.run_states[
                    0
                ].iteration
                reference_copies += 1
            elapsed = shared_elapsed()
            row = {
                "iteration": session.run_states[0].iteration,
                "iteration_seconds": time.monotonic() - iteration_start,
                "training_seconds": elapsed,
                "entropy_coefficient_used": coefficient,
                "entropy_coefficient_next": clock.entropy_coefficient(control),
                "reference_copied_after_iteration": due,
                "reference_copies": reference_copies,
                "clock": clock.to_dict(),
                "global_roots": global_roots,
                "metrics": metrics,
                "posterior_exposure_local": posterior,
                "posterior_exposure_global": {
                    "fresh_player_decisions": decisions,
                    "sum_exp_entropy": raw_breadth,
                    "sum_sqrt_exp_entropy": sqrt_breadth,
                    "scheduled_exposure": effective,
                    "exponent": exponent,
                    "clock": control["clock"],
                    "entropy_clock": control.get("entropy_clock", control["clock"]),
                    "reference_clock": control.get("reference_clock", control["clock"]),
                    "completion_ESS_used": False,
                },
            }
            with (output / f"metrics.rank_{rank:04d}.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
            if rank == 0:
                atomic_json(
                    budget_path, {"spec_sha256": spec_hash, "training_seconds": elapsed}
                )
                print(
                    json.dumps(
                        {
                            key: row[key]
                            for key in (
                                "iteration",
                                "training_seconds",
                                "entropy_coefficient_used",
                                "reference_copies",
                                "clock",
                            )
                        }
                    ),
                    flush=True,
                )
            snapshot_due = (
                time.monotonic() - last_snapshot >= control["snapshot_seconds"]
            )
            checkpoint_due = (
                time.monotonic() - last_checkpoint >= control["checkpoint_seconds"]
            )
            if session._agree_flag(checkpoint_due or snapshot_due):
                checkpoint(elapsed, snapshot=session._agree_flag(snapshot_due))
        elapsed = shared_elapsed()
        checkpoint(elapsed, snapshot=True)
        final_hash = model_state_sha256(model)
        hashes = [final_hash] * world
        if world > 1:
            dist.all_gather_object(hashes, final_hash)
        if len(set(hashes)) != 1:
            raise RuntimeError("Distributed model replicas diverged")
        if rank == 0:
            inference = output / "inference.pt"
            save_snapshot(session, inference, spec, sha256_file(last_native))
            result = {
                "complete": status == "complete",
                "status": status,
                "spec_sha256": spec_hash,
                "training_seconds": elapsed,
                "snapshot": str(inference),
                "checkpoint": str(last_native),
                "checkpoint_manifest": str(last_native.parent / "state.pt"),
                "iteration": session.run_states[0].iteration,
                "model_sha256": final_hash,
                "rank_model_sha256": hashes,
                "world_size": world,
                "visible_devices": visible,
                "clock": clock.to_dict(),
                "reference_copies": reference_copies,
                "global_roots": global_roots,
                "entropy_coefficient_final": clock.entropy_coefficient(control),
                "metrics": metrics,
                "budget_includes": "rollout,completion,learner,checkpoint and telemetry overhead; excludes worker initialization",
                "limitations": "training stability does not establish policy strength",
            }
            atomic_json(output / "result.json", result)
        barrier()
        return 0 if status == "complete" else 75
    finally:
        if monitor is not None:
            monitor.terminate()
            monitor.wait(timeout=10)
        if telemetry_stream is not None:
            telemetry_stream.close()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--spec", type=Path)
    inputs.add_argument("--profile", type=Path)
    parser.add_argument("--mode", choices=("normal", "rb"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--seed", type=int, default=918273)
    parser.add_argument(
        "--interrupt-after-iterations",
        type=int,
        help="Checkpoint and return75 after N new iterations; resume verification only",
    )
    args = parser.parse_args()
    if args.spec:
        spec = json.loads(args.spec.read_text())
    else:
        if args.mode is None or args.output is None or args.hours <= 0:
            parser.error("--profile requires --mode, --output and positive --hours")
        profile = yaml.safe_load(args.profile.read_text())
        spec = {
            "kind": "self_play",
            "profile": profile,
            "arm": "control",
            "overrides": {},
            "private_estimator": "sampled"
            if args.mode == "normal"
            else "rao_blackwell",
            "seed": args.seed,
            "output_dir": str(args.output.resolve()),
            "budget_seconds": args.hours * 3600,
            "initialization": "model",
            "smoke": False,
        }
    raise SystemExit(
        run(spec, interrupt_after_iterations=args.interrupt_after_iterations)
    )


if __name__ == "__main__":
    main()
