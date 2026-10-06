"""Paired normalized-reward CUDA continuations of the July checkpoint.

Uses repaired production targets, fresh Adam state, and experiment-local legacy
input encoding. A short continuation diagnoses stability, not poker strength.
The saved config needs this driver to preserve the legacy checkpoint semantics.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch
import yaml

from experiments.training_diagnosis.probe_policies import OLD
from experiments.training_diagnosis.restoration_smoke import historical_adapter, policy_snapshot
from pbg.config import load_config_mapping
from pbg.core.optimization import LossScaler
from pbg.core.persistence import model_state_sha256
from pbg.games.poker.queued.replica import encode_model_evaluation
from pbg.runtime import TrainingSession


def posterior_observer(session, observations):
    """Measure fresh decision posteriors once, before learner resampling."""
    factory = session._iteration_engine

    def instrumented_factory():
        engine = factory()
        prepare = engine._prepare_learner

        def observed_prepare(owners, state, lane):
            prepared = prepare(owners, state, lane)
            parts = {}
            with torch.no_grad():
                for name, store in prepared.sampler.stores.items():
                    if session.distributed is not None and not observations:
                        batch = store.gather(torch.arange(min(2, store.rows), device=session.device))
                        fields = batch.fields
                        owner = fields['owner']
                        private = fields.get('sampled_private_state', owner[:, None])
                        direct = session.modules.lane_module(0).rollout_operation(name, fields['model_input'], private, owner)
                        functional = session.inference(name, fields['model_input'][None], private[None], owner[None],
                            active_lanes=torch.zeros(1, device=session.device, dtype=torch.int64))
                        direct_values = encode_model_evaluation(direct)
                        functional_values = encode_model_evaluation(functional)
                        error = max(float((value - functional_values[key][0]).abs().max()) for key, value in direct_values.items())
                        if error > 1e-4:
                            raise RuntimeError(f'Functional legacy adapter mismatch: {name}: {error}')
                        session._experiment_functional_parity[name] = error
                    columns = store.slabs['ActionTarget'].columns
                    if 'posterior' in columns:
                        p = columns['posterior'][store.lane, :store.rows].float()
                        mass = p.sum(-1, keepdim=True)
                        torch._assert_async((torch.isfinite(p).all() & mass.gt(0).all()), 'Invalid posterior mass')
                        p = p / mass
                        entropy = -(p * p.clamp_min(1e-30).log()).sum(-1)
                        breadth = entropy.exp()
                        exact_entropy = entropy
                        if name == 'pre':
                            if p.shape[-1] != 169:
                                raise RuntimeError('Class-uniform lift requires full-deck preflop classes')
                            counts = torch.bincount(session.modules.lane_module(0).combo_class, minlength=p.shape[-1]).float()
                            exact_entropy = entropy + (p * counts.clamp_min(1).log()).sum(-1)
                        parts[name] = {'rows': store.rows, 'sum_exp_entropy': float(breadth.sum()),
                                       'mean_exp_entropy': float(breadth.mean()),
                                       'sum_exact_hand_exp_entropy': None if name == 'pre' else float(breadth.sum()),
                                       'sum_class_uniform_lift_exp_entropy': float(exact_entropy.exp().sum()) if name == 'pre' else None,
                                       'posterior_available': True}
                    else:
                        parts[name] = {'rows': store.rows, 'sum_exp_entropy': float(store.rows),
                                       'mean_exp_entropy': 1.0,
                                       'sum_exact_hand_exp_entropy': None, 'posterior_available': False,
                                       'interpretation': 'one sampled hand per decision; no posterior measured'}
            observations.append(parts)
            return prepared

        engine._prepare_learner = observed_prepare
        return engine

    session._iteration_engine = instrumented_factory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('normal', 'rb'), default='rb')
    parser.add_argument('--tag', required=True)
    parser.add_argument('--checkpoint', type=Path, default=OLD / 'ckpt_0001110.pt')
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--roots', type=int, default=256)
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--seed', type=int, default=918273)
    parser.add_argument('--entropy', type=float, default=0.0015625)
    parser.add_argument('--entropy-horizon', type=int, default=2)
    parser.add_argument('--kl', type=float, default=0.00625)
    parser.add_argument('--value-coef', type=float, default=0.016)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--rollout-pack', type=int, default=64)
    parser.add_argument('--completion-pack', type=int, default=128)
    parser.add_argument('--learner-pack', type=int, default=12)
    parser.add_argument('--samples', type=int, default=1024)
    parser.add_argument('--reference-cache', type=int, default=0)
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--thermal-step-guard', action='store_true', help='Pause between optimizer updates at75C or a thermal flag, resume at70C')
    args = parser.parse_args()
    if min(args.iterations, args.roots, args.steps, args.entropy_horizon) <= 0:
        parser.error('iteration, root, step and entropy horizon counts must be positive')
    if '/' in args.tag or '..' in args.tag:
        parser.error('tag must be a filename component')
    torch.set_num_threads(2)
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if int(os.environ.get('LOCAL_WORLD_SIZE', str(world))) != world:
        raise RuntimeError('This experiment driver supports one node only')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible:
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES explicitly to idle GPU UUIDs')
    selected = visible.split(',')[rank]
    busy = subprocess.check_output(['nvidia-smi', '-i', selected, '--query-compute-apps=pid',
                                   '--format=csv,noheader,nounits'], text=True).strip()
    if busy:
        raise RuntimeError(f'Refusing occupied GPU {selected}: compute PIDs {busy}')
    if not torch.cuda.is_available():
        raise RuntimeError('An explicitly selected CUDA GPU is required')
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    torch.cuda.set_device(rank)
    stem = f'nlhe_hu_32bb_{args.mode}_hyperparam_20260928_{args.tag}'
    output = Path('runs/experiments') / stem
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f'result_rank{rank}.json'
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    template = Path('configs/experiments/nlhe_hu_32bb_rb_dualclip5_60h_20260814.yaml')
    source = yaml.safe_load(template.read_text())
    scientific = source['scientific']
    scientific['mode']['private_estimator'] = 'rao_blackwell' if args.mode == 'rb' else 'sampled'
    if args.mode == 'normal':
        scientific.pop('completion')
    else:
        scientific['completion']['samples'] = args.samples
    objective = scientific['objective']
    objective.pop('negative_advantage_ratio_cap', None)
    objective['value'] = {'kind': 'clipped', 'epsilon': 0.00625, 'coefficient': args.value_coef}
    objective['entropy'] = {'kind': 'linear_decay', 'coefficient': args.entropy,
                            'iterations': args.entropy_horizon}
    objective['reference_kl'] = {'coefficient': args.kl, 'refresh': {'iterations': 1000000000}}
    scientific['optimizer']['learning_rate'] = args.lr
    scientific['schedule'] = {'batch_size': args.batch, 'steps': args.steps}
    run = source['run']
    run.update(seed=args.seed, iterations=args.iterations, utility='stack_normalized',
               output_dir=str(output), reproducibility='strict')
    run['workload']['roots'] = args.roots
    run['recovery']['minimum_roots'] = args.roots
    execution = run['execution']
    execution.update(device=f'cuda:{rank}', reference_cache=(
        {'budget_bytes': args.reference_cache} if args.reference_cache else 'off'))
    if world > 1:
        execution['distributed'] = True
    execution['microbatch'] = {'rollout_rows': args.rollout_pack,
                               'completion_rows': args.completion_pack,
                               'learner_rows': args.learner_pack}
    if rank == 0:
        (Path('configs/experiments') / f'{stem}.yaml').write_text(yaml.safe_dump(source, sort_keys=False))
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    # Checkpoint I/O is host orchestration; scaling the live model happens on CUDA.
    session = TrainingSession(load_config_mapping(source), initial_model_state=payload['model'],
                              benchmark_telemetry=True)
    del payload
    observations = []
    monitor = None
    telemetry_file = None
    try:
        model = session.modules.lane_module(0)
        historical_adapter(model, session.geometry)
        historical_adapter(session.modules._code, session.geometry)
        historical_adapter(session.reference_modules._code, session.geometry)
        session._experiment_functional_parity = {}
        historical_adapter(session.reference_modules.lane_module(0), session.geometry)
        raw = policy_snapshot(session)
        scaled_names = []
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith(('value_head.affine.weight', 'value_head.affine.bias')):
                    parameter.div_(32.0)
                    scaled_names.append(name)
        if len(scaled_names) != 4:
            raise RuntimeError(f'Unexpected value head tensors: {scaled_names}')
        session.reference_modules.lane_module(0).load_state_dict(model.state_dict(), strict=True)
        session.initial_model_sha256 = (model_state_sha256(model),)
        before = policy_snapshot(session)
        migration = {}
        for key in raw:
            policy_error = max(abs(a-b) for a,b in zip(raw[key]['mean_event_probabilities'], before[key]['mean_event_probabilities']))
            value_error = abs(raw[key]['value_mean_native_units']/32 - before[key]['value_mean_native_units'])
            migration[key] = {'policy_max_error': policy_error, 'value_scale_error': value_error}
            if policy_error > 1e-6 or value_error > 1e-6:
                raise RuntimeError(f'Failed normalized checkpoint migration: {migration}')
        optimizer = session.optimizers.optimizers[0]
        optimizer.scaler = LossScaler(enabled=True, scale=1024.0, growth_interval=2000)
        thermal_pauses = []
        if args.thermal_step_guard:
            original_step = optimizer.step
            def guarded_step(*step_args, **step_kwargs):
                started = time.monotonic()
                def heat_status():
                    parts = subprocess.check_output(['nvidia-smi', '-i', selected,
                        '--query-gpu=temperature.gpu,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown',
                        '--format=csv,noheader,nounits'], text=True).strip().split(',')
                    heat = torch.tensor([int(parts[0]), int(any(p.strip() == 'Active' for p in parts[1:]))], device=session.device)
                    if world > 1:
                        torch.distributed.all_reduce(heat, op=torch.distributed.ReduceOp.MAX)
                    return heat.tolist()
                temperature, throttled = heat_status()
                if temperature >= 75 or throttled:
                    first_temperature = temperature
                    while temperature > 70 or throttled:
                        if time.monotonic() - started > 180:
                            raise RuntimeError('Step guard cooldown exceeded180seconds')
                        time.sleep(1)
                        temperature, throttled = heat_status()
                    thermal_pauses.append({'from_temperature_C': first_temperature, 'seconds': time.monotonic()-started})
                return original_step(*step_args, **step_kwargs)
            optimizer.step = guarded_step
        posterior_observer(session, observations)
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        selected = visible.split(',')[rank] if visible else str(rank)
        telemetry_file = (output / f'gpu_rank{rank}.csv').open('w')
        monitor = subprocess.Popen(['nvidia-smi', '-i', selected,
            '--query-gpu=timestamp,index,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown',
            '--format=csv', '-l', '2'], stdout=telemetry_file, stderr=subprocess.DEVNULL)
        results = {'arguments': {**vars(args), 'checkpoint': str(args.checkpoint)},
                   'git_revision': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                   'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                   'config': source, 'visible_devices': visible, 'world_size': world,
                   'migration': migration, 'scaled_parameters': scaled_names,
                   'optimizer': 'fresh AdamW; loss scale 1024',
                   'reference': 'frozen normalized checkpoint1110 policy; refreshed once at iteration0',
                   'input_encoding': 'experiment-local verified July exact-hand preflop gauge',
                   'scope': 'stability/throughput, not strength evaluation',
                   'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'thermal_step_pauses': thermal_pauses,
                   'before': before, 'iterations': []}
        for iteration in range(args.iterations):
            torch.cuda.synchronize(session.device)
            torch.cuda.reset_peak_memory_stats(session.device)
            start = time.monotonic()
            outcome = session.step()
            torch.cuda.synchronize(session.device)
            elapsed = time.monotonic() - start
            metrics = copy.deepcopy(session.run_states[0].metrics_prefix[-1])
            lane = session._lane_results(outcome)[0]
            row = {'iteration': iteration+1, 'seconds': elapsed, 'metrics': metrics,
                   'entropy_coefficient': args.entropy * max(0., 1-iteration/args.entropy_horizon),
                   'posterior_breadth': observations[-1],
                   'telemetry': dataclasses.asdict(lane.telemetry) if lane.telemetry else None,
                   'policy': policy_snapshot(session)}
            results['iterations'].append(row)
            destination.write_text(json.dumps(results, indent=2)+'\n')
            print(json.dumps({'tag': args.tag, 'rank': rank, 'iteration': iteration+1,
                'seconds': elapsed, 'nodes': metrics['player_nodes'], 'gradient': metrics.get('gradient'),
                'objective': metrics.get('objective'), 'root_raise': row['policy']['root']['raise']}), flush=True)
            failed = torch.tensor(int(not metrics['gradient']['finite'] or metrics['gradient']['applied_steps'] != args.steps), device=session.device)
            if world > 1:
                torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
            if int(failed):
                raise RuntimeError('Nonfinite gradients or skipped optimizer steps on a rank')
            temperature = int(subprocess.check_output(['nvidia-smi','-i',selected,
                '--query-gpu=temperature.gpu','--format=csv,noheader,nounits'], text=True).strip())
            # Coordinate thermal pauses across ranks before the next collective.
            heat = torch.tensor(temperature, device=session.device)
            if world > 1:
                torch.distributed.all_reduce(heat, op=torch.distributed.ReduceOp.MAX)
            cooldown_start = time.monotonic()
            if int(heat) >= 82:
                while int(heat) > 74:
                    if time.monotonic() - cooldown_start > 180:
                        raise RuntimeError('GPU did not cool below 75 C within 180 seconds')
                    time.sleep(2)
                    heat.fill_(int(subprocess.check_output(['nvidia-smi','-i',selected,
                        '--query-gpu=temperature.gpu','--format=csv,noheader,nounits'], text=True).strip()))
                    if world > 1:
                        torch.distributed.all_reduce(heat, op=torch.distributed.ReduceOp.MAX)
            row['cooldown_seconds'] = time.monotonic() - cooldown_start
        final_hash = model_state_sha256(model)
        rank_hashes = [final_hash]
        if world > 1:
            rank_hashes = [None] * world
            torch.distributed.all_gather_object(rank_hashes, final_hash)
            if len(set(rank_hashes)) != 1:
                raise RuntimeError('Distributed model parameters disagree')
        results['validation'] = {'final_model_sha256_by_rank': rank_hashes,
                                 'functional_adapter_max_error': session._experiment_functional_parity,
                                 'finite_parameters': bool(torch.stack([p.isfinite().all() for p in model.parameters()]).all()),
                                 'applied_steps': optimizer.step_count, 'completed': True}
        if args.save_model and rank == 0:
            torch.save({'model': model.state_dict(), 'scientific_config': session.config.scientific_mapping(),
                        'requires_legacy_input_adapter': True}, output / 'experimental_model.pt')
        destination.write_text(json.dumps(results, indent=2)+'\n')
    finally:
        if monitor is not None:
            monitor.terminate()
            monitor.wait(timeout=10)
        if telemetry_file is not None:
            telemetry_file.close()
        session.close()


if __name__ == '__main__':
    main()
