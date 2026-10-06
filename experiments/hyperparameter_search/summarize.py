"""Aggregate saved CUDA pilots without loading models or running GPU work."""
from pathlib import Path
import csv
import json
import statistics

OUT = Path('runs/experiments/nlhe_hu_32bb_rb_hyperparam_20260928_report')


def main():
    rows = []
    for path in sorted(Path('runs/experiments').glob('nlhe_hu_32bb_*_hyperparam_20260928_*/result_rank*.json')):
        data = json.loads(path.read_text())
        iterations = data['iterations']
        if not iterations:
            continue
        rank = int(path.stem.removeprefix('result_rank'))
        csv_path = path.parent / f'gpu_rank{rank}.csv'
        telemetry = list(csv.DictReader(csv_path.open())) if csv_path.exists() else []
        def values(key):
            return [float(row[key].strip().split()[0]) for row in telemetry]
        warm = iterations[1:] or iterations
        last = iterations[-1]
        breadth = last['posterior_breadth']
        pre = breadth.get('pre', {})
        if 'sum_class_uniform_lift_exp_entropy' not in pre and pre.get('sum_exact_hand_exp_entropy') is not None:
            pre['sum_class_uniform_lift_exp_entropy'] = pre['sum_exact_hand_exp_entropy']
            pre['sum_exact_hand_exp_entropy'] = None
            pre['legacy_label_correction'] = 'Raw pilot used exact-hand label for uniform-within-class lift upper bound; within-class posterior is lost in target aggregation.'
        rows.append({
            'mode': data['arguments']['mode'], 'tag': data['arguments']['tag'], 'rank': rank,
            'world_size': data['world_size'], 'complete': data.get('validation', {}).get('completed', False),
            'source': str(path), 'iterations': len(iterations),
            'seconds_by_iteration': [row['seconds'] for row in iterations],
            'warm_mean_seconds': statistics.mean(row['seconds'] for row in warm),
            'warm_global_decisions_per_second': sum(row['metrics']['global_player_nodes'] for row in warm) / sum(row['seconds'] for row in warm),
            'peak_allocated_GiB': max(row['metrics']['memory']['peak_allocated_bytes'] for row in iterations) / 2**30,
            'peak_reserved_GiB': max(row['metrics']['memory']['peak_reserved_bytes'] for row in iterations) / 2**30,
            'all_finite_gradients': all(row['metrics']['gradient']['finite'] for row in iterations),
            'applied_steps': sum(row['metrics']['gradient']['applied_steps'] for row in iterations),
            'max_recorded_action_approximate_kl': max(row['metrics']['objective']['approximate_kl'] for row in iterations),
            'initial_root_raise': data['before']['root']['raise'],
            'final_root_raise': last['policy']['root']['raise'],
            'final_root_hand_std': last['policy']['root']['raise_std_over_hands'],
            'final_postflop_decision_fraction': last['metrics']['structure']['decision_fractions_by_street']['postflop'],
            'entropy_coefficients': [row['entropy_coefficient'] for row in iterations],
            'last_posterior_breadth': breadth,
            'last_completion_health': last['metrics']['completion_health'],
            'max_temperature_C': max(values(' temperature.gpu')) if telemetry else None,
            'mean_sampled_utilization_percent': statistics.mean(values(' utilization.gpu [%]')) if telemetry else None,
            'thermal_throttle_samples': sum(any(value.strip() == 'Active' for key, value in row.items() if 'thermal_slowdown' in key) for row in telemetry),
            'telemetry_samples': len(telemetry),
            'step_guard_pause_seconds': sum(row['seconds'] for row in data.get('thermal_step_pauses', [])),
            'thermal_throttle_observed': any('Active' == value.strip() for row in telemetry for key, value in row.items() if 'thermal_slowdown' in key),
            'cooldown_seconds': sum(row.get('cooldown_seconds', 0) for row in iterations),
            'validation': data.get('validation'),
        })
    OUT.mkdir(exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    lines = ['| Mode / pilot | Rank | Updates | Warm s/iteration | Global decisions/s | Peak allocated GiB | Max °C | Throttle samples | Final root raise |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['mode']} / {row['tag']} | {row['rank']} | {row['applied_steps']} | {row['warm_mean_seconds']:.2f} | {row['warm_global_decisions_per_second']:.1f} | {row['peak_allocated_GiB']:.2f} | {row['max_temperature_C']:.0f} | {row['thermal_throttle_samples']}/{row['telemetry_samples']} | {100*row['final_root_raise']:.2f}% |")
    (OUT/'pilot_table.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
