"""CLI for strict matched-work queued benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pbg.benchmark import benchmark_training, compare_benchmark
from pbg.config import load_config


def run_cli(*, require_replica: bool = False) -> None:
    parser = argparse.ArgumentParser(description="Benchmark queued PBG training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", default=None)
    parser.add_argument("--phase-telemetry", action="store_true",
                        help="Instrument synchronized phase boundaries; report separately from throughput")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--max-regression", type=float, default=0.10)
    parser.add_argument("--allow-environment-mismatch", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if require_replica and config.execution.manifest.n_replica <= 1:
        parser.error("replica benchmark requires manifest.n_replica greater than one")
    result = benchmark_training(
        config,
        warmup=args.warmup,
        repeats=args.repeats,
        checkpoint=args.checkpoint,
        phase_telemetry=args.phase_telemetry,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        failures = compare_benchmark(
            result,
            baseline,
            max_regression=args.max_regression,
            allow_environment_mismatch=args.allow_environment_mismatch,
        )
        if failures:
            parser.exit(1, "performance gate failed:\n- " + "\n- ".join(failures))


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
