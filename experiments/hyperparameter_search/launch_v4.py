"""Persistent v4 launcher with CUDA validation gates and visible failure state."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from experiments.hyperparameter_search.campaign import (
    PYTHON,
    ROOT,
    atomic_json,
    digest,
    resolve_gpus,
    source_digest,
)
from experiments.hyperparameter_search.protocol import sha256_file

PROFILE = ROOT / "configs/profiles/nlhe_hu_32bb_shared_loss_search_v4.yaml"
OUTPUT = ROOT / "runs/experiments/nlhe_hu_32bb_rb_shared_loss_search_v4_20261006"
OLD = ROOT / "runs/experiments/nlhe_hu_32bb_rb_shared_loss_search_v3_20260928"


class Launcher:
    def __init__(self, output: Path, gpus: list[str]):
        self.output = output
        self.gpus = gpus
        self.started = time.time()
        self.child = None
        self.stage = "starting"
        self.interrupted = False
        self.log = output / "launcher.log"

    def status(self, state, **details):
        atomic_json(
            self.output / "launcher_status.json",
            {
                "state": state,
                "stage": self.stage,
                "launcher_pid": os.getpid(),
                "child_pid": None if self.child is None else self.child.pid,
                "started_at_unix": self.started,
                "updated_at_unix": time.time(),
                "gpus": self.gpus,
                **details,
            },
        )

    def say(self, message):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S %z')} {message}"
        print(line, flush=True)
        with self.log.open("a") as stream:
            stream.write(line + "\n")

    def stop(self, _sig, _frame):
        self.interrupted = True
        if self.child is not None and self.child.poll() is None:
            # The campaign supervisor owns orderly worker checkpoint/shutdown.
            self.child.send_signal(signal.SIGTERM)

    def wait_idle(self):
        self.stage = "waiting_for_idle_gpus"
        while True:
            occupied = {}
            for gpu in self.gpus:
                pids = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "-i",
                        gpu,
                        "--query-compute-apps=pid",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                ).strip()
                if pids:
                    occupied[gpu] = pids.splitlines()
            if not occupied:
                return
            self.status("waiting", occupied=occupied)
            if self.interrupted:
                raise KeyboardInterrupt
            time.sleep(15)

    def run(self, stage, command, *, environment=None, log=None):
        if self.interrupted:
            raise KeyboardInterrupt
        self.stage = stage
        destination = log or self.output / f"{stage}.log"
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.say(f"Starting {stage}; log: {destination}")
        with destination.open("a") as stream:
            self.child = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            while self.child.poll() is None:
                self.status("running", log=str(destination), command=command)
                time.sleep(2)
            code = self.child.returncode
        self.child = None
        if self.interrupted:
            raise KeyboardInterrupt
        if code:
            raise RuntimeError(f"{stage} exited {code}; see {destination}")
        self.say(f"Completed {stage}")

    def validate(self):
        tests = sorted((ROOT / "tests").glob("test_hyperparameter_*.py"))
        identity = {
            "source_sha256": source_digest(),
            "profile_sha256": sha256_file(PROFILE),
            "test_sha256": digest(
                {str(p.relative_to(ROOT)): sha256_file(p) for p in tests}
            ),
            "gpu_uuids": self.gpus,
        }
        directory = self.output / "validation" / digest(identity)[:16]
        passed = directory / "passed.json"
        if (
            passed.exists()
            and json.loads(passed.read_text()).get("identity") == identity
        ):
            self.say(f"Using completed validation: {passed}")
            return
        directory.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=self.gpus[0])
        self.run(
            "cuda_tests",
            [str(PYTHON), "-m", "pytest", "-q", *map(str, tests)],
            environment=environment,
            log=directory / "cuda_tests.log",
        )
        # Validate the actual 12 x 16 dense-gradient workload that broke v3.
        # The old artifacts retain their generating source hash, independently
        # of the repaired analyzer hash. No old result/manifest is overwritten.
        manifest = json.loads((OLD / "noise/manifest.json").read_text())
        manifest["probe_source_sha256"] = json.loads(
            (OLD / "manifest.json").read_text()
        )["source_sha256"]
        path = directory / "full_size_analysis_manifest.json"
        atomic_json(path, manifest)
        result = directory / "full_size_analysis.json"
        self.run(
            "full_size_cuda_analysis",
            [
                str(PYTHON),
                "-m",
                "experiments.hyperparameter_search.gradient_noise",
                "--analyze",
                str(path),
                "--output",
                str(result),
                "--device",
                "cuda:0",
            ],
            environment=environment,
            log=directory / "full_size_analysis.log",
        )
        analysis = json.loads(result.read_text())
        if (
            len(analysis["groups"]) != 12
            or len(analysis["comparisons"]) != 6
            or analysis["peak_cuda_allocated_bytes"] > 5 * 1024**3
        ):
            raise RuntimeError(
                "Full-size analysis did not meet the 12-group, 6-pair, <=5 GiB gate"
            )
        self.run(
            "integration_smoke",
            [
                str(PYTHON),
                "-m",
                "experiments.hyperparameter_search.league_campaign",
                "--profile",
                str(PROFILE),
                "--output",
                str(directory / "integration"),
                "--gpus",
                ",".join(self.gpus),
                "--smoke",
                "--resume",
            ],
            log=directory / "integration.log",
        )
        if source_digest() != identity["source_sha256"]:
            raise RuntimeError("Source changed during validation")
        atomic_json(
            passed,
            {
                "complete": True,
                "identity": identity,
                "analysis_peak_allocated_bytes": analysis["peak_cuda_allocated_bytes"],
                "completed_at_unix": time.time(),
            },
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if len(args.gpus.split(",")) != 3 and not args.validate_only:
        parser.error("The registered training campaign requires all three GPUs")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".launcher.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("A launcher already owns this output directory")
        runner = Launcher(output, resolve_gpus(args.gpus.split(",")))
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, runner.stop)
        try:
            runner.say(
                "Waiting for the requested GPUs to be idle; existing jobs are never stopped."
            )
            runner.wait_idle()
            runner.validate()
            if not args.validate_only:
                runner.wait_idle()
                runner.run(
                    "campaign",
                    [
                        str(PYTHON),
                        "-m",
                        "experiments.hyperparameter_search.league_campaign",
                        "--profile",
                        str(PROFILE),
                        "--output",
                        str(output),
                        "--gpus",
                        ",".join(runner.gpus),
                        "--resume",
                    ],
                )
            runner.status("complete", validation_only=args.validate_only)
            runner.say(
                "Validation complete." if args.validate_only else "Campaign complete."
            )
            return 0
        except KeyboardInterrupt:
            runner.status("interrupted")
            runner.say("Interrupted; rerun the same command to resume.")
            return 130
        except Exception as exc:  # noqa: BLE001 - durable launcher failure boundary
            runner.status("failed", error=str(exc))
            runner.say(f"FAILED: {exc}")
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
