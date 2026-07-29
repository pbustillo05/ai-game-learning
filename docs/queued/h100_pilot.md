# One-H100 Normal/RB pilot

This pilot chooses a production geometry separately for sequential Normal and
sequential Rao--Blackwell (RB) HU 32 BB training before a longer Quest run. It
uses one visible H100, BF16, minibatch size 256, and 116 gradient steps in every
paid arm. It does not compare FP16, change minibatch size, or tune learning
rate.

## Experiment stages

The BF16 gate runs one very small Normal iteration and one very small RB
iteration. Its purpose is only to prove H100 admission, BF16 forward/backward,
completion, targets, and optimizer commit before spending time on throughput
measurements.

Replica scaling then runs $K\in\{1,2,4,8\}$ separately for Normal and RB with
512 roots per lane, one warmup, and two measured fresh sessions. Every lane
starts from the same mode-specific trained model, while optimizer and RNG state
are fresh and lane-local. The selected $K$ remains the smaller value unless a
larger value improves aggregate player nodes per second by at least 10%, and
any arm above the 70 GiB peak-reserved ceiling is rejected.

Geometry selection runs three root counts appropriate to the selected $K$ and
chooses the passing arm with the highest aggregate player-node rate. Root
recovery is disabled in benchmark configs, so an undersized immutable
capacity fails instead of silently changing the compared work.

Finally, RB runs 256, 512, and 1024 completion samples at its selected
replica/root geometry. These rows report the runtime cost of posterior quality;
they are not treated as interchangeable estimators by the automatic selector.
The final sample-count decision should consider both this cost and downstream
variance/exploitability evidence.

## Prepared artifacts

The checked pilot bundle is:

- plan and model-only initializers:
  `runs/experiments/nlhe_hu_32bb_h100_pilot/`
- strict paid-run configs:
  `configs/experiments/nlhe_hu_32bb_h100_pilot/`
- local smoke reports:
  `runs/experiments/nlhe_hu_32bb_h100_pilot_local/local_smoke.json`

Regenerate it after changing a source checkpoint or the pilot code:

```bash
source /home/pablob/.venvs/poker/bin/activate
pbg-h100-pilot prepare \
  --directory runs/experiments/nlhe_hu_32bb_h100_pilot \
  --configs-directory configs/experiments/nlhe_hu_32bb_h100_pilot
```

The initializer artifacts contain only model tensors. Their metadata binds the
source-checkpoint hash and a device-independent model-state hash. Pilot arms do
not import legacy optimizer, RNG, iteration, or reference-cadence state.

## Local CUDA gate

The RTX 2080 Ti does not support BF16, so the local gate uses FP32, one root,
batch size 8, one gradient step, and four RB completion samples. It still runs
the full HU 32 BB model and the same queued Normal/RB and K=1/K=2 runtime paths.

```bash
CUDA_VISIBLE_DEVICES=0 pbg-h100-pilot local-smoke \
  --normal-initializer \
    runs/experiments/nlhe_hu_32bb_h100_pilot/artifacts/nlhe_hu_32bb_normal_initializer.pt \
  --rb-initializer \
    runs/experiments/nlhe_hu_32bb_h100_pilot/artifacts/nlhe_hu_32bb_rb_initializer.pt
```

Local throughput numbers are smoke evidence only and must not be used to select
H100 geometry.

## RunPod launch

Use one 80 GB H100 SXM and expose exactly that GPU to the process. From the
repository root, with the project and evaluator installed in the active
environment:

```bash
export CUDA_VISIBLE_DEVICES=0
pbg-h100-pilot run \
  --plan runs/experiments/nlhe_hu_32bb_h100_pilot/plan.json
```

The runner checks for exactly one visible H100, compute capability 9.0 or
newer, at least 80,000,000,000 bytes of VRAM, and PyTorch BF16 support. Every
arm gets a fresh subprocess to isolate CUDA allocator state. State is written
atomically after every arm to
`runs/experiments/nlhe_hu_32bb_h100_pilot/run_state.json`.

The default deadline is 170 minutes with a 10-minute reserve for copying
reports and terminating the pod. `PILOT_COMPLETE` means the staged workflow
finished; it does not terminate the RunPod instance automatically.

## Quest follow-up

Quest General Access H100 jobs use the `gengpu` partition and
`--gres=gpu:h100:<count>`. The maximum H100 wall time is 48 hours; H100 nodes
have four GPUs, while the per-user QoS ceiling is eight concurrent H100s.
Start with one GPU and the selected per-mode geometry from `run_state.json`.
A minimal allocation header is:

```bash
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=48:00:00
```

Queue time and immediate GPU count remain subject to Quest fair share. Check
the live partition before submission rather than assuming all four GPUs on a
node are available.
