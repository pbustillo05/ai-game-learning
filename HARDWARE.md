# Hardware Profiles

Training code must be configurable by hardware profile. Keep algorithm,
checkpoint, rollout, and evaluator logic independent of the selected profile;
only launch, batching, memory limits, and scheduler integration should change.

## Active Profile: `local_3x2080ti`

This is the current baseline hardware target.

- One dedicated always-on server.
- `3 x NVIDIA GeForce RTX 2080 Ti` GPUs, available 24 x 7.
- Treat each GPU as an independent 11 GB VRAM class device with the same
  approximately 10 GB PBG envelope whether one, two, or three ranks
  participate. Detect exact usable memory at runtime and leave headroom for
  CUDA/NCCL state, evaluator kernels, allocator variance, and display use.
- CUDA architecture target: Turing / `sm_75`.
- Do not assume cluster scheduling, preemption, shared filesystem semantics, or
  fast inter-node networking.
- Use the validated PyTorch SDPA path for production attention on this profile;
  automatic/FlexAttention dispatch is not the production baseline.
- Multi-GPU training uses one process per GPU with CUDA/NCCL. Production
  rollout, model, targets, completion, and learner work remain on the selected
  accelerator; there are no CPU rollout workers or CPU fallbacks.
- The default implementation profile, tests, smoke runs, and initial paper
  timing logs should use this profile unless explicitly overridden.

## Available but Deferred Profile: `quest_slurm`

Northwestern Quest access is already available. Its use is intentionally
deferred while the codebase is hardened and validated on the local 2080 Ti
server; lack of access is not a blocker. The local profile remains the default
for implementation, tests, smoke runs, and initial profiling. Quest must be
selected explicitly and must never be used as an automatic overflow or
fallback target.

Substantive Quest GPU runs begin only after a local readiness review confirms:

- the relevant CPU and local-CUDA correctness suites pass;
- exact-attention, physical-batching, checkpoint/resume, and DDP paths have
  passed staged local smoke tests;
- local peak-memory and runtime measurements are reproducible and leave the
  GPUs cleanly idle; and
- critical execution findings are resolved or have an explicit accepted
  exception.

Passing this gate authorizes staged Quest validation, not an immediate large
run: start with one H100 and tiny forward/backward probes, then validate two
and four GPUs before increasing workloads. The Quest account/allocation value
is launch metadata and should be supplied by the user environment or launch
template rather than committed to this repository.

Public Northwestern RCDS docs currently describe Quest as a Slurm-managed
compute environment with General Access GPU jobs submitted to the `gengpu`
partition via `--gres`. The current public GPU guide lists A100 and H100 GPU
types, 40 GB and 80 GB GPU-memory tiers, optional SXM constraints for 80 GB /
high-bandwidth interconnect jobs, a 48-hour maximum for General Access GPU jobs,
and an eight-GPU per-user running-job limit. Re-verify these details at the
time of migration because Quest policy and hardware can change.

References checked on 2026-06-24:

- Northwestern IT Research Computing and Data Services:
  `https://www.it.northwestern.edu/departments/it-services-support/research/`
- Northwestern RCDS documentation:
  `https://rcdsdocs.it.northwestern.edu/`
- GPUs on Quest:
  `https://rcdsdocs.it.northwestern.edu/systems/quest/user-guide/gpu/gpu.html`

Use this explicitly selected profile for Slurm launch templates only:

```text
hardware_profile: quest_slurm
scheduler: slurm
partition: gengpu
account: <quest_account>
gres: gpu:<type_or_blank>:<count>
constraint: <optional, e.g. sxm>
time_limit: <= 48:00:00 for General Access GPU jobs unless allocation says otherwise
```

Implementation requirements:

- Keep all Quest-specific settings in config files or launch templates, not in
  model, evaluator, or PPO code.
- Support choosing GPU type only when needed. A generic `--gres=gpu:<count>`
  request should remain valid for availability-oriented runs.
- Compile CUDA extensions for both the active local target and the selected
  Quest target when building portable wheels. For Quest A100/H100 portability,
  include `sm_80` and `sm_90` when the build environment supports them.
- Quest A100/H100 runs may prefer FlashAttention when it supports the required
  MHA semantics, but the same dense/SDPA fallback path must stay tested.
- Log the resolved profile, visible devices, GPU names, CUDA version, driver
  version, VRAM, scheduler job ID, and effective time limit in every run.

## Profile Toggle Contract

All long-running commands must accept a profile value, for example:

```text
--hardware-profile local_3x2080ti
--hardware-profile quest_slurm
```

The training CLI defaults to `local_3x2080ti` when the flag is omitted. Quest
therefore always requires the explicit `--hardware-profile quest_slurm`
selection. Python callers should likewise pass `hardware_profile="quest_slurm"`
explicitly; no hardware-detection path may opt into cluster execution.

Configs should keep hardware knobs separate from scientific knobs:

```text
configs/
  hardware/
    local_3x2080ti.yaml
    quest_slurm.yaml
  train/
  model/
  eval/
```

Changing the hardware profile must not change the poker rules, model contract,
rollout estimator, PPO loss, random seed semantics, or black-box `F_theta`
constraint. It may change batch sizes, worker counts, launch commands,
checkpoint cadence, mixed-precision policy, and scheduler metadata.

The active execution files are `configs/hardware/local_3x2080ti.yaml` and
`configs/hardware/quest_slurm.yaml`; the scientific NLHE preset is
`configs/train/nlhe6.yaml`. Although Quest access is available, its profile
deliberately retains the local-safe policy batch ceilings until the local
readiness gate and staged Quest measurements establish explicit H100 40/80 GiB
tiers.
