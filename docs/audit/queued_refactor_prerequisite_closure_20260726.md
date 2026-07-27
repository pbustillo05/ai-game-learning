# Queued-refactor prerequisite closure

Date: 2026-07-26  
Clean starting revision: `2e19bd89279df31d5491d089ab4cb5f2a5a52ff1`

## Approval and authority

The user approved the complete refactor plan and clarified that its changes
are intentional and supersede the stale repository conventions. The following
clauses are therefore authoritative for the rewrite:

- `root_count >= 1` is an execution dimension for every rollout mode. Each row
  owns one local distinguished root and a disjoint logical RNG address.
- Sequential execution is the exact
  $N=N_{\rm tgt}=\widetilde N=1$ specialization.
- Only sampled branched mode uses one joint profile for the complete tree. Its
  sibling correlation and estimator bias are intentional and observable. It
  is implemented as a negative-control experiment to demonstrate that this
  estimator is a bad idea, not as a recommended training law.
- Branched Rao--Blackwell sampling gives child zero the inherited profile and
  draws exactly $N-1$ fresh posterior profiles.
- `ai-game-learning` owns profile sampling, folded-summary construction,
  policy composition, logical RNG, chance generation, and public-tree
  navigation.
- `poker-gpu-evaluator` contract v2 remains the exact boundary for showdown
  evaluation and `Leaf -> LayerWork` completion involving `hand_rank`.

The corresponding stale ownership and independent-tree-only wording was
updated in `AGENTS.md`, `docs/theory/core_math.tex`,
`docs/theory/MHA_model.md`, and `docs/theory/folder_impact.md`. No sibling
evaluator source or API document was modified.

## Execution and failure decisions

The public mode identity is the product of:

```text
training_regime   self_play | best_response
rollout_mode      sequential | branched.fixed | branched.KKT
private_estimator sampled | rao_blackwell
```

All twelve rows use one learner operation. Ordinary, distributed, and replica
execution change ownership and scheduling dimensions, not this semantic mode.
Fixed and KKT branched lanes are physically compatible when their training
regime, estimator, game partitions, capacities, and owner field signatures
match. Their semantic identity and lane-local controller state remain distinct.

Overflow recovery is whole-attempt transactional recovery. A failed lane
publishes none of that attempt, releases its attempt-owned storage, advances to
a disjoint RNG address, and lowers only `root_count`, fixed beta, or KKT scale.
Completed lanes retain their published stores, capacities never change, ranks
agree before retry, and failure at the minimum control terminates the run.

## Capacity source

Every fixed capacity is public run configuration in an immutable capacity
manifest:

- player-node, trained-player, and action capacities by declared partition;
- leaf, topology-node, topology-edge, and folded-range capacities;
- family pack, chance frontier, completion pack, learner sigma/action pack,
  topology packet, and movement-scratch bounds;
- two target-work arenas, library workspaces, evaluator workspaces, and a
  memory guard.

Phase 2 will validate the manifest before allocation against `(P,S,R)`, the
mode schema, `root_count`, replica count, the exact history bound
$SP+3P-S+1$, partition widths, `beta_max`, hardware memory/token limits, and
bounded external workspaces. Hardware profiles may calibrate capacities and
launch shapes but may not select algorithms or alter sampling laws.

## Artifact disposition

- Current `pbg-checkpoint-v8` checkpoints in the four named run families under
  `runs/` are selected for one-time offline conversion in their first consumer
  phase. Source SHA-256, source schema, mode, and conversion result are
  recorded. There is no runtime compatibility loader.
- The two tracked range-database `.pt`/`.json` pairs under
  `experiments/range_database/` remain calibration evidence and receive a
  versioned offline dataset conversion before a queued consumer uses them.
- Checked-in run/config/benchmark JSON and YAML are retained as read-only
  evidence. They are not executable queued configuration and need no runtime
  loader.
- Unselected checkpoint schemas and artifacts are deleted after their callers
  and compatibility tests are removed. A newly discovered artifact defaults
  to unselected until the user explicitly adds it to the conversion inventory.

## Frozen CUDA gates

Comparisons use matched logical work, the same three seeds `104830`, `209559`,
and `314288`, family pack maximum 84, 1,024 completion samples per leaf, and
the fixed-$\beta=3$ HU 32 bb identity from
`nlhe_hu_32bb_family_timing_attribution_20260724.md`. A baseline is rejected
when code/config identity, device/environment, RNG version, logical work, or
required metrics do not match.

The accepted median limits on the reference RTX 2080 Ti class are:

| Metric | Frozen limit |
|---|---:|
| sequential-RB rollout | at most 4.364 ms per retained decision |
| fixed branched-RB rollout | at most 9.433 ms per retained decision |
| fixed branched routing traffic | at most 12.838 GB total and 0.841 MB per retained decision |
| fixed branched host synchronizations | at most 64 plus one per realized target layer |
| fixed branched child reservation plus commit | at most 17.71 seconds |
| fixed branched target propagation | at most 0.500 seconds |
| sequential target propagation | at most 0.050 seconds |
| peak CUDA memory | no configured-budget breach and no more than 5% above the matched legacy peak |

For every other currently delivered consumer (sampled sequential training,
fixed/KKT branched training, the shared-response probe, distributed training,
replica training, checkpoint/resume, benchmark, BR evaluation, inference
export, CUDA inference, and MPS inference), median end-to-end throughput may
regress by at most 10% at matched logical work. Strict-profile FP32 numerical
comparisons use `atol=1e-6`, `rtol=1e-5`; FP16/BF16 model-facing comparisons
use `atol=5e-3`, `rtol=5e-3`. Integer topology, RNG labels, publication
addresses, checkpoint identity, and strict-resume state compare exactly.

Every measured CUDA gate records revision, dirty state, resolved configs,
hardware/driver/CUDA/PyTorch/evaluator identities, seeds, logical work, timing,
allocated/reserved/high-water bytes, and pre/post GPU state. Warmup and
diagnostic kernels are excluded from measured throughput.

## Saved H100 branch

The saved branch is evidence only and is never applied wholesale:

- `7c8ff1d` replica batching and selection harness: rewrite requirement for
  Phase 9; no old replica representation crosses the firewall.
- `4fd5220` range dataset and H100 profiles: range artifacts are calibration
  evidence; hardware values are calibration inputs, not schema authority.
- `0fa6305` fail-fast parity checks and `f952104` progress reporting:
  behavioral evidence for Phase 9 observability and failure gates.
- `279c283` pre-refactor snapshot: historical baseline only.

Any individual hunk encountered later is classified again in the migration
ledger as evidence, rewrite, pre-existing change, or delete before use.
