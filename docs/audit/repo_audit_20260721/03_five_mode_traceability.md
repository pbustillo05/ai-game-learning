# Five-mode traceability

## Verdict

Strict validation yields exactly five semantic combinations. Internally they
correctly share three target/storage/learner labels. The missing piece is a
closed semantic identity used consistently by configs, manifests, acceptance
tests, and benchmarks.

| Semantic combination | Strict configuration | Storage/training label | CUDA status | Principal gaps |
|---|---|---|---|---|
| normal | `training.mode=normal`, no probe/branching, null completion samples | `normal` | Reachable; tiny run environment-blocked before completion | C01/C02/C05/C08/C11/C12; no canonical preset; normal evaluator JIT needs local CUDA toolkit |
| sequential RB | `training.mode=rao_blackwell`, null probe, positive completion samples | `sequential_rb` | Tiny real update passed, 22 MiB reserved peak | C01/C02/C05/C08/C11/C12 |
| branched RB, fixed $\beta$ | `branched_rao_blackwell` + `branching_mode=fixed`, one root, required T/C | `branched_rb` | Tiny real update passed, 24 MiB | C01/C02/C05/C06/C07/C08/C09/C11/C12 |
| branched RB, KKT | same base + `branching_mode=dynamic` and adaptive KKT config | `branched_rb` | Tiny real update passed, 24 MiB | C01/C02/C05/C06/C07/C08/C09/C11/C12; no canonical preset |
| BR probe | `rao_blackwell` + non-null `probe.target_checkpoint` | `sequential_rb` | No CUDA acceptance test | C03/C04/C08/C12/C16; no authoritative contract or preset |

The full source trace is [batch 10](batches/10_five_mode_end_to_end_traces.md).
Entry is `pbg-train` → `pbg.cli.train.main` → `load_config` →
`pbg.runtime.train` → `run_training` (`pyproject.toml:14`,
`pbg/cli/train.py:11-40`, `pbg/config.py:11-21`,
`pbg/runtime.py:24-38`, `pbg/core/training.py:2529-2538`).

The strict parser exposes three base strings (`pbg/core/config.py:282-287`),
with probe and branching selectors supplying the two semantic splits. Fixed
and KKT share equations and stores but differ in future geometry/controller
state; BR shares sequential-RB storage but changes policy ownership and the
trained model. Checkpoint resume distinguishes fixed from KKT controller state
(`pbg/core/training.py:2788-2800`).

All five enter the same problematic frozen lifetime:

```text
paged rollout owners -> SampledTree -> post-hoc compact topology
  -> exact TrainingStorage -> bounded learner subspans -> optimizer/checkpoint
```

The last-consumer deletions around that path are useful, but they do not cure
the source-plus-destination overlap or make its conversion transactionally
admitted (C01).

BR specifically mixes a frozen opponent with a trainable probe and retains
probe-seat decisions. For $P>2$, `PROBE_FOLD` identity is lost before terminal
settlement (`pbg/core/rollout.py:565-610,860-868`;
`pbg/games/poker/rollout_backend.py:772-788,2590-2610`), confirming C04.
