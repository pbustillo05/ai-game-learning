# Mathematics and model traceability

## Mathematical verdict

The implemented **fixed-intensity PPO subset** is substantially faithful to the
constructive estimator mathematics in `docs/theory/core_math.tex`. This is not a
confirmation of every theorem or research extension in that note. Its own audit,
`docs/theory/core_math_audit.md`, classifies the document as a research draft
whose named-method and convergence claims still need assumptions and proofs.

The production implementation does not implement NeuRD, R-NaD/DeepNash,
adaptive variance-guided branching, public-moment losses, all-action Q losses,
or suit consistency. Those are correctly outside the fixed v1 estimator, though
the reference-KL and optional-head documentation has become stale.

## Core estimator trace

| Mathematical requirement | Implementation | Assessment |
|---|---|---|
| Curried policy emits all actor private types | Preflop outputs 169 class rows then lifts to combos; postflop outputs every compatible actor combo and scatters to 1326 | Conformant for full deck |
| Public likelihood reach updated for every action/chance | `pbg/core/rollout.py`, `branching.py`, poker backend likelihood updates and board masks | Conformant |
| Sequential rollout has one sampled profile/path per start | family scheduler retains one active occurrence/profile for each start | Conformant |
| Normal player-clock lambda return | backward owner-depth recurrence in `pbg/core/normal.py:134-152` | Conformant |
| RB stops at pre-terminal public leaves and completes under posterior | sibling completion adapters and streamed completion in training | Conformant |
| Preserve posterior mass `P` and moment `M=P V` before division | `pbg/core/values.py` fp32 frontier and `conditionalize` | Conformant |
| Parent is uniform child mixture | child `P/M` are summed and divided by child count | Conformant |
| Child Q is `M_child/P_child` | `values.emit`, lines 123-126 | Conformant |
| Node V is `M_parent/P_parent` | `values.emit`, lines 128-133 | Conformant |
| Leave-one-out sibling baseline | parent total times count minus selected child; conditionalize; lambda mix with one cached old V | Conformant |
| Single child baseline is cached old V | `multi` mask leaves baseline as old V | Conformant |
| Bootstrap only on the owner clock | reducer updates owner row with `lambda M + (1-lambda) P V_old` | Conformant |
| One distinguished branched root | `branched_root_state`, one-root `DeviceTreeBuffer` per rank | Conformant |
| Root uses ordinary randomized rounding | root enters the same `offspring_for` path as other non-preterminal nodes | Conformant |
| `floor(beta)+Bernoulli(frac(beta))`, beta at least one | `pbg/core/branching.py:143-165,404-411` | Conformant |
| Independently addressed temporary profiles per abstract child | logical child/profile keys include rank-tree and abstract path/ordinal | Conformant |
| Duplicate event labels remain distinct children | stable parent-local edge ordinal and duplicate-preserving CSR | Conformant |
| Branched terminal/decision path law uses product `1/N` | `tree.path_weight`, alias sampling, family/local/global mass | Conformant |
| Pool trees/ranks by total decision mass, not equal rank means | asynchronous/global SUM mass and local/global objective scale | Conformant |
| Flat gradient all-reduce uses SUM with no world division | `_sync_scaled_gradients` | Conformant until finite-step gate |
| Common nonfinite result controls the update | boolean is calculated but ignored by the optimizer path | **Nonconformant; B-03** |

The implementation also respects the important warning against generating a
whole branch family from one shared true private profile. Branched children get
separate posterior-predictive logical draws, preserving the sibling-independence
assumption used by the leave-one-out baseline.

## Loss-law trace

For `rb_ppo`, code matches the production equation in `NOTATION.md`:

$$
\frac1B\sum_b\left[
\frac1{K_b}\sum_{a\in\mathrm{children}(b)}\sum_x
\widehat P_b(x\mid a)\,\mathcal L_{\mathrm{clip}}(x,a)
+\sum_x\widehat P_b(x\mid b)
  (\mathcal L_V+\mathcal L_H+\mathcal L_{\mathrm{KL}})
\right].
$$

`pbg/core/losses.py:137-209` uses child posterior mass divided by child count
for policy terms and parent posterior mass for value, entropy, and optional
reference KL. Duplicate abstract children remain separate columns. The normal
composite selects the sampled actor combo and applies the ordinary PPO clipped
surrogate and scalar lambda target.

The fixed loss equations are correct apart from the `linear_decay` coefficient
lookup in B-04. Invalid lambda and coefficient ranges can also enter through the
validation gap in B-05.

## Model architecture verdict

For the full 52-card deck, the two networks substantially follow
`docs/theory/MHA_model.md`.

| Architecture requirement | Code evidence | Assessment |
|---|---|---|
| One weight-only object with nonshared pre/post nets | `PokerModelBundle`, `PreflopNet`, `PostflopNet` | Conformant |
| 169 native preflop classes | class aggregation and `N_CLASSES=169` | Conformant for `R=13` |
| Compatible postflop combo rows for live players | `build_compact_map`, live-seat gather, full scatter | Conformant for `R=13` |
| Folded players absent from postflop attention rows | status/live-seat masks and compaction | Conformant |
| Public, exact arithmetic, line/history, seat/status inputs | `PokerPublicBatch`, packed history and public token builders | Conformant, subject to B-02 |
| Button-relative seats | relative-seat conversion before embeddings/readout | Conformant |
| Pre-LN residual transformer | per-layer LN, MHA residual, incidence/FFN residual | Conformant |
| Learned positive per-head content temperature | `MHA.temperature` | Conformant |
| Key-side log-reach bias is not temperature-scaled | `MHA.forward_biased` adds bias after content scaling | Conformant |
| Exact incidence residual including equal-combo channel | `IncidenceResidual` and numerical reconstruction tests | Conformant |
| Factorized passive/call/raise plus DLogisticTick and all-in atom | `TypeHead`, `RaiseMixtureHead`, policy adapter legal normalization | Conformant |
| One actor-private-type V, no learned per-action Q heads | `ValueHead` on actor row | Conformant |
| Optional global player-vector and squared-moment heads | `MomentHeads` behind `use_moment_heads` | Structurally present, not trained by any registered loss |
| Suit-consistency auxiliary | no implementation | Deferred as documented in `MHA_model.md` |
| Variable compact deck geometry | model document itself fixes 52/1326/169; product docs generalize R | **Not implemented; B-01** |

## Folded-player statistics

The folded-player architecture is genuinely connected, not a placeholder.

### Ownership and production flow

1. Poker rollout stores each fold's board, reach row, seat, line metadata, and a
   partition dependency key.
2. Postflop routing groups rows by street and number of live players.
3. Because `PostflopNet.requires_partition_summary = True`, the rollout backend
   calls `PokerGPUPartitionAdapter.partition_folders` for dependency-distinct
   rows.
4. The adapter validates versioned fields returned by the sibling evaluator. No
   local collision posterior, partition ratio, or showdown implementation is
   substituted.
5. `PostflopNet._folded_geometry` consumes unary hand corrections and signed
   pair-kernel factors. `_folder_context` consumes per-folder log mass, single
   card/card-marginal pressure, optional tight delta, seat identity, validity,
   and public line metadata.
6. Those features affect corrected log reach, row geometry, pressure channels,
   tight-folder products, and folded-seat readout summaries.
7. Only live players contribute hand rows to attention. Folded seats are restored
   to the seat summary through folder features.

This is exactly the relevant architecture for `P>2`: in heads-up a fold ends the
hand, so there is no postflop continuation with a folded observer; with three or
more players, one or more folded seats can continue to influence the live-hand
posterior.

### Tests

- `tests/test_p7_postflop.py:176-204` changes a semantic per-folder card field and
  verifies the network output changes.
- `tests/test_p7_postflop.py:207-226` verifies a folded seat creates no postflop
  attention row.
- `tests/test_p13a_gpu_rollout.py:442-465` uses a three-player state and verifies
  partition reuse across an action and invalidation on a fold.
- Folder/partition adapter tests validate exact field shapes, validity, device,
  and evaluator provenance.

### Qualification

The cache currently reaches the correct semantic result but its Python/device
boundary is a performance and observability defect (M-01). Preflop intentionally
retains all seated ranges, including folded seats, because the preflop network
can afford all class rows; evaluator folder statistics begin postflop, matching
`core_math.tex:1420-1425` and `MHA_model.md`.

## Mathematical features that are not production capabilities

- NeuRD and the synchronous R-NaD/DeepNash analogues;
- adaptive/Neyman branching intensity and any intensity loss;
- public `W/S/U/V-vector` target losses;
- stored all-action Q targets or learned per-action value heads;
- suit-consistency loss;
- MMD or sampled-KL losses;
- an exploitability or approximate-best-response solver.

Their absence is acceptable only because the fixed v1 boundary explicitly
defers them. The docs should consistently separate an optional untrained output
head from an implemented loss/control feature.
