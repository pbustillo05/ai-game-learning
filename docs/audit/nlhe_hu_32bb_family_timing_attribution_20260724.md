# Why is NLHE HU 32 bb branching slow? — rollout diagnosis, 2026-07-24

## Answer

Fixed-$\beta=3$ branching is slow for several independent reasons. The
rejection sampler matters, but it is not the main explanation.

Across three established seeds, legal branched rollout took a median
8.575 ms/decision versus 3.967 ms/decision for sequential
Rao--Blackwellization (RB). The mean attribution of that paired per-decision
gap was:

| Cause | Mean share of legal paired gap |
|---|---:|
| More expensive rollout model mix | 25.8% |
| Extra rejection-sampler work | 13.3% |
| Internal rollout sealing | 2.8% |
| Child materialization, range/topology movement, queueing, and other rollout work | 58.1% |

The strongest conclusions are:

1. **Removing nearly all rejection sampling makes branched rollout only about
   9.7% faster per decision in the two geometry-matched seeds.** The third seed
   improved 14.9%, but its ablated tree was also 11.0% smaller.
2. **Branched child materialization is expensive.** At the median legal shape,
   device-side child reservation and commit took about 16.1 seconds, while
   event sampling, state transition, and reach update took another 5.6 seconds.
3. **The generic paged live-reach path does enormous device-internal routing
   work.** The legal branched median recorded 51.35 GB of routing metadata
   traffic, 47.82 million moved rows, 73.22 million candidate rows, and
   3.25 million routing tensor operations. Sequential RB recorded no generic
   routing bytes.
4. **PCIe bandwidth is not the problem.** Branched rollout copied only about
   14.2 KB device-to-host and zero bytes host-to-device. It did, however, make
   about 2,494 host synchronization calls.
5. **One scalar synchronization serializes every branched policy pack.** The
   exact ragged action-row reservation reads the offspring total after every
   forward. There were 163--188 such barriers, exactly one per model pack. Their
   host wait was 52.2--60.6 seconds, but this overlaps the preceding model and
   offspring CUDA work and must not be added as an independent cost.
6. **After rollout, branched target propagation is also a major problem.**
   Completion itself took about 1.97 seconds, but branched propagation took
   14.67 seconds versus 0.040 seconds for sequential RB.
7. **The old `phase_seal_seconds` metric hid almost all sealing.** Final seal
   was only 4.2 ms, but sealing already performed inside rollout took about
   2.01 seconds, mostly the decision/action stores.

So the direct answer is: **branching is slow mainly because it turns each
public decision into ragged child work that repeatedly gathers, duplicates,
updates, routes, and seals dense private-state rows through generic paged
owners.** The postflop-heavy model mix is the next largest contributor.
Rejection sampling is real but secondary.

## Experiment

This was a rollout-only diagnosis: sample, chance-close, seal, complete, and
propagate targets, with no learner construction or optimizer step.

The legal comparison used:

- sequential RB with 3,072 roots per GPU;
- fixed-$\beta=3$ branched RB with one distinguished root;
- family pack maximum 84 and storage pages of 256 rows;
- the existing diagnostic upper trigger at 10,000 nodes and protected lower
  refill threshold at 1,000 nodes;
- 1,024 Monte Carlo completion samples per leaf;
- one full warmup followed by one profiled iteration;
- known seeds 104830, 209559, and 314288;
- the iteration-1,110 sequential checkpoint and its matching branched
  dummy-$T/C$-head checkpoint.

Each seed stayed on one physical RTX 2080 Ti for sequential, legal branched,
and ablated branched trials. The job order was counterbalanced across GPUs,
with a 30-second cooldown between jobs. All three GPUs ran concurrently.

GPU 1 did exhibit the suspected thermal behavior: its legal branched segment
reached 86 °C with a median 1,725 MHz SM clock. The other cards also became
hot under simultaneous load. Therefore, absolute cross-card seconds are
reported but causal comparisons use within-GPU paired rates. Complete
one-second temperature, power, clock, utilization, and memory telemetry is in
the raw artifact.

## Timing semantics

CUDA events were recorded asynchronously around the actual work and resolved
only at an existing synchronized phase boundary. Component timing therefore
does not add a synchronization per call.

Important interpretation rules:

- `rollout` is wall time and already includes internal arena sealing.
- `final seal` is the later conversion to immutable `RolloutStorage`.
- `completion` is the nested evaluator completion CUDA interval.
- `target propagation` is total target-builder CUDA time minus the nested
  completion interval.
- `chance closure` is a nested scope. It includes the work performed while
  closing chance-only frontiers and must not be added to sampler, transition,
  or child-work timers that occur inside it.
- synchronization wait is overlapping time: a scalar read may wait for prior
  model work that is also measured by model CUDA events.
- `storage_routing_bytes` measures temporary device-internal routing metadata,
  not PCIe traffic and not unique retained payload bytes.

## Legal sequential RB versus branched RB

The table reports independent medians across the three seeds.

| Metric | Sequential RB | Branched RB |
|---|---:|---:|
| Nodes | 18,773 | 23,912 |
| Decisions | 12,428 | 15,273 |
| Leaves | 3,072 | 6,365 |
| Rollout wall | 48.580 s | 132.436 s |
| Rollout per decision | 3.967 ms | 8.575 ms |
| Internal seal, included in rollout | 0.025 s | 2.008 s |
| Final seal | 0.0027 s | 0.0042 s |
| Completion | 0.882 s | 1.972 s |
| Target propagation excluding completion | 0.040 s | 14.670 s |
| Sample + seal + completion + targets | 49.484 s | 149.176 s |

The complete rollout/closure/target process was about 3.0 times slower in raw
wall time and about 2.45 times slower per retained decision. More leaves explain
the roughly doubled completion total: completion cost per leaf was similar.
They do not explain the 366-fold propagation-time ratio.

## Preflop and postflop forwards

| Mode | Family | Rows | Total forward time | Average per node |
|---|---|---:|---:|---:|
| Sequential RB | preflop | 4,797 | 1.721 s | 0.363 ms |
| Sequential RB | postflop | 7,613 | 34.053 s | 4.645 ms |
| Branched RB | preflop | 3 | 0.019 s | 6.312 ms |
| Branched RB | postflop | 15,270 | 62.994 s | 4.121 ms |

The branched preflop average is a singleton/tiny-pack artifact and is not a
family-speed estimate. The useful comparison is postflop: fuller branched
packs were not slower per row than sequential packs. Branching is model-heavy
because almost every decision is postflop, not because its postflop forward is
intrinsically inefficient.

Across paired seeds, model mix accounted for a mean 25.8% of the legal
per-decision rollout gap. This confirms the earlier family-attribution result,
but gives it a smaller and more precise role than “branching is slow because
postflop is expensive.”

## Rejection-sampler ablation

The ablation samples one valid root-conditioned private profile once and
reuses it throughout the tree. This preserves card validity for the resulting
chance draws but deliberately violates the branched posterior and sibling
independence law. It is diagnostic only and is not a training mode.

| Seed / GPU | Legal nodes | Ablated nodes | Legal rollout | Ablated rollout | Per-decision improvement | Sampler time saved |
|---|---:|---:|---:|---:|---:|---:|
| 104830 / GPU 0 | 24,095 | 24,068 | 8.575 ms | 7.744 ms | 9.7% | 11.303 s |
| 209559 / GPU 1 | 21,368 | 19,009 | 8.390 ms | 7.142 ms | 14.9% | 9.815 s |
| 314288 / GPU 2 | 23,912 | 24,042 | 8.750 ms | 7.917 ms | 9.5% | 11.533 s |

The two geometry-matched trials agree almost exactly: rejection sampling costs
about 9.6% of branched rollout per decision. The legal sampler consumed a
median 11.334 seconds for 23,911 profiles. The ablation consumed about
0.031 seconds for one profile.

The ablation left the expensive child/reach/topology machinery in place.
Its median residual non-model/non-sampler rollout cost remained
3.507 ms/decision versus 3.615 ms/decision legally. This is strong evidence
that simply making profile sampling free will not cure branching.

## What makes sample + chance closure complicated?

For legal branched rollout, median CUDA component totals were:

| Component | Time | Interpretation |
|---|---:|---|
| Rejection sampler | 11.334 s | One conditional profile per sampled child |
| Event sampling | 0.827 s | Player action or public chance event |
| State transition/classification | 2.577 s | Poker state/history mutation |
| Reach update | 2.202 s | Dense conditional range update |
| Logical child reservation | 7.951 s | Topology, occurrence, controller, and reach-slot reservation |
| Child commit | 8.158 s | State/topology commit, reach write, path factors, action capture |
| Chance-closure scope | 19.034 s | Nested scope; not additive with rows above |

The reservation and commit pair alone cost about 16.1 seconds. The sampler is
therefore not hiding a cheap closure mechanism: duplicating the relevant
parents, assigning exact child addresses, updating dense reaches, writing
ragged action rows, and closing chance nodes is itself expensive.

The untimed residual after subtracting model forward, rejection sampling, and
internal sealing was 55.83 seconds (3.615 ms/decision) for branched rollout
versus 11.16 seconds (0.891 ms/decision) for sequential RB. The component
timers above explain part of that residual. Host-side scheduling, Python launch
overhead, capture-store work, and generic page routing account for the rest.

## Data movement and synchronization

| Metric | Sequential RB | Branched RB |
|---|---:|---:|
| H2D transfers / bytes | 0 / 0 B | 0 / 0 B |
| D2H transfers / bytes | 600 / 12.1 KB | 2,494 / 14.2 KB |
| Host synchronization calls | 600 | 2,494 |
| Measured host wait, overlapping prior CUDA | 0.017 s | 60.44 s |
| Generic device routing bytes | 0 B | 51.35 GB |
| Storage rows moved | 10.86 M | 47.82 M |
| Candidate rows inspected | 0 | 73.22 M |
| Routing tensor operations | 55,377 | 3.25 M |

The device-to-host payload is tiny, so transfer bandwidth is irrelevant. The
problem is many control dependencies. In the typical legal tree, the action
reservation performed 186 scalar offspring-total reads for 186 postflop packs
plus two preflop packs. That makes the currently asynchronous-looking pipeline
effectively pack-serial.

The 60-second synchronization wait is not 60 seconds of extra copy time. The
scalar read follows the forward and waits for that forward and offspring
calculation to finish. It is nevertheless important: exact ragged allocation
currently prevents the host from scheduling beyond that boundary.

The largest device-internal routing sources in seed 104830 were:

- 36.48 GB from generic paged live-reach writes;
- 11.59 GB from generic paged live-reach reads;
- 1.50 GB from generic controller-row writes;
- 0.90 GB from generic topology-node writes.

Thus about 48.1 GB of 51.4 GB was live-reach page routing. This is routing
metadata caused by generic page access, not 48 GB of scientifically necessary
reach payload.

## Completion, propagation, and sealing

Completion is not the post-rollout bottleneck. Branched RB had about twice as
many leaves and took about twice as long to complete them. The evaluator cost
per leaf remained close to sequential RB.

Target propagation is qualitatively different:

- sequential RB uses a batched segmented player-clock recurrence and took
  about 40 ms;
- branched RB reduces a ready tree, writes ragged edge posteriors/advantages,
  and advances wide private-state moments through Python-driven node/action
  loops; it took about 14.67 seconds.

Sealing also needs correct phase attribution:

| Seal component | Sequential RB | Branched RB |
|---|---:|---:|
| Internal leaf seal | 0.0017 s | 0.323 s |
| Internal decision/action seal | 0.0236 s | 1.697 s |
| Final `RolloutStorage` seal | 0.0027 s | 0.0042 s |

The final seal is cheap. The large difference is the internal decision/action
seal already charged to rollout wall.

## Diagnosis and next optimization order

The evidence supports this order:

1. **Replace generic live-reach page routing in the branched hot path.** It is
   the dominant source of routing metadata and is unique to branching.
2. **Reduce reservation/commit work per child wave without changing stable
   logical addresses or the last-consumer storage contract.**
3. **Remove avoidable validation synchronizations and redesign the exact
   ragged action reservation so one scalar total does not serialize every
   model pack.** Any redesign must still reserve exact child/action spans
   before publication.
4. **Vectorize or batch branched target propagation.** This is separate from
   rollout but costs almost 15 seconds per iteration at this shape.
5. **Only then optimize the rejection sampler.** A perfect sampler is worth
   roughly 10% of branched rollout here.
6. **Do not prioritize model batch tuning first.** Branched postflop forwards
   are already faster per row than sequential postflop forwards.

These are implementation targets, not permission to weaken the authoritative
sampling law, stable FIFO family scheduling, exact reservation, or
last-consumer lifetime rules.

## Artifacts and reproduction

Raw and reduced evidence:

- `runs/experiments/nlhe_hu_32bb_rollout_diagnosis_20260724/summary.json`;
- `runs/experiments/nlhe_hu_32bb_rollout_diagnosis_20260724/gpu0_seed104830/`;
- `runs/experiments/nlhe_hu_32bb_rollout_diagnosis_20260724/gpu1_seed209559/`;
- `runs/experiments/nlhe_hu_32bb_rollout_diagnosis_20260724/gpu2_seed314288/`.

The artifact root includes the exact Git HEAD, dirty-worktree status, tracked
binary-capable patch, a copy of the measured launcher, pre/post
`nvidia-smi -q`, empty per-job stderr logs, per-job JSON, boundary snapshots,
and one-second GPU telemetry. The status file records the pre-existing
untracked files that are necessarily absent from `git diff`.

Reproduction and reduction:

- `tools/launch-nlhe-hu32-rollout-diagnosis.sh`;
- `tools/analyze-nlhe-hu32-rollout-diagnosis.py`.

The working-tree patch stored before launch is the source authority for these
numbers. Later audit wording changes are not part of the measured code.
