# Postflop rejection sampling and chance closure

Status: implementation proposal.
## 1. Contract and ownership

```text
mode        branched Rao--Blackwell only
H           1326 at R=13
P           dealt seats, 2..6
B           parent rows in the post_A pack
N_b         executed player-child count for parent b
E           sum_b N_b
G           total logical chance children
q           floor(32/P) profile groups per warp
beta_max    rollout.beta_max, common maximum ordinary-node branching intensity
N_max       ceil(beta_max), ordinary player/chance degree bound
```

| Owner | Fields | Lifetime |
|---|---|---|
| `FamilyWork` parent | normalized live exact ranges `[A,H]`, public state | through child materialization |
| `FoldedRanges[pre]` | normalized folded `r_pi[K]` rows | tree generation |
| `FoldedRanges[post]` | normalized folded `r_pi[H]` rows | tree generation |
| `FamilyWork[pre].hot` | `folded_range_ref_by_seat[P]`, `-1` for live seats | tree generation |
| `FamilyWork[post_A].hot` | ordered `folded_range_ref[P-A]` into `FoldedRanges` | tree generation |
| `FamilyWork[post_A].retained` | compact model-facing folded statistics | through learner |
| inference workspace | factorized `policy_raw[H,F]` | model forward through last child materializer |
| sampler shared memory | packed per-seat base CDFs of width `K` or `H`, optional actor CDFs | one parent CTA |
| sampler registers | combo ID, card mask, binary-search bounds, RNG counter | one proposal |
| descriptor workspace | logical child, action, chance event, destination | transaction |
| `DeviceFailure` | minimum failing logical address and attempt count | transaction |

`policy_raw = {type_logits, omega, mu, s, iota}` uses native inference dtype. No `[H,event_count]` expansion,
accepted-combo array, or child-range staging exists.

`FoldedRanges` contains only the normalized folded reach factor `r_pi`, the stored per-player factor of the
$\mathfrak r_{\mathsf s}$ reach cache in `core_math.tex`; it contains no folded statistics or public metadata. `pre` rows
use the class domain `K` and store total probability mass per class; all `post_A` families share the exact-hand `post`
partition. A folded-range reference identifies its partition and row. Rejection sampling and folded-stat construction
resolve these references; neither the references nor `r_pi` are model inputs. A fold appends one immutable row while
writing its child destinations.
Preflop rows duplicate their small `K`-domain factor in `natural_input` and `FoldedRanges[pre]` so the reference passes
directly to postflop. Fold and chance events separately recompute the compact folded statistics stored in each
resulting `natural_input`; that engine consumes only resolved `r_pi` rows and the resulting board.

## 2. Sampler kernel

```text
sample_and_close<<<B,W*32>>>

CTA             one parent
warps/CTA       W, calibrated by P, hardware, registers, and shared-memory variant
warp lane       group*P + seat
group           lane/P
seat            lane%P
active lanes    q*P per sampling warp
```

| $P$ | $q$ | active lanes |
|---:|---:|---:|
| 2 | 16 | 32 |
| 3 | 10 | 30 |
| 4 | 8 | 32 |
| 5 | 6 | 30 |
| 6 | 5 | 30 |

All CTA threads cooperatively build CDFs. Sampling warps then process logical children independently. Physical work
order may vary; RNG keys and descriptor addresses use logical child and attempt ordinals, and routing scans logical
order.

Choose `W` to maximize measured eligible warps subject to resident-parent coverage, register pressure, and the
combined L1/shared-memory partition. Calibrate feasible `W in {1,2,4,8,16}` variants; one-warp CTAs are not a design
requirement. One CTA remains the maximum for one parent because admitted player degree is expected to be at most
`N_max` for both player and chance nodes. Calibrate the launch table from GPU architecture, `P`, `N_max`, registers,
and the selected shared-memory variant.

### 2.1 Independent range proposal

Preconditions:

```text
range[p,h] >= 0
live-range board-conflicting mass == 0
sum_h range[p,h] == 1 under the range-normalization contract
range pointer resolves live row or a FoldedRanges `r_pi` row
```

The CTA constructs `cdf[p,h] = sum_{j=0}^h range[p,j]` once in shared memory using a canonical-order,
block-cooperative fp32 prefix scan, then synchronizes. No permutation is stored. Define
`total[p]` from the final CDF entry so sampling follows the realized reduction. Folded rows are masked against the
current board during CDF construction; their stored rows are not rewritten on later streets. For `r_pi[K]`, class
entry `k` has board-conditioned mass
`r_pi[k] * legal_combo_count[k] / orbit_size[k]`. After drawing `k`, draw one of its legal exact combos uniformly.
Its CDF stays `K`-dimensional; exact-hand rows use the ordinary `H`-dimensional CDF.

For each group assigned an unfinished logical child:

```text
target = min(U(logical_child, attempt, seat, private) * total[seat], nextafter(total[seat],0))
lo = 0; hi = width[seat]
repeat ceil(log2(width[seat])) times:
    mid = (lo + hi) >> 1
    advance = cdf[seat,mid] <= target
    (lo,hi) = advance ? (mid+1,hi) : (lo,mid)
selected = lo
leader:
    gather P selected hand masks
    accept iff popcount(board | hands) == board_count + 2P
```

The fixed upper-bound search selects the first `cdf > target`; zero-weight plateaus are unselectable. `K` rows use
eight steps and `H` rows use eleven. Rejection increments the child's attempt ordinal; exhaustion writes
`DeviceFailure`.
When fewer children than profile groups remain, groups may evaluate consecutive attempts for the same child and
select the lowest successful attempt.

Random choices use versioned logical counter addresses and the fastest validated device generator. Compatibility with
the legacy sampler's seeded output is not required. Distinct logical child, attempt, seat, and purpose addresses must
produce independently usable choices and remain invariant under pack subdivision and physical scheduling.

### 2.2 Player event

For an accepted profile:

```text
leader loads policy_raw[acting_hand]
leader evaluates the legal action distribution with the canonical policy operator
leader draws action with RNG purpose player_event
write (logical_action_child, action)
retain the profile through transition classification
```

The canonical policy operator evaluates and samples the factorized head without an `[H,event_count]` expansion. The
same fp32 operator supplies selected-action probabilities and log-probabilities to action sampling, chance sampling,
materialization, and the learner. Its learner entry points provide autograd through `policy_raw`.
Its API is specified in [`poker_policy_cuda_operator.md`](poker_policy_cuda_operator.md).

### 2.3 Chance event strategies

If a sampled action produces a chance parent, its originating profile already has the action-conditioned posterior
law. Use it for the first logical chance child. Additional chance children need independent profiles. Benchmark these
CTA-local alternatives:

| Strategy | Additional chance profile |
|---|---|
| likelihood thinning | sample a collision-free parent profile and accept with `pi(action|actor_hand)` |
| tight-envelope thinning | accept with `pi(action|actor_hand) / max_h pi(action|h)` |
| post-action CDF | overwrite the base actor CDF with `range_actor[h] * pi(action|h)`, then reject only collisions |
| overlapped CDF | while other warps sample player actions, build extra actor CDFs for the at most two actions that can produce chance |

The last two strategies sample the exact action-conditioned posterior because only the actor's independent proposal
row changes. Post-action overwrite retains the base `[P,H]` shared footprint; overlapped construction uses up to two
extra `[H]` CDF rows and trades earlier availability for shared-memory/L1 pressure. A deterministic calibrated selector
may choose among these variants from hardware, `P`, and realized bounded chance work. Chance degree and variant
selection must not depend on the originating private profile or its RNG outcomes.

Originating-profile reuse intentionally contradicts the authoritative pseudocode's literal fresh-profile draw at the
chance node. It is an explicitly approved design simplification because the reused profile has the required
action-conditioned law; authoritative theory remains unchanged pending reconciliation before production.

For each accepted or reused chance profile, form the remaining-deck bit mask. Draw one card, or three cards without
replacement for the flop, with unbiased bounded integers and set-bit selection; sort the flop cards into its canonical
label. Write `(logical_chance_child, source_action_child, chance_event)`, then discard the profile registers.

Duplicate labels share neither RNG addresses nor topology. Poker RB preterminal stopping bounds the closure before
completion work.

## 3. Routing and materialization

```text
sample_and_close
-> stable destination count/scan
-> exact destination reservation
-> materialize
-> stream-ordered commit
```

For each distinct selected action $a$, `materialize` streams `policy_raw[h]` and parent ranges once over
`h=0..H-1`:

$$
\mathtt{oldLogP}[h]=\log\pi(a\mid h),
\qquad
r'_{\rm actor}(h)\propto r_{\rm actor}(h)\pi(a\mid h).
$$

It reuses this arithmetic across duplicate action occurrences, applies each child's chance/board mask, normalizes or
safely rescales, and writes the final row. A fold appends the actor's updated `r_pi` to the matching `FoldedRanges`
partition and writes its reference into resulting hot rows. Fold and chance destinations recompute compact folded
statistics into `natural_input`. Atomics may count destinations but never determine insertion order. Commit waits for
all writes and `DeviceFailure`.

## 4. Working sets

Per-parent exact ranges:

$$
M_{\rm range}=M_{\rm cdf}=4PH=5304P\ \text{bytes}.
$$

| $P$ | range/CDF KiB |
|---:|---:|
| 2 | 10.36 |
| 3 | 15.54 |
| 4 | 20.72 |
| 5 | 25.90 |
| 6 | 31.08 |

Optional overlapped action CDFs add

$$
M_{\rm actionCDF}=4C_aH,\qquad 0\le C_a\le2.
$$

Post-action overwrite and likelihood thinning add no CDF storage.

Factorized policy storage:

$$
M_{\rm policy}=HFb_{\rm inf},
\qquad
F=4+3m_{\rm raise}.
$$

For `m_raise=2`, `b_inf=2`, `M_policy=25.90 KiB` per parent. Policy access is pointwise and excluded from the shared
CDF working set. Exact ranges are read once to construct it; no global CDF is stored.

## 5. Device utilization and variant selection

| Item | RTX 2080 Ti | H100 SXM |
|---|---:|---:|
| SMs | 68 | 132 |
| maximum resident warps/SM | 32 | 64 |
| maximum blocks/SM | 16 | 32 |
| combined L1/shared pool | 64 KiB | 256 KiB |
| L2 | 5.5 MiB | 50 MiB |
| example $B$ | 84 | 512 |

Aggregate $P=6$ range storage: `2.55 MiB` and `15.54 MiB`; with `m_raise=2`, two-byte `policy_raw`: `4.67 MiB` and
`28.49 MiB`. Cache residency is not a correctness condition.

Expected sampling-lane utilization within each active warp is 93.75--100%. Select `W` jointly with the chance-profile
strategy: extra warps improve latency hiding, while overlapped CDFs reduce the shared/L1 budget and may reduce resident
parents. Complete family-transaction time is the primary selector; also record rejection tail latency, eligible
warps/cycle, long-scoreboard stalls, and cache/DRAM traffic.

## 6. Acceptance gates

- Exact collision-conditioned target law, including actual folded-player ranges.
- Exact board conditioning of class-total `r_pi[K]` through legal-orbit fractions.
- Fixed-seed invariance under pack subdivision and physical scheduling.
- Fixed 8/11-step upper-bound search for `K/H`; zero-mass hands are never selected.
- No local-memory spills or global CDF storage for proposal state.
- No accepted-profile, expanded-action, or child-range staging output.
- Canonical policy forward agreement across sampling, materialization, and learner; valid learner gradients.
- Distribution agreement for originating-profile reuse and every additional-chance strategy.
- Stable descriptor and destination order.
- Terminal exhaustion with no published destination.
- CUDA coverage for $P=2..6$, every street, concentrated/zero-heavy ranges, and duplicate children.
- Benchmark CTA warps and chance-profile strategies by hardware, `P`, chance degree, action frequency, and range
  concentration.
- Enforce `N <= N_max = ceil(beta_max)` for every ordinary player and chance node.
- Record registers/thread, resident blocks/warps, active lanes, eligible warps/cycle, L1/L2/DRAM bytes, collision and
  action-thinning attempts, tail latency, and complete family-transaction time on each target hardware profile.

Initial tuning targets RTX 2080 Ti with `P in {2,6}`. A data-creation pass loads existing checkpoints, generates short
trajectories, and captures their actual boards, `r_pi`, `policy_raw`, legality, logical degrees, action/chance
descriptors, and realized work. Synthetic inputs are used only for corner correctness and controlled stress. Selected
launch settings live in `configs/hardware/`; their exact schema is finalized with the kernel benchmark.

Root private-profile generation, including refill, never calls this rejection sampler. With an empty board and
uniform root `r_pi`, generate a uniform random permutation of the `4R`-card deck, take the first `2P` cards, assign
consecutive pairs to seats, and canonicalize each hand. An equivalent standard random-subset primitive is valid if it
also randomizes the pairing and seat assignment.

For the few deals that initialize or refill one branched tree, use a basic host primitive and transfer the compact
deals as orchestration; a dedicated CUDA launch is unnecessary. This is a bounded preprocessing exception, not a CPU
rollout backend, and the host retains no rollout state. Sequential initialization of many trajectories uses the
batched GPU equivalent. Both paths derive independent seeds from the logical root/refill address. Neither path builds
range CDFs, performs collision rejection, or enters chance closure.

## 7. Implementation ownership

Proposed boundary: this repository owns rollout rejection sampling, folded statistics, policy composition, RNG,
exhaustion, and chance-card generation. `poker-gpu-evaluator` owns operations involving `hand_rank`, including
showdown and completion. This supersedes the current repository contract and requires an explicit boundary change
before implementation.
