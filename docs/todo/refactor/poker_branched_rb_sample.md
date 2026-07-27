# Poker branched-RB `sample`

Status: implementation proposal.

## 1. Contract

This file specifies the non-trivial poker implementation of `sample` in
`model -> sample -> commit`. Other modes need only categorical draws from their stored `true_profile` and stable
labels.

```text
B           parent rows in one post_A pack
P           dealt seats, 2..6
H           exact-hand width; 1326 at R=13
K           preflop class width
N_max       ceil(rollout.beta_max)
G           logical children in the pack
q           floor(32/P) profile groups per warp
```

`sample` reads:

- raw model heads in the family workspace;
- `FamilyWork.hot.true_profile` for every parent;
- public state, live exact ranges, and resolved folded `r_pi` rows;
- legality, logical RNG addresses, and realized child ordinals.

For each logical child $g$ and seat $p$, the bounded handoff is:

```text
SampleOutput <G>
  logical_child[g]
  label_kind[g]                         # action or chance
  label[g]
  source_action_child[g]                # chance only
  true_profile[g] : int32 <P>
  attempt_count[g]
```

`true_profile[g]` is the profile used to draw child $g$'s label. It may remain in registers when `sample` and `commit`
are fused; otherwise it occupies bounded family workspace. `commit` writes it to a non-leaf child's
`FamilyWork.hot.true_profile`; no second profile owner exists. The reuse child has `attempt_count[0] = 0`.

`sample` stages a compact asynchronous D2H packet containing degrees, labels, logical edges, and unresolved edge
handles. The CPU accepts it with `commit`'s endpoint patches and transaction marker.

| Owner | Contents | Last consumer |
|---|---|---|
| parent `FamilyWork.hot.true_profile` | inherited private profile | `sample` |
| family parent | public transition state, live exact ranges, folded references | `commit` |
| family workspace | raw model heads, `SampleOutput` | `commit` |
| sampler shared memory | base CDFs and selected actor-proposal scratch | parent CTA |
| sampler registers | combo IDs, masks, search bounds, RNG counters | proposal or fused `commit` |
| child `FamilyWork.hot` | committed `true_profile` | child's `sample` |
| topology packet | degrees, labels, unresolved edges | CPU `Topology` builder |
| `DeviceFailure` | minimum failing logical address and attempt count | transaction |

Raw policy heads use their configured inference dtype. Proposal arithmetic is fp32. Sampling never expands policy to
`<H,event_count>` and never writes `FamilyWork`, `Leaf`, `SigmaHead`, `ActionHead`, `FoldedRanges`, or destination
tails.

`FoldedRanges[pre]` stores normalized class-mass rows of width $K$.
All `post_A` families share `FoldedRanges[post]`, whose rows have width $H$.

## 2. Parent CTA

```text
sample_branched_rb<<<B,W*32>>>

CTA             one parent
warps/CTA       W
warp lane       group*P + seat
group           lane/P
seat            lane%P
active lanes    q*P per sampling warp
```

With replicas, $B$ concatenates equal-width stable prefixes of one `input_type`; each CTA and every destination it
reserves keep one replica owner.

| $P$ | $q$ | active lanes |
|---:|---:|---:|
| 2 | 16 | 32 |
| 3 | 10 | 30 |
| 4 | 8 | 32 |
| 5 | 6 | 30 |
| 6 | 5 | 30 |

Logical child $0$ uses the parent's `true_profile` and bypasses rejection. If $N>1$, all CTA threads cooperatively
build CDFs and sampling groups generate children $1,\ldots,N-1$. Physical order may vary; output addresses and RNG
keys use logical child and attempt ordinals.

Calibrate feasible $W\in\{1,2,4,8,16\}$ by architecture, $P$, `N_max`, registers, shared memory, and resident-parent
coverage. One CTA is the maximum for one parent; one warp is not required.

## 3. Profile reuse and additional profiles

For parent $\mathsf s$ with $N\geq1$, let $x_j$ be the profile used for logical child $j$:

$$
x_0=\mathtt{FamilyWork.hot.true\_profile}[\mathsf s],
\qquad
x_j\sim\mu_{\mathsf s}\ \text{ independently for }1\leq j<N.
$$

The inherited $x_0$ already has law $\mu_{\mathsf s}$. The kernel therefore rejection-samples $N-1$, not $N$,
profiles. When $N=1$, it builds no proposal CDF.

For each parent $b$, seat $p$, and hand $h$:

```text
range <B,P,H>
cdf <P,H>                                # parent-CTA workspace when N > 1
```

The input contract is:

$$
\mathtt{range}[b,p,h]\geq0,\qquad
\sum_h\mathtt{range}[b,p,h]=1,
$$

with zero mass on board-conflicting live hands. A range pointer resolves either a live row or a folded `r_pi` row.

The CTA constructs each canonical-order CDF once with a block-cooperative fp32 prefix scan. It stores no permutation.
The final realized CDF entry defines the sampling total.

Folded rows are masked against the current board during CDF construction. For each preflop class $k$:

$$
\mathtt{conditionedMass}[k]
=\mathtt{rPi}[k]\,
\frac{\mathtt{legalComboCount}[k]}{\mathtt{orbitSize}[k]}.
$$

After drawing a class, draw one legal exact combo uniformly. The CDF remains $K$-wide; exact-hand rows use width $H$.

For each logical child $1\leq j<N$, attempt, and seat:

```text
target = min(U(logical_child, attempt, seat, private) * total[seat],
             nextafter(total[seat], 0))
lo = 0
hi = width[seat]
repeat ceil(log2(width[seat])) times:
    mid = (lo + hi) >> 1
    if cdf[seat,mid] <= target:
        lo = mid + 1
    else:
        hi = mid
selected = lo
```

The group leader gathers the $P$ selected card masks and accepts exactly when they and the board are disjoint. The
fixed upper-bound search selects the first CDF entry greater than `target`; zero-mass plateaus are unreachable.
$K$ rows use eight steps and $H$ rows use eleven.

Rejection increments the logical attempt ordinal. Groups test fixed consecutive attempt waves, synchronize, and
publish the lowest success. Exhaustion writes `DeviceFailure`.

RNG addresses contain version, logical child, attempt, seat, and purpose. Results are invariant under pack
subdivision and physical scheduling.

## 4. Labels

### 4.1 Player label

For each player-child profile, whether inherited or sampled:

```text
load raw policy head for the acting combo
evaluate legality with the canonical policy operator
draw the action with purpose player_event
write the action label and the profile that generated it
```

The policy operator is [`poker_policy_cuda_operator.md`](poker_policy_cuda_operator.md). The sampled scalar
log-probability may be used by an action-conditioned proposal, but `commit` constructs the retained
`ActionHead.old_log_pi` row after the action is known.

### 4.2 Chance label

Each chance parent inherits the `true_profile` that generated its incoming action. This profile has the exact
action-conditioned law after conditioning on that action, so logical chance child $0$ reuses it. Only chance children
$1,\ldots,N-1$ draw additional profiles from the action-conditioned posterior.

If $C$ player children enter chance, their $C$ inherited profiles each seed one chance child. The fused frontier saves
$C$ posterior draws before sampling further children.

For an observed action $a$, the actor proposal is proportional to:

$$
r_{\rm actor}(h)\pi(a\mid h).
$$

Poker permits at most two distinct action labels from one frontier to reach chance; duplicate sampled labels reuse
the same proposal. Prototype thinning and actor-CDF samplers, select one by complete `sample -> commit` time, and
delete the others. Production has no sampler selector.

For each inherited or sampled chance profile, form the remaining-deck mask. Draw one card, or three cards without
replacement for the flop, with unbiased bounded integers and set-bit selection. Sort flop cards into the canonical
label. The same profile becomes the resulting child's `true_profile`; conditioned on the chance label, it has the
correct posterior at that child.

Duplicate labels retain distinct logical addresses and topology edges.

## 5. Handoff and physical fusion

The semantic boundary is:

```text
model -> sample -> commit
```

`sample` ends after writing labels and their generating `true_profile` rows to `SampleOutput`, then staging its
topology packet. `commit` performs destination classification, device count/scan/reservation, range
materialization, folded-stat updates, retained-head writes, child `true_profile` writes, endpoint patches, and atomic
publication.

For a one-CTA pack, registers may cross the semantic boundary and the implementation may fuse:

```text
sample -> classify/count -> stable offsets -> materialize -> commit
```

Larger packs materialize `SampleOutput` in bounded workspace before the stable scan. Neither form changes ownership.
Chance closure repeats `sample -> commit` in a bounded device frontier without host synchronization per generation.

## 6. Working sets

For a parent with $N>1$, exact ranges or base CDFs require:

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

The largest prototype workspace adds at most:

$$
M_{\rm actorCDF}=4C_aH,\qquad 0\leq C_a\leq2.
$$

The final schema retains only the selected sampler's exact bound.

For factorized policy width $F=4+3m_{\rm raise}$ and inference bytes $b_{\rm inf}$:

$$
M_{\rm policy}=HFb_{\rm inf}.
$$

For `m_raise=2` and `b_inf=2`, this is 25.90 KiB per parent. Policy access is pointwise; no global CDF is stored.
An $N=1$ parent uses its inherited profile and needs no proposal CDF.
Sampler experiments use `experiments/range_database`; the evaluator may test the same CDF/rejection design separately.

## 7. Root profiles

The separate root sampler uses a batched GPU uniform-permutation or equivalent random-subset kernel. For each root it
assigns the first $2P$ cards in seat order, canonicalizes each hand, and writes
`FamilyWork.hot.true_profile`.

Both sequential and branched modes accept multiple roots. Prefer one branched root when roots differ only by RNG;
multiple branched roots remain useful for intrinsic differences such as sampled `true_profile` or stack size.
Root generation needs neither range CDFs nor collision rejection.

## 8. Acceptance

- Exact collision-conditioned law for live and folded ranges.
- Logical child $0$ uses its parent's `true_profile`; an $N$-child node samples exactly $N-1$ additional profiles.
- Every non-leaf child receives the exact profile used to draw its label.
- Every chance parent reuses its incoming action-conditioned profile for one chance child.
- Exact board conditioning of class-mass `r_pi` rows.
- Fixed-seed invariance under pack subdivision and scheduling.
- Fixed 8/11-step upper-bound search; zero-mass hands are never selected.
- Stable logical labels, descriptors, and topology packets.
- No policy child-axis expansion, global proposal CDF, or retained profile owner besides `FamilyWork.hot`.
- No retained-array, destination, or endpoint-patch write by the logical `sample` stage.
- Minimum-address terminal failure with no published destination.
- CUDA coverage for $P=2,\ldots,6$, every street, duplicate labels, and concentrated or zero-heavy ranges.
- Select one profile sampler using complete `sample -> commit` time, rejection tails, spills, eligible warps, and
  memory traffic; delete every losing implementation.

## 9. Ownership

This repository owns rejection sampling, logical RNG, action/chance labels, exhaustion, and chance-card generation.
`poker-gpu-evaluator` owns showdown and completion operations consuming `Leaf`.
