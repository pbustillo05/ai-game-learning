# Synthetic Poker Game Design

## Status and purpose

This document proposes a replacement for the current Kuhn/Leduc-anchored synthetic
dial. It is a research design, not an implemented contract. The goal is a granular
poker-like family for testing RB and branched RB without a second poker implementation
or a large rewrite of the GPU hand evaluator.

The family should be understandable as poker at every point. It does not interpolate
through Kuhn or Leduc, and it is always heads-up. Multi-player NLHE remains separate.

## Why two dials

There are three training estimators of interest:

1. **Normal sampling** samples a private-card profile, a public trajectory, and
   any terminal runout. Its target contains variance from all three sources.
2. **Rao--Blackwellization** conditions on the sampled public trajectory, sums
   the loss over compatible actor private types, and uses completion to estimate
   the conditional terminal value. By the law of total variance, integrating
   private states and terminal chance cannot increase target variance.
3. **Branched RB** additionally samples several public children at a node. The
   current branching rule is

   \[
   K_\beta(\sigma)=\lfloor\beta(\sigma)\rfloor+
   \operatorname{Bernoulli}(\beta(\sigma)-\lfloor\beta(\sigma)\rfloor),
   \qquad \beta(\sigma)\ge 1.
   \]

   Thus \(\beta=1\) retains one public path, while larger \(\beta\) spends more work on
   independent children. Fixed or public-stage-dependent \(\beta\) exists now. A
   variance-guided same-budget allocation is a future experiment, not an implemented controller.

These estimators should be stressed by two different game properties:

- The **chance/type dial** increases the private type space and the amount of
  terminal chance left for completion. This should increase the potential gain
  from RB.
- The **betting dial** increases the width and depth of the public player tree.
  This should increase the potential gain from branching.

The underlying parameters must still be logged: a scalar dial is a named parameter
path, not permission to hide the deck, board horizon, stack, or bet grid. Scientific
comparisons should primarily vary one dial while holding the other fixed.

## Fixed game rules

Every main-family point has the following invariants:

- two players;
- two private cards per player;
- four suits;
- Hold'em-style fold, check/call, and raise-to actions;
- the usual poker hand classes and kicker rules;
- collision-free dealing from an explicitly active deck;
- chip payoffs with ties split exactly.

Varying suits is not a main dial. Four suit bitmasks are the evaluator's natural
fast path. With one to three suits, straight flushes may exist when quads cannot,
straights may exist when trips cannot, and flushes can become more common than
straights. That would require several rule modes instead of one controlled family.

## Deck geometry and private tokens

Let \(R\) be the number of active ranks and use the highest \(R\) ranks of the
standard deck. The recommended clean range is

\[
R\in\{8,9,10,11,12,13\}.
\]

With four suits:

- the active deck size is \(D=4R\);
- the exact two-card private space is \(H=\binom{4R}{2}\);
- the suit-canonical preflop class count is \(C=R^2\);
- after \(b\) real public cards, the compatible postflop width is
  \(H_b=\binom{4R-b}{2}\).

For example, \(R=8\) gives \(D=32\), \(H=496\), and \(C=64\). Full Hold'em
geometry, \(R=13\), gives \(D=52\), \(H=1326\), and \(C=169\).

Here \(C\) denotes preflop classes; it is deliberately not called \(K\), because
\(K_\beta\) already denotes the number of sampled branch children.

The model must use these intrinsic widths, not mask mostly inactive 169/1326 rows.
Preflop uses exactly \(R^2\) canonical tokens and postflop exactly the compatible
\(H_b\) combos, with compact-to-full maps only at external boundaries.

## Hand-class ordering

For four suits and five to seven total cards, exhaustive counting over
\(R=3,\ldots,13\) found only one adjacent obstruction to the standard
cumulative rarity order: on short decks, a full-house pattern is more common
than a flush pattern. The conventional short-deck fix is therefore sufficient
over the proposed range:

- \(R=13\): standard order, with full house above flush;
- \(R=8,\ldots,12\): short-deck order, with flush above full house.

Straight flush, quads, straight, trips, two pair, pair, and high card retain
their usual relative order. This statement treats categories cumulatively--for
example, "flush" includes straight flush and "trips" includes full houses and
quads--which is the relevant check for rarity-consistent category strength.

The GPU ranker can compute the usual rank and swap only the flush/full-house category
code before comparison, preserving within-category tie breakers. CPU and CUDA must
apply identical remapping.

## Chance/type dial

Let:

- \(b\) be the number of real board cards visible at the completion leaf;
- \(T\) be the number of real board cards at terminal showdown;
- \(q=T-b\) be the hidden terminal runout length integrated by completion.

The chance/type dial is a named monotone schedule over \((R,b,T)\). A simple
default keeps a public flop, \(b=3\), and walks from \((R,T)=(8,3)\) toward
\((13,5)\), incrementing either rank count or terminal horizon at each level.
For example:

| level | \(R\) | \(b\) | \(T\) | hidden cards \(q\) |
|---:|---:|---:|---:|---:|
| 0 | 8 | 3 | 3 | 0 |
| 1 | 9 | 3 | 3 | 0 |
| 2 | 9 | 3 | 4 | 1 |
| 3 | 10 | 3 | 4 | 1 |
| 4 | 11 | 3 | 4 | 1 |
| 5 | 11 | 3 | 5 | 2 |
| 6 | 12 | 3 | 5 | 2 |
| 7 | 13 | 3 | 5 | 2 |

This recommended path does not erase primitive values from configs or results.
Fixed-\(R\) and fixed-\(q\) slices help attribute observed variance changes.

Normal training samples the \(q\) cards; RB completion integrates them and returns
private-type-conditioned targets. Later levels add controlled terminal chance variance.

## Betting dial

The existing poker engine discretizes raise-to amounts in chip units. If the
stack is \(S\) and the chip unit is \(u\), its validated static action grid has
exactly \(N=S/u\) raise ticks before state-dependent legality masking.

Use integer \(N\) as the betting dial and report \(S\) and \(u\) alongside it.
Increasing \(N\):

- exposes more distinct legal bet sizes at many player states;
- permits longer legal raise/re-raise sequences before an all-in;
- increases both public-tree width and player-state count;
- directly stresses the storage and compute trade-off controlled by \(\beta\).

Board horizon is not silently folded into this dial. Street/runout choices belong to
the reported chance schedule, so stack effects are not confused with added chance
streets. Street-count ablations are not a third headline axis.

## Terminal boards with five to seven real cards

With two private cards, \(T=3,4,5\) produces five, six, or seven real cards. The
low-level `hand_rank` accepts four suit/rank bitmasks and selects the best five-card
hand in all three cases; the PBG showdown kernel does not use the seven-card wrapper.

There is nevertheless a real sibling-repository change. The current showdown
facade requires five real board entries, and its CUDA loop unconditionally adds
all five entries to the masks. Supporting \(T=3\) or \(T=4\) directly requires:

1. allowing trailing `255` board padding at terminal validation;
2. skipping entries outside the 0--51 card range before calling `add_card`;
3. making completion draw exactly \(T-b\) active cards rather than always
   completing to a five-card board;
4. adding CPU/CUDA parity tests for five-, six-, and seven-real-card holdings.

This modifies the evaluator facade and its showdown caller, but not the
`hand_rank` algorithm. The CPU path already ignores `255` when decoding a
board. Below five total real cards, however, the ranker's kicker assumptions
are not part of its supported contract; this proposal does not depend on such
partial-hand behavior.

Logical public cards, hidden completion cards, and evaluator padding are
different objects. Only logical public cards become model tokens. Padding must
never appear in the deck, reach state, chance law, or public observation.

## Optional dummy-board extension

Valid low cards can fill the evaluator's five board slots and avoid partial-hand
ranking. The kernel reads five board cards, so a logical two-card board needs three
dummies and a logical one-card board needs four.

A restricted candidate construction is:

- reserve ranks 2, 3, and 4 for evaluator-only dummies;
- leave rank 5 absent as a straight-breaking gap;
- use at most the real ranks 6 through A;
- for a two-card logical board, append three distinct dummy ranks 2/3/4;
- for a one-card logical board, append a low pair plus two distinct lows,
  such as 2/2/3/4 using distinct physical cards;
- assign dummy suits so
  `real_board_count[suit] + dummy_count[suit] <= 2` for every suit.

The rank gap prevents a dummy from completing a straight, and the suit bound
prevents two private cards from turning the augmented board into a synthetic
flush. For four real cards, the three distinct low cards act only as constant
low kickers. For three real cards, the low dummy pair deliberately embeds the
natural ordering as follows:

| natural three-card pattern | augmented evaluator pattern |
|---|---|
| high card | low dummy pair plus three real kickers |
| one pair | real pair plus low dummy pair |
| trips | real trips plus low dummy pair: full house |

Winner and tie ordering can therefore be preserved even though the evaluator's
displayed category is artificial. Dummies are never model inputs or sampled
chance cards.

This optional construction supports only compatible small decks: the simple 6--A
version has at most nine active ranks. Every geometry needs exhaustive comparison
with an independent logical-hand reference; "low and rainbow" is not proof.

## Reuse and implementation boundary

The generic `pbg/core` rollout, reach, branching, completion-target, value,
loss, minibatch, and training machinery should be reused unchanged. The new
family should also reuse the poker engine's betting transitions, raise-tick
action adapter, payoff assembly, and MHA design concepts. The current synthetic
game's one-card showdown rule is not an appropriate semantic base, though its
small exact solvers remain useful as generic test oracles.

Use one geometry object as the source of truth for active card IDs, compact
combos, preflop classes, board-compatible combos, and mappings to the sibling's
52-card encoding. The same MHA architectural family may be used across levels,
but every model is a separate geometry-sized instance. Hidden width and layer
count may be held fixed as experimental controls; token counts and action-head
widths may not be faked with permanent 169/1326 padding.

Poker-private mathematics remains evaluator-owned. This repository must not
reimplement hand ranking, collision-conditioned joint profiles, folded-card
partition corrections, or completion equities.

## Practical evaluator paths

The smallest plausible v1 keeps compact local reach/model tensors and uses the
existing 52-card/1326-combo representation only at the evaluator boundary:

1. scatter active compact ranges into full combo rows;
2. pass an active-card mask so completion cannot sample excluded ranks;
3. apply the standard/short-deck category mode consistently;
4. support `terminal_board_count` plus `255` padding for the direct main path,
   leaving validated dummies as a restricted alternative;
5. gather completion results back to active compact rows immediately.

At the smallest main point, the full boundary costs \(1326/496\approx2.67\) times
the minimal combo width. This bounded overhead may beat rewriting tuned kernels,
but realistic completion batches must establish whether memory becomes limiting.

If that boundary dominates runtime or memory, the next step is a versioned
compact evaluator API accepting active cards and an explicit \([H,2]\) combo
table. That is a performance optimization, not a prerequisite for validating
the game design.

The deliberately avoided paths are:

- a new CUDA partial-hand ranker;
- arbitrary suit counts in the main family;
- duplicated poker-private math in this repository;
- full 1326-row neural attention for every small-deck state;
- a monolithic scalar that silently mixes deck, board, stack, and street rules.

## Required validation before experiments

- Enumerate category frequencies for every supported \((R,T)\) and confirm the
  selected standard or short-deck order.
- Compare CPU and CUDA winner masks, tie shares, and short-deck remapping on
  five-, six-, and seven-real-card holdings.
- Verify compact/full combo maps, active-deck sampling, collision masks, and
  preflop class multiplicities for every \(R\).
- Exhaustively validate every enabled dummy geometry against an independent
  logical reference.
- Assert model token counts are \(R^2\) preflop and \(\binom{4R-b}{2}\)
  postflop, with no inactive rows entering attention.
- Report estimator variance, effective completion samples, nodes per tree,
  mean \(\beta\), wall-clock throughput, peak memory, and learning quality.
- Compare normal, RB, and branched RB at matched environment work and at matched
  wall-clock budgets; a variance reduction bought by unreported extra compute
  is not by itself an algorithmic win.
