# Request: compact-deck `pbg-v3` evaluator API

## Status and ownership

This is the PBG consumer's decision-complete change request for the sibling
`poker-gpu-evaluator` repository.  The sibling's authoritative current contract
is `/home/pablob/poker-gpu-evaluator/docs/api_pbg.md`; this repository must not
modify that file or the sibling implementation.  The sibling should update its
own API document and implementation together.

The current installed API is `pbg-v2`, fixed to 52 cards, 1,326 private combos,
169 preflop classes, and conventional full-deck category ordering.  PBG needs a
versioned `pbg-v3` contract.  Keep `pbg-v2` available for existing consumers,
but this repository will use `pbg-v3` for every deck, including `R=13`.

## Deck representation

- Add required integer `rank_count` (`R`) to every public PBG operation.  Accept
  exactly `4 <= R <= 13`.
- The active deck has `D=4R` cards and uses the existing global card IDs
  `52-D, ..., 51`.  Thus `R=4` is `J,Q,K,A` in all four suits.  Board padding
  remains `255`.
- Define `H=C(D,2)`.  Export `combo_cards(rank_count)`, returning a defensive
  `uint8 [H,2]` tensor in lexicographic global-card order.  Combo IDs are row
  indices in this tensor.
- Card-indexed output axes are compact active-card ordinals `0,...,D-1`, where
  ordinal `i` denotes global card `52-D+i`.  Document this mapping explicitly.
- Reject inactive cards, duplicate cards, wrong compact widths, and combo IDs
  outside `[0,H)` as hard input errors.

## Operation changes

Add `rank_count` to:

- `complete`
- `sample_joint_profiles`
- `partition_many`
- `partition_folders`
- `evaluate_showdowns`

For all operations, replace every combo axis of length 1,326 with `H`.
For partition and folder summaries, replace every card axis of length 52 with
`D`; pair/covariance fields become `[D,D]`.  Restrict `beta_rank` to `[0,D]`.
All collision tests, proposal draws, completion runouts, and board validation
must use only the active deck.  Existing dtype, device, contiguity, mask,
seeding, and error behavior otherwise remains unchanged.

Every result's metadata must contain:

- `api_version="pbg-v3"`
- `rank_count`
- `deck_size=D`
- `combo_count=H`
- `category_order_version="cumulative-five-card-v1"`
- existing backend/package/git version fields

## Hand-category ordering

Use cumulative five-card containment probabilities.  For
`T=C(4R,5)` and `q=max(0,R-4)+1[R=13]`, the exact numerators are:

| Category | Numerator |
|---|---:|
| high card | `T` |
| pair | `T - 4^5 C(R,5)` |
| two pair | `144(R-2)C(R,2) + 24R(R-1)` |
| trips | `64R C(R-1,2) + 28R(R-1)` |
| straight | `4^5 q` |
| flush | `4C(R,5)` |
| full house | `24R(R-1)` |
| quads | `4R(R-1)` |
| straight flush | `4q` |

Here a full house contains trips, two pair, and pair; a straight flush contains
straight and flush; quads contains trips and pair but not two pair; and every
holding contains high card.  Rank feasible categories by increasing numerator.
On exact ties retain conventional poker strength.  Impossible categories need
no relative order because they cannot be produced by the active deck.

Required strongest-to-weakest feasible order:

- `R=4`: quads, full house, trips, two pair, pair, high card.
- `R=5`: straight flush, flush, quads, full house, straight, trips, two pair,
  pair, high card.
- `R=6,7`: straight flush, flush, quads, full house, straight, trips, two pair,
  pair, high card.
- `R=8,...,11`: straight flush, quads, flush, full house, straight, trips, two
  pair, pair, high card.
- `R=12,13`: straight flush, quads, full house, flush, straight, trips, two
  pair, pair, high card.

Apply the selected category code remapping identically on CPU and CUDA after
recognizing the usual best-five category.  Preserve all existing within-category
kicker comparisons.  The active global IDs mean the existing 13-rank pattern
recognizer can remain in use; absent lower ranks simply cannot occur.

Important degeneracies and tie-breaks:

- `R=4`: straight, flush, and straight flush counts are zero; pair and high
  card both have probability one.
- `R=5`: flush and straight flush both have numerator four; every flush is a
  straight flush.
- `R=8`: flush and quads both have numerator 224; conventional order keeps
  quads stronger.
- `R=12`: flush and full house both have numerator 3,168; conventional order
  keeps full house stronger.

## Required sibling tests and acceptance gates

- Verify the formulas and category order for every `R=4,...,13`.
- Verify `combo_cards(R)` bijections, active-card mapping, validation failures,
  collision-free joint profiles, and seeded repeatability for every R.
- Verify exact joint-profile sampling against small independently enumerated
  distributions, especially `(P=2,R=4)` and `(P=6,R=5)`.
- Add CPU/CUDA showdown parity fixtures that cross each changed boundary:
  flush versus quads for `R=6,7,8,9`, and flush versus full house for
  `R=11,12,13`.
- Exercise completion and all partition/folder fields at `R=4,5,13`, including
  zero-folder and maximum-`beta_rank` cases.
- For `R=13`, require the `pbg-v3` combo table, accepted samples, completion
  statistics (within existing stochastic tolerances), showdown masks/shares,
  and partition fields to match `pbg-v2` for identical inputs and seeds.
- Update the sibling's authoritative `docs/api_pbg.md` only when implementation
  and CPU/CUDA tests pass.

## Consumer handoff

Publish/install the sibling version containing `pbg-v3` into
`/home/pablob/.venvs/poker`.  PBG will then replace its temporary version gate
with full compact evaluator integration and run the three corner configurations
`(P,S,R)=(2,8,4),(6,8,5),(6,128,13)` on CPU and an idle GPU.
