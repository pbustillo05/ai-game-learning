# Public-action Bayesian game learning

`pbg` is a GPU-oriented research framework for learning strategies in
imperfect-information games. It implements policy-gradient self-play for finite
public-action Bayesian games, with exact synthetic card games for validation
and no-limit Texas hold'em as the main large-scale domain.

The implementation plan in `Prompts/` is complete. Those files remain as
development history; this README describes the repository as a normal,
implemented project.

## What is implemented

- Strict configuration, explicit RNGs, deterministic checkpointing, AMP, and
  gradient-only DDP.
- Device-resident sequential and fixed-intensity public-tree rollout.
- Duplicate-preserving labeled branching with randomized-rounded offspring.
- Monte Carlo leaf completion and posterior-moment backward propagation.
- Rao–Blackwellized PPO over all actor private types.
- Configurable synthetic games, Kuhn and Leduc presets, exact posteriors, best
  responses, and exploitability.
- A public NLHE engine with correct betting, short-all-in reopening, chance
  transitions, side pots, and affine chip-payoff assembly.
- Preflop 169-class and compact postflop 1326-combo transformer models.
- CPU/CUDA tensor-native poker rollout and paired duplicate-deal evaluation.
- A strict adapter boundary to `poker-gpu-evaluator` for all private poker
  combinatorics and showdown evaluation.

## Architecture

The generic package under `pbg/core/` operates on opaque state handles and
game-neutral tensor protocols. A game factory returns one typed `GameBundle`
containing its model, rollout backend, policy adapter, and completion backend.
Synthetic and poker implementations live under `pbg/games/`; evaluation is
read-only under `pbg/eval/`.

The active training pipeline is:

```text
frozen model -> sampled public trees -> leaf completion
             -> posterior P/M backward pass -> aligned decision minibatches
             -> posterior-weighted clipped policy + value + entropy update
```

Poker ownership is intentionally split. This repository owns public game
navigation, policies, ranges, pots, and chip-EV assembly.
`poker_gpu_evaluator.pbg` owns collision-conditioned private profiles,
folded-card partition summaries, completion/equity, hand comparison, ties, and
per-pot winner shares. See
[`POKER_GPU_EVALUATOR_IO.md`](POKER_GPU_EVALUATOR_IO.md).

## Requirements and installation

- Python 3.11+
- PyTorch 2.2+
- NumPy, PyYAML, and pytest
- The sibling `poker-gpu-evaluator` repository for poker completion and
  integration paths

On the shared GPU server, use the persistent poker virtual environment and do
not install into system Python:

```bash
/home/pablob/.venvs/poker/bin/python -m pip install -e .
/home/pablob/.venvs/poker/bin/python -m pip install -e ../poker-gpu-evaluator
```

The sibling install is optional for synthetic-only work and required for real
poker evaluator calls.

## Configuration

Complete presets are provided in `configs/`:

- `kuhn_smoke.yaml` — small sequential Kuhn run.
- `leduc.yaml` — fixed-intensity Leduc branching.
- `synthetic_dial_example.yaml` — scalable three-player synthetic example.
- `nlhe6_h100.yaml` — conservative first-profile six-player CUDA NLHE run.

Configuration loading rejects unknown or missing fields. Game-owned mappings
are validated when their bundle is constructed. See
[`docs/training_and_evaluation.md`](docs/training_and_evaluation.md) for the
schema and preset details.

## Training

There is no console-script wrapper yet. Run a preset through the Python API:

```bash
/home/pablob/.venvs/poker/bin/python - <<'PY'
from pbg.core.config import load_config
from pbg.core.training import train

train(load_config("configs/kuhn_smoke.yaml"))
PY
```

Training writes append-only `metrics.jsonl` rows and atomic checkpoints to the
configured output directory. Checkpoints contain model, optimizer, gradient
scaler, configuration, iteration, and explicit Torch/NumPy RNG state.

For DDP, launch the same code with `torchrun`. Tree and minibatch sizes are per
rank. Before CUDA work on a shared server, inspect `nvidia-smi`, use only an
idle explicitly selected GPU, stage small smoke runs first, and verify the GPU
returns to idle afterward.

## Evaluation

`pbg.eval.exploitability` performs exact read-only evaluation of feasible
synthetic checkpoints and writes iteration-sorted exploitability curves.

`pbg.eval.arena` compares poker checkpoints using paired duplicate deals,
complement runouts, seat rotation, and paired Student-t confidence intervals.
Every showdown crosses the evaluator facade.

## Tests

Run the full suite from the repository root:

```bash
/home/pablob/.venvs/poker/bin/python -m pytest -q
```

The tests cover strict contracts, exact synthetic oracles, sampled terminal
laws, posterior targets, loss weighting, public poker rules, model behavior,
CPU/tensor parity, evaluator boundaries, device residency, DDP, checkpointing,
and short integration runs. CUDA- and sibling-dependent tests skip when their
prerequisites are unavailable.

## Documentation

The implementation documentation starts at [`docs/README.md`](docs/README.md):

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/core_algorithms.md`](docs/core_algorithms.md)
- [`docs/synthetic_games.md`](docs/synthetic_games.md)
- [`docs/poker.md`](docs/poker.md)
- [`docs/training_and_evaluation.md`](docs/training_and_evaluation.md)
- [`docs/file_reference.md`](docs/file_reference.md), covering every tracked
  file

Authoritative technical sources are
[`CONTRACTS.md`](CONTRACTS.md), [`NOTATION.md`](NOTATION.md), and
[`docs/latex/core_math.tex`](docs/latex/core_math.tex). Historical implementation
status and decisions are recorded in
[`IMPLEMENTATION_NOTES.md`](IMPLEMENTATION_NOTES.md).

## Project status

All implementation and correction prompts through P13B are implemented. The
current default is the corrected fixed-intensity RB-PPO path. Learned Neyman
allocation, reference-policy KL, NeuRD/R-NaD objectives, and public moment
losses remain explicit research extensions rather than unfinished requirements.
