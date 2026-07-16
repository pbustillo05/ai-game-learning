# AGENTS.md

## Repository goal

This repository implements and tests a GPU-oriented framework for learning
strategies in imperfect-information games, with configurable NLHE cash poker
as its sole active game.

## Where to find things

- `docs/theory/`: authoritative mathematics and ultimate source of design choices here.
  `core_math.tex` is the mathematical abstraction, `MHA_model.md` is the model architecture and `pseudocode/` contains multi-GPU training implementation design.
- `pbg/core/`: generic rollout, branching, value, loss, and training code.
- `pbg/games/poker/`: NLHE-specific implementation.
- `archive/synthetic_games/`: inert historical Kuhn/Leduc/synthetic snapshot;
  it is not importable and is excluded from CI.
- `tests/`: behavior and contract tests.
- `/home/pablob/poker-gpu-evaluator/docs/api_pbg.md` (read-only rights, you are not authorized to modify that file): binding interface to the sibling poker evaluator. We **have** to use that sibling repo for posterior sampling and Monte Carlo equity/posterior simulations.
- `IMPLEMENTATION_NOTES.md`: historical implementation ledger; use
  `docs/audit/` for current readiness and open issues.

## PBG tree terminology and rollout convention

- A PBG public game tree has exactly one distinguished root,
  $\varnothing$.
- In branched mode, each GPU samples exactly one labeled rollout tree rooted at
  $\varnothing$, not a forest. The root is treated by the same branching rule as every other
  non-pre-terminal node: $N(\varnothing)$ is obtained from the configured
  intensity by the ordinary randomized-rounding rule. Do not give
  $\varnothing$ special multiplicity or branch there more aggressively.
- In sequential mode (normal or Rao-Blackwell), this logic is reversed: only $\varnothing$ has branching.
## Actual Games
- Kuhn, Leduc, and synthetic games are retired and archived because they are
  too small for this setup.
- **NLHE cash poker** is the only active game. It has three per-run dials:
  players $P\in\{2,\ldots,6\}$, equal stack depth $S\in[8,128]$ bb in 0.5 bb
  increments, and the number of highest ranks $R\in\{4,\ldots,13\}$ with all
  four suits (for example, $R=4$ is AKQJ only).
- Require $2P+5<4R$ so every private hand and the complete board fit with at
  least one undealt card.
- We always use sb = 0.5 and bb = 1 for the blinds, with half-integer stacks. No rake, no ante.
- Each player has the exact same stack, making side pots impossible.
- For $P$ players and $S$bb, the theoretical maximal amount of players nodes in a trajectory is $4P + (S-1)(P-1) = SP+3P-S+1$.

## Shared server hygiene

This repo is often used on a shared GPU server, and we must used these GPUs for testing, before doing any serious runs.
Do use the GPUs but be conservative:
- Do NOT modify servers in the other repo (GPU-evaluator). You should only treat it as read only
- Use a virtualenv for Python work. Do not install packages into the server's system Python. Prefer one persistent user-level venv shared by the poker repos (there should be one available at `/home/pablob/.venvs/poker` and if unavailable, ask the user to create it), so dependencies are not reinstalled every session.
- Check GPU state before CUDA work with `nvidia-smi`. If sandboxed access fails, request explicit approval rather than guessing.
- Do not touch other users' GPU jobs. After checking GPU state, make sure to only launch tasks in idle available GPUs.
- After a GPU task is done, make sure the GPU has returned to a clean idle state, without any strange lingering workload or memory usage.
- Avoid OOM at all cost. Stage CUDA tests from tiny smoke runs to larger benchmarks, watch memory between runs, precompute estimated memory usage before anything sizable, and keep batch sizes/partials modest until usage is known. An OOM can require someone to physically reset the server, which is remote and difficult to access.
- Prefer commands that make resource targeting explicit.

You are by default approved and encouraged by the user to use the GPUs, as long you respect these rules. Escalate approval requests if the sandbox prevents you from using them.
