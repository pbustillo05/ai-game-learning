# AGENTS.md

## Repository goal

This repository implements and tests a GPU-oriented framework for learning
strategies in imperfect-information games, with synthetic games and poker as
the main domains.

## Where to find things

- `CONTRACTS.md`, `NOTATION.md`, and `docs/theory/core_math.tex`: authoritative
  contracts, notation, and mathematics.
- `archive/prompts/`: historical implementation and correction prompts.
- `pbg/core/`: generic rollout, branching, value, loss, and training code.
- `pbg/games/`: synthetic-game and poker-specific implementations.
- `tests/`: behavior and contract tests.
- `archive/POKER_GPU_EVALUATOR_IO.md`: binding interface to the sibling poker evaluator.
- `IMPLEMENTATION_NOTES.md`: historical implementation ledger; use
  `docs/audit/` for current readiness and open issues.

## Shared server hygiene

This repo is often used on a shared GPU server, and we must used these GPUs for testing, before doing any serious runs.
Do use the GPUs but be conservative:
- Use a virtualenv for Python work. Do not install packages into the server's system Python. Prefer one persistent user-level venv shared by the poker repos (there should be one available at `/home/pablob/.venvs/poker` and if unavailable, ask the user to create it), so dependencies are not reinstalled every session.
- Check GPU state before CUDA work with `nvidia-smi`. If sandboxed access fails, request explicit approval rather than guessing.
- Do not touch other users' GPU jobs. After checking GPU state, make sure to only launch tasks in idle available GPUs.
- After a GPU task is done, make sure the GPU has returned to a clean idle state, without any strange lingering workload or memory usage.
- Avoid OOM at all cost. Stage CUDA tests from tiny smoke runs to larger benchmarks, watch memory between runs, precompute estimated memory usage before anything sizable, and keep batch sizes/partials modest until usage is known. An OOM can require someone to physically reset the server, which is remote and difficult to access.
- Prefer commands that make resource targeting explicit.

You are by default approved and encouraged by the user to use the GPUs, as long you respect these rules. Escalate approval requests if the sandbox prevents you from using them.
