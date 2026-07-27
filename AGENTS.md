# AGENTS.md

## Repository goal

This repository implements and tests a GPU-oriented framework for learning
strategies in imperfect-information games, with configurable NLHE cash poker
as its sole active game.

## GPU-accelerator-only execution

- Production execution is intentionally GPU-accelerator-only. The only valid
  execution-device backends are `cuda` and `mps`; do not add or restore `cpu`,
  `auto`, implicit device selection, or fallback to CPU or another backend.
- CUDA supports every production workflow. Training, distributed training,
  best-response evaluation, performance benchmarks, variance runners, and
  inference must execute on CUDA when those workflows require operations not
  available on MPS. Distributed training is CUDA/NCCL-only.
- MPS is a valid backend, but is currently implemented only for the supported
  inference paths. MPS training and evaluator-dependent inference operations
  that contract v2 does not provide must fail clearly instead of staging work
  through CPU.
- CPU remains appropriate only for small orchestration work such as parsing
  configuration, checkpoint I/O, hashing, deterministic seed construction, and
  report formatting. It must not own production rollout state, model forwards,
  losses, backward passes, optimizer state, posterior/completion work, or
  showdown evaluation.
- All changes must be tested with CUDA, escalating approval outside the sandbox if needed.

## Where to find things

- `docs/theory/`: authoritative mathematics, you must ask for explicit approval to modify anything there:
  - `core_math.tex` is the mathematical abstraction, 
  - `MHA_model.md` is the model architecture
  - `pseudocode/` is demoted and is to be replaced by `docs/todo/refactor`
- `pbg/core/`: legacy implementation of generic rollout, branching, value, loss, and training code.
- `pbg/games/poker/`: NLHE-specific implementation.
- `tests/`: behavior and contract tests.
- `/home/pablob/poker-gpu-evaluator/docs/api.md` (read-only rights, you are not authorized to modify that file): binding interface to the sibling poker evaluator. We **have** to use that sibling repo for posterior sampling and Monte Carlo equity/posterior simulations.
- In Markdown, write math with `$...$` and `$$...$$` rather than `\(...\)` or `\[...\]` so it renders in VS Code.

## Actual Games
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

This repo is on a shared GPU server, and we must use these GPUs for testing, after any changes, escalading sandbox approval if needed.
Follow these rules:
- Use `/home/pablob/.venvs/poker` for Python work.
- The shared venv owns the CUDA toolkit; it is not installed under `/usr/local/cuda`.
  Before compiling or loading CUDA extensions, activate the venv and export its toolkit location:
  ```bash
  source /home/pablob/.venvs/poker/bin/activate
  export CUDA_HOME=/home/pablob/.venvs/poker/lib/python3.10/site-packages/nvidia/cu13
  export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
  ```
  In particular, `nvcc` is at
  `/home/pablob/.venvs/poker/lib/python3.10/site-packages/nvidia/cu13/bin/nvcc`
  and `ninja` is at `/home/pablob/.venvs/poker/bin/ninja`. Do not report either
  as missing until those exact locations have been checked with the environment
  above; invoking the venv's Python by absolute path does not itself add the
  venv's `bin` directory to `PATH`.
- Check GPU state before CUDA work with `nvidia-smi`.
- Do not touch other users' GPU jobs. After checking GPU state, make sure to only launch tasks in idle available GPUs.
- After a GPU task is done, make sure the GPU has returned to a clean idle state, without any strange lingering workload or memory usage.
- Prefer commands that make resource targeting explicit.

## RUNS folder hygiene
The `runs/` and `config/` folders can get out of hand fast. Follow these rules:
- name runs folders and `config/train/` files using: `nlhe_{hu or {P}max}_{{S}bb}_{normal or rb or branched_fixed or branched_KKT}_{optional short identifier}`
- delete failed runs, failed configs,...
- all experimental runs live in `runs/experiments/` and `configs/experiments/`, they don't clogg the root of the `runs/` and `config/` folders.
