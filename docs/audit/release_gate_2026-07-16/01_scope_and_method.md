# Scope and method

## Scope

The audit covered:

- every project-authored Markdown or TeX document under `docs/`;
- all six root Markdown files: `AGENTS.md`, `CONTRACTS.md`, `HARDWARE.md`,
  `IMPLEMENTATION_NOTES.md`, `NOTATION.md`, and `README.md`;
- every Python module under `pbg/` (15,689 source lines);
- the configuration presets under `configs/` as executable documentation;
- the test suite under `tests/`;
- the read-only sibling evaluator contract at
  `/home/pablob/poker-gpu-evaluator/docs/api_pbg.md`.

The 138 files under `docs/references/` are vendored third-party papers, source
archives, figures, and bibliography assets. They were inventoried as provenance,
not treated as repository requirements. No claim is made that `pbg` implements
every algorithm discussed in those papers.

Historical content under `archive/` was consulted only where active documents
point to a frozen interface or explain provenance. It was not treated as active
code. The sibling evaluator repository was read only and was not modified.

## Authority order

Conflicts were resolved using this order:

1. `docs/theory/core_math.tex` for mathematical abstraction and estimator laws;
2. `docs/theory/MHA_model.md` for model architecture;
3. `docs/theory/folder_impact.md` and the sibling `pbg-v3` API for folded-card
   input semantics and ownership;
4. theory pseudocode for production scheduling and distributed updates;
5. `CONTRACTS.md`, `NOTATION.md`, and project documentation for the supported
   product surface;
6. `IMPLEMENTATION_NOTES.md` as historical evidence only.

`docs/theory/core_math_audit.md` already establishes that the authoritative math
note is still a research draft around named-method and convergence claims. This
audit does not promote those unproved claims. It evaluates the fixed-intensity
PPO/RB subset that the implementation says it ships.

## Method

The review used five passes:

1. **Inventory and reachability.** Enumerate documents, modules, configs, tests,
   composition roots, imports, registries, and archived boundaries.
2. **Requirement extraction.** Convert game geometry, tree laws, posterior
   recursions, model equations, evaluator ownership, loss definitions, memory,
   synchronization, checkpoint, and DDP text into traceable requirements.
3. **Static trace.** Follow each requirement through construction, rollout,
   completion, target generation, model forward, loss reduction, optimizer,
   checkpoint, benchmark, and evaluation code.
4. **Behavioral verification.** Run syntax, full unit/integration, real sibling,
   CUDA, single-GPU training, and two-rank training checks; explicitly probe
   advertised boundary conditions and optional loss settings.
5. **Adversarial release review.** Look for accepted-but-unimplemented config
   values, late validation, maximum legal geometry, nonfinite behavior, hidden
   host synchronization, stale documentation, and untrained optional heads.

## Severity

- **Release blocker:** contradicts a required production capability, can corrupt
  training state, or makes an advertised valid run fail.
- **Major:** materially violates an active contract or creates a clear hot-path
  production risk, but does not by itself make the fixed audited smoke invalid.
- **Moderate:** documentation, observability, maintainability, or validation gap
  that should be corrected in the minimal release hardening pass.
- **Minor:** cleanup or precision issue with no current correctness effect.

## Limits of the evidence

The successful live training matrix used heads-up, 8 bb, full-deck acceptance
workloads to stay conservative on shared RTX 2080 Ti hardware. Existing tests and
prior audit evidence cover six-player execution and boundary rule arithmetic,
but this audit did not attempt an exhaustive run of every `(P,S,R)` tuple. Such a
claim is currently impossible because the code rejects every `R != 13` tuple and
because legal long histories exceed the default model capacity.

Only GPUs 1 and 2 were used. GPU 0 had an existing 2.9 GiB process and was left
untouched. No large training or performance-tuning run was attempted after the
release blockers were established.
