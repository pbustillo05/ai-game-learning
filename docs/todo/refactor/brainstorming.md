# Refactor brainstorming

Status: resolved; decisions are merged into the refactor proposals.

1. Storage uses fixed-capacity aligned SoA arenas. Family `model_input` prefixes are direct model batches.
2. Rows are append-only within an iteration. `[0,head)` remains live until `align-and-seal`.
3. The mode matrix is:

   ```text
   training_regime   self_play | best_response
   rollout_mode      sequential | branched.fixed | branched.KKT
   private_estimator sampled | rao_blackwell
   ```

4. One `model + sample` batch runs at once; its commit may overlap the next batch. Counts, scans, and reservations
   stay on-device.
5. Caps are fixed hyperparameters. Whole-attempt overflow reduces sequential root count or branched beta, records
   diagnostics, and retries with a disjoint RNG address.
6. Sampled branching intentionally uses one joint profile per tree. Branched RB reuses child zero and samples
   `N - 1` independent posterior profiles.
7. Action-conditioned sampler variants are prototypes. Select one by complete `sample -> commit` time and delete the
   rest. Poker needs at most two distinct action-conditioned actor ranges per frontier.
8. Topology packets are staged by transaction and become visible only after commit.
9. Target creation follows [`layered_target_creation.md`](layered_target_creation.md): one asynchronous CUDA kernel
   per layer, initially one CTA per parent or carry row.
10. Sampled target work keeps lambda-return and trace-value carriers separate.
11. A compressed edge contributes

    $$\mathtt{path\_length}
      +\mathtt{cost\_multiplier}\,C_{\rm child}.$$

12. Frozen best-response opponents own no learner heads or targets. Unary rows compress; branched rows remain
    headless structural reductions.
13. Learner indices are final when created. Self-play rows align directly; best response compacts only trained
    inputs.
14. Acceptance targets the audited routing, reservation, synchronization, sealing, and target-propagation costs
    while preventing sequential regression.
15. Experiments may select an implementation. Production retains one path and deletes every loser, adapter, legacy
    path, stale option, and compatibility surface.
