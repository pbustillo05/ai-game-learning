1. preflop FamilyWork stores 1326-wide instead of 169-wide
2. no poker CUDA kernels, see "docs/audit/high_priority_cuda_kernels_20260728.md"
3. in fixed beta mode, we need a preflop and postflop beta
4. there seems to be a few redundant parameters like "trained_player_cap" (redundant in non-BR mode) which should ONLY be required in some modes
5. safe_tree_size = None should be allowed to remove that functionality
6. does completion happen AFTER all rollouts?