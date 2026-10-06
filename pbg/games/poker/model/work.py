"""Shape-derived major-matrix work estimates for the production shared model."""

from pbg.games.poker.model.capacity import max_policy_partition_sequence_tokens
from pbg.games.poker.queued.model import model_configuration_from_scientific, model_storage_bytes


def shared_model_work_ledger(scientific):
    geometry, config = model_configuration_from_scientific(scientific)
    tokens = max_policy_partition_sequence_tokens(n_players=geometry.players,
        stack_bb=geometry.stack_bb, rank_count=geometry.rank_count, k_act=config.k_act)
    d, hidden, layers = config.d_model, config.ffn_hidden or 4 * config.d_model, config.layers
    sampled = scientific["mode"]["private_estimator"] == "sampled"
    result = {}
    for family, sequence in tokens.items():
        private = geometry.private_classes if family == "pre" else geometry.private_hands
        players = geometry.players if family == "pre" else int(family.split("_")[1])
        private_tokens = players * private
        static = (config.class_embedding_dim + 4 + geometry.deck_cards if family == "pre" else
                  config.class_embedding_dim + 3 + geometry.deck_cards + 4 * config.folder_beta_rank + 2 * d)
        row_width = d + config.class_embedding_dim
        context_width = (config.k_act + 1) * d
        heads = (18 * hidden * (row_width + context_width + d) if sampled else
                 18 * hidden * (private * (row_width + d) + context_width))
        major = {
            "attention": 12 * layers * sequence**2 * d,
            "qkvo": 24 * layers * sequence * d**2,
            "ffn": 6 * layers * sequence * hidden * (4 * d + static + d),
            "incidence_projection": 30 * layers * private_tokens * d**2,
            "incidence_fixed_geometry": 8 * layers * private_tokens * geometry.deck_cards * d,
            "three_readout_ffns": heads,
        }
        result[family] = {"sequence_envelope": sequence, "actor_private_rows": private,
                          "active_private_tokens": private_tokens, "ffn_input_width": 4 * d + static,
                          "readout_width": row_width + context_width,
                          "forward_dx_dw_major_flops_per_presentation": major,
                          "major_flops_per_presentation": sum(major.values()),
                          "attention_score_recompute_flops_separate": 2 * layers * sequence**2 * d}
    parameters, buffers = model_storage_bytes(scientific)
    return {"schema": "pbg-shared-model-work-v1", "fma_flops": 2,
            "parameter_bytes_per_replica": parameters, "buffer_bytes_per_replica": buffers,
            "families": result,
            "interpretation": "Conservative full-width major-matrix envelope, not measured issued work or MFU.",
            "excluded_work": ["scalar policy/loss operations", "minor public/history/pooling projections",
                              "rollout sampling and state transitions", "reference no-grad calls",
                              "evaluator completion", "target propagation", "packing and copies"],
            "other_phases": "Use measured phase telemetry and actual logical work; do not infer pipeline time from learner FLOPs."}
