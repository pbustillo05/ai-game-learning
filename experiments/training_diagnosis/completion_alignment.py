"""Experiment-only application of the existing leaf-to-LayerWork movement plan.

Production completion appends each partition in ascending Leaf FIFO row order.
LayerPlan addresses that partition in topology traversal order. This adapter
copies the completed prefix into the destination slots already described by
LayerPlan before any target kernel runs. It changes no production source.
"""
from __future__ import annotations
import torch


def install_completion_alignment(session, audit=None):
    """Wrap one TrainingSession's completion phase; optional list records maps."""
    factory = session._iteration_engine

    def aligned_factory():
        engine = factory()
        original = engine._execute_completion

        def complete(owners, state):
            result = original(owners, state)
            plan = state.topology.plan
            for lane in range(owners.plan.manifest.n_replica):
                for output, name in enumerate(plan.output_names):
                    selected = (plan.movement_replica == lane) & (
                        plan.movement_output_type == output
                    )
                    leaf_ids = plan.movement_source_leaf_row[selected]
                    destination = plan.movement_destination_work_row[selected]
                    order = torch.argsort(leaf_ids, stable=True)
                    destination = destination[order].to(torch.int64)
                    count = destination.numel()
                    if not count:
                        continue
                    if sorted(destination.tolist()) != list(range(count)):
                        raise RuntimeError('leaf destinations must permute completion prefix')
                    slab = owners.slab('LayerWork', name)
                    device_destination = destination.to(device=owners.plan.device)
                    verified_columns = []
                    for column_name, column in slab.columns.items():
                        source = column[lane, :count].clone()
                        column[lane].index_copy_(0, device_destination, source)
                        if audit is not None:
                            verified = torch.equal(column[lane].index_select(0, device_destination), source)
                            if not verified:
                                raise RuntimeError(f'completion movement failed for {name}.{column_name}')
                            verified_columns.append(column_name)
                    if audit is not None:
                        audit.append({
                            'lane': lane, 'output': name, 'leaf_count': count,
                            'all_columns_verified': True, 'column_names': verified_columns,
                            'moved_rows': int((destination != torch.arange(count)).sum()),
                            'source_leaf_ids': leaf_ids[order].tolist(),
                            'destination_rows': destination.tolist(),
                        })
            return result

        engine._execute_completion = complete
        return engine

    session._iteration_engine = aligned_factory
