"""Representation-independent structural diagnostics for queued consumers."""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Mapping

from pbg.queued.contracts import PartitionContract
from pbg.queued.topology import Topology, TopologyBindings


def topology_diagnostics(
    topology: Topology,
    bindings: TopologyBindings,
    partitions: PartitionContract,
) -> Mapping[str, object]:
    """Summarize family/street decisions and per-root retained lengths."""

    trained = topology.owner.ge(0)
    family_counts: dict[str, int] = {}
    for partition_id, partition in enumerate(partitions.inputs):
        family_counts[partition.name] = int(
            (trained & bindings.input_type.eq(partition_id)).sum()
        )
    street_counts = {
        "preflop": family_counts.get("pre", 0),
        "postflop": sum(
            count
            for name, count in family_counts.items()
            if name.startswith("post_")
        ),
    }
    total = sum(street_counts.values())
    street_fractions = {
        name: (count / total if total else 0.0)
        for name, count in street_counts.items()
    }
    parents = topology.parent.tolist()
    roots: list[int] = []
    root_for: list[int] = []
    for node, parent in enumerate(parents):
        if parent == -1:
            roots.append(node)
            root_for.append(node)
        else:
            root_for.append(root_for[parent])
    root_index = {root: index for index, root in enumerate(roots)}
    lengths = [0] * len(roots)
    for node, keep in enumerate(trained.tolist()):
        if keep:
            lengths[root_index[root_for[node]]] += 1
    ordered = sorted(lengths)
    p95 = (
        0
        if not ordered
        else ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
    )
    return MappingProxyType(
        {
            "decision_counts_by_family": family_counts,
            "decision_counts_by_street": street_counts,
            "decision_fractions_by_street": street_fractions,
            "trajectory_length": {
                "count": len(lengths),
                "minimum": min(lengths, default=0),
                "maximum": max(lengths, default=0),
                "mean": sum(lengths) / len(lengths) if lengths else 0.0,
                "p95": p95,
            },
        }
    )


__all__ = ["topology_diagnostics"]
