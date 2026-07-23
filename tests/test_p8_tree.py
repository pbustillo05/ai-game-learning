from __future__ import annotations

import pytest
import torch

from pbg.core.game import TERMINAL
from pbg.core.execution import ByteAccountant
from pbg.core.tree import PagedRolloutTopology


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
def test_device_tree_depth_uses_the_same_inclusive_convention(device: str) -> None:
    buffer = PagedRolloutTopology(1, 2, device, page_rows=2)
    root = buffer.add_roots(
        torch.tensor([0], dtype=torch.int32, device=device),
        torch.tensor([0], dtype=torch.int32, device=device),
    )
    child = buffer.append_children(
        root,
        torch.tensor([1], dtype=torch.int32, device=device),
        torch.tensor([1], dtype=torch.int32, device=device),
        torch.tensor([10], dtype=torch.int32, device=device),
        torch.tensor([1.0], device=device),
    )
    buffer.append_children(
        child,
        torch.tensor([2], dtype=torch.int32, device=device),
        torch.tensor([TERMINAL], dtype=torch.int32, device=device),
        torch.tensor([11], dtype=torch.int32, device=device),
        torch.tensor([1.0], device=device),
    )

    assert torch.equal(
        buffer.depth_p[torch.arange(buffer.node_count, device=device)],
        torch.tensor(
            [[1, 0], [1, 1], [1, 1]], dtype=torch.int32, device=device
        ),
    )
    assert buffer.depth_player[torch.arange(buffer.node_count, device=device)].tolist() == [1, 2, 2]


def test_paged_edges_keep_stable_parent_local_ordinals_without_legacy_finalize() -> None:
    buffer = PagedRolloutTopology(2, 3, "cpu", page_rows=2)
    roots = buffer.add_roots(
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0, 0], dtype=torch.int32),
    )
    later_root_child = buffer.append_children(
        roots[1:2],
        torch.tensor([20], dtype=torch.int32),
        torch.tensor([TERMINAL], dtype=torch.int32),
        torch.tensor([200], dtype=torch.int32),
        torch.tensor([1.0]),
    )
    earlier_root_children = buffer.append_children(
        roots[0:1].expand(2),
        torch.tensor([10, 11], dtype=torch.int32),
        torch.tensor([TERMINAL, TERMINAL], dtype=torch.int32),
        torch.tensor([100, 101], dtype=torch.int32),
        torch.tensor([0.25, 0.75]),
    )

    edge_rows = torch.arange(buffer.edge_count)
    assert buffer.edge_parent[edge_rows].tolist() == [1, 0, 0]
    assert buffer.edge_ordinal[edge_rows].tolist() == [0, 0, 1]
    assert buffer.child_node[edge_rows].tolist() == [
        *later_root_child.tolist(), *earlier_root_children.tolist()
    ]
    assert buffer.child_event[edge_rows].tolist() == [200, 100, 101]
    assert torch.equal(buffer.edge_weight[edge_rows], torch.tensor([1.0, 0.25, 0.75]))


def test_child_addresses_can_be_reserved_before_physical_materialization() -> None:
    buffer = PagedRolloutTopology(1, 2, "cpu", page_rows=2)
    root = buffer.add_roots(
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
    )
    children, edges = buffer.reserve_children(root.expand(3), torch.full((3,), 1 / 3))

    assert children.tolist() == [1, 2, 3]
    assert edges.tolist() == [0, 1, 2]
    assert buffer.child_count[root.to(torch.int64)].tolist() == [3]
    assert buffer.state_ref[children.to(torch.int64)].tolist() == [-1, -1, -1]

    buffer.commit_children(
        1,
        0,
        torch.tensor([10, 11], dtype=torch.int32),
        torch.tensor([1, TERMINAL], dtype=torch.int32),
        torch.tensor([100, 101], dtype=torch.int32),
    )
    buffer.commit_children(
        3,
        2,
        torch.tensor([12], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([102], dtype=torch.int32),
    )

    assert buffer.state_ref[children.to(torch.int64)].tolist() == [10, 11, 12]
    assert buffer.child_event[edges.to(torch.int64)].tolist() == [100, 101, 102]
    assert buffer.depth_p[children.to(torch.int64)].tolist() == [
        [1, 1],
        [1, 0],
        [2, 0],
    ]


def test_child_wave_page_admission_rolls_back_every_owner_before_mutation() -> None:
    node_page = PagedRolloutTopology.node_schema(2).capacity_bytes(1, chunks=1)
    edge_page = PagedRolloutTopology.edge_schema().capacity_bytes(1, chunks=1)
    accountant = ByteAccountant(node_page + max(node_page, edge_page))
    buffer = PagedRolloutTopology(
        1, 2, "cpu", page_rows=1, accountant=accountant
    )
    root = buffer.add_roots(
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
    )
    before = (
        buffer.node_count,
        buffer.edge_count,
        int(buffer.child_count[root.to(torch.int64)][0]),
        accountant.reserved_bytes,
    )
    with pytest.raises(MemoryError):
        buffer.append_children(
            root,
            torch.tensor([1], dtype=torch.int32),
            torch.tensor([TERMINAL], dtype=torch.int32),
            torch.tensor([7], dtype=torch.int32),
            torch.tensor([1.0]),
        )
    assert (
        buffer.node_count,
        buffer.edge_count,
        int(buffer.child_count[root.to(torch.int64)][0]),
        accountant.reserved_bytes,
    ) == before
