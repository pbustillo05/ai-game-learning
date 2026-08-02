from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from pbg.config import load_config_mapping
from pbg.games.poker.queued.hooks import QueuedPokerError, SequentialPokerHooks
from pbg.h100_pilot import (
    Arm,
    pilot_config_mapping,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize(
    "device_type",
    (
        "cpu",
        pytest.param("cuda", marks=CUDA),
    ),
)
def test_route_casts_only_model_input_at_bf16_storage_boundary(
    device_type: str,
) -> None:
    device = torch.device(
        "cpu" if device_type == "cpu" else f"cuda:{torch.cuda.current_device()}"
    )
    destination = {
        "model_input": torch.zeros((2, 2, 3), dtype=torch.bfloat16, device=device)
    }
    source = {
        "model_input": torch.tensor(
            [[[1.25, 2.5, 3.75], [4.0, 5.0, 6.0]]],
            dtype=torch.float32,
            device=device,
        )
    }

    SequentialPokerHooks._gather_route(
        destination,
        source,
        torch.tensor([[1, 0]], dtype=torch.int64, device=device),
        torch.tensor([1], dtype=torch.int64, device=device),
    )

    assert destination["model_input"].dtype == torch.bfloat16
    torch.testing.assert_close(
        destination["model_input"][1].float(),
        source["model_input"][0, [1, 0]],
    )
    assert torch.count_nonzero(destination["model_input"][0]) == 0


@CUDA
@pytest.mark.parametrize(
    ("name", "source_dtype", "destination_dtype"),
    (
        ("owner", torch.int64, torch.int32),
        ("owner", torch.float32, torch.bfloat16),
        ("model_input", torch.int64, torch.int32),
    ),
)
def test_route_keeps_every_other_dtype_mismatch_fatal(
    name: str,
    source_dtype: torch.dtype,
    destination_dtype: torch.dtype,
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    destination = {name: torch.zeros((1, 1), dtype=destination_dtype, device=device)}
    source = {name: torch.ones((1, 1), dtype=source_dtype, device=device)}

    with pytest.raises(QueuedPokerError, match="has dtype"):
        SequentialPokerHooks._gather_route(
            destination,
            source,
            torch.zeros((1, 1), dtype=torch.int64, device=device),
            torch.zeros(1, dtype=torch.int64, device=device),
        )


@pytest.mark.parametrize("mode", ("normal", "rb"))
def test_512_root_configs_compile_all_capacity_from_workload(mode: str) -> None:
    arm = Arm(
        f"nlhe_hu_32bb_{mode}_h100_roots512",
        "geometry",
        mode,
        1,
        512,
        None if mode == "normal" else 256,
    )

    mapping = pilot_config_mapping(arm, output_dir="/tmp/h100-roots512")

    assert "manifest" not in mapping["run"]
    config = load_config_mapping(mapping)
    assert config.execution.manifest.family_pack_cap == 256
    assert config.execution.manifest.partition_rows("folded_range_cap") == {}


@pytest.mark.parametrize("mode", ("normal", "rb"))
def test_checked_in_512_root_k1_config_has_no_authored_capacity(
    mode: str,
) -> None:
    path = (
        Path("configs/experiments/nlhe_hu_32bb_h100_pilot")
        / f"nlhe_hu_32bb_{mode}_h100_replica_k1.yaml"
    )
    mapping = yaml.safe_load(path.read_text(encoding="utf-8"))

    config = load_config_mapping(mapping)

    assert "manifest" not in mapping["run"]
    assert "layout" not in mapping["run"]
    assert mapping["run"]["execution"]["reference_cache"] == "auto"
    assert mapping["run"]["execution"]["microbatch"] == {
        "rollout_rows": 256,
        "completion_rows": 512,
        "learner_rows": 256,
    }
    assert config.scientific.common.root_count == 512
    assert config.execution.manifest.partition_rows("folded_range_cap") == {}
