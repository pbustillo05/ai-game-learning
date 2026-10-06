from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.hyperparameter_search import evaluation


def _snapshot(
    path: Path,
    value: float,
    *,
    legacy: bool = True,
    initialization_anchor: Path | None = None,
) -> None:
    payload = {
        "model": {"weight": torch.tensor([value])},
        "scientific_config": {
            "game": {"players": 2, "stack_bb": 32.0, "rank_count": 13},
            "model": {"width": 8},
        },
        "iteration": 7,
        "source_checkpoint_sha256": f"native-{value}",
        "legacy_preflop_inputs": legacy,
        "input_encoding": "july_exact_hand_preflop_v1",
        "utility": "stack_normalized",
    }
    if initialization_anchor is not None:
        payload["initialization_anchor"] = str(initialization_anchor)
    torch.save(payload, path)


def test_campaign_snapshot_requires_explicit_legacy_input_contract(
    tmp_path: Path,
) -> None:
    accepted = tmp_path / "accepted.pt"
    rejected = tmp_path / "rejected.pt"
    _snapshot(accepted, 1.0)
    _snapshot(rejected, 1.0, legacy=False)

    source = evaluation.load_campaign_snapshot(accepted)
    assert source["iteration"] == 7
    assert source["metadata"]["legacy_preflop_inputs"] is True
    assert source["checkpoint_metadata"] == {
        "source_checkpoint_sha256": "native-1.0",
        "checkpoint": None,
        "utility": "stack_normalized",
        "input_encoding": "july_exact_hand_preflop_v1",
    }
    assert len(source["file_sha256"]) == 64
    assert len(source["model_sha256"]) == 64

    with pytest.raises(RuntimeError, match="legacy_preflop_inputs=true"):
        evaluation.load_campaign_snapshot(rejected)


class _FakeSession:
    def __init__(self, source: dict, *, device: torch.device) -> None:
        self.tag = float(source["model"]["weight"].item())
        self.device = device


def _fake_play_one(
    probe: _FakeSession,
    _target: _FakeSession,
    *,
    deal: object,
    button: int,
    probe_seat: int,
    mixed: bool,
) -> float:
    assert mixed is True
    assert button == deal.logical_deal_id % 2  # type: ignore[attr-defined]
    assert probe_seat == (deal.logical_deal_id // 2) % 2  # type: ignore[attr-defined]
    return probe.tag + float(deal.logical_deal_id)  # type: ignore[attr-defined]


def _install_fakes(monkeypatch: pytest.MonkeyPatch) -> list[_FakeSession]:
    adapted: list[_FakeSession] = []
    monkeypatch.setattr(
        evaluation,
        "resolve_execution_device",
        lambda _device, *, workload: torch.device("cuda:0"),
    )
    monkeypatch.setattr(evaluation, "PokerInferenceSession", _FakeSession)
    monkeypatch.setattr(evaluation, "_play_one", _fake_play_one)
    monkeypatch.setattr(
        evaluation,
        "install_legacy_preflop_inputs",
        lambda session: adapted.append(session),
    )
    return adapted


def test_evaluation_resumes_identical_deals_and_installs_all_adapters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate.pt"
    opponent = tmp_path / "opponent.pt"
    baseline = tmp_path / "baseline.pt"
    output = tmp_path / "evaluation.json"
    _snapshot(candidate, 3.0)
    _snapshot(opponent, 2.0)
    _snapshot(baseline, 1.0)
    adapted = _install_fakes(monkeypatch)

    partial = evaluation.evaluate_fixed_opponent(
        candidate,
        opponent_path=opponent,
        baseline_path=baseline,
        hands=4,
        seed=19,
        split="validation-a",
        output_path=output,
        max_new_hands=2,
    )
    assert partial["status"] == "incomplete"
    assert partial["completed_hands"] == 2
    assert len(adapted) == 4
    with pytest.raises(RuntimeError, match="resume invariant differs for seed"):
        evaluation.evaluate_fixed_opponent(
            candidate,
            opponent_path=opponent,
            baseline_path=baseline,
            hands=4,
            seed=20,
            split="validation-a",
            output_path=output,
        )

    complete = evaluation.evaluate_fixed_opponent(
        candidate,
        opponent_path=opponent,
        baseline_path=baseline,
        hands=4,
        seed=19,
        split="validation-a",
        output_path=output,
    )
    assert complete["status"] == "complete"
    assert complete["completed_hands"] == 4
    assert len(adapted) == 8
    assert complete["mean_candidate_minus_baseline_bb_per_hand"] == 2.0
    assert complete["confidence_interval_candidate_minus_baseline_bb_per_hand"] == (
        2.0,
        2.0,
    )
    assert complete["paired_candidate_minus_baseline"]["independent_cluster_count"] == 4
    assert complete["paired_candidate_minus_baseline"]["sampling_unit"] == (
        "logical_deal_cluster_seat_stratified"
    )
    assert complete["candidate_path"] == str(candidate.resolve())
    assert complete["opponent_path"] == str(opponent.resolve())
    assert complete["baseline_path"] == str(baseline.resolve())
    assert [row["logical_deal_id"] for row in complete["observations"]] == list(
        range(4)
    )
    assert [row["probe_seat"] for row in complete["observations"]] == [0, 0, 1, 1]
    assert [row["button"] for row in complete["observations"]] == [0, 1, 0, 1]
    assert json.loads(output.read_text())["status"] == "complete"

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        evaluation.evaluate_fixed_opponent(
            candidate,
            opponent_path=opponent,
            baseline_path=baseline,
            hands=4,
            seed=19,
            split="validation-a",
            output_path=output,
        )


def test_four_hand_identity_smoke_has_exact_direct_output_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = tmp_path / "policy.pt"
    opponent = tmp_path / "opponent.pt"
    output = tmp_path / "identity.json"
    _snapshot(policy, 4.0, initialization_anchor=policy)
    _snapshot(opponent, 2.0)
    _install_fakes(monkeypatch)

    complete = evaluation.evaluate_fixed_opponent(
        policy,
        opponent_path=opponent,
        hands=4,
        seed=23,
        split="identity-smoke",
        output_path=output,
    )
    assert complete["direct_output_parity"] == {
        "same_model": True,
        "all_payoffs_identical": True,
        "hands_checked": 4,
    }
    assert complete["mean_candidate_minus_baseline_bb_per_hand"] == 0.0


def test_target_as_baseline_reports_achieved_attack_gain_without_exploitability_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = tmp_path / "response.pt"
    target = tmp_path / "target.pt"
    output = tmp_path / "response-evaluation.json"
    _snapshot(response, 5.0)
    _snapshot(target, 2.0)
    _install_fakes(monkeypatch)

    complete = evaluation.evaluate_fixed_opponent(
        response,
        opponent_path=target,
        baseline_path=target,
        hands=4,
        seed=31,
        split="response-screen",
        output_path=output,
    )
    assert complete["mean_attack_gain_bb_per_hand"] == 3.0
    assert complete["confidence_interval_attack_gain_bb_per_hand"] == (3.0, 3.0)
    assert "not exploitability" in complete["achieved_response_interpretation"]
    assert "exploitability" not in complete


def test_validation_happens_before_cuda_session_construction() -> None:
    with pytest.raises(ValueError, match="multiple of 4"):
        evaluation.evaluate_fixed_opponent(
            "candidate.pt",
            opponent_path="opponent.pt",
            baseline_path="baseline.pt",
            hands=6,
            seed=1,
            split="invalid",
            output_path="output.json",
        )


def test_raw_match_skips_baseline_sessions_and_reports_raw_payoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate.pt"
    opponent = tmp_path / "opponent.pt"
    output = tmp_path / "raw-match.json"
    _snapshot(candidate, 3.0)
    _snapshot(opponent, 2.0)
    adapted = _install_fakes(monkeypatch)

    partial = evaluation.evaluate_fixed_opponent(
        candidate,
        opponent_path=opponent,
        hands=4,
        seed=43,
        split="league-fresh",
        output_path=output,
        max_new_hands=2,
        raw_match=True,
    )
    assert partial["status"] == "incomplete"
    assert len(adapted) == 2

    complete = evaluation.evaluate_fixed_opponent(
        candidate,
        opponent_path=opponent,
        hands=4,
        seed=43,
        split="league-fresh",
        output_path=output,
        raw_match=True,
    )
    assert complete["status"] == "complete"
    assert len(adapted) == 4
    assert complete["raw_match"] is True
    assert complete["baseline_evaluated"] is False
    assert complete["baseline_path"] is None
    assert complete["baseline_sha256"] is None
    assert complete["mean_candidate_bb_per_hand"] == pytest.approx(4.5)
    assert complete["candidate_payoff_vs_fixed_opponent"][
        "mean_bb_per_hand"
    ] == pytest.approx(4.5)
    assert all("baseline_payoff_bb" not in row for row in complete["observations"])
    assert all(
        "candidate_minus_baseline_bb" not in row for row in complete["observations"]
    )

    with pytest.raises(ValueError, match="does not accept a baseline"):
        evaluation.evaluate_fixed_opponent(
            candidate,
            opponent_path=opponent,
            baseline_path=opponent,
            hands=4,
            seed=43,
            split="league-fresh",
            output_path=tmp_path / "invalid.json",
            raw_match=True,
        )
