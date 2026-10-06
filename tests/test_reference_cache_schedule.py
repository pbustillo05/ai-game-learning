"""Cache admission counts scheduled row reuse without advancing sampler RNG."""

from types import SimpleNamespace

import pytest
import torch

from pbg.core.iteration import IterationEngine
from pbg.core.learner import LearnerSchedule, ModelEvaluation, SequentialBatchSampler


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("batch_size,epochs,steps,counts,admitted", (
    (256, None, 116, None, True),
    (256, 1, None, None, False),
    (256, 2, None, None, True),
    (64, None, 1, None, False),
    (256, None, 116, {"pre": 220, "post_2": 36}, True),
))
def test_cache_admission_uses_entire_schedule_and_preserves_rng(
    batch_size, epochs, steps, counts, admitted,
):
    if not torch.cuda.is_bf16_supported():
        pytest.skip("native BF16 unavailable")
    device = torch.device("cuda", torch.cuda.current_device())
    weight = torch.arange(6, device=device, dtype=torch.float32).reshape(3, 2) / 7
    calls = []

    def reference(family, inputs, private, owner):
        calls.append((family, inputs.shape[0]))
        raw = inputs @ weight
        return ModelEvaluation({"raw": raw}, raw[:, 0])

    stores = {}
    for family, rows in (("pre", 118), ("post_2", 51)):
        columns = {"model_input": torch.arange(rows * 3, device=device,
                                               dtype=torch.float32).reshape(rows, 3) / 19,
                   "sampled_private_state": torch.zeros(rows, 2, device=device),
                   "owner": torch.zeros(rows, dtype=torch.int64, device=device)}
        stores[family] = SimpleNamespace(rows=rows, device=device, lane=0,
                                        sealed_inputs=SimpleNamespace(columns=columns))
    schedule = LearnerSchedule(batch_size=batch_size, epochs=epochs, gradient_steps=steps)
    generator = torch.Generator(device=device).manual_seed(774)
    sampler = SequentialBatchSampler(stores, schedule, generator=generator,
                                    statistical_counts=counts)
    rng_before = generator.get_state()
    engine = IterationEngine.__new__(IterationEngine)
    engine.config = SimpleNamespace(cache_reference_outputs=True,
        lanes=[SimpleNamespace(reference_operation=reference)], precision="bf16",
        learner_family_rows={"pre": 32, "post_2": 8})
    owners = SimpleNamespace(plan=SimpleNamespace(
        layout=SimpleNamespace(reference_cache_bytes=1 << 30),
        manifest=SimpleNamespace(learner_sigma_pack_cap=32)))
    caches = engine._reference_caches(owners, 0, stores, sampler=sampler)
    torch.testing.assert_close(generator.get_state(), rng_before, rtol=0, atol=0)
    assert (caches is not None) is admitted
    if not admitted:
        # A known unprofitable schedule must not launch a wasted model forward.
        assert calls == []
        return
    assert sum(rows for _, rows in calls) == 169
    assert len(calls) == 11  # Exact physical caps: ceil(118/32) + ceil(51/8).
    for family, cache in caches.items():
        indices = torch.tensor([0, stores[family].rows - 1, 0], device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected = stores[family].sealed_inputs.columns["model_input"][indices] @ weight
        torch.testing.assert_close(cache.gather(indices)["raw"], expected)
        cache.release()
