# Poker inference on Apple Silicon

`PokerInferenceSession` is the model-only serving entry point. It loads a
training checkpoint or compact inference artifact without constructing rollout
storage, completion, an optimizer, or any CUDA subsystem. Public NLHE rules run
on CPU, folded-player statistics use the sibling evaluator's CPU implementation,
and all policy/value network operations run on Metal through PyTorch MPS.

## Installation and diagnostics

Use native arm64 Python on macOS 14 or newer and install the Mac dependency
set in a virtual environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[mac]'
.venv/bin/python -m pip install -e ../poker-gpu-evaluator
```

Verify the accelerator before loading a production checkpoint:

```bash
.venv/bin/python -c \
  'import torch; print(torch.__version__, torch.backends.mps.is_available())'
```

Do not set `PYTORCH_ENABLE_MPS_FALLBACK=1`. The inference loader rejects that
setting because it can silently execute unsupported model operators on CPU.
The explicit folded-statistics CPU call is unaffected.

## Export and load

Create a smaller model-only artifact on the training machine:

```bash
pbg-export-inference runs/example/ckpt_0000100.pt policy.pt
```

The `pbg-inference-v1` artifact contains CPU model weights, scientific game and
model configuration, iteration, schema metadata, and the source checkpoint
SHA-256. It omits optimizer, scaler, RNG, controller, and hardware state.

```python
from pbg.inference import PokerInferenceSession

session = PokerInferenceSession.from_checkpoint(
    "policy.pt",
    device="mps",       # "auto" selects MPS when available
    precision="auto",   # fp16 on MPS; use "fp32" for strict comparison
    mode="auto",        # singleton latency path, batched throughput path
)

hand = session.new_hand(button=0)
session.advance(hand, ("raise_to", 6.0))
session.advance(hand, "call")
session.deal(hand, ["Ah", "7d", "2c"])

decision = session.predict(hand, ["Ks", "Qs"], action_mode="argmax")
for action, probability in zip(
    decision.actions, decision.probabilities, strict=True
):
    print(action, float(probability))
print("value (bb):", decision.value_bb)
```

`advance` accepts `fold`, `check`, `call`, `check_call`, `all_in`, or an exact
half-big-blind `("raise_to", amount)` pair. Card strings use conventional ranks
and the suit letters `c`, `d`, `h`, and `s`. Compact-deck sessions reject cards
outside their configured highest-rank deck.

## Batches and full strategies

Use `new_hands`, `advance_many`, `deal_many`, and `predict_many` to amortize
input construction and Metal dispatch. Requests are grouped by street and live
player count, split under the MPS memory budget, and restored to caller order.

```python
hands = session.new_hands([0, 1, 0, 1])
results = session.predict_many(
    hands,
    [["As", "Kd"], ["Qh", "Qs"], ["Ac", "Jc"], ["Kh", "Kc"]],
)
```

Normal predictions gather the requested private-hand row on MPS before copying
the result to CPU. For analysis, `session.full_strategy(hands)` explicitly
returns CPU tensors containing `[B,H,E]` probabilities, `[B,H]` values, legal
masks, compatibility masks, and optional seat values.

`mode="latency"` executes singleton physical forwards. `mode="throughput"`
packs larger physical batches. `mode="auto"` selects the former for one spot
and the latter for larger requests. `max_batch_rows` can impose a stricter cap;
otherwise the session uses Metal recommended-memory and allocator telemetry and
keeps at least 25% or 2 GiB outside its inferred batch allowance.

Run warmed latency, throughput, and allocator measurements against an exported
artifact with:

```bash
pbg-inference-benchmark policy.pt --device mps --street flop \
  --batches 1 2 8 32 --warmup 2 --repeats 10
```
