"""Bounded compilation/capture ownership for the same tensor accumulation program."""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor
import torch._functorch.config as functorch_config


@dataclass
class _Entry:
    operation: Callable
    static: tuple[Tensor, ...] = ()
    graph: torch.cuda.CUDAGraph | None = None
    outputs: tuple[Tensor, ...] = ()
    resident_bytes: int = 0
    addresses: tuple[int, ...] = ()


class TensorProgramRunner:
    """Bounded exact-shape compilation with optional CUDA graph ownership.

    The callable returns report tensors followed by a scalar objective. Backward
    runs outside its autocast context. Captures own their input/output storage;
    mutable gradients/buffers and CUDA RNG are restored after warmup/capture.
    A warmed program remains compiled if its measured graph pool exceeds the
    budget; other rare/unadmitted shapes execute the same eager function.
    """

    def __init__(self, *, mode: str = "eager", max_entries: int = 8,
                 byte_budget: int = 512 << 20, reserve_bytes: int = 0,
                 peak_workspace_bytes: int = 0) -> None:
        if mode not in ("eager", "compiled", "captured"):
            raise ValueError("tensor program mode must be eager, compiled, or captured")
        if max_entries <= 0 or byte_budget < 0:
            raise ValueError("tensor program cache limits are invalid")
        self.mode, self.max_entries, self.byte_budget = mode, max_entries, byte_budget
        self.reserve_bytes, self.peak_workspace_bytes = reserve_bytes, peak_workspace_bytes
        self.denied: OrderedDict[object, None] = OrderedDict()
        self.entries: dict[object, _Entry] = {}
        self.seen: OrderedDict[object, int] = OrderedDict()
        self.calls = self.replays = self.uncaptured = 0
        self.work_counts: dict[str, dict[str, int]] = {}
        self._capture_stream: torch.cuda.Stream | None = None

    @property
    def resident_bytes(self) -> int:
        return sum(entry.resident_bytes for entry in self.entries.values())

    def _deny(self, key) -> None:
        self.denied[key] = None
        while len(self.denied) > 4 * self.max_entries:
            self.denied.popitem(last=False)

    @staticmethod
    def _invoke(operation, inputs):
        with functorch_config.patch(backward_pass_autocast="off"):
            outputs = operation(*inputs)
            outputs[-1].backward()
        return tuple(value.detach() for value in outputs[:-1])

    def run(self, key, operation, inputs, *, mutable):
        inputs, mutable = tuple(inputs), tuple(mutable)
        if self.mode != "captured":
            return self._run(key, operation, inputs, mutable=mutable)
        if not inputs or any(value.device.type != "cuda" or value.device != inputs[0].device
                             for value in inputs + mutable):
            raise ValueError("CUDA capture inputs and mutable tensors must share a CUDA device")
        device = inputs[0].device
        if self._capture_stream is None:
            self._capture_stream = torch.cuda.Stream(device=device)
        if self._capture_stream.device != device:
            raise ValueError("a captured tensor program runner must retain its CUDA device")
        caller = torch.cuda.current_stream(device)
        stream = self._capture_stream
        stream.wait_stream(caller)
        try:
            # Autograd can retain AccumulateGrad nodes between compiled shapes.
            # Their stream is fixed when first created, so eager admission,
            # warmup and every capture must use the same non-default stream.
            with torch.cuda.stream(stream):
                for value in inputs + mutable:
                    value.record_stream(stream)
                return self._run(key, operation, inputs, mutable=mutable)
        finally:
            caller.wait_stream(stream)

    def _run(self, key, operation, inputs, *, mutable):
        self.calls += 1
        inputs, mutable = tuple(inputs), tuple(mutable)
        signature = tuple((tuple(value.shape), tuple(value.stride()), value.dtype, value.device)
                          for value in inputs)
        key = (key, signature)
        entry = self.entries.get(key)
        addresses = tuple(value.data_ptr() for value in mutable)
        if entry is not None:
            if entry.addresses != addresses:
                raise RuntimeError("tensor program lost its gradient/buffer addresses")
            if entry.graph is not None:
                for destination, source in zip(entry.static, inputs, strict=True):
                    destination.copy_(source)
                entry.graph.replay()
                self.replays += 1
                # Reports are consumed immediately; callers must not retain an
                # alias across the next replay of this exact graph.
                return entry.outputs
            self.uncaptured += 1
            return self._invoke(entry.operation, inputs)
        count = self.seen.pop(key, 0) + 1
        self.seen[key] = count
        while len(self.seen) > 4 * self.max_entries:
            self.seen.popitem(last=False)
        if self.mode == "eager" or count < 2 or key in self.denied or len(self.entries) >= self.max_entries:
            self.uncaptured += 1
            return self._invoke(operation, inputs)
        compiled = torch.compile(operation, fullgraph=True, dynamic=False,
                                 mode="max-autotune-no-cudagraphs")
        entry = _Entry(compiled, addresses=addresses)
        if self.mode == "compiled":
            self.entries[key] = entry
            self.uncaptured += 1
            return self._invoke(compiled, inputs)
        if not inputs or any(value.device.type != "cuda" for value in inputs):
            raise ValueError("CUDA capture inputs must be accelerator tensors")
        # Temporary rollback copies count against admission as well as retained
        # static inputs. Actual graph allocator growth is measured below.
        staging_bytes = sum(value.numel() * value.element_size() for value in inputs)
        snapshot_bytes = sum(value.numel() * value.element_size() for value in mutable)
        available = self.byte_budget - self.resident_bytes
        free_bytes, _ = torch.cuda.mem_get_info(inputs[0].device)
        peak = staging_bytes + snapshot_bytes + self.peak_workspace_bytes
        if staging_bytes + snapshot_bytes > available or peak + self.reserve_bytes > free_bytes:
            self._deny(key)
            self.uncaptured += 1
            return self._invoke(operation, inputs)
        device = inputs[0].device
        current = torch.cuda.current_stream(device)
        warmup = current
        snapshots = tuple(value.detach().clone() for value in mutable)
        rng = torch.cuda.get_rng_state(device)
        entry.static = tuple(value.detach().clone() for value in inputs)
        warmup.wait_stream(current)
        try:
            with torch.cuda.stream(warmup):
                for _ in range(2):
                    self._invoke(compiled, entry.static)
            current.wait_stream(warmup)
            # Capture mutations are discarded too. First replay therefore
            # performs exactly one accumulation, including on first admission.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=warmup):
                entry.outputs = self._invoke(compiled, entry.static)
            current.wait_stream(warmup)
            entry.graph = graph
        finally:
            current.wait_stream(warmup)
            with torch.no_grad():
                for destination, snapshot in zip(mutable, snapshots, strict=True):
                    destination.copy_(snapshot)
            torch.cuda.set_rng_state(rng, device)
        entry.resident_bytes = staging_bytes + sum(
            segment["total_size"] for segment in torch.cuda.memory_snapshot(
                mempool_id=graph.pool(), include_traces=False)
        )
        if entry.resident_bytes + snapshot_bytes > available:
            # Release the entire oversized graph pool, but retain the already
            # warmed compiled program. Recurring large shapes should not lose
            # compilation merely because capture cannot fit this cache budget.
            current.synchronize()
            entry.outputs = ()
            entry.static = ()
            entry.graph = None
            graph.reset()
            del graph, snapshots
            entry.resident_bytes = 0
            self.entries[key] = entry
            self._deny(key)
            self.uncaptured += 1
            return self._invoke(entry.operation, inputs)
        del snapshots
        self.entries[key] = entry
        entry.graph.replay()
        self.replays += 1
        return entry.outputs

    def manifest(self) -> dict[str, object]:
        return {"work_counts": {key: dict(value) for key, value in self.work_counts.items()},
                "mode": self.mode, "max_entries": self.max_entries,
                "byte_budget": self.byte_budget, "entries": len(self.entries),
                "captured_entries": sum(entry.graph is not None for entry in self.entries.values()),
                "resident_bytes": self.resident_bytes, "calls": self.calls,
                "replays": self.replays, "uncaptured": self.uncaptured,
                "denied_shapes": len(self.denied), "reserve_bytes": self.reserve_bytes,
                "peak_workspace_bytes": self.peak_workspace_bytes,
                "fullgraph": True, "dynamic": False,
                "compile_mode": "max-autotune-no-cudagraphs", "layout_version": 2}
