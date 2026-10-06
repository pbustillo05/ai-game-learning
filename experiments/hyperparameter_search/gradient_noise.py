"""Fixed-policy CUDA gradient-noise probes; no scheduler multiplier is inferred.

One independent public-trajectory batch is one replicate. The captured vector
is the actual pooled-decision learner gradient, before clipping/Adam/decay,
with KL, entropy and critic objectives disabled. RB and sampled gradients may
have different means; variance reduction alone does not establish equivalence.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import torch

from experiments.hyperparameter_search.campaign import source_digest
from experiments.hyperparameter_search.protocol import (
    atomic_json,
    canonical_sha256,
    resolve_spec,
    sha256_file,
)


def read_json(path):
    return json.loads(Path(path).read_text())


def immutable_json(path, value):
    path = Path(path)
    if path.exists() and read_json(path) != value:
        raise RuntimeError(f"Immutable diagnostic specification changed: {path}")
    atomic_json(path, value)


def atomic_tensor(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def require_cuda_device(device):
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(
            "Gradient diagnostics require CUDA; CPU statistics are forbidden"
        )
    if device.index not in (None, 0):
        raise ValueError("Expose exactly one GPU and use cuda:0")
    return torch.device("cuda:0")


def preflight(device):
    device = require_cuda_device(device)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Gradient diagnostics require a single GPU process")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(visible) != 1 or not visible[0]:
        raise ValueError("Set CUDA_VISIBLE_DEVICES to exactly one idle GPU")
    busy = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            visible[0],
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    if busy:
        raise RuntimeError(f"Refusing occupied GPU {visible[0]}: {busy}")
    torch.set_num_threads(2)
    torch.cuda.set_device(device)
    return device


def probe_config(spec):
    if spec.get("gradient_noise") is not True or spec.get("kind") != "self_play":
        raise ValueError("Probe requires gradient_noise:true and kind:self_play")
    if spec.get("initialization") not in ("model", "reference_model"):
        raise ValueError("Probe requires a fixed model or reference_model initializer")
    source, _ = resolve_spec(spec, 0, 1)
    # Resolve smoke first: its ordinary worker default is TWO steps.
    source["scientific"]["schedule"]["steps"] = 1
    objective = source["scientific"]["objective"]
    objective.pop("reference_kl", None)
    objective["entropy"] = {"kind": "constant", "coefficient": 0.0}
    objective["value"]["coefficient"] = 0.0
    source["run"]["execution"]["reference_cache"] = "off"
    return source


def capture_gradient(optimizer, destination):
    original = optimizer.step

    def step():
        if destination:
            raise RuntimeError(
                "Diagnostic unexpectedly attempted multiple optimizer steps"
            )
        gradient = optimizer.gradients.flat.detach().clone()
        if optimizer.scaler.enabled:
            gradient.div_(float(optimizer.scaler.scale))
        destination.append(gradient)
        return original()

    optimizer.step = step


def run_probe(spec, device="cuda:0"):
    from experiments.hyperparameter_search.exposure import (
        install_posterior_exposure_observer,
    )
    from experiments.hyperparameter_search.worker import install_adapters
    from pbg.config import load_config_mapping
    from pbg.core.optimization import LossScaler
    from pbg.core.persistence import model_state_sha256
    from pbg.runtime import TrainingSession

    source = probe_config(spec)
    output = Path(spec["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_hash = source_digest()
    identity = {"spec_sha256": canonical_sha256(spec), "source_sha256": source_hash}
    immutable_json(
        output / "probe_spec.json", {"spec": spec, **identity, "config": source}
    )
    result_path = output / "result.json"
    if result_path.exists():
        previous = read_json(result_path)
        if (
            previous.get("complete")
            and all(previous.get(k) == v for k, v in identity.items())
            and sha256_file(previous["gradient"]) == previous["gradient_sha256"]
        ):
            return previous
        raise RuntimeError(
            "Existing probe result failed immutable identity/hash validation"
        )
    device = preflight(device)
    initializer = spec["profile"]["initialization"]
    checkpoint = Path(initializer["checkpoint"])
    if sha256_file(checkpoint) != initializer["sha256"]:
        raise RuntimeError("Initializer checkpoint content changed")
    torch.manual_seed(int(source["run"]["seed"]))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    session = TrainingSession(
        load_config_mapping(source),
        initial_model_state=payload[spec["initialization"]],
        benchmark_telemetry=True,
    )
    del payload
    try:
        if len(session.run_states) != 1:
            raise RuntimeError("Probe requires one policy replica")
        install_adapters(session)
        model = session.modules.lane_module(0)
        scaled = []
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith(
                    ("value_head.affine.weight", "value_head.affine.bias")
                ):
                    parameter.div_(float(initializer["value_scale"]))
                    scaled.append(name)
        if len(scaled) != 4:
            raise RuntimeError(f"Unexpected value-head migration layout: {scaled}")
        if session.reference_modules is not None:
            session.reference_modules.lane_module(0).load_state_dict(model.state_dict())
        initial_hash = model_state_sha256(model)
        session.initial_model_sha256 = (initial_hash,)
        optimizer = session.optimizers.optimizers[0]
        optimizer.scaler = LossScaler(enabled=True, scale=1024.0, growth_interval=2000)
        named = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        if [id(p) for _, p in named] != [id(p) for p in optimizer.gradients.parameters]:
            raise RuntimeError(
                "Named parameter order differs from dense optimizer layout"
            )
        layout = [
            {"name": name, "shape": list(parameter.shape), "numel": parameter.numel()}
            for name, parameter in named
        ]
        observations, captured = [], []
        install_posterior_exposure_observer(session, observations)
        capture_gradient(optimizer, captured)
        torch.cuda.synchronize(device)
        started = time.monotonic()
        session.step()
        torch.cuda.synchronize(device)
        seconds = time.monotonic() - started
        metrics = copy.deepcopy(session.run_states[0].metrics_prefix[-1])
        gradient_metrics = metrics.get("gradient", {})
        if (
            len(captured) != 1
            or gradient_metrics.get("finite") is not True
            or gradient_metrics.get("applied_steps") != 1
            or not bool(torch.isfinite(captured[0]).all().item())
        ):
            raise RuntimeError(
                f"Probe did not apply exactly one finite update: {gradient_metrics}"
            )
        if not observations:
            raise RuntimeError("Probe produced no posterior exposure observations")
        if source_digest() != source_hash:
            raise RuntimeError("Source changed during gradient probe")
        path = output / "gradient.pt"
        metadata = {
            **identity,
            "initial_model_sha256": initial_hash,
            "seed": int(source["run"]["seed"]),
            "private_estimator": source["scientific"]["mode"]["private_estimator"],
            "roots": source["run"]["workload"]["roots"],
            "batch": source["scientific"]["schedule"]["batch_size"],
            "parameter_layout": layout,
        }
        atomic_tensor(path, {"gradient": captured[0].cpu(), **metadata})
        result = {
            "complete": True,
            "kind": "gradient_noise_probe",
            **metadata,
            "gradient": str(path),
            "gradient_sha256": sha256_file(path),
            "config": source,
            "training_seconds": seconds,
            "metrics": metrics,
            "posteriorbreadth": observations,
            "gradient_semantics": "AMP_unscaled_pre_clip_pre_Adam_actual_pooled_decision_PG",
            "independent_replicate_unit": "fresh_public_trajectory_batch",
            "checkpoint_io_in_training_seconds": False,
        }
        atomic_json(result_path, result)
        return result
    finally:
        session.close()


def cuda_replicates(gradients):
    if (
        not isinstance(gradients, torch.Tensor)
        or gradients.device.type != "cuda"
        or gradients.ndim != 2
        or gradients.shape[0] < 2
        or gradients.shape[1] < 1
    ):
        raise ValueError(
            "Expected CUDA [at least two replicates, parameters] gradients"
        )
    for begin in range(0, gradients.shape[1], 65536):
        if not bool(torch.isfinite(gradients[:, begin : begin + 65536]).all().item()):
            raise ValueError("Non-finite gradient replicate")
    return gradients


def covariance_summary(gradients):
    """CUDA Welford mean and trace(sample covariance), avoiding CPU statistics."""
    gradients = cuda_replicates(gradients)
    mean = torch.zeros(gradients.shape[1], dtype=torch.float64, device=gradients.device)
    trace_m2 = mean.new_zeros(())
    for count, row in enumerate(gradients, 1):
        row = row.to(torch.float64)
        delta = row - mean
        mean += delta / count
        trace_m2 += torch.dot(delta, row - mean)
    trace = (trace_m2 / (gradients.shape[0] - 1)).clamp_min(0)
    mean_square = mean.square().sum()
    n = gradients.shape[0]
    summary = {
        "replicates": n,
        "parameters": gradients.shape[1],
        "trace_sample_covariance": trace.item(),
        "mean_gradient_squared_norm": mean_square.item(),
        "noise_trace_of_mean": (trace / n).item(),
        "single_replicate_signal_to_noise": (mean_square / trace).item()
        if trace > 0
        else None,
        "mean_signal_to_noise": (mean_square / (trace / n)).item()
        if trace > 0
        else None,
        "noise_corrected_mean_squared_norm_unclipped": (mean_square - trace / n).item(),
        "snr_definition": "squared norm of sample mean divided by trace covariance; descriptive",
    }
    return mean, trace, summary


def paired_comparison(normal, rb, *, repetitions=1000, seed=20260928):
    """Same-index resampling across modes; no independent-sample or bias claim."""
    normal, rb = cuda_replicates(normal), cuda_replicates(rb)
    if normal.shape != rb.shape or normal.device != rb.device:
        raise ValueError(
            "Paired mode gradients must have identical CUDA shapes/devices"
        )
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 1
    ):
        raise ValueError("bootstrap_repetitions must be a positive integer")
    mn, vn, _ = covariance_summary(normal)
    mr, vr, _ = covariance_summary(rb)
    delta = mn - mr
    # Accumulate small Gram matrices over parameter chunks. Never materialize
    # either full [replicate, parameter] matrix in float64, or their difference.
    n = normal.shape[0]
    gn = normal.new_zeros((n, n), dtype=torch.float64)
    gr = torch.zeros_like(gn)
    difference_m2 = gn.new_zeros(())
    for begin in range(0, normal.shape[1], 65536):
        end = begin + 65536
        cn = normal[:, begin:end].double() - mn[begin:end]
        cr = rb[:, begin:end].double() - mr[begin:end]
        gn.add_(cn @ cn.T)
        gr.add_(cr @ cr.T)
        difference_m2.add_((cn - cr).square().sum())
    vd = difference_m2 / (n - 1)
    noise = vd / normal.shape[0]
    difference_square = delta.square().sum()
    denominator = mn.norm() * mr.norm()
    generator = torch.Generator(device=normal.device).manual_seed(seed)
    n = normal.shape[0]
    draws = torch.randint(
        n, (repetitions, n), device=normal.device, generator=generator
    )
    counts = torch.zeros((repetitions, n), dtype=torch.float64, device=normal.device)
    counts.scatter_add_(1, draws, torch.ones_like(draws, dtype=torch.float64))

    def bootstrap_trace(gram):
        quadratic = ((counts @ gram) * counts).sum(dim=1) / n
        return ((counts @ gram.diag() - quadratic) / (n - 1)).clamp_min(0)

    bn, br = bootstrap_trace(gn), bootstrap_trace(gr)
    valid = (br > 0) & ((counts > 0).sum(dim=1) > 1)
    ratios = bn[valid] / br[valid]
    interval = None
    if ratios.numel():
        interval = torch.quantile(ratios, ratios.new_tensor([0.025, 0.975])).tolist()
    return {
        "variance_ratio_normal_over_rb": (vn / vr).item() if vr > 0 else None,
        "variance_ratio_paired_bootstrap_95_percentile_interval": interval,
        "bootstrap_repetitions": repetitions,
        "bootstrap_valid_replicates": int(valid.sum().item()),
        "bootstrap_degenerate_replicates": int((~valid).sum().item()),
        "bootstrap_pairing": "same seed-index multiplicities across modes",
        "mean_l2_difference": delta.norm().item(),
        "mean_squared_difference": difference_square.item(),
        "mean_cosine_similarity": (torch.dot(mn, mr) / denominator).item()
        if denominator > 0
        else None,
        "paired_estimated_noise_of_mean_difference": noise.item(),
        "mean_squared_difference_over_estimated_noise": (
            difference_square / noise
        ).item()
        if noise > 0
        else None,
        "noise_corrected_squared_mean_difference_unclipped": (
            difference_square - noise
        ).item(),
        "interpretation": "Descriptive paired batch-gradient diagnostic; not a test of unbiasedness, not an independent-hands multiplier.",
    }


def validate_groups(manifest):
    groups = manifest.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("Analysis manifest requires nonempty groups")
    labels, cells = set(), set()
    for group in groups:
        label = group.get("label")
        mode = group.get("private_estimator")
        if not isinstance(label, str) or not label or label in labels:
            raise ValueError("Group labels must be unique nonempty strings")
        if mode not in ("sampled", "rao_blackwell"):
            raise ValueError("Unknown private estimator")
        if any(
            isinstance(group.get(k), bool)
            or not isinstance(group.get(k), int)
            or group[k] < 1
            for k in ("roots", "batch")
        ):
            raise ValueError("roots and batch must be positive integers")
        cell = (group["roots"], group["batch"], mode)
        if cell in cells:
            raise ValueError("Duplicate mode at a roots/batch setting")
        artifacts = group.get("artifacts")
        if not isinstance(artifacts, list) or len(artifacts) < 2:
            raise ValueError(
                "Each group needs at least two independent replicate artifacts"
            )
        if len(set(artifacts)) != len(artifacts):
            raise ValueError("Duplicate replicate artifact path")
        labels.add(label)
        cells.add(cell)
    for roots, batch, mode in cells:
        other = "sampled" if mode == "rao_blackwell" else "rao_blackwell"
        if (roots, batch, other) not in cells:
            raise ValueError("Each roots/batch setting requires both estimator modes")
    return groups


def comparable_config(config):
    """Remove only the registered mode distinction and replicate/run identity."""
    value = copy.deepcopy(config)
    value["scientific"]["mode"].pop("private_estimator", None)
    # Sampled mode has no private-hand completion estimator configuration.
    value["scientific"].pop("completion", None)
    value["run"].pop("seed", None)
    value["run"].pop("output_dir", None)
    return value


def run_analysis(manifest, output, device="cuda:0"):
    groups = validate_groups(manifest)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    source_hash = source_digest()
    identity = {
        "manifest_sha256": canonical_sha256(manifest),
        "source_sha256": source_hash,
    }
    immutable_json(
        output.with_suffix(".manifest.json"), {"manifest": manifest, **identity}
    )
    if output.exists():
        previous = read_json(output)
        if (
            previous.get("complete")
            and all(previous.get(k) == v for k, v in identity.items())
            and sha256_file(previous["mean_gradients"])
            == previous["mean_gradients_sha256"]
        ):
            for artifact in previous["input_artifacts"]:
                if (
                    sha256_file(artifact["result"]) != artifact["result_sha256"]
                    or sha256_file(artifact["gradient"]) != artifact["gradient_sha256"]
                ):
                    raise RuntimeError("Completed analysis input content changed")
            return previous
        raise RuntimeError(
            "Existing analysis failed immutable identity/hash validation"
        )
    device = preflight(device)
    torch.cuda.reset_peak_memory_stats(device)
    loaded, summaries, means, inputs, comparisons = {}, {}, {}, [], []
    shared_identity = None
    configurations = {}
    probe_source_hash = manifest.get("probe_source_sha256", source_hash)
    for group in sorted(
        groups, key=lambda g: (g["roots"], g["batch"], g["private_estimator"])
    ):
        rows = []
        for result_path in group["artifacts"]:
            result_path = str(Path(result_path).resolve())
            result = read_json(result_path)
            if (
                not result.get("complete")
                or result.get("kind") != "gradient_noise_probe"
            ):
                raise ValueError(f"Incomplete or wrong-kind probe: {result_path}")
            if result.get("source_sha256") != probe_source_hash:
                raise ValueError(
                    "Probe source identity differs from registered probe source"
                )
            for key in ("private_estimator", "roots", "batch"):
                if result.get(key) != group[key]:
                    raise ValueError(f"Probe {key} differs from manifest group")
            gradient_path = result["gradient"]
            if sha256_file(gradient_path) != result["gradient_sha256"]:
                raise ValueError("Gradient artifact hash mismatch")
            artifact = torch.load(gradient_path, map_location=device, weights_only=True)
            for key in (
                "source_sha256",
                "spec_sha256",
                "initial_model_sha256",
                "seed",
                "private_estimator",
                "roots",
                "batch",
                "parameter_layout",
            ):
                if artifact.get(key) != result.get(key):
                    raise ValueError(f"Gradient artifact metadata mismatch: {key}")
            probe_spec = read_json(Path(result_path).parent / "probe_spec.json")
            if (
                canonical_sha256(probe_spec["spec"]) != result["spec_sha256"]
                or probe_spec["config"] != result["config"]
                or probe_config(probe_spec["spec"]) != result["config"]
            ):
                raise ValueError("Probe specification/configuration identity mismatch")
            cell = (group["roots"], group["batch"])
            comparable = comparable_config(result["config"])
            if cell in configurations and configurations[cell] != comparable:
                raise ValueError(
                    "Same roots/batch probe configurations differ beyond mode/seed"
                )
            configurations[cell] = comparable
            current_identity = (
                result["initial_model_sha256"],
                result["parameter_layout"],
            )
            if shared_identity is None:
                shared_identity = current_identity
            if current_identity != shared_identity:
                raise ValueError(
                    "Initial model or parameter layout differs across replicates"
                )
            gradient = artifact["gradient"]
            if (
                gradient.device.type != "cuda"
                or gradient.ndim != 1
                or gradient.numel()
                != sum(p["numel"] for p in result["parameter_layout"])
            ):
                raise ValueError("Invalid dense gradient artifact shape/device")
            rows.append((result["seed"], gradient))
            inputs.append(
                {
                    "result": result_path,
                    "result_sha256": sha256_file(result_path),
                    "gradient": gradient_path,
                    "gradient_sha256": result["gradient_sha256"],
                }
            )
        rows.sort(key=lambda item: item[0])
        seeds = [seed for seed, _ in rows]
        if len(set(seeds)) != len(seeds):
            raise ValueError("Replicates must have distinct random seeds")
        stacked = torch.stack([gradient for _, gradient in rows])
        mean, _, summary = covariance_summary(stacked)
        label = group["label"]
        summaries[label] = {
            **summary,
            "private_estimator": group["private_estimator"],
            "roots": group["roots"],
            "batch": group["batch"],
            "seeds": seeds,
        }
        # CPU storage here is checkpoint serialization only; all statistics run on CUDA.
        means[label] = mean.cpu()
        del mean, rows, gradient, artifact
        loaded[(group["roots"], group["batch"], group["private_estimator"])] = (
            label,
            seeds,
            stacked,
        )
        roots, batch = group["roots"], group["batch"]
        normal_key, rb_key = (roots, batch, "sampled"), (roots, batch, "rao_blackwell")
        if normal_key in loaded and rb_key in loaded:
            nl, ns, ng = loaded.pop(normal_key)
            rl, rs, rg = loaded.pop(rb_key)
            if ns != rs:
                raise ValueError(
                    "Normal/RB paired bootstrap requires identical replicate seeds"
                )
            comparison = paired_comparison(
                ng,
                rg,
                repetitions=manifest.get("bootstrap_repetitions", 1000),
                seed=int(manifest.get("seed", 20260928)),
            )
            comparisons.append(
                {
                    "roots": roots,
                    "batch": batch,
                    "normal_label": nl,
                    "rb_label": rl,
                    **comparison,
                }
            )
            # Each pair is released before the next pair's artifacts are loaded.
            del ng, rg, stacked
    if loaded:
        raise RuntimeError("Unpaired gradient groups remain")
    if source_digest() != source_hash:
        raise RuntimeError("Source changed during gradient analysis")
    mean_path = output.with_suffix(".means.pt")
    atomic_tensor(
        mean_path,
        {
            "means": {key: value.cpu() for key, value in means.items()},
            "parameter_layout": shared_identity[1],
            **identity,
        },
    )
    result = {
        "complete": True,
        "kind": "gradient_noise_analysis",
        "analysispath": str(output),
        **identity,
        "initial_model_sha256": shared_identity[0],
        "probe_source_sha256": probe_source_hash,
        "analysis_memory_strategy": "one_mode_pair_fp32_parameter_chunked_float64_gram_v2",
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "mean_gradients": str(mean_path),
        "mean_gradients_sha256": sha256_file(mean_path),
        "input_artifacts": inputs,
        "groups": summaries,
        "comparisons": comparisons,
        "calibration_status": "diagnostic_only_no_validated_schedule_multiplier",
        "automatic_schedule_multiplier": None,
        "limitations": [
            "Each replicate is a fresh public-trajectory batch; decisions within a root are correlated.",
            "Captures the actual pooled-decision learner gradient, including minibatch and completion randomness.",
            "Matched RNG seeds pair modes but do not guarantee identical public trajectories or completion draws.",
            "Nonlinear RB targets can shift gradient means; low variance alone cannot establish unbiasedness.",
            "Small-sample bootstrap intervals are exploratory; degenerate resamples are reported and excluded.",
            "No effective-hands exponent, scheduling multiplier, Nash certificate, or strength conclusion is inferred.",
        ],
    }
    atomic_json(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--spec", type=Path)
    action.add_argument("--analyze", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    require_cuda_device(args.device)
    if args.spec:
        if args.output is not None:
            parser.error(
                "--output belongs to --analyze; probe output comes from its spec"
            )
        run_probe(read_json(args.spec), args.device)
    else:
        if args.output is None:
            parser.error("--analyze requires --output")
        run_analysis(read_json(args.analyze), args.output, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
