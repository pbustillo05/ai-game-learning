# Quest H100 reference snapshot

Retrieved on 2026-07-24 for planning the HU 32 bb Normal-versus-Rao--Blackwell
training study. The HTML files are local snapshots of the linked authoritative
pages; the PDF is the archived NVIDIA Hopper tuning guide recommended by
Quest.

## Capacity and scheduling facts

- Quest lists 24 General Access Quest 13 GPU nodes, each with four 80 GB
  NVIDIA H100 SXM5 cards: 96 General Access H100 cards in total.
- Each H100 node has 64 schedulable CPU cores and 976 GB of schedulable host
  memory. The cards use HBM3 and the nodes use NDR InfiniBand.
- General Access H100 jobs use `gengpu`, request
  `--gres=gpu:h100:<count>`, and have a 48-hour maximum wall time.
- One user may have at most eight General Access GPUs running at once. This is
  a QoS ceiling, not a promise that eight H100s will be immediately available.
- H100 nodes are four-GPU SXM nodes. A single-node training launch can
  therefore use at most four H100s; two separate four-GPU jobs could reach the
  per-user ceiling when scheduling and Fairshare permit.
- Actual free capacity and queue time are dynamic and are not published on the
  public web pages. Inspect them from Quest with `sinfo`, `squeue`, and
  `scontrol` before choosing a request size.

Suggested live inventory commands:

```bash
sinfo -p gengpu -N -o '%N %t %G %C %m %f'
squeue -p gengpu -o '%.18i %.9P %.8u %.2t %.10M %.6D %R %b'
scontrol show partition gengpu
scontrol show nodes
```

For the intended matched experiment, two one-GPU arms are the easiest useful
request. Four paired seeds would consume the eight-GPU per-user maximum and
will generally be harder to schedule.

## Environment compatibility warning

The downloaded Quest pages disagree about the installed CUDA driver:

- `quest-gpus.html` says driver 570.86.15 is installed and supports
  applications built with CUDA Toolkit 12.8 or earlier.
- `quest-release-notes.html` says the June 2026 maintenance updated the GPU
  driver to "CUDA 13.3."

The current local environment uses PyTorch 2.12.1+cu130 and CUDA Toolkit 13.0.
NVIDIA's compatibility table requires an R580-or-newer driver for CUDA 13.x.
Do not copy the local environment to Quest until an allocated H100 reports its
actual driver through `nvidia-smi`. If it is still R570, build the Quest
environment against CUDA 12.8 or earlier. If it is R580 or newer, the current
CUDA 13 line is eligible, subject to a smoke test.

## Source manifest

### Northwestern Quest

- `quest-technical-specifications.html`:
  [Quest Technical Specifications](https://www.it.northwestern.edu/departments/it-services-support/research/computing/quest/specs.html)
- `quest-gpus.html`:
  [GPUs on Quest](https://rcdsdocs.it.northwestern.edu/systems/quest/user-guide/gpu/gpu.html)
- `quest-slurm.html`:
  [Slurm Job Scheduler](https://rcdsdocs.it.northwestern.edu/systems/quest/user-guide/slurm/slurm.html)
- `quest-release-notes.html`:
  [Quest maintenance and update notes](https://rcdsdocs.it.northwestern.edu/systems/quest/specs/release-notes.html)
- `quest-filesystems.html`:
  [Quest filesystems, storage, and permissions](https://rcdsdocs.it.northwestern.edu/systems/quest/user-guide/filesystem/filesystem.html)
- `quest-resources.html`:
  [Quest resources](https://rcdsdocs.it.northwestern.edu/systems/quest/resources/quest-resource.html)
- `quest-python.html`:
  [Python on Quest](https://rcdsdocs.it.northwestern.edu/systems/quest/software/python/python-quest.html)
- `quest-python-gpus.html`:
  [GPUs in Python](https://rcdsdocs.it.northwestern.edu/tutorials/python/python-gpus.html)
- `quest-mamba-conda.html`:
  [Mamba or Conda virtual environments](https://rcdsdocs.it.northwestern.edu/tutorials/software-management/conda-mamba-quest/mamba-conda-quest.html)
- `quest-modules.html`:
  [Quest software module system](https://rcdsdocs.it.northwestern.edu/systems/quest/user-guide/modules/modules.html)

### H100, CUDA, and PyTorch

- `nvidia-hopper-tuning-guide-12.2.pdf`:
  [NVIDIA Hopper Tuning Guide, CUDA 12.2](https://docs.nvidia.com/cuda/archive/12.2.2/pdf/Hopper_Tuning_Guide.pdf)
- `nvidia-cuda-compatibility.html`:
  [NVIDIA CUDA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- `pytorch-sdpa.html`:
  [PyTorch scaled-dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- `pytorch-cuda-backends.html`:
  [PyTorch backend controls](https://docs.pytorch.org/docs/stable/backends.html)

### RunPod

- `runpod-gpu-types.html`:
  [RunPod GPU types](https://docs.runpod.io/references/gpu-types)
- `runpod-pricing.html`:
  [RunPod GPU pricing](https://www.runpod.io/pricing)

RunPod currently distinguishes H100 SXM, H100 PCIe, and H100 NVL. Use one
80 GB H100 SXM for the closest single-GPU match to Quest's H100 SXM5 nodes.

