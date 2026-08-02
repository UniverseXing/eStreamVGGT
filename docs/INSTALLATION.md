# Installation

This guide installs the environment required for eStreamVGGT inference and
evaluation. The project currently targets Linux with an NVIDIA CUDA GPU.
Camera-pose, reconstruction, and long-sequence evaluators are not CPU
fallbacks.

## Reference environment

The frozen resource tables were produced with:

| Component | Reference value |
|---|---|
| GPU | NVIDIA RTX 6000 Ada Generation |
| Python | 3.11.9 |
| PyTorch | 2.3.1+cu121 |
| CUDA reported by PyTorch | 12.1 |
| Input size | 518 |

Exact GPU memory and runtime values depend on the driver, device, library
versions, and other processes. A different supported NVIDIA GPU can be used,
but resource numbers should only be compared within one controlled
environment.

## Clone the extension repository

```bash
git clone https://github.com/UniverseXing/eStreamVGGT.git
cd eStreamVGGT
```

Do not clone the upstream repository when attempting to reproduce the bounded
K4/K6/K8 results; the bounded-cache code and public wrappers live in this
extension.

## Create the Conda environment

The download and verification helpers also expect the standard command-line
tools `git`, `wget`, `unzip`, `tar`, and `sha256sum` to be available.

```bash
conda create -n StreamVGGT python=3.11 cmake=3.14.0
conda activate StreamVGGT
python -m pip install --upgrade pip
python -m pip install -r requirements_eval.txt
conda install 'llvm-openmp<16'
```

`requirements_eval.txt` includes the core `requirements.txt` environment and
adds the pose, reconstruction, and paper-asset dependencies (`evo`, `open3d`,
and `imageio`). The requirements pin the core PyTorch and NumPy versions used
by the project. Use `python -m pip` after activation so packages are installed
into the same interpreter that runs the evaluation.

The public reproduction scripts intentionally do not activate Conda. This
keeps them portable across local machines and schedulers and prevents a batch
shell from silently using a different Python installation.

## Download the checkpoint

eStreamVGGT uses the original StreamVGGT checkpoint without additional
training. Download `checkpoints.pth` from either upstream location:

- [StreamVGGT on Hugging Face](https://huggingface.co/lch01/StreamVGGT/)
- [StreamVGGT Tsinghua cloud mirror](https://cloud.tsinghua.edu.cn/d/d6ad8f36fcd541bcb246/)

Place it at:

```bash
hf download lch01/StreamVGGT checkpoints.pth --local-dir ckpt
```

This produces `eStreamVGGT/ckpt/checkpoints.pth`. The `hf` command is installed
with `huggingface_hub`, which is included in `requirements.txt`. The exact
upstream file can also be downloaded manually from
[`checkpoints.pth`](https://huggingface.co/lch01/StreamVGGT/blob/main/checkpoints.pth).

The VGGT teacher `model.pt` is only needed for the inherited training or
fine-tuning workflows; it is not required for the frozen bounded-cache
evaluation.

The repository does not redistribute a new eStreamVGGT checkpoint because the
method changes inference-time state retention rather than model parameters.

## Verify the active environment

From the repository root:

```bash
which python
python - <<'PY'
import sys
import numpy
import scipy
import torch

print("Python:", sys.executable)
print("NumPy:", numpy.__version__)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

test -f ckpt/checkpoints.pth
bash scripts/reproduce/smoke_test.sh
```

Run the smoke test before downloading every evaluation dataset or submitting a
long batch job.

For a formal result, also capture the fully resolved environment rather than
assuming that unpinned transitive packages stayed unchanged:

```bash
mkdir -p eval_results/reproduce
python -m pip freeze > eval_results/reproduce/pip-freeze.txt
nvidia-smi > eval_results/reproduce/nvidia-smi.txt
```

The four pinned core versions above describe the reference baseline;
`pip-freeze.txt` is the exact lock record for the machine that produced a new
table.

## Scheduler use

The files under `scripts/reproduce/` are portable task commands, not scheduler
submission files. In a SLURM script, activate the environment with the setup
that is valid for your cluster and invoke a reproduction wrapper afterwards:

```bash
# Use your site's normal module and Conda initialization here.
conda activate StreamVGGT
bash scripts/reproduce/run_video_depth.sh
```

Do not copy a site-specific `module load`, Conda installation path, account,
partition, or node name into the public wrappers.

## Troubleshooting

### `ModuleNotFoundError` for an evaluation package

The batch process is normally using a different interpreter from the activated
environment. Compare:

```bash
which python
python -c 'import sys; print(sys.executable)'
python -m pip --version
```

All three paths should belong to the same Conda environment.

### `libGL.so.1` while importing OpenCV

This repository uses `opencv-python-headless` for non-GUI evaluation. Remove a
conflicting GUI OpenCV wheel and reinstall the requirements in the active
environment if an older environment still imports `opencv-python`.

### NVML driver/library mismatch

An NVML driver/library mismatch is a host or cluster-node driver problem, not a
missing Python package. Confirm with `nvidia-smi` and move the job to a healthy
node or contact the system administrator. Reinstalling NumPy, PyTorch, or this
repository does not repair the host driver.

### CUDA out of memory

First check that no unrelated process occupies the device. Full cache is
expected to grow with sequence length and can fail on long inputs. K4, K6, and
K8 bound cached frame states, but a task can still retain input images or dense
outputs unless it uses the streaming-release path. Use
`scripts/reproduce/run_long_sequence.sh` for the end-to-end bounded test.

### Reproducing runtime or memory values

Use one process and one GPU, keep the resolution and method set fixed, and do
not mix measurements from different GPU models. Report both the hardware and
software versions stored by the evaluator.
