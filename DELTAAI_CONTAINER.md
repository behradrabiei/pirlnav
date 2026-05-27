# PIRLNav Apptainer Container for NCSA Delta AI

This document covers the Delta AI–specific workflow for the dependency-only
PIRLNav image. Delta AI nodes are aarch64 (NVIDIA GH200 / Grace Hopper), so
the image and PyTorch wheels differ from the x86 build in
[GREATLAKE_CONTAINER.md](GREATLAKE_CONTAINER.md). The PIRLNav source tree
and MP3D assets are not baked into the image; clone the repo and stage data
on the cluster, then bind them into the container at runtime.

## What is different from the x86 build

- Base image: `nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04` (multi-arch;
  Apptainer pulls the arm64 manifest on Delta AI automatically).
- Miniforge: `Miniforge3-Linux-aarch64.sh`, pinned to a specific release
  for reproducibility.
- PyTorch: installed from `https://download.pytorch.org/whl/cu126`. There
  are no `cu118` arm64 wheels, and Delta AI's host driver supports CUDA
  12.x with forward compatibility ([DeltaAI PyTorch docs](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/python/pytorch.html)).
- `habitat-sim 0.2.5` is **built from source** (`--headless --bullet`)
  inside `%post`. The `aihabitat` and `conda-forge` channels do not ship
  aarch64 builds of habitat-sim.

Everything else (PYTHONPATH-based dependency-only design, headless logging,
runtime bind layout) matches `containers/pirlnav.def`.

## Build on a Delta AI Login Node

Delta AI's Apptainer (1.4+) has user namespaces enabled, so unprivileged
builds work via `--fakeroot` directly on the login nodes
(`gh-login01..04`). Login nodes have outbound network to GitHub and PyPI,
which is required because the def clones `habitat-sim` and pulls Python
wheels during `%post`.

```bash
ssh you@gh-login.delta.ncsa.illinois.edu
cd /path/to/pirlnav
bash containers/build_pirlnav_deltaai_sif.sh
```

The build script automatically:

- Defaults to `apptainer build --fakeroot --ignore-fakeroot-command`.
  The `--ignore-fakeroot-command` flag is needed because most Delta AI
  users are not provisioned in `/etc/subuid`; without it, Apptainer
  bind-mounts the host's `libfakeroot.so` (built against a newer glibc)
  into the Ubuntu 22.04 container and `%post` fails with
  ``GLIBC_2.38' not found``.
- Sets `APPTAINER_CACHEDIR=$WORK/.apptainer` and
  `APPTAINER_TMPDIR=$WORK/apptainer-tmp` (falling back to `$HOME` if
  `$WORK` is unset) so the build does not overflow the small home quota.
- Writes `pirlnav-deltaai.sif` into the current directory.

Override paths if needed:

```bash
IMAGE=/work/$USER/images/pirlnav-deltaai.sif \
APPTAINER_CACHEDIR=/work/$USER/.apptainer \
APPTAINER_TMPDIR=/work/$USER/apptainer-tmp \
bash containers/build_pirlnav_deltaai_sif.sh
```

Expect a 20–40 minute wall time on a login node: most of that is compiling
Habitat-Sim, Magnum, and the vendored Bullet on Grace ARM. The image is
~6–8 GB.

### If `--fakeroot` is not provisioned for your account

If you see `failed to create user namespace`, user namespaces themselves
are not enabled — ask NCSA support to enable fakeroot for your user, or
build inside a `ghx4-interactive` allocation (same command works there):

```bash
srun --account=YOUR_ACCOUNT --partition=ghx4-interactive \
     --nodes=1 --gpus-per-node=1 --cpus-per-task=8 --mem=32g \
     --time=01:00:00 --pty bash
cd /path/to/pirlnav
bash containers/build_pirlnav_deltaai_sif.sh
```

### Build is fully offline-free?

The image build pulls from GitHub (Habitat-Sim source, Miniforge installer)
and PyPI (PyTorch + Python deps). If a future change blocks egress on the
login nodes, build inside `ghx4-interactive` instead.

## Stage the Source Tree and Data

```bash
git clone <repo-url> /path/to/pirlnav
cd /path/to/pirlnav
git submodule update --init --recursive
```

Stage MP3D, the ObjectNav subset, OVRL DINO encoder weights, and any
DINOv2 cache under the repo's `data/` layout (or via symlinks) following
[SETUP_NOTES.md](SETUP_NOTES.md). On Delta AI, `/projects/<group>` and
`/work/$USER` are the right places for large datasets and run outputs.

## Interactive Smoke Test

Grab one GPU on `ghx4-interactive` (max 2 hours, max 4 nodes per job):

```bash
SIF=/work/$USER/images/pirlnav-deltaai.sif
REPO=/path/to/pirlnav

srun --account=YOUR_ACCOUNT --partition=ghx4-interactive \
     --nodes=1 --gpus-per-node=1 --tasks=1 --tasks-per-node=1 \
     --cpus-per-task=8 --mem=32g --time=00:30:00 \
  apptainer exec --nv \
    --bind "${REPO}:/workspace/pirlnav" \
    --pwd /workspace/pirlnav \
    "${SIF}" \
    python - <<'PY'
import habitat_sim
import torch

print("habitat_sim", habitat_sim.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0))
PY
```

Then a short PIRLNav training smoke test:

```bash
srun --account=YOUR_ACCOUNT --partition=ghx4-interactive \
     --nodes=1 --gpus-per-node=1 --tasks=1 --tasks-per-node=1 \
     --cpus-per-task=8 --mem=64g --time=00:30:00 \
  apptainer exec --nv \
    --bind "${REPO}:/workspace/pirlnav" \
    --pwd /workspace/pirlnav \
    "${SIF}" \
    bash -lc 'NUM_UPDATES=3 NUM_ENVIRONMENTS=2 TAG=deltaai_smoke bash scripts/run_il_mp3d_1scene.sh'
```

If your data lives outside the repo, add bind mounts:

```bash
apptainer exec --nv \
  --bind "${REPO}:/workspace/pirlnav" \
  --bind /projects/your_proj/mp3d:/workspace/pirlnav/data/scene_datasets/mp3d \
  --bind /work/$USER/pirlnav_outputs:/workspace/pirlnav/data/new_checkpoints \
  --pwd /workspace/pirlnav \
  "${SIF}" \
  bash -lc 'NUM_UPDATES=3 NUM_ENVIRONMENTS=2 TAG=deltaai_smoke bash scripts/run_il_mp3d_1scene.sh'
```

## Multi-GPU SLURM Run

Use the template at
[scripts/slurm_train_pirlnav_deltaai.sh](scripts/slurm_train_pirlnav_deltaai.sh).
At minimum, edit:

```bash
#SBATCH --account=YOUR_ACCOUNT          # required on Delta AI
SIF=/work/$USER/images/pirlnav-deltaai.sif
REPO=/path/to/pirlnav
EXTRA_BINDS=/projects/your_proj/mp3d:/workspace/pirlnav/data/scene_datasets/mp3d,/work/$USER/pirlnav_outputs:/workspace/pirlnav/data/new_checkpoints
```

Submit:

```bash
SIF=/work/$USER/images/pirlnav-deltaai.sif \
REPO=/path/to/pirlnav \
EXTRA_BINDS=/projects/your_proj/mp3d:/workspace/pirlnav/data/scene_datasets/mp3d,/work/$USER/pirlnav_outputs:/workspace/pirlnav/data/new_checkpoints \
sbatch scripts/slurm_train_pirlnav_deltaai.sh
```

The template uses one SLURM task per GPU (4 per node on a GH200 box).
Habitat's distributed helper reads `SLURM_LOCALID`, `SLURM_PROCID`, and
`SLURM_NTASKS`, then assigns each process to its local GPU.
`RL.DDPPO.force_distributed True` stays enabled for multi-GPU runs.

## Full MP3D Run (online DINOv2 + online Object Cloud)

For training IL across the full MP3D THDA dataset filtered down to the
canonical 21 ObjectNav classes (56 demo scenes, 60085 episodes, online
cloud + online DINOv2, larger point transformer), use the dedicated
experiment config and the 2-node SLURM template:

- Task config: [configs/tasks/objectnav_mp3d_object_cloud_full.yaml](configs/tasks/objectnav_mp3d_object_cloud_full.yaml)
- Experiment config: [configs/experiments/il_objectnav_mp3d_dinov2_object_cloud_full.yaml](configs/experiments/il_objectnav_mp3d_dinov2_object_cloud_full.yaml)
- Single-node launcher (smoke / interactive probe): [scripts/run_il_mp3d_full_dinov2_object_cloud.sh](scripts/run_il_mp3d_full_dinov2_object_cloud.sh)
- 2-node x 4-GPU SLURM template: [scripts/slurm_train_pirlnav_deltaai_full.sh](scripts/slurm_train_pirlnav_deltaai_full.sh)
- 21-class filter helper: [scripts/filter_mp3d_thda_to_21cat.py](scripts/filter_mp3d_thda_to_21cat.py)

### Why the 21-class filter is required

The `axel81/habitat-web` THDA 70k bundle ships with a 28-entry
`category_to_task_category_id` table. The first 21 entries match the
canonical MP3D ObjectNav benchmark classes (and `pirlnav.task.semantic_map.OBJECTNAV_CATEGORIES`);
the last 7 (`foodstuff, stationery, fruit, plaything, hand_tool,
game_equipment, kitchenware`) are THDA-paper "treasure hunt" augmented
goals that have no `mpcat40_idx -> task_id` mapping in pirlnav's table,
so they would never appear in the object-cloud sensor's class-id stream.
Worse, the bundle's per-scene `goals_by_category` tables only contain
canonical-21 entries, so loading any THDA-extra episode would crash
`ObjectNavDatasetV2.from_json` with a `KeyError` on
`self.goals_by_category[episode.goals_key]`.

The filter helper drops THDA-extra episodes (10091 of 70176, ~14%) and
trims the category table to the canonical 21, writing a sibling bundle
that the rest of this section refers to as *the 21-class bundle*.

### Bind-mount layout (no symlinks)

The full-MP3D run replaces the `EXTRA_BINDS` envvar pattern with three
explicit Apptainer bind flags. The host-side data lives under
`/projects/bgon/brabiei/MP3D` and is mapped into the repo's `data/` paths
inside the container at runtime:

| Host path | Container path | Used by |
| --- | --- | --- |
| `/projects/bgon/brabiei/MP3D/scenes/mp3d` | `/workspace/pirlnav/data/scene_datasets/mp3d` | habitat-sim (90 MP3D scene assets + `mp3d.scene_dataset_config.json`; only the 56 referenced by demo episode_ids are actually loaded) |
| `/projects/bgon/brabiei/MP3D/demo_episodes/data/datasets/objectnav/objectnav_mp3d_thda_70k_21cat/objectnav/objectnav_mp3d_thda_70k_21cat` | `/workspace/pirlnav/data/datasets/objectnav/objectnav_mp3d/objectnav_mp3d_thda_70k_21cat` | training (21-class filtered THDA demos with `reference_replay`) |
| `/projects/bgon/brabiei/MP3D/eval_episodes` | `/workspace/pirlnav/data/datasets/objectnav/objectnav_mp3d/objectnav_mp3d_v1` | future periodic eval only (no `reference_replay`; never used as a demo source) |

The SLURM script sets these as `DEFAULT_FULL_BINDS` and `mkdir -p`s the
matching empty directories under the repo so Apptainer's bind mounts have
landing pads. Override `EXTRA_BINDS` on submit if your data layout differs.

### Resolving the THDA 70k Git LFS pointers

The `axel81/habitat-web` HF Hub dataset uses Git LFS for the `.json.gz`
demos. A bare `git clone` only pulls 132-byte pointer files; the trainer
will hit `gzip.BadGzipFile` until the real blobs are fetched. On Delta AI:

```bash
module load git-lfs/3.6.1
# git-lfs over the HF Hub HTTPS endpoint requires an HF access token --
# https://huggingface.co/settings/tokens. SSH alone is not enough.
export HF_TOKEN=hf_xxx
python3.12 -m pip install --user huggingface_hub  # if not already installed
python3.12 - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="axel81/habitat-web",
    repo_type="dataset",
    local_dir="/projects/bgon/brabiei/MP3D/demo_episodes/data/datasets/objectnav/objectnav_mp3d_thda_70k",
    allow_patterns=["objectnav/**"],
    token=os.environ["HF_TOKEN"],
)
PY
```

The pulled dataset is ~400 MB (56 scenes plus the top-level table files).
After this completes, the `.json.gz` files will be real gzip blobs.

### Filter the bundle down to the canonical 21 classes

Run once after the LFS pull completes; takes ~9 min on a login node and
uses only the Python stdlib:

```bash
python3 scripts/filter_mp3d_thda_to_21cat.py \
  --src /projects/bgon/brabiei/MP3D/demo_episodes/data/datasets/objectnav/objectnav_mp3d_thda_70k/objectnav/objectnav_mp3d_thda_70k \
  --dst /projects/bgon/brabiei/MP3D/demo_episodes/data/datasets/objectnav/objectnav_mp3d_thda_70k_21cat/objectnav/objectnav_mp3d_thda_70k_21cat \
  --splits train
```

Expected output:

```
split=train  scenes=56  src=70176  kept=60085 (85.62%)  dropped_class=10091  dropped_no_replay=0
  goals_by_category: 857 / 857
  dropped THDA-extra classes: game_equipment=1489, hand_tool=1482, kitchenware=1475, fruit=1470, foodstuff=1459, stationery=1381, plaything=1335
```

The filtered bundle is ~308 MB on disk and is what the bind table above
points at. The original 70k bundle is left untouched.

### Recompute the inflection coefficient

The IL `InflectionWeightSensor` is sensitive to the dataset's
inflection-point density, so it must be recomputed any time the demo set
changes (the 1-scene 6-cat subset value `3.234951275740812` and the
unfiltered THDA 70k value `3.556007765129` are both wrong for the
21-class filtered bundle):

```bash
python scripts/compute_inflection_coef.py \
  --data /projects/bgon/brabiei/MP3D/demo_episodes/data/datasets/objectnav/objectnav_mp3d_thda_70k_21cat/objectnav/objectnav_mp3d_thda_70k_21cat/train
```

The current `objectnav_mp3d_object_cloud_full.yaml` is pre-baked with
`INFLECTION_COEF: 3.513870128085` (computed over 60085 episodes /
14.1M total steps / 28.459% inflection rate on the filtered bundle).
Re-run the helper if you ever change the demo dataset.

### Interactive VRAM probe before the long run

The point transformer enlargement plus `MAX_OBJECTS=300` plus online cloud
construction makes per-step memory non-trivial on a single GH200. Probe
the largest `NUM_ENVIRONMENTS` that fits before submitting the multi-node
job:

```bash
salloc --account=bgon-dtai-gh --partition=ghx4-interactive \
       --gres=gpu:1 --cpus-per-task=64 --mem=0 --time=01:00:00
# inside the alloc, in the container with the three bind flags:
NUM_ENVIRONMENTS=8  NUM_UPDATES=20 bash scripts/run_il_mp3d_full_dinov2_object_cloud.sh
NUM_ENVIRONMENTS=16 NUM_UPDATES=20 bash scripts/run_il_mp3d_full_dinov2_object_cloud.sh
NUM_ENVIRONMENTS=24 NUM_UPDATES=20 bash scripts/run_il_mp3d_full_dinov2_object_cloud.sh
NUM_ENVIRONMENTS=32 NUM_UPDATES=20 bash scripts/run_il_mp3d_full_dinov2_object_cloud.sh
# In a second shell on the same node:
nvidia-smi -l 5
# Pick the largest N where peak memory stays below ~85% and steps/sec is
# still trending up; back-solve NUM_UPDATES so total env steps approach
# 500M across 8 ranks: NUM_UPDATES = 500_000_000 / (8 * N * 64).
```

### Submitting the 2-node job

```bash
SIF=/work/$USER/images/pirlnav-deltaai.sif \
REPO=/u/brabiei/projects/pirlnav \
sbatch scripts/slurm_train_pirlnav_deltaai_full.sh
```

The defaults inside the script are paper-faithful (`NUM_ENVIRONMENTS=8`,
`NUM_UPDATES=125000`, lr=1e-3 with linear decay, `num_steps=64`,
`num_mini_batch=2`). Override on submit:

```bash
NUM_ENVIRONMENTS=16 NUM_UPDATES=62500 \
SIF=/work/$USER/images/pirlnav-deltaai.sif \
REPO=/u/brabiei/projects/pirlnav \
sbatch scripts/slurm_train_pirlnav_deltaai_full.sh
```

### Throughput caveat

Online DINOv2 + online cloud construction is the most expensive sensor
combination in this codebase. Expect ~2-3x lower steps/sec than the cached
variants at equal `NUM_ENVIRONMENTS`. If `NUM_ENVIRONMENTS=8` does not
saturate the GH200 GPU memory, raise it as the probe allows; if it
overflows, drop to 4 and adjust `NUM_UPDATES` upward to keep the env-step
budget on target.

## Notes

- Always use `apptainer exec --nv` for GPU access. Without `--nv`, the
  container will not see the host CUDA driver.
- The host driver is CUDA 12.x with forward compatibility, so the image's
  `cu126` PyTorch wheels run correctly under both newer and slightly older
  host driver versions.
- `habitat-sim` is built from source for aarch64. If a future habitat-sim
  release adds an official aarch64 conda build, the def can be simplified
  back to the conda install used in `containers/pirlnav.def`.
- If habitat-sim's `v0.2.5` tag fails to compile on aarch64 (upstream PR
  #2626 added explicit aarch64 support on newer revisions), retry with the
  `stable` branch of the same 0.2.x line by editing
  `containers/pirlnav-deltaai.def`:
  `git clone --branch stable --recursive ...`. PIRLNav is exercised against
  the conda 0.2.5 build today, so 0.2.x source should be a near drop-in.
- For multi-node jobs the SLURM template sets `MAIN_ADDR` from the first
  allocated host. If you need to constrain NCCL to a specific Slingshot
  interface, export `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` (e.g.
  `hsn0`) in the job script.
- Single-GPU launchers under `scripts/run_il_*.sh` still work inside the
  container for smoke tests; use the SLURM template above for full
  multi-GPU runs.
