# PIRLNav Singularity Container

This repository uses a dependency-only Singularity/Apptainer image for HPC runs.
The image does not include the source tree or datasets; clone the repo and stage
MP3D assets on the cluster, then bind them into the container at runtime.

## Build Locally

Install Apptainer or Singularity on a machine where you can build images, then
run:

```bash
bash containers/build_pirlnav_sif.sh
```

This creates `pirlnav.sif` from `containers/pirlnav.def`.

Equivalent commands:

```bash
apptainer build pirlnav.sif containers/pirlnav.def
singularity build pirlnav.sif containers/pirlnav.def
```

The definition installs `habitat-sim=0.2.5` with the conda `headless` feature,
so it does not require an X display on the HPC. GPU libraries are supplied by the
host at runtime via `--nv`.

## Transfer To The HPC

Copy the image to the cluster, for example:

```bash
rsync -av pirlnav.sif user@hpc.example.edu:/path/to/images/pirlnav.sif
```

Clone this repository separately on the HPC:

```bash
git clone <repo-url> /path/to/pirlnav
cd /path/to/pirlnav
git submodule update --init --recursive
```

Make sure the MP3D data, ObjectNav subset, visual encoders, and optional DINOv2
cache are available under the repo's `data/` layout or via symlinks matching the
paths used in `SETUP_NOTES.md`.

## Interactive Smoke Test

On an interactive GPU node:

```bash
SIF=/path/to/images/pirlnav.sif
REPO=/path/to/pirlnav

singularity exec --nv \
  --bind "${REPO}:/workspace/pirlnav" \
  --pwd /workspace/pirlnav \
  "${SIF}" \
  python - <<'PY'
import habitat_sim
import torch

print("habitat_sim", habitat_sim.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY
```

Then run a short PIRLNav smoke test:

```bash
singularity exec --nv \
  --bind "${REPO}:/workspace/pirlnav" \
  --pwd /workspace/pirlnav \
  "${SIF}" \
  bash -lc 'NUM_UPDATES=3 NUM_ENVIRONMENTS=2 TAG=hpc_smoke bash scripts/run_il_mp3d_1scene.sh'
```

If your data lives outside the repo, add extra bind mounts. Example:

```bash
singularity exec --nv \
  --bind "${REPO}:/workspace/pirlnav" \
  --bind /cluster/datasets/mp3d:/workspace/pirlnav/data/scene_datasets/mp3d \
  --bind /cluster/scratch/pirlnav_outputs:/workspace/pirlnav/data/new_checkpoints \
  --pwd /workspace/pirlnav \
  "${SIF}" \
  bash -lc 'NUM_UPDATES=3 NUM_ENVIRONMENTS=2 TAG=hpc_smoke bash scripts/run_il_mp3d_1scene.sh'
```

## Multi-GPU SLURM Run

Use the template in `scripts/slurm_train_pirlnav_singularity.sh`.

At minimum, edit these variables for your cluster:

```bash
SIF=/path/to/images/pirlnav.sif
REPO=/path/to/pirlnav
EXTRA_BINDS=/optional/data/path:/workspace/pirlnav/data
```

Submit:

```bash
SIF=/path/to/images/pirlnav.sif \
REPO=/path/to/pirlnav \
sbatch scripts/slurm_train_pirlnav_singularity.sh
```

The template uses one SLURM task per GPU. Habitat's distributed helper reads
`SLURM_LOCALID`, `SLURM_PROCID`, and `SLURM_NTASKS`, then assigns each process
to its local GPU. Keep `RL.DDPPO.force_distributed True` enabled for multi-GPU
training.

## Notes

- Always use `singularity exec --nv` or `apptainer exec --nv` for GPU access.
- The container defaults to a CUDA 11.8 PyTorch wheel for broad HPC compatibility.
  If the cluster requires a different CUDA runtime, adjust `containers/pirlnav.def`
  and rebuild.
- For multi-node jobs, the SLURM template sets `MAIN_ADDR` from the first
  allocated host. If your cluster requires specific NCCL interfaces, export
  `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` in the job script.
- The existing single-GPU launchers are still useful inside the container for
  smoke tests. Use the SLURM template for multi-GPU runs.
