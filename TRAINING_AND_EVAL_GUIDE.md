# PIRLNav — Training & Evaluation Guide

This guide covers how to train the IL agent, evaluate a checkpoint, and interpret
the results.  All commands assume the `pirlnav` conda environment is active and the
working directory is the repo root (`/root/Projects/World-Modelling/pirlnav`).

---

## 1. Dataset & Prerequisites

| Asset | Path |
|---|---|
| MP3D 1-scene 6-category dataset | `data/datasets/objectnav/objectnav_mp3d/objectnav_mp3d_1scene_6cat/` |
| OVRL ResNet-50 pretrained encoder | `data/visual_encoders/omnidata_DINO_02.pth` |
| DINOv2 precomputed feature cache | `data/dinov2_cache/17DRP5sb8fy/<episode_id>.pt` |
| Scene mesh / navmesh | `data/scene_datasets/mp3d/17DRP5sb8fy/` |

The dataset has two splits:

- **train** — 302 episodes (used during training and overfitting check at eval)
- **val** — 53 held-out episodes from the same scene (generalization check)

---

## 2. Training

### 2a. OVRL ResNet-50 variant

**Launcher:** `scripts/run_il_mp3d_1scene.sh`

```bash
# Smoke run (200 updates, 2 envs) — quick sanity check
bash scripts/run_il_mp3d_1scene.sh

# Full run (20 000 updates, 4 envs) — what we use for real experiments
bash scripts/run_il_mp3d_1scene.sh --full
```

**Environment-variable overrides** (prefix any of these before the command):

| Variable | Default | Meaning |
|---|---|---|
| `NUM_UPDATES` | 200 (smoke) / 20000 (full) | Total gradient updates |
| `NUM_ENVIRONMENTS` | 2 (smoke) / 4 (full) | Parallel Habitat-Sim environments |
| `NUM_CHECKPOINTS` | 10 | Number of checkpoints saved evenly across training |
| `TAG` | `mp3d_1scene_6cat_smoke` | Name tag; controls where logs/ckpts land |
| `INFLECTION_COEF` | 3.2349… | Weight on action-change steps in inflection loss |

**Output locations:**

- Checkpoints: `data/new_checkpoints/objectnav_il/<TAG>/ckpt.{0…N}.pth`
- TensorBoard: `tb/objectnav_il/<TAG>/`

**Experiment config:** `configs/experiments/il_objectnav_mp3d.yaml`
(ResNet-50 backbone, pretrained OVRL weights, augmentations **on** by default)

---

### 2b. OVRL ResNet-50 — no augmentation variant

**Launcher:** `scripts/run_il_mp3d_1scene_noaug.sh`

Identical to 2a except it always passes:
```
POLICY.RGB_ENCODER.use_augmentations False
POLICY.RGB_ENCODER.use_augmentations_test_time False
```

```bash
bash scripts/run_il_mp3d_1scene_noaug.sh --full

# Custom tag to keep separate from the augmented run
TAG=overfit_v1_noaug bash scripts/run_il_mp3d_1scene_noaug.sh --full
```

Output lands at `data/new_checkpoints/objectnav_il/overfit_v1_noaug/`.

---

### 2c. DINOv2-cached variant

**Step 1 — Precompute features** (one-time, ~5 min):

```bash
python scripts/precompute_dinov2_features.py \
  --config configs/experiments/il_objectnav_mp3d_dinov2_cached.yaml \
  --split train \
  --out-dir data/dinov2_cache
```

This runs the frozen `facebook/dinov2-base` ViT over every training episode's
RGB frames (center-crop 476×630, ImageNet normalisation) and saves one
`<episode_id>.pt` tensor file per episode.

**Step 2 — Train:**

**Launcher:** `scripts/run_il_mp3d_1scene_dinov2_cached.sh`

```bash
bash scripts/run_il_mp3d_1scene_dinov2_cached.sh --full
```

Additional override:

| Variable | Default | Meaning |
|---|---|---|
| `CACHE_ROOT` | `data/dinov2_cache` | Root directory of precomputed `.pt` files |

**Experiment config:** `configs/experiments/il_objectnav_mp3d_dinov2_cached.yaml`

The policy reads precomputed CLS tokens during training (no augmentation —
features were computed from plain center-crops).  The DINOv2 backbone weights
are **frozen** and **not** saved into checkpoints.

Output lands at
`data/new_checkpoints_dinov2_cached/objectnav_il/mp3d_1scene_6cat_dinov2_cached/`.

---

### 2d. DINOv2-cached + goal-compass variant

This variant adds an optional **12-bin oracle goal-direction compass** on top
of the cached-DINOv2 pipeline.  At every step the `GoalCompassSensor` reads
the agent pose and `episode.goals`, and emits a 12-D rectified,
distance-weighted cosine vector (same math as `global_test.py`).  A
`Linear(12, 32)` embedding is concatenated to the RNN input alongside the
existing GPS / compass / objectgoal streams.

The toggle is YAML-only: the sensor is listed in `TASK.SENSORS` of
`configs/tasks/objectnav_mp3d_cached_goalcompass.yaml`; the policy
auto-detects it via the observation space and builds the branch.  Removing
the sensor from the task yaml disables the whole thing, and the existing
cached-DINOv2 checkpoint remains load-compatible with the original config.

**Step 1 — Precompute DINOv2 features** (same as 2c, skip if already done):

```bash
python scripts/precompute_dinov2_features.py \
  --config configs/experiments/il_objectnav_mp3d_dinov2_cached.yaml \
  --split train \
  --out-dir data/dinov2_cache
```

**Step 2 — Train:**

**Launcher:** `scripts/run_il_mp3d_1scene_dinov2_cached_goalcompass.sh`

```bash
bash scripts/run_il_mp3d_1scene_dinov2_cached_goalcompass.sh --full
```

**Experiment config:** `configs/experiments/il_objectnav_mp3d_dinov2_cached_goalcompass.yaml`

Output lands at
`data/new_checkpoints_dinov2_cached_gc/objectnav_il/mp3d_1scene_6cat_dinov2_cached_gc/`.

**Eval:**

```bash
ALLOW_SLIDING=False SUCCESS_DISTANCE=1.0 \
  NUM_ENVIRONMENTS=1 VIDEO_ENABLED=false \
  bash scripts/eval_il_mp3d_1scene_full.sh configs/eval_dinov2_cached_goalcompass.env
```

The eval env file points to the **online-DINOv2 + goal-compass** twin config
(`configs/experiments/il_objectnav_mp3d_dinov2_goalcompass.yaml`) so val
episodes without precomputed features still run; the goal-compass branch is
identical in both configs.

---

### 2e. Monitoring training with TensorBoard

All three run variants write to `tb/objectnav_il/<TAG>/`.  Point TensorBoard at
the parent to overlay all runs on the same plots:

```bash
tensorboard --logdir tb/objectnav_il --port 6006 --bind_all
```

Key scalar panels:

| Panel | What to look for |
|---|---|
| `losses/action_loss` | Should approach 0 on train split within a few k updates |
| `metrics/success` | Train success (teacher-forced) should rise towards ~0.9+ |
| `perf/fps` | Steps per second — DINOv2 cached is ~10× faster than online |

---

## 3. Evaluation

### 3a. How the eval launcher works

**Script:** `scripts/eval_il_mp3d_1scene_full.sh`

```bash
bash scripts/eval_il_mp3d_1scene_full.sh [env-file]
```

The script sources an **env file** (default `configs/eval_overfit.env`) that sets
all knobs, then runs the eval loop on each split in `EVAL_SPLITS` back-to-back.
Any variable exported before the command takes priority over the env-file default
(`:=` semantics).

---

### 3b. Eval env files

| Env file | Model |
|---|---|
| `configs/eval_overfit.env` | OVRL ResNet-50 (any checkpoint) |
| `configs/eval_dinov2_cached.env` | DINOv2-cached (runs online DINOv2 at eval time) |

#### Variables common to both env files

| Variable | Default (OVRL) | Meaning |
|---|---|---|
| `EVAL_CKPT` | `…/overfit_v1/ckpt.9.pth` | Path to the `.pth` checkpoint |
| `EVAL_SPLITS` | `train val` | Space-separated splits to evaluate |
| `TEST_EPISODE_COUNT` | `-1` | Episodes per split; `-1` = all |
| `NUM_ENVIRONMENTS` | `1` | Parallel envs. **Keep at 1** to avoid episode skipping |
| `SUCCESS_DISTANCE` | `1.0` | Success radius in metres (must match demo-recording setting) |
| `ALLOW_SLIDING` | `True` (OVRL) / `True` (DINOv2) | Collision-sliding behaviour (must match demo-recording) |
| `VIDEO_ENABLED` | `false` | Write mp4 videos to disk |
| `VIDEO_FAILED_ONLY` | `false` | Only record failed episodes |
| `VIDEO_FPS` | `10` | Frames per second for saved videos |
| `VIDEO_RENDER_TOP_DOWN` | `true` | Overlay top-down map on each video frame |
| `OUT_ROOT` | `data/eval_out/overfit_v1_ckpt9` | Output root; sub-dirs created automatically |

#### `EXTRA_OPTS` — pass arbitrary YACS overrides

Any YACS key-value pair can be appended via `EXTRA_OPTS`, which is forwarded
directly to the Python `run` module.  Pairs must be space-separated (no quotes
around the whole string).

```bash
# Disable test-time augmentation for OVRL
EXTRA_OPTS="POLICY.RGB_ENCODER.use_augmentations_test_time False" \
  bash scripts/eval_il_mp3d_1scene_full.sh configs/eval_overfit.env

# Enable test-time augmentation for DINOv2
EXTRA_OPTS="POLICY.RGB_ENCODER.use_augmentations_test_time True" \
  bash scripts/eval_il_mp3d_1scene_full.sh configs/eval_dinov2_cached.env
```

---

### 3c. Canonical eval commands for each trained model

All commands below use:
- `NUM_ENVIRONMENTS=1` (avoids episode skipping)
- `VIDEO_ENABLED=false` (fastest)
- `ALLOW_SLIDING=False`, `SUCCESS_DISTANCE=1.0` (matches demonstration recording)

**OVRL ckpt.9 (trained with augmentation) — aug ON at eval:**

```bash
EXTRA_OPTS="POLICY.RGB_ENCODER.use_augmentations_test_time True" \
  ALLOW_SLIDING=False SUCCESS_DISTANCE=1.0 \
  NUM_ENVIRONMENTS=1 VIDEO_ENABLED=false \
  EVAL_CKPT=data/new_checkpoints/objectnav_il/overfit_v1/ckpt.9.pth \
  OUT_ROOT=data/eval_out/ovrl_ckpt9_aug_seeded \
  bash scripts/eval_il_mp3d_1scene_full.sh configs/eval_overfit.env
```

**OVRL ckpt.9 (trained with augmentation) — aug OFF at eval:**

```bash
EXTRA_OPTS="POLICY.RGB_ENCODER.use_augmentations_test_time False" \
  ALLOW_SLIDING=False SUCCESS_DISTANCE=1.0 \
  NUM_ENVIRONMENTS=1 VIDEO_ENABLED=false \
  EVAL_CKPT=data/new_checkpoints/objectnav_il/overfit_v1/ckpt.9.pth \
  OUT_ROOT=data/eval_out/ovrl_ckpt9_noaug \
  bash scripts/eval_il_mp3d_1scene_full.sh configs/eval_overfit.env
```

**OVRL ckpt.10 (trained WITHOUT augmentation) — aug ON at eval:**

```bash
EXTRA_OPTS="POLICY.RGB_ENCODER.use_augmentations_test_time True" \
  ALLOW_SLIDING=False SUCCESS_DISTANCE=1.0 \
  NUM_ENVIRONMENTS=1 VIDEO_ENABLED=false \
  EVAL_CKPT=data/new_checkpoints/objectnav_il/overfit_v1_noaug/ckpt.10.pth \
  OUT_ROOT=data/eval_out/ovrl_noaug_ckpt10_aug_seeded \
  bash scripts/eval_il_mp3d_1scene_full.sh configs/eval_overfit.env
```

**OVRL ckpt.10 (trained WITHOUT augmentation) — aug OFF at eval:**

```bash
EXTRA_OPTS="POLICY.RGB_ENCODER.use_augmentations_test_time False" \
  ALLOW_SLIDING=False SUCCESS_DISTANCE=1.0 \
  NUM_ENVIRONMENTS=1 VIDEO_ENABLED=false \
  EVAL_CKPT=data/new_checkpoints/objectnav_il/overfit_v1_noaug/ckpt.10.pth \
  OUT_ROOT=data/eval_out/ovrl_noaug_ckpt10_noaug \
  bash scripts/eval_il_mp3d_1scene_full.sh configs/eval_overfit.env
```

**DINOv2-cached ckpt.10 — aug OFF at eval (matched training condition):**

```bash
ALLOW_SLIDING=False SUCCESS_DISTANCE=1.0 \
  NUM_ENVIRONMENTS=1 VIDEO_ENABLED=false \
  OUT_ROOT=data/eval_out/dinov2_cached_ckpt10_noaug \
  bash scripts/eval_il_mp3d_1scene_full.sh configs/eval_dinov2_cached.env
```

**DINOv2-cached ckpt.10 — aug ON at eval (seeded):**

```bash
EXTRA_OPTS="POLICY.RGB_ENCODER.use_augmentations_test_time True" \
  ALLOW_SLIDING=False SUCCESS_DISTANCE=1.0 \
  NUM_ENVIRONMENTS=1 VIDEO_ENABLED=false \
  OUT_ROOT=data/eval_out/dinov2_cached_ckpt10_aug_seeded \
  bash scripts/eval_il_mp3d_1scene_full.sh configs/eval_dinov2_cached.env
```

> **Note on DINOv2 eval:** The cached training pipeline does not save DINOv2
> backbone weights into the checkpoint (they are frozen).  At eval time the
> `eval_dinov2_cached.env` uses the **online** DINOv2 config
> (`configs/experiments/il_objectnav_mp3d_dinov2.yaml`), which re-loads the
> frozen backbone from HuggingFace and runs it live on each step.  The
> preprocessing is identical to what built the cache (center-crop 476×630,
> ImageNet normalisation), so this is a faithful replica of the training
> distribution.

---

## 4. Reading the Results

### 4a. Output directory layout

```
data/eval_out/<run_name>/
├── summary.tsv          ← tab-separated table: one row per split
├── train/
│   ├── eval.log         ← full trainer log with per-episode progress + final averages
│   ├── tb/              ← TensorBoard scalars for this split
│   └── videos/          ← mp4 files (only if VIDEO_ENABLED=true)
└── val/
    ├── eval.log
    ├── tb/
    └── videos/
```

### 4b. summary.tsv columns

```
split   success   spl   softspl   dist_to_goal   infer_ms_per_step   n_episodes
```

| Column | Meaning |
|---|---|
| `split` | `train` or `val` |
| `success` | Fraction of episodes where the agent stopped within `SUCCESS_DISTANCE` of the goal |
| `spl` | Success weighted by Path Length — penalises taking longer paths than optimal |
| `softspl` | Soft SPL — partial credit even for near-successes |
| `dist_to_goal` | Average final distance to goal (metres); lower is better even on failures |
| `infer_ms_per_step` | Average wall-clock time per policy step in milliseconds |
| `n_episodes` | Total episodes evaluated (302 for train, 53 for val) |

### 4c. How to print a formatted table

```bash
column -t -s$'\t' data/eval_out/<run_name>/summary.tsv
```

### 4d. Finding the raw per-episode averages

```bash
grep "Average episode" data/eval_out/<run_name>/train/eval.log
```

---

## 5. Results Summary

All evals: `ALLOW_SLIDING=False`, `SUCCESS_DISTANCE=1.0 m`, `NUM_ENVIRONMENTS=1`,
seeded RNGs (`il_trainer.py` seeds `random`, `numpy`, and `torch` from
`TASK_CONFIG.SEED` at eval start for reproducible augmentation).

### Train split (302 episodes — overfitting check)

| Model | Trained aug | Eval aug | Success | SPL | Infer ms/step |
|---|---|---|---|---|---|
| OVRL ResNet-50 ckpt.9 | yes | on  | 67.22% | 35.95% | 3.7 |
| OVRL ResNet-50 ckpt.9 | yes | off | 67.55% | 36.10% | 3.0 |
| OVRL ResNet-50 ckpt.10 (no-aug) | no | on  | 20.53% | 7.94%  | 3.8 |
| OVRL ResNet-50 ckpt.10 (no-aug) | no | off | 67.55% | 36.10% | 3.0 |
| DINOv2-cached ckpt.10 | no | on  | 45.70% | 22.16% | 23.8 |
| DINOv2-cached ckpt.10 | no | off | 67.55% | 36.10% | 23.3 |

### Val split (53 held-out episodes — generalisation check)

| Model | Trained aug | Eval aug | Success | SPL | Infer ms/step |
|---|---|---|---|---|---|
| OVRL ResNet-50 ckpt.9 | yes | on  | 37.74% | 20.14% | 3.9 |
| OVRL ResNet-50 ckpt.9 | yes | off | **45.28%** | **24.37%** | 3.1 |
| OVRL ResNet-50 ckpt.10 (no-aug) | no | on  | 22.64% | 7.05%  | 3.8 |
| OVRL ResNet-50 ckpt.10 (no-aug) | no | off | 26.42% | 14.76% | 3.0 |
| DINOv2-cached ckpt.10 | no | on  | 26.42% | 11.96% | 23.9 |
| DINOv2-cached ckpt.10 | no | off | **32.08%** | **14.75%** | 23.3 |

### Key takeaways

1. **Matched comparison (no-aug trained, no-aug eval):** DINOv2 (32.1%) vs OVRL
   (26.4%) — DINOv2 generalises better by +5.7 points despite identical train
   success, suggesting the frozen ViT features are more transferable.

2. **Augmentation matters for OVRL:** The aug-trained OVRL ckpt.9 evaluated
   without aug reaches 45.3% val success — the best result overall.
   Augmentation during training acts as regularisation for the ResNet-50.

3. **Augmentation hurts a model that wasn't trained with it:** OVRL-noaug with
   aug-on at eval collapses to 22.6% train success, confirming the distribution
   mismatch is harmful.

4. **Inference speed:** OVRL (~3 ms/step) is ~8× faster than DINOv2 online
   (~23 ms/step) at eval time.  The cached training pipeline avoids this overhead
   during training only; eval always runs the ViT live.

5. **Train numbers are identical across no-aug models:** Both OVRL-noaug and
   DINOv2-noaug score exactly 67.55% / 36.10% on the train split because both
   have effectively memorised the 302 expert trajectories (BC loss ≈ 0).
   The val split is where backbone quality separates them.

---

## 6. Important Configuration Notes

### SUCCESS_DISTANCE and ALLOW_SLIDING

The MP3D demonstrations were recorded with `SUCCESS_DISTANCE=1.0 m` and
`ALLOW_SLIDING=True`.  The upstream PIRLNav YAML defaults to `SUCCESS_DISTANCE=0.1`
and `ALLOW_SLIDING=False`, which causes the expert replay itself to fail ~75%
of the time.  **Always override these to `1.0` / `False` (sliding off in eval)
for meaningful numbers.**

### NUM_ENVIRONMENTS=1 during eval

Using `NUM_ENVIRONMENTS > 1` causes the Habitat `VectorEnv` episode iterator to
silently drop or repeat episodes at split boundaries, inflating the episode count
to `N × num_envs`.  Always evaluate with `NUM_ENVIRONMENTS=1`.

### Reproducibility (seeded eval)

`il_trainer.py` seeds `random`, `numpy.random`, `torch`, and `torch.cuda` at the
start of `_eval_checkpoint` using `TASK_CONFIG.SEED`.  This ensures
`RandomShiftsAug` and `ColorJitter` produce the same augmentations across
repeated runs when `use_augmentations_test_time=True`.
