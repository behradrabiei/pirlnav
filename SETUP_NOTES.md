# PIRLNav — MP3D Imitation Learning Setup Notes

End-to-end record of everything that was done to get upstream
[PIRLNav](https://github.com/Ram81/pirlnav) running ObjectNav **imitation
learning** on a single MP3D training scene on an RTX 5090, including every
pitfall and the fix we landed on. Written so the next person can reproduce or
extend this without re-debugging.

---

## 1. Goal

Run PIRLNav's IL trainer on **MP3D** (not HM3D), restricted to **one training
scene** and the **6 HM3D ObjectNav categories** (`chair, bed, plant, toilet,
tv_monitor, sofa`), using the OVRL DINO visual encoder as initialization.
Pretrained IL checkpoint (`objectnav_il_hd.ckpt`) is intentionally skipped —
upstream S3 bucket is gone.

Hardware: single RTX 5090 (Blackwell). This forces PyTorch ≥ 2.7 + CUDA 12.8.

---

## 2. Environment

A dedicated conda env `pirlnav` is used so the user's existing `vln` env (newer
habitat-lab 0.3.x with Hydra config) is **not touched**.

| Component | Version | Source |
| --- | --- | --- |
| Python | 3.9 | conda |
| habitat-sim | 0.2.5 | conda-forge |
| habitat-lab | 0.2.2 | the `habitat-lab/` submodule in this repo |
| habitat-baselines | 0.2.2 | also the submodule (installed from `habitat-lab/habitat_baselines`) |
| PyTorch | 2.7 + cu128 | pip (Blackwell support) |
| tensorboard / moviepy / ifcfg | latest | pip |
| webdataset | **< 0.2** | pip (see pitfall) |

### Why this combination

- PIRLNav's Python code (config schema, `ObjectNavDatasetV2`, IL trainer) is
  locked to the **YACS**-based habitat-lab 0.2.2 API. Newer habitat-lab uses
  Hydra/OmegaConf and the imports `_C`, `Config`, etc. don't exist anymore.
  So we must use the 0.2.2 submodule.
- habitat-sim 0.2.2's prebuilt wheel doesn't install on Blackwell GPUs. The
  conda-forge 0.2.5 wheel does. Mixing 0.2.5 sim with 0.2.2 lab is mostly fine
  **except for the pitfalls below**.

---

## 3. Data layout

Symlinks inside the repo point at the user's existing MP3D data on `/data`:

```
data/
├── scene_datasets/
│   └── mp3d -> /data/hm3d_datasets/MP3D/v1/tasks/mp3d_habitat/mp3d
├── datasets/
│   └── objectnav/objectnav_mp3d/objectnav_mp3d_1scene_6cat/   # built by our script
├── visual_encoders/
│   └── omnidata_DINO_02.pth   # from huggingface.co/gunjan050/ZSON
└── new_checkpoints/           # training output
```

The MP3D scene dataset config
(`data/scene_datasets/mp3d/mp3d.scene_dataset_config.json`) is critical —
habitat-sim 0.2.5 asserts on an empty string (see pitfall §5.3).

---

## 4. Files we added / modified

### Added

- `scripts/make_mp3d_1scene_6cat_subset.py` — builds the 1-scene 6-class MP3D
  subset from the THDA 70k demonstrations, remaps task IDs, stamps
  `scene_dataset_config` onto every episode.
- `configs/tasks/objectnav_mp3d.yaml` — MP3D task config.
- `configs/experiments/il_objectnav_mp3d.yaml` — IL experiment config.
- `scripts/run_il_mp3d_1scene.sh` — non-SLURM single-GPU launcher
  (smoke / full modes, env-var overrides).
- `scripts/eval_il_mp3d_1scene.sh` — single-GPU eval launcher.

### Patched in submodule (`habitat-lab/`)

- `habitat-lab/habitat/tasks/rearrange/__init__.py` — `_try_register_rearrange_task`
  wrapped in `try/except (ImportError, ModuleNotFoundError)` so the rearrange
  task registration becomes optional.
- `habitat-lab/habitat/tasks/rearrange/rearrange_sim.py` — `from
  habitat_sim.robots import FetchRobot, FetchRobotNoWheels` wrapped in
  `try/except`, with `FetchRobot = FetchRobotNoWheels = None` as fallbacks.

### Patched in pirlnav

- `pirlnav/utils/utils.py` — `torch.load(..., weights_only=False)` for
  PyTorch 2.6+ compatibility with the DINO checkpoint.
- `pirlnav/algos/agent.py` — `DecentralizedDistributedMixin` got sane defaults
  (`find_unused_params = False`, `reducer = None`), and `before_backward` now
  short-circuits when `reducer is None`. Required because single-GPU runs
  never call `init_distributed`.

Nothing in the `vln` conda env is touched.

---

## 5. Pitfalls encountered and fixes

### 5.1 `ImportError: _C from habitat.config.default` / `No module named 'habitat_baselines'`

**Cause.** Tried to use the user's `vln` env (habitat-lab 0.3.3, Hydra).
PIRLNav hard-imports the YACS `_C` object and `habitat_baselines` as a
separate package — neither exists there.

**Fix.** New `pirlnav` conda env, install habitat-lab 0.2.2 **and**
`habitat_baselines` from the repo's submodule. Left `vln` untouched.

### 5.2 `ModuleNotFoundError: No module named 'habitat_sim.robots'` at `import habitat`

**Cause.** habitat-lab 0.2.2 auto-registers its `rearrange` task on import,
and that task imports `habitat_sim.robots` — a module that was only added
after 0.2.2 and then removed/renamed again by 0.2.5.

**Fix.** Wrap the two offending import sites in `try/except`. PIRLNav doesn't
use rearrange at all, so silently skipping the registration is safe.

### 5.3 `ESP_CHECK failed: Scene Dataset "" does not exist`

**Cause (two separate bugs stacked).**

1. The upstream `configs/tasks/objectnav_hm3d.yaml` declares `SCENE_DATASET`
   as a **top-level** key. YACS accepts it (the config schema has
   `new_allowed=True`) but habitat-lab actually reads
   `SIMULATOR.SCENE_DATASET`. The top-level key was a no-op in 0.2.2 and is
   still a no-op; the simulator was simply picking up its default of
   `"default"`, which habitat-sim 0.2.2 tolerated.
2. habitat-sim 0.2.5 got stricter and now asserts unless a valid
   `scene_dataset_config_file` is given.
3. Even worse: `habitat/core/env.py` overwrites `SIMULATOR.SCENE_DATASET =
   self.current_episode.scene_dataset_config` on every episode reset. If the
   episode JSON doesn't carry that field, the simulator config gets clobbered
   with `""`.

**Fix.**

- `configs/tasks/objectnav_mp3d.yaml` puts `SCENE_DATASET` under `SIMULATOR:`
  where it is actually consumed.
- `make_mp3d_1scene_6cat_subset.py` writes
  `ep["scene_dataset_config"] = "data/scene_datasets/mp3d/mp3d.scene_dataset_config.json"`
  onto every episode so the per-episode override stays valid.

### 5.4 Category-name filter dropped `tv_monitor`

**Cause.** First cut of `make_mp3d_1scene_6cat_subset.py` did
`key.rsplit("_", 1)` on goal-table keys to peel off the category. That splits
`17DRP5sb8fy.glb_tv_monitor` into `(..._tv, monitor)` → `monitor` ∉ target
classes → dropped. `chest_of_drawers` has the same disease.

**Fix.** Iterate over the target category set and use
`key.endswith(f"_{cat}")` instead of string-splitting. Rebuild shows
`5 goals_by_category entries kept` and `tv_monitor: 72 episodes`.

### 5.5 HM3D vs MP3D category count mismatch

**Cause.** MP3D's ObjectNav ships **21 categories**; HM3D ships **6**. The
PIRLNav policy's `obj_categories_embedding` is sized by
`category_to_task_category_id`, and the pretrained weights expect 6.

**Fix.** The subset script writes a fresh top-level
`category_to_task_category_id = {chair:0, bed:1, plant:2, toilet:3,
tv_monitor:4, sofa:5}`, drops all other episodes, and the MP3D strings for
the six classes already match HM3D's, so no string remap is needed. Net
result: 355 episodes in `17DRP5sb8fy` across 5 present classes (no `plant`
exists in this scene).

### 5.6 `webdataset.Dataset` attribute error

**Cause.** Modern `webdataset` (≥ 0.2) removed the `Dataset` class;
habitat-baselines 0.2.2 still imports it.

**Fix.** `pip install 'webdataset<0.2'`.

### 5.7 `torch.load` weights-only failure on DINO checkpoint

**Cause.** PyTorch 2.6 flipped the default of `weights_only` to `True`, which
refuses to unpickle the numpy scalars embedded in the OVRL DINO state dict.

**Fix.** In `pirlnav/utils/utils.py`:
```python
state_dict = torch.load(path, map_location="cpu", weights_only=False)["teacher"]
```

### 5.8 `num_environments < num_mini_batch`

**Cause.** PIRLNav's `RolloutStorage.recurrent_generator` asserts
`num_envs >= num_mini_batch`. Config default is `num_mini_batch = 2`, so a
`NUM_ENVIRONMENTS=1` smoke run crashes.

**Fix.** Use at least `NUM_ENVIRONMENTS=2`. The launcher's smoke default was
raised accordingly.

### 5.9 `AttributeError: 'DDPILAgent' object has no attribute 'find_unused_params'`

**Cause.** `DDPILAgent` inherits from `DecentralizedDistributedMixin` whose
`before_backward` reads `self.find_unused_params` and `self.reducer`. Those
are only set inside `init_distributed(...)`, which is only called when DDP is
active. Single-GPU non-SLURM training never calls it, so the attributes are
missing and `before_backward` crashes on the first update.

**Fix.** In `pirlnav/algos/agent.py`:

```python
class DecentralizedDistributedMixin:
    find_unused_params: bool = False   # class-level default
    reducer = None

    def before_backward(self, loss):
        super().before_backward(loss)
        if self.reducer is None:
            return
        ...
```

### 5.10 Missing pretrained IL checkpoint

**Cause.** `habitat-on-web.s3.amazonaws.com/pirlnav_release/...` is **gone**
(S3 `NoSuchBucket`). `objectnav_il_hd.ckpt` has no public mirror.

**Fix.** Skipped. We train IL from the OVRL-initialized encoder instead. The
OVRL weights `omnidata_DINO_02.pth` were recovered from
`huggingface.co/gunjan050/ZSON`.

---

## 6. What the subset script does

`scripts/make_mp3d_1scene_6cat_subset.py --scene <scene_id>`

1. Loads upstream THDA 70k train: `train.json.gz` and `content/<scene>.json.gz`.
2. Keeps only episodes whose `object_category` is in the 6 HM3D classes.
3. Stamps each kept episode with `scene_dataset_config` pointing at
   `data/scene_datasets/mp3d/mp3d.scene_dataset_config.json`.
4. Filters `goals_by_category` by **suffix match** (handles `tv_monitor`,
   `chest_of_drawers`, etc. correctly).
5. Rewrites `category_to_task_category_id` as the canonical HM3D 0..5 map
   and projects `category_to_mp3d_category_id` to only the 6 classes.
6. Writes:
   - `data/datasets/objectnav/objectnav_mp3d/objectnav_mp3d_1scene_6cat/train/train.json.gz`
   - `data/datasets/objectnav/objectnav_mp3d/objectnav_mp3d_1scene_6cat/train/content/<scene>.json.gz`

Verified output for `17DRP5sb8fy`: 355 episodes across `{chair:68, bed:78,
sofa:67, toilet:70, tv_monitor:72}` (plant absent in this scene), 5 goal-
category entries.

---

## 7. Configs summary

`configs/tasks/objectnav_mp3d.yaml` highlights:

- `SIMULATOR.SCENE_DATASET: "data/scene_datasets/mp3d/mp3d.scene_dataset_config.json"`
- Sensors: RGB (640×480, HFOV 79°) + Compass + GPS + ObjectGoal + Demonstration + InflectionWeight.
- `TASK.TYPE: ObjectNav-v2`, `DATASET.TYPE: ObjectNav-v2`.
- Measurements: `DISTANCE_TO_GOAL`, `SUCCESS`, `SPL`, `SOFT_SPL`, `SPARSE_REWARD`.

`configs/experiments/il_objectnav_mp3d.yaml` highlights:

- `TRAINER_NAME: pirlnav-il`
- `POLICY.RGB_ENCODER.backbone: resnet50` + `pretrained_encoder: data/visual_encoders/omnidata_DINO_02.pth`
- `IL.POLICY.name: ObjectNavILMAEPolicy`
- Checkpoint / tensorboard dirs overridable at CLI.

---

## 8. How to run

### Build the subset

```bash
conda activate pirlnav
cd /root/Projects/World-Modelling/pirlnav
python scripts/make_mp3d_1scene_6cat_subset.py --scene 17DRP5sb8fy
```

### Smoke test (~10 s end-to-end)

```bash
TAG=smoke NUM_UPDATES=3 NUM_ENVIRONMENTS=2 bash scripts/run_il_mp3d_1scene.sh
```

### Full IL run

```bash
TAG=mp3d_1scene_v1 NUM_UPDATES=20000 NUM_ENVIRONMENTS=4 \
  bash scripts/run_il_mp3d_1scene.sh --full
tensorboard --logdir tb/objectnav_il/mp3d_1scene_v1
```

### Evaluation

```bash
bash scripts/eval_il_mp3d_1scene.sh data/new_checkpoints/objectnav_il/mp3d_1scene_v1/ckpt.9.pth
```

---

## 9. Smoke-test evidence

With `NUM_UPDATES=3`, `NUM_ENVIRONMENTS=2`, OVRL encoder initialized from
`data/visual_encoders/omnidata_DINO_02.pth`:

| Update | action_loss | entropy | SPL | success |
| --- | --- | --- | --- | --- |
| 0 | 1.77 | 1.79 | 0.00 | 0.0 |
| 1 | 1.20 | 1.18 | 0.63 | 1.0 |
| 2 | 1.03 | 0.80 | 0.31 | 0.5 |

~217 FPS on the 5090 with 2 envs. Checkpoints land in
`data/new_checkpoints/objectnav_il/<TAG>/ckpt.*.pth`. Tensorboard scalars in
`tb/objectnav_il/<TAG>/`.

---

## 10. Known remaining caveats

- **Plant class has zero episodes in `17DRP5sb8fy`.** Policy will never see
  that class. If you extend to another scene, make sure the class mix is
  what you want.
- **No pretrained IL checkpoint.** Training from OVRL-initialized encoder
  only; convergence will be slower than the paper's numbers. If a mirror
  surfaces, just drop it anywhere and pass it via `--checkpoint` to the
  trainer (or set `RL.DDPPO.pretrained_weights` in the experiment YAML).
- **Single-scene IL.** Great for sanity / architecture work, useless for
  generalization benchmarks. Extending to more scenes is just calling the
  subset script per scene and concatenating `content/`.
- **Deprecation spam.** `Gym has been unmaintained since 2022 ...` prints on
  every env startup. Harmless — habitat-lab 0.2.2 still uses `gym`.
- **NumPy 2.x** works, but some habitat-baselines code emits deprecation
  warnings. No functional impact so far.

---

## 11. Where to plug in a new architecture

The policy class is `ObjectNavILMAENet` in
`pirlnav/policy/visual_policy.py`. It's registered via
`@baseline_registry.register_policy(name="ObjectNavILMAEPolicy")`. To swap:

1. Write a new `class MyPolicy(Policy)` in the same file (or a new one that
   gets imported).
2. Decorate with `@baseline_registry.register_policy(name="MyPolicy")`.
3. Change `IL.POLICY.name: "MyPolicy"` in
   `configs/experiments/il_objectnav_mp3d.yaml`.

The IL trainer and rollout storage are architecture-agnostic, so no other
changes are needed as long as the policy exposes the usual
`act / evaluate_actions / build_distribution` surface.
