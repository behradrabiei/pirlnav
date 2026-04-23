# Important Notes — PIRLNav Scripting Gotchas

These issues recur whenever writing new scripts against this codebase.

---

## 1. Always activate the conda environment first

```bash
conda activate pirlnav
cd /root/Projects/World-Modelling/pirlnav
```

All scripts must run from the **repo root** so relative config paths (e.g. `configs/tasks/objectnav_mp3d.yaml`) resolve correctly.

---

## 2. Always `import pirlnav` before constructing a Habitat env

```python
import pirlnav  # noqa: F401
```

PIRLNav registers its custom task (`ObjectNav-v2`), dataset, sensors
(`DemonstrationSensor`, `InflectionWeightSensor`, etc.) and measurements
(`SparseReward`, etc.) via `@registry` decorators that only fire when the
modules are imported.  Without this line you get:

```
AssertionError: Could not find dataset ObjectNav-v2
```
or
```
AttributeError: SPARSE_REWARD
```

---

## 3. Use `pirlnav.config.get_task_config`, not `habitat.get_config`

```python
# WRONG — stock habitat schema doesn't know about SPARSE_REWARD etc.
config = habitat.get_config(config_paths="configs/tasks/objectnav_mp3d.yaml")

# CORRECT — PIRLNav-extended schema includes SPARSE_REWARD, SIMPLE_REWARD,
#            CACHED_DINOV2_SENSOR, and all other custom keys
from pirlnav.config import get_task_config
config = get_task_config(config_paths="configs/tasks/objectnav_mp3d.yaml")
```

The parameter is `config_paths` (plural) — `config_path` (singular) is silently
ignored by YACS (it's a `**kwargs` sink and won't raise, but also won't load
anything).

---

## 4. This is habitat-lab **0.2.2 YACS** — not the Hydra/OmegaConf API

The `pirlnav` conda env uses habitat-lab 0.2.2 with a YACS config system.
**None** of the following exist here:

| Hydra / newer API | YACS 0.2.2 equivalent |
|---|---|
| `habitat.config.read_write(cfg)` | `cfg.defrost()` … `cfg.freeze()` |
| `cfg.habitat.task.measurements` | `cfg.TASK.MEASUREMENTS` |
| `TopDownMapMeasurementConfig(...)` | set fields on `cfg.TASK.TOP_DOWN_MAP.*` |
| `habitat.get_config(config_path=...)` | `get_task_config(config_paths=...)` |

To add `TOP_DOWN_MAP` to a task config at runtime:

```python
config = get_task_config(config_paths="configs/tasks/objectnav_mp3d.yaml")
config.defrost()
config.TASK.MEASUREMENTS = list(config.TASK.MEASUREMENTS) + ["TOP_DOWN_MAP"]
config.TASK.TOP_DOWN_MAP.MAP_RESOLUTION = 512
config.TASK.TOP_DOWN_MAP.DRAW_SHORTEST_PATH = False
# ... other TOP_DOWN_MAP fields ...
config.freeze()
env = habitat.Env(config=config)
```

---

## 5. Dataset / config paths

| Asset | Path |
|---|---|
| MP3D 6-cat episode split | `data/datasets/objectnav/objectnav_mp3d/objectnav_mp3d_1scene_6cat/{split}/{split}.json.gz` |
| MP3D scene dataset config | `data/scene_datasets/mp3d/mp3d.scene_dataset_config.json` |
| OVRL ResNet-50 encoder | `data/visual_encoders/omnidata_DINO_02.pth` |
| Task config (MP3D) | `configs/tasks/objectnav_mp3d.yaml` |
| Experiment config (OVRL) | `configs/experiments/il_objectnav_mp3d.yaml` |

The scene is `17DRP5sb8fy`. Train split: 302 episodes, val split: 53 episodes.

---

## 6. `cv2.imshow` requires an X11 display

If running headlessly (SSH without X forwarding), skip `cv2.imshow` /
`cv2.waitKey` entirely.  Use `cv2.imwrite` to save frames and `input()` for
keyboard prompts instead.

---

## 7. Action names are **uppercase**

`env.step(action)` takes the canonical action name, which in this codebase is
uppercase:

```python
# WRONG — raises "Can't find 'turn_left' action"
env.step("turn_left")

# CORRECT
env.step("TURN_LEFT")
```

Valid names (from `configs/tasks/objectnav_mp3d.yaml`): `STOP`, `MOVE_FORWARD`,
`TURN_LEFT`, `TURN_RIGHT`, `LOOK_UP`, `LOOK_DOWN`.

---

## 8. TopDownMap fog-of-war makes the map look black at step 0

The default `TASK.TOP_DOWN_MAP.FOG_OF_WAR.DRAW = True` hides every tile the
agent hasn't seen yet, so before the agent moves, the entire map renders
black.  For a fully-visible reference map (useful for teleop / visualization),
disable it:

```python
config.TASK.TOP_DOWN_MAP.FOG_OF_WAR.DRAW = False
```

Keep it `True` only when you actually want to visualize agent exploration.

---

## 9. TopDownMap never draws goals — `hasattr(episode, "goal")` is always False

In `habitat-lab/habitat/tasks/nav/nav.py`, `TopDownMap.reset_metric` guards
all goal drawing (view points, AABB, positions, shortest path) behind:

```python
if hasattr(episode, "goal"):   # BUG — should be "goals" (plural)
```

`NavigationEpisode` only has `goals` (plural), so this condition is always
`False` and nothing is ever drawn.

**Fix** (already applied to the submodule):

```python
if hasattr(episode, "goals"):
```

After this fix and with `DRAW_SHORTEST_PATH = True`, the map shows:

| Color | What |
|---|---|
| Green line | Shortest path to goal (recomputed each step) |
| Red dot | Goal object centroid |
| Pink dots | View points (success positions within `SUCCESS_DISTANCE`) |
| Green rectangle | Object bounding box |
