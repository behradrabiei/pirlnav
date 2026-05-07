import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

TB_BASE = "tb/objectnav_il"

runs = {
    "DINOv2":                   f"{TB_BASE}/mp3d_1scene_6cat_dinov2_cached",
    "DINOv2 w/ Global Compass": f"{TB_BASE}/mp3d_1scene_6cat_dinov2_cached_gc_v1",
    "PIRLNav":                  f"{TB_BASE}/overfit_v1_noaug",
}

TAG = "losses/action_loss"
SMOOTH = 0.95  # EMA smoothing factor (0 = no smoothing, closer to 1 = more smoothing)


def ema(values, alpha):
    smoothed, last = [], values[0]
    for v in values:
        last = alpha * last + (1 - alpha) * v
        smoothed.append(last)
    return smoothed


fig, ax = plt.subplots(figsize=(10, 5))

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

for (label, path), color in zip(runs.items(), colors):
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    events = ea.Scalars(TAG)
    steps  = [e.step  for e in events]
    values = [e.value for e in events]
    half   = len(steps) // 5
    steps, values = steps[:half], values[:half]
    smooth = ema(values, SMOOTH)

    ax.plot(steps, values,  color=color, alpha=0.15, linewidth=1.5)
    ax.plot(steps, smooth,  color=color, alpha=1.0,  linewidth=3.0, label=label)

ax.set_xlabel("Step", fontsize=16)
ax.set_ylabel("Action Loss", fontsize=16)
ax.tick_params(axis="both", labelsize=14)
ax.legend(fontsize=14)
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
fig.tight_layout()
fig.savefig("loss_curves.png", dpi=150)
plt.show()
print("Saved loss_curves.png")
