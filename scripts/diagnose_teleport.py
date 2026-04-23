"""
Teleport-to-final-pose diagnostic.

For each episode: reset the env, teleport the agent to the LAST recorded
agent_state from reference_replay, then call STOP.  This isolates the
question 'are the recorded demo endpoints themselves successful under our
current task config?' from 'does our sim reach those endpoints when we
replay the actions?'.
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import gzip, json
import habitat  # noqa
from habitat.config.default import get_config as get_task_config
from habitat.core.env import Env
import pirlnav  # noqa


def load_raw(path):
    with gzip.open(path, "rt") as f:
        d = json.load(f)
    return {ep["episode_id"]: ep for ep in d["episodes"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--success-distance", type=float, default=1.0)
    ap.add_argument("--allow-sliding", dest="allow_sliding",
                    action="store_true")
    ap.add_argument("--no-allow-sliding", dest="allow_sliding",
                    action="store_false")
    ap.set_defaults(allow_sliding=False)
    args = ap.parse_args()

    cfg = get_task_config("configs/tasks/objectnav_mp3d.yaml")
    cfg.defrost()
    cfg.DATASET.SPLIT = args.split
    cfg.TASK.MEASUREMENTS = [
        m for m in cfg.TASK.MEASUREMENTS
        if m in {"DISTANCE_TO_GOAL", "SUCCESS", "SPL", "SOFT_SPL"}
    ]
    cfg.TASK.SUCCESS.SUCCESS_DISTANCE = args.success_distance
    cfg.TASK.SUCCESS_DISTANCE = args.success_distance
    cfg.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING = args.allow_sliding
    cfg.freeze()

    path = (
        f"data/datasets/objectnav/objectnav_mp3d/objectnav_mp3d_1scene_6cat/"
        f"{args.split}/content/17DRP5sb8fy.json.gz"
    )
    raw_by_id = load_raw(path)

    env = Env(config=cfg)
    n = env.number_of_episodes
    succ = 0
    d_sum = 0.0
    per_cat = {}
    per_cat_s = {}
    no_pose = 0

    for _ in range(n):
        env.reset()
        ep = env.current_episode
        raw = raw_by_id.get(ep.episode_id)
        if raw is None:
            continue
        last = raw["reference_replay"][-1]["agent_state"]
        if last is None:
            no_pose += 1
            continue
        pos = np.array(last["position"], dtype=np.float32)
        rot_xyzw = last["rotation"]  # [x, y, z, w]
        # habitat_sim expects rotation as a quaternion; set_agent_state
        # accepts a list in [x, y, z, w] order as used in episode JSON.
        import quaternion as qlib
        q = qlib.quaternion(rot_xyzw[3], rot_xyzw[0], rot_xyzw[1],
                            rot_xyzw[2])  # w, x, y, z

        env.sim.set_agent_state(pos, q)
        obs = env.step(0)  # STOP
        info = env.get_metrics()
        s = float(info["success"])
        d = float(info["distance_to_goal"])
        succ += int(s > 0.5)
        d_sum += d
        per_cat.setdefault(ep.object_category, 0)
        per_cat_s.setdefault(ep.object_category, 0)
        per_cat[ep.object_category] += 1
        if s > 0.5:
            per_cat_s[ep.object_category] += 1

    print(f"\nTeleport-to-final-pose on {args.split} split "
          f"(SUCCESS_DISTANCE={args.success_distance}, "
          f"ALLOW_SLIDING={args.allow_sliding}):")
    print(f"  Total episodes   : {n}")
    print(f"  No recorded pose : {no_pose}")
    print(f"  Success          : {succ}/{n} = {100*succ/n:.2f}%")
    print(f"  Mean d2goal      : {d_sum/max(n,1):.3f} m")
    print("\nPer-category:")
    for c in sorted(per_cat):
        print(f"  {c:<10} {per_cat_s[c]}/{per_cat[c]} "
              f"({100*per_cat_s[c]/per_cat[c]:.1f}%)")
    env.close()


if __name__ == "__main__":
    main()
