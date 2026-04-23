"""
Compare the expert's recorded end-pose to what the sim produces when we
replay the same action sequence.  If poses diverge by a lot, the sim
config (FORWARD_STEP_SIZE, TURN_ANGLE, ALLOW_SLIDING, agent radius/
height) at eval differs from what was used to record the demos.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import gzip, json

import habitat  # noqa
from habitat.config.default import get_config as get_task_config
from habitat.core.env import Env

import pirlnav  # noqa - register custom dataset / sensors


def load_raw_replays(path):
    """Load the raw JSON so we can inspect agent_state (the PIRLNav
    dataset loader nulls out agent_state on each replay step)."""
    with gzip.open(path, "rt") as f:
        d = json.load(f)
    return {ep["episode_id"]: ep for ep in d["episodes"]}

A = {"STOP": 0, "MOVE_FORWARD": 1, "TURN_LEFT": 2, "TURN_RIGHT": 3,
     "LOOK_UP": 4, "LOOK_DOWN": 5}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-sliding", dest="allow_sliding",
                    action="store_true")
    ap.add_argument("--no-allow-sliding", dest="allow_sliding",
                    action="store_false")
    ap.set_defaults(allow_sliding=None)
    args = ap.parse_args()

    cfg = get_task_config("configs/tasks/objectnav_mp3d.yaml")
    cfg.defrost()
    cfg.DATASET.SPLIT = "train"
    if args.allow_sliding is not None:
        cfg.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING = args.allow_sliding
    # SPARSE_REWARD is defined only in the PIRLNav extension of the yacs
    # tree; when using habitat.Env directly we never loaded those extras,
    # so strip measurements that can't be resolved.
    cfg.TASK.MEASUREMENTS = [
        m for m in cfg.TASK.MEASUREMENTS
        if m in {"DISTANCE_TO_GOAL", "SUCCESS", "SPL", "SOFT_SPL"}
    ]
    cfg.freeze()

    raw_by_id = load_raw_replays(
        "data/datasets/objectnav/objectnav_mp3d/objectnav_mp3d_1scene_6cat/"
        "train/content/17DRP5sb8fy.json.gz"
    )

    env = Env(config=cfg)
    print(f"Running on {env.number_of_episodes} train episodes (cap=30)")
    print(f"{'ep':>4} {'cat':<11} {'|d_pos|(m)':>10} {'|d_rot|':>8} "
          f"{'d2goal':>7} {'succ':>4}  rec_end_pos            sim_end_pos")

    n = min(30, env.number_of_episodes)
    fail_tight = 0
    fail_loose = 0
    ok = 0
    pose_errs = []

    for _ in range(n):
        obs = env.reset()
        ep = env.current_episode
        replay = list(ep.reference_replay)
        for r in replay[1:]:
            an = str(r.action).upper().split(".")[-1]
            a = A[an]
            obs = env.step(a)
            if env.episode_over:
                break
        info = env.get_metrics()
        ag = env.sim.get_agent_state()
        sim_pos = np.array(ag.position)
        sim_rot = np.array([ag.rotation.w, ag.rotation.x, ag.rotation.y, ag.rotation.z])
        raw = raw_by_id.get(ep.episode_id)
        if raw is None:
            print(f"{ep.episode_id:>4}  <raw not found>")
            continue
        raw_replay = raw["reference_replay"]
        rec_state = raw_replay[-1]["agent_state"]
        rec_pos = np.array(rec_state["position"])
        rr = rec_state["rotation"]  # stored as [x,y,z,w] in JSON
        rec_rot = np.array([rr[3], rr[0], rr[1], rr[2]])
        dp = float(np.linalg.norm(sim_pos - rec_pos))
        dq = float(min(
            np.linalg.norm(sim_rot - rec_rot),
            np.linalg.norm(sim_rot + rec_rot),
        ))
        pose_errs.append((dp, dq))
        d2g = float(info["distance_to_goal"])
        s = float(info["success"])
        if s > 0.5:
            ok += 1
        else:
            if d2g < 1.0:
                fail_loose += 1
            else:
                fail_tight += 1
        print(
            f"{ep.episode_id:>4} {ep.object_category:<11} "
            f"{dp:10.4f} {dq:8.4f} {d2g:7.3f} {int(s>0.5):>4}  "
            f"{str(list(np.round(rec_pos,3))):<22} "
            f"{str(list(np.round(sim_pos,3))):<22}"
        )
    pose_errs = np.array(pose_errs)
    print()
    print(f"Pose-error (sim vs recorded) across {n} train episodes:")
    print(f"  median |dp|  = {np.median(pose_errs[:,0]):.4f} m")
    print(f"  max    |dp|  = {np.max(pose_errs[:,0]):.4f} m")
    print(f"  median |dq|  = {np.median(pose_errs[:,1]):.4f}")
    print(f"  max    |dq|  = {np.max(pose_errs[:,1]):.4f}")
    print()
    print("Summary:")
    print(f"  success (d<0.1 to viewpoint)   : {ok}/{n} = {100*ok/n:.1f}%")
    print(f"  would-succ with d<1.0 threshold: {(ok+fail_loose)}/{n} = "
          f"{100*(ok+fail_loose)/n:.1f}%")
    print(f"  genuinely far (d>1.0)           : {fail_tight}/{n}")
    env.close()


if __name__ == "__main__":
    main()
