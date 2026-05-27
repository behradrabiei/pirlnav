"""
Per-scene scene-loadability + expert-replay smoke test.

Picks the first episode in `--scene` that has a `reference_replay`, builds an
`Env` with the FULL training sensor stack (RGB + depth + semantic +
EGO_OBJECT_CLOUD_SENSOR at MAX_OBJECTS=300), then steps through the entire
demonstration action by action. Designed to be invoked **as a subprocess**
(one per scene) so a native crash (SIGSEGV/SIGABRT inside habitat-sim or its
GL/EGL stack) is contained to a single scene rather than killing a whole
sweep.

Exit codes / signals (interpreted by scripts/test_all_scenes_replay.sh):
  0   - success (replay finished or hit episode-step limit cleanly)
  2   - no episode with reference_replay found for this scene
  3   - dataset / config setup error (path, gzip, episode list, etc)
  4   - Python exception during env construction or stepping
  -N  - process killed by signal N (e.g. -11 = SIGSEGV, -6 = SIGABRT,
        -9 = SIGKILL/OOM-killer); in bash this surfaces as 128+N.

Single-scene usage (inside the container, with the standard binds):
  python scripts/test_scene_replay.py --scene 17DRP5sb8fy
"""

import argparse
import gzip
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

ACTION_NAME_TO_ID = {
    "STOP": 0,
    "MOVE_FORWARD": 1,
    "TURN_LEFT": 2,
    "TURN_RIGHT": 3,
    "LOOK_UP": 4,
    "LOOK_DOWN": 5,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True,
                   help="MP3D scene id, e.g. 17DRP5sb8fy")
    p.add_argument(
        "--config",
        default="configs/experiments/il_objectnav_mp3d_dinov2_object_cloud_full.yaml",
        help="Experiment yaml (only TASK_CONFIG portion is exercised here).",
    )
    p.add_argument("--max-actions", type=int, default=500,
                   help="Hard cap on actions replayed per episode.")
    p.add_argument(
        "--data-path",
        default=(
            "data/datasets/objectnav/objectnav_mp3d/"
            "objectnav_mp3d_thda_70k_21cat/{split}/{split}.json.gz"
        ),
        help="DATA_PATH template; {split} is filled in by habitat-lab.",
    )
    p.add_argument("--split", default="train")
    return p.parse_args()


def find_first_replay_episode(content_path):
    """Quick gzip read just to confirm the per-scene content file decodes
    and contains at least one episode with a non-empty reference_replay.
    Returns the count of (total_eps, with_replay)."""
    if not os.path.exists(content_path):
        raise FileNotFoundError(content_path)
    with gzip.open(content_path, "rb") as f:
        data = json.loads(f.read().decode("utf-8"))
    eps = data.get("episodes", [])
    n_with_replay = sum(
        1 for e in eps if e.get("reference_replay")
    )
    return len(eps), n_with_replay


def main():
    args = parse_args()
    scene = args.scene

    # --- Phase 0: sanity-check the per-scene content file exists & decodes.
    content_root = os.path.dirname(args.data_path.format(split=args.split))
    content_path = os.path.join(content_root, "content", f"{scene}.json.gz")
    try:
        n_eps, n_replay = find_first_replay_episode(content_path)
    except Exception as exc:
        print(f"[test_scene_replay] {scene}: SETUP_FAIL "
              f"could not read {content_path}: {exc}", flush=True)
        return 3
    if n_eps == 0 or n_replay == 0:
        print(f"[test_scene_replay] {scene}: NO_REPLAY "
              f"({n_eps} episodes, {n_replay} with reference_replay)",
              flush=True)
        return 2
    print(f"[test_scene_replay] {scene}: content/{scene}.json.gz OK "
          f"({n_eps} episodes, {n_replay} with reference_replay)", flush=True)

    # --- Phase 1: build the env exactly the way training does, but locked
    #              to this single scene via CONTENT_SCENES.
    try:
        from habitat.core.environments import get_env_class
        from habitat.utils.env_utils import construct_envs
        import pirlnav  # noqa: F401  (registers ObjectNav-v2, sensors, etc.)
        from pirlnav.config import get_config
    except Exception:
        print(f"[test_scene_replay] {scene}: SETUP_FAIL imports", flush=True)
        traceback.print_exc()
        return 3

    try:
        cfg = get_config(args.config, opts=None)
        cfg.defrost()
        cfg.RUN_TYPE = "eval"
        cfg.NUM_ENVIRONMENTS = 1
        cfg.TEST_EPISODE_COUNT = 1
        cfg.TASK_CONFIG.DATASET.SPLIT = args.split
        cfg.TASK_CONFIG.DATASET.DATA_PATH = args.data_path
        cfg.TASK_CONFIG.DATASET.CONTENT_SCENES = [scene]
        cfg.freeze()
    except Exception:
        print(f"[test_scene_replay] {scene}: SETUP_FAIL config", flush=True)
        traceback.print_exc()
        return 3

    t0 = time.time()
    try:
        envs = construct_envs(cfg, get_env_class(cfg.ENV_NAME))
    except Exception:
        print(f"[test_scene_replay] {scene}: ENV_CONSTRUCT_FAIL "
              f"after {time.time() - t0:.1f}s", flush=True)
        traceback.print_exc()
        return 4
    construct_s = time.time() - t0
    print(f"[test_scene_replay] {scene}: env constructed in {construct_s:.1f}s",
          flush=True)

    try:
        envs.reset()
        cur = envs.current_episodes()[0]
        replay = list(cur.reference_replay)
        # reference_replay[0] is the start pose (action not meaningful);
        # actions start at index 1 (matches diagnose_expert_replay.py).
        start_idx = 1 if len(replay) > 0 and hasattr(replay[0], "action") else 0

        steps_taken = 0
        done = False
        last_info = None
        replay_actions = replay[start_idx:args.max_actions + start_idx]
        t1 = time.time()
        for r in replay_actions:
            action_name = r.action
            if action_name not in ACTION_NAME_TO_ID:
                action_name = str(action_name).upper().split(".")[-1]
            a = ACTION_NAME_TO_ID[action_name]
            outs = envs.step([a])
            _obs, _rew, dones, infos = [list(x) for x in zip(*outs)]
            last_info = infos[0]
            steps_taken += 1
            if dones[0]:
                done = True
                break
        step_s = time.time() - t1

        d2g = float((last_info or {}).get("distance_to_goal", float("nan")))
        succ = float((last_info or {}).get("success", 0.0))
        print(
            f"[test_scene_replay] {scene}: PASS "
            f"ep={cur.episode_id} cat={cur.object_category} "
            f"steps={steps_taken}/{len(replay)} done={done} succ={int(succ>0.5)} "
            f"d2g={d2g:.3f}m construct={construct_s:.1f}s step={step_s:.1f}s",
            flush=True,
        )
    except Exception:
        print(f"[test_scene_replay] {scene}: STEP_FAIL after "
              f"{time.time() - t0:.1f}s", flush=True)
        traceback.print_exc()
        try:
            envs.close()
        except Exception:
            pass
        return 4

    try:
        envs.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
