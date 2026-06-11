"""
Replay an expert episode and visualize the progressive oracle object cloud.

For each scene, replays one reference episode under the ORACLE_REVEAL config
(configs/experiments/il_objectnav_mp3d_dinov2_oracle_reveal_full_252.yaml) and
writes an MP4 of [RGB | top-down ego cloud] per step, so you can watch
objects pop into the cloud as the camera sweeps over them. Reveal events
(step, category, distance) are also printed, and the final frame is saved as
a PNG next to the video.

Run inside the container with the standard full-MP3D binds (GPU needed for
the RGB render):
  python scripts/visualize_oracle_reveal.py                       # 1 scene
  python scripts/visualize_oracle_reveal.py --scenes 17DRP5sb8fy --max-actions 250
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from verify_semantic_segmentation import (  # noqa: E402
    DEFAULT_DATA_PATH,
    discover_scenes,
    replay_action_names,
)

DEFAULT_CONFIG = (
    "configs/experiments/il_objectnav_mp3d_dinov2_oracle_reveal_full_252.yaml"
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes", nargs="*", default=None,
                   help="MP3D scene ids; auto-picks --num-scenes if unset.")
    p.add_argument("--num-scenes", type=int, default=1)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    p.add_argument("--split", default="train")
    p.add_argument("--max-actions", type=int, default=250)
    p.add_argument("--max-objects", type=int, default=100,
                   help="EGO_OBJECT_CLOUD_SENSOR.MAX_OBJECTS (training value).")
    p.add_argument("--video-fps", type=int, default=10)
    p.add_argument("--out-dir", default="probe_logs/oracle_reveal_viz")
    return p.parse_args()


def _annotate(frame, lines):
    import cv2
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (8, 24 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def visualize_scene(scene, args):
    import cv2
    import numpy as np
    import habitat
    import pirlnav  # noqa: F401  (registers ObjectNav-v2, sensors, etc.)
    from pirlnav.config import get_config
    from pirlnav.task.object_cloud import TASK_NAMES, render_ego_cloud_topdown

    cfg = get_config(args.config, opts=None)
    cfg.defrost()
    tc = cfg.TASK_CONFIG
    tc.DATASET.SPLIT = args.split
    tc.DATASET.DATA_PATH = args.data_path
    tc.DATASET.CONTENT_SCENES = [scene]
    tc.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    tc.TASK.EGO_OBJECT_CLOUD_SENSOR.MAX_OBJECTS = args.max_objects
    cfg.freeze()

    env = habitat.Env(config=tc)
    try:
        obs = env.reset()
        names = replay_action_names(env.current_episode, args.max_actions)
        if not names:
            print(f"[reveal-viz] {scene}: SKIP episode has no reference_replay")
            return False
        goal = str(env.current_episode.object_category)

        sensor = env.task.sensor_suite.sensors["ego_object_cloud"]
        reveal = sensor._reveal
        prev_mask = reveal.revealed.copy()
        cam_pos = np.asarray(env.sim.get_agent_state().position)

        os.makedirs(args.out_dir, exist_ok=True)
        side = tc.SIMULATOR.RGB_SENSOR.HEIGHT
        video_path = os.path.join(args.out_dir, f"{scene}_reveal.mp4")
        writer = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"mp4v"), args.video_fps,
            (tc.SIMULATOR.RGB_SENSOR.WIDTH + side, side),
        )

        def log_new_reveals(step):
            nonlocal prev_mask
            new = np.nonzero(reveal.revealed & ~prev_mask)[0]
            for idx in new:
                d = float(np.linalg.norm(reveal.obj_pos[idx] - cam_pos))
                print(f"[reveal-viz]   step {step:3d}: + "
                      f"{TASK_NAMES[reveal.task_ids[idx]]:<14s} at {d:.1f}m")
            prev_mask = reveal.revealed.copy()

        def write_frame(obs, step, action):
            rgb = np.asarray(obs["rgb"])[..., :3].astype(np.uint8)
            top = render_ego_cloud_topdown(
                np.asarray(obs["ego_object_cloud"]), side_px=side
            )
            n = int(reveal.revealed.sum())
            frame = np.concatenate([rgb, top], axis=1)
            _annotate(frame, [
                f"step {step}  {action}",
                f"goal: {goal}",
                f"revealed: {n}/{len(reveal.revealed)}",
            ])
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            return frame

        print(f"[reveal-viz] {scene}: episode "
              f"{env.current_episode.episode_id}, goal={goal}, "
              f"{len(names)} actions, {len(reveal.revealed)} oracle objects")
        log_new_reveals(0)
        frame = write_frame(obs, 0, "RESET")
        for step, name in enumerate(names, start=1):
            obs = env.step({"action": name})
            cam_pos = np.asarray(env.sim.get_agent_state().position)
            log_new_reveals(step)
            frame = write_frame(obs, step, name)
            if env.episode_over:
                break
        writer.release()

        png_path = os.path.join(args.out_dir, f"{scene}_reveal_final.png")
        cv2.imwrite(png_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"[reveal-viz] {scene}: revealed "
              f"{int(reveal.revealed.sum())}/{len(reveal.revealed)} objects; "
              f"saved {video_path} and {png_path}")
        return True
    finally:
        env.close()


def main():
    args = parse_args()
    scenes = args.scenes or discover_scenes(
        args.data_path, args.split, args.num_scenes
    )
    ok = all([visualize_scene(scene, args) for scene in scenes])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
