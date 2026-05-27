import glob
import gzip
import json
import os
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Union

import cv2
import matplotlib
# Use a non-interactive backend so eval video rendering works headlessly.
# Set before importing pyplot; harmless if the host already configured one.
if matplotlib.get_backend().lower() != "agg":
    matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
from habitat.utils import profiling_wrapper
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import (
    append_text_to_image,
    draw_collision,
    images_to_video,
)
from numpy import ndarray
from torch import Tensor

from pirlnav.policy.models.resnet_gn import ResNet
from pirlnav.task.object_cloud import render_ego_cloud_topdown
from pirlnav.task.semantic_map import label_map_to_rgb


# Knuth multiplicative hash constant (floor(2**32 / phi)). Multiplying an
# integer by this and taking the low 8 bits spreads consecutive inputs across
# [0, 256), so adjacent instance ids land on perceptually distant colours
# after a downstream HSV colormap lookup. Used to colourise the raw
# first-person semantic observation for eval videos.
_INSTANCE_HASH_MULTIPLIER: np.uint32 = np.uint32(2654435761)


def render_goal_compass_panel(
    compass: np.ndarray,
    side_px: int,
    title: str,
    max_val: Optional[float] = None,
) -> np.ndarray:
    """Render a 12-bin goal-compass vector as a polar bar chart.

    Mirrors the convention used by `global_test.py.render_compass_panel` and
    by `GoalCompassSensor`: bin 0 points up (agent forward), positive angles
    advance counter-clockwise, bar heights are the raw (non-negative) bin
    scores.  Pass `max_val` to pin the radial scale -- the eval video uses
    this to keep the GT and predicted panels on the same scale so bar heights
    are directly comparable across the two columns.

    Args:
        compass: (12,) non-negative float vector.
        side_px: output panel side length in pixels (square).
        title: text shown above the polar plot.
        max_val: radial axis cap.  None -> auto-scale to `compass.max()`.

    Returns:
        (side_px, side_px, 3) uint8 RGB image.
    """
    compass = np.asarray(compass, dtype=np.float32).reshape(-1)
    n_bins = compass.shape[0]
    bin_angles = np.arange(n_bins) * (2.0 * np.pi / n_bins)

    if max_val is None or not np.isfinite(max_val) or max_val <= 1e-9:
        max_val = float(compass.max()) if compass.size and compass.max() > 1e-9 else 1.0
    max_val = float(max_val)

    # `colors` is normalised against `max_val` so identical bar heights across
    # GT and pred panels also receive identical hues.
    colors = plt.cm.viridis(np.clip(compass / max_val, 0.0, 1.0))

    fig = plt.figure(figsize=(side_px / 100.0, side_px / 100.0), dpi=100)
    try:
        ax = fig.add_subplot(111, projection="polar")
        ax.bar(
            bin_angles,
            compass,
            width=(2.0 * np.pi / n_bins),
            bottom=0.0,
            color=colors,
            edgecolor="white",
            linewidth=0.5,
            align="center",
        )
        ax.plot([0.0, 0.0], [0.0, max_val * 1.05], color="red", linewidth=2.0)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(1)
        ax.set_ylim(0.0, max_val * 1.15)
        ax.set_yticklabels([])
        ax.set_xticks(bin_angles)
        ax.set_xticklabels(
            [f"{int(np.rad2deg(a))}" for a in bin_angles], fontsize=7
        )
        ax.set_title(title, fontsize=10, pad=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    finally:
        plt.close(fig)

    if buf.shape[0] != side_px or buf.shape[1] != side_px:
        buf = cv2.resize(buf, (side_px, side_px), interpolation=cv2.INTER_AREA)
    return buf


def load_encoder(encoder, path):
    assert os.path.exists(path)
    if isinstance(encoder.backbone, ResNet):
        state_dict = torch.load(path, map_location="cpu", weights_only=False)["teacher"]
        state_dict = {
            k.replace("module.", ""): v for k, v in state_dict.items()
        }
        return encoder.load_state_dict(state_dict=state_dict, strict=False)
    else:
        raise ValueError("unknown encoder backbone")


def observations_to_image(observation: Dict, info: Dict) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    render_obs_images: List[np.ndarray] = []
    for sensor_name in observation:
        if "rgb" in sensor_name:
            rgb = observation[sensor_name]
            if not isinstance(rgb, np.ndarray):
                rgb = rgb.cpu().numpy()

            render_obs_images.append(rgb)
        elif "depth" in sensor_name:
            depth_map = observation[sensor_name].squeeze() * 255.0
            if not isinstance(depth_map, np.ndarray):
                depth_map = depth_map.cpu().numpy()

            depth_map = depth_map.astype(np.uint8)
            depth_map = np.stack([depth_map for _ in range(3)], axis=2)
            render_obs_images.append(depth_map)
        elif sensor_name == "semantic":
            # Habitat-sim returns per-pixel instance ids. We don't have the
            # per-scene instance->category table here (it lives inside
            # SemanticMapSensor), so paint each instance with a hash-derived
            # HSV colour. Adjacent instances land on distant hues; same
            # instance always paints the same colour within an episode.
            sem = observation[sensor_name]
            if not isinstance(sem, np.ndarray):
                sem = sem.cpu().numpy()
            sem = sem.squeeze()
            keys = (sem.astype(np.uint32) * _INSTANCE_HASH_MULTIPLIER) % np.uint32(256)
            bgr = cv2.applyColorMap(keys.astype(np.uint8), cv2.COLORMAP_HSV)
            render_obs_images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        elif sensor_name == "semantic_map":
            # Egocentric (H, W) int8 label map from SemanticMapSensor with
            # labels already in {-1, 0, 1, 2..22}. Resize to the running
            # panel height with nearest-neighbour so cell boundaries stay
            # crisp and no intermediate label values are invented.
            label_map = observation[sensor_name]
            if not isinstance(label_map, np.ndarray):
                label_map = label_map.cpu().numpy()
            sem_map_rgb = label_map_to_rgb(label_map)
            if render_obs_images:
                target_h = render_obs_images[0].shape[0]
                if sem_map_rgb.shape[0] != target_h:
                    new_w = max(
                        1,
                        int(round(sem_map_rgb.shape[1] * target_h / sem_map_rgb.shape[0])),
                    )
                    sem_map_rgb = cv2.resize(
                        sem_map_rgb,
                        (new_w, target_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
            render_obs_images.append(sem_map_rgb)
        elif sensor_name == "ego_object_cloud":
            # Packed (MAX_OBJECTS, 4) float32 from EgoObjectCloudSensor.
            # Render as an agent-frame top-down panel (forward = up).
            packed = observation[sensor_name]
            if not isinstance(packed, np.ndarray):
                packed = packed.cpu().numpy()
            target_h = (
                render_obs_images[0].shape[0] if render_obs_images else 480
            )
            cloud_rgb = render_ego_cloud_topdown(
                packed.astype(np.float32, copy=False), side_px=target_h
            )
            render_obs_images.append(cloud_rgb)
        elif sensor_name == "goal_compass":
            # Privileged 12-bin direction-to-goal feature from
            # GoalCompassSensor.  Rendered as the GT column for the
            # compass-aux variants; harmless if the policy doesn't consume
            # it (it is only a training label).
            gt = observation[sensor_name]
            if not isinstance(gt, np.ndarray):
                gt = gt.cpu().numpy()
            target_h = (
                render_obs_images[0].shape[0] if render_obs_images else 480
            )
            render_obs_images.append(
                render_goal_compass_panel(
                    gt.astype(np.float32, copy=False),
                    side_px=target_h,
                    title="Goal compass (GT)",
                )
            )
        elif sensor_name == "compass_pred":
            # Not a real sensor: injected into the per-env observation dict by
            # the IL eval loop from ILPolicy.get_last_aux()["compass_pred"].
            # Pin the radial scale to the GT max (if available) so bar heights
            # are directly comparable across the GT and predicted panels.
            pred = observation[sensor_name]
            if not isinstance(pred, np.ndarray):
                pred = pred.cpu().numpy()
            target_h = (
                render_obs_images[0].shape[0] if render_obs_images else 480
            )
            gt_max: Optional[float] = None
            if "goal_compass" in observation:
                gt = observation["goal_compass"]
                if not isinstance(gt, np.ndarray):
                    gt = gt.cpu().numpy()
                gt_arr = np.asarray(gt, dtype=np.float32).reshape(-1)
                if gt_arr.size:
                    gt_max = float(gt_arr.max())
            render_obs_images.append(
                render_goal_compass_panel(
                    pred.astype(np.float32, copy=False),
                    side_px=target_h,
                    title="Goal compass (pred)",
                    max_val=gt_max,
                )
            )

    # add image goal if observation has image_goal info
    if "imagegoal" in observation or "imagegoalrotation" in observation:
        if "imagegoal" in observation:
            rgb = observation["imagegoal"]
        else:
            rgb = observation["imagegoalrotation"]
        if not isinstance(rgb, np.ndarray):
            rgb = rgb.cpu().numpy()

        render_obs_images.append(rgb)

    assert (
        len(render_obs_images) > 0
    ), "Expected at least one visual sensor enabled."

    # shapes_are_equal = len(set(x.shape for x in render_obs_images)) == 1
    # if not shapes_are_equal:
    #     render_frame = tile_images(render_obs_images)
    # else:
    #     render_frame = np.concatenate(render_obs_images, axis=1)

    render_frame = np.concatenate(render_obs_images, axis=1)
    # draw collision
    if "collisions" in info and info["collisions"]["is_collision"]:
        render_frame = draw_collision(render_frame)

    if "top_down_map" in info:
        top_down_map = maps.colorize_draw_agent_and_fit_to_height(
            info["top_down_map"], render_frame.shape[0]
        )
        render_frame = np.concatenate((render_frame, top_down_map), axis=1)
    return render_frame


def generate_video(
    video_option: List[str],
    video_dir: Optional[str],
    images: List[np.ndarray],
    episode_id: Union[int, str],
    checkpoint_idx: int,
    metrics: Dict[str, float],
    fps: int = 10,
    verbose: bool = True,
) -> None:
    r"""Generate video according to specified information.

    Args:
        video_option: string list of "tensorboard" or "disk" or both.
        video_dir: path to target video directory.
        images: list of images to be converted to video.
        episode_id: episode id for video naming.
        checkpoint_idx: checkpoint index for video naming.
        metric_name: name of the performance metric, e.g. "spl".
        metric_value: value of metric.
        tb_writer: tensorboard writer object for uploading video.
        fps: fps for generated video.
    Returns:
        None
    """
    if len(images) < 1:
        return

    metric_strs = []
    for k, v in metrics.items():
        metric_strs.append(f"{k}={v:.2f}")

    video_name = f"episode={episode_id}-ckpt={checkpoint_idx}-" + "-".join(
        metric_strs
    )
    if "disk" in video_option:
        assert video_dir is not None
        images_to_video(images, video_dir, video_name, verbose=verbose)


def add_info_to_image(frame, info):
    string = "d2g: {} | a2g: {} |\nsimple reward: {} |\nsuccess: {} | angle success: {}".format(
        round(info["distance_to_goal"], 3),
        round(info["angle_to_goal"], 3),
        round(info["simple_reward"], 3),
        round(info["success"], 3),
        round(info["angle_success"], 3),
    )
    frame = append_text_to_image(frame, string)
    return frame


def write_json(data, path):
    with open(path, "w") as file:
        file.write(json.dumps(data))


def load_dataset(path):
    with gzip.open(path, "rb") as file:
        data = json.loads(file.read(), encoding="utf-8")
    return data


def load_json_dataset(path):
    file = open(path, "r")
    data = json.loads(file.read())
    return data


def _to_tensor(v: Union[Tensor, ndarray]) -> torch.Tensor:
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        if v.dtype == np.uint32:
            return torch.from_numpy(v.astype(int))
        else:
            return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)


@torch.no_grad()
@profiling_wrapper.RangeContext("batch_obs")
def batch_obs(
    observations: List[Dict],
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    r"""Transpose a batch of observation dicts to a dict of batched
    observations.

    Args:
        observations:  list of dicts of observations.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None

    Returns:
        transposed dict of torch.Tensor of observations.
    """
    batch: DefaultDict[str, List] = defaultdict(list)

    for obs in observations:
        for sensor in obs:
            batch[sensor].append(_to_tensor(obs[sensor]))

    batch_t: Dict[str, torch.Tensor] = {}

    for sensor in batch:
        batch_t[sensor] = torch.stack(batch[sensor], dim=0).to(device=device)

    return batch_t


def linear_warmup(
    epoch: int, start_update: int, max_updates: int, start_lr: int, end_lr: int
) -> float:
    r"""Returns a multiplicative factor for linear value decay

    Args:
        epoch: current epoch number
        total_num_updates: total number of

    Returns:
        multiplicative factor that decreases param value linearly
    """
    # logger.info("policy: {}, {}, {}, {}, {}".format(epoch, start_update, max_updates, start_lr, end_lr))
    if epoch < start_update:
        return 1.0

    if epoch > max_updates:
        return end_lr

    if max_updates == start_update:
        return end_lr

    pct_step = (epoch - start_update) / (max_updates - start_update)
    step_lr = (end_lr - start_lr) * pct_step + start_lr
    if step_lr > end_lr:
        step_lr = end_lr
    # logger.info("{}, {}, {}, {}, {}, {}".format(epoch, start_update, max_updates, start_lr, end_lr, step_lr))
    return step_lr


def critic_linear_decay(
    epoch: int, start_update: int, max_updates: int, start_lr: int, end_lr: int
) -> float:
    r"""Returns a multiplicative factor for linear value decay

    Args:
        epoch: current epoch number
        total_num_updates: total number of

    Returns:
        multiplicative factor that decreases param value linearly
    """
    # logger.info("critic lr: {}, {}, {}, {}, {}".format(epoch, start_update, max_updates, start_lr, end_lr))
    if epoch <= start_update:
        return 1

    if epoch >= max_updates:
        return end_lr

    if max_updates == start_update:
        return end_lr

    pct_step = (epoch - start_update) / (max_updates - start_update)
    step_lr = start_lr - (start_lr - end_lr) * pct_step
    if step_lr < end_lr:
        step_lr = end_lr
    # logger.info("{}, {}, {}, {}, {}, {}".format(epoch, start_update, max_updates, start_lr, end_lr, step_lr))
    return step_lr
