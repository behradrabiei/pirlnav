"""Benchmark frozen vision encoders on one demonstration episode.

DINOv2 (patch 14): small / base / large at 252x252 and 476x630, each timed with
both HuggingFace ``transformers`` (HF) and Meta ``torch.hub`` (hub) backends.
EUPE ViT (patch 16, HuggingFace weights + torch.hub): ViT-T/S/B at 256x256 and
480x624 (patch-16-aligned analog of 476x630).

Reports per-frame mean and std latency (ms) for preprocess + encoder forward
only (Habitat sim excluded), batch size 1.

Usage (``conda activate pirlnav``)::

    python scripts/benchmark_dinov2_inference.py --device cuda
    python scripts/benchmark_dinov2_inference.py --skip-eupe
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import habitat  # noqa: E402
from habitat.core.dataset import Dataset  # noqa: E402

import pirlnav  # noqa: F401,E402
from pirlnav.config import get_config  # noqa: E402
from pirlnav.policy.dinov2_encoder import DINOv2VisualEncoder  # noqa: E402
from pirlnav.policy.transforms import (  # noqa: E402
    DINOv2Transform,
    resize_and_center_crop,
)


ACTION_NAME_TO_ID = {
    "STOP": 0,
    "MOVE_FORWARD": 1,
    "TURN_LEFT": 2,
    "TURN_RIGHT": 3,
    "LOOK_UP": 4,
    "LOOK_DOWN": 5,
}

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

DINOV2_HUB_REPO = "facebookresearch/dinov2"

# label, HF transformers id, hub entrypoint, CLS dim
DINOV2_MODELS: Sequence[Tuple[str, str, str, int]] = (
    ("dinov2-small", "facebook/dinov2-small", "dinov2_vits14", 384),
    ("dinov2-base", "facebook/dinov2-base", "dinov2_vitb14", 768),
    ("dinov2-large", "facebook/dinov2-large", "dinov2_vitl14", 1024),
)

DINOV2_SIZES: Sequence[Tuple[str, Tuple[int, int]]] = (
    ("252x252", (252, 252)),
    ("476x630", (476, 630)),
)

EUPE_HUB_REPO = "facebookresearch/EUPE"

# label, torch.hub entrypoint, HF repo id, checkpoint filename on the Hub
EUPE_MODELS: Sequence[Tuple[str, str, str, str]] = (
    ("eupe-vitt16", "eupe_vitt16", "facebook/EUPE-ViT-T", "EUPE-ViT-T.pt"),
    ("eupe-vits16", "eupe_vits16", "facebook/EUPE-ViT-S", "EUPE-ViT-S.pt"),
    ("eupe-vitb16", "eupe_vitb16", "facebook/EUPE-ViT-B", "EUPE-ViT-B.pt"),
)

EUPE_SIZES: Sequence[Tuple[str, Tuple[int, int]]] = (
    ("256x256", (256, 256)),
    ("480x624", (480, 624)),
)


@dataclass
class BenchmarkResult:
    family: str
    label: str
    backend: str
    size_label: str
    mean_ms: float
    std_ms: float
    num_frames: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark DINOv2 and EUPE ViT inference on one demo episode.",
    )
    p.add_argument(
        "--config",
        default="configs/experiments/il_objectnav_mp3d_dinov2_object_cloud.yaml",
        help="Experiment config that loads MP3D demonstration episodes.",
    )
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--scene", default="17DRP5sb8fy")
    p.add_argument(
        "--episode-id",
        default=None,
        help="Episode id to replay (default: first episode in scene/split).",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--warmup-frames",
        type=int,
        default=20,
        help="Frames to run before timing (discarded).",
    )
    p.add_argument(
        "--timing-runs",
        type=int,
        default=1,
        help="Repeat the timed frame loop this many times (same frames).",
    )
    p.add_argument(
        "--dinov2-backend",
        default="both",
        choices=["hf", "hub", "both"],
        help="DINOv2 runtime: HF transformers, Meta torch.hub, or both (default).",
    )
    p.add_argument(
        "--skip-eupe",
        action="store_true",
        help="Skip the EUPE table.",
    )
    return p.parse_args()


def download_eupe_weights(hf_repo_id: str, filename: str) -> str:
    """Download EUPE checkpoint from Hugging Face (cached under ~/.cache/huggingface)."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=hf_repo_id, filename=filename)


def _patch_hub_cache_for_py39(cache_dir_name: str) -> None:
    """Meta hub repos use PEP-604 unions; patch cached sources for Python 3.9."""
    import sys

    if sys.version_info >= (3, 10):
        return

    hub_dir = os.path.join(torch.hub.get_dir(), cache_dir_name)
    if not os.path.isdir(hub_dir):
        return

    future = "from __future__ import annotations\n"
    for dirpath, _, filenames in os.walk(hub_dir):
        for name in filenames:
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            if future.strip() in text.splitlines()[:3]:
                continue
            with open(path, "w", encoding="utf-8") as f:
                f.write(future + text)


def build_task_config_for_split(base_config_path: str, split: str):
    cfg = get_config(base_config_path, opts=None)
    task_cfg = cfg.TASK_CONFIG.clone()
    task_cfg.defrost()
    task_cfg.DATASET.SPLIT = split
    task_cfg.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    task_cfg.ENVIRONMENT.ITERATOR_OPTIONS.CYCLE = False
    task_cfg.ENVIRONMENT.ITERATOR_OPTIONS.NUM_EPISODE_SAMPLE = -1
    task_cfg.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = 10**12
    task_cfg.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = -1
    task_cfg.freeze()
    return task_cfg


def _scene_name(episode) -> str:
    return Dataset.scene_from_scene_path(episode.scene_id)


def select_episode(episodes: list, scene: str, episode_id: Optional[str]):
    scene_eps = [ep for ep in episodes if _scene_name(ep) == scene]
    if not scene_eps:
        raise RuntimeError(
            f"No episodes found for scene {scene!r} in this split."
        )
    if episode_id is not None:
        for ep in scene_eps:
            if ep.episode_id == episode_id:
                return ep
        raise RuntimeError(
            f"episode_id {episode_id!r} not found in scene {scene!r}."
        )
    return scene_eps[0]


def collect_episode_rgb_frames(
    config_path: str,
    split: str,
    episode,
) -> Tuple[str, List[np.ndarray]]:
    """Replay one demonstration episode; return episode id and RGB frames."""
    task_cfg = build_task_config_for_split(config_path, split)
    env = habitat.Env(config=task_cfg)
    frames: List[np.ndarray] = []
    try:
        env.episode_iterator = iter([episode])
        obs = env.reset()
        cur = env.current_episode
        if cur.episode_id != episode.episode_id:
            raise RuntimeError(
                "episode iterator gave a different episode than expected: "
                f"wanted {episode.episode_id}, got {cur.episode_id}"
            )

        replay = list(cur.reference_replay)
        if len(replay) == 0:
            raise RuntimeError(
                f"episode {cur.episode_id} has empty reference_replay"
            )

        frames.append(np.ascontiguousarray(obs["rgb"]))

        start_idx = 1 if hasattr(replay[0], "action") else 0
        for r in replay[start_idx:]:
            action_name = r.action
            if action_name not in ACTION_NAME_TO_ID:
                action_name = str(action_name).upper().split(".")[-1]
            a = ACTION_NAME_TO_ID[action_name]
            step_obs = env.step(a)
            frames.append(np.ascontiguousarray(step_obs["rgb"]))
            if env.episode_over:
                break

        return cur.episode_id, frames
    finally:
        env.close()


def _cuda_sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


class CenterCropImageNetTransform:
    """Resize+center-crop uint8 BHWC -> normalized BCHW (matches DINOv2Transform)."""

    def __init__(self, size_hw: Tuple[int, int]) -> None:
        self.size_hw = (int(size_hw[0]), int(size_hw[1]))
        self._mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        self._std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        x = resize_and_center_crop(x, self.size_hw)
        x = x.float() / 255.0
        mean = self._mean.to(device=x.device, dtype=x.dtype)
        std = self._std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std


ForwardFn = Callable[[np.ndarray], torch.Tensor]


@torch.no_grad()
def _run_forward(
    forward_fn: ForwardFn,
    rgb_uint8: np.ndarray,
    device: str,
) -> torch.Tensor:
    return forward_fn(rgb_uint8)


@torch.no_grad()
def warmup_frames(
    forward_fn: ForwardFn,
    frames: Sequence[np.ndarray],
    device: str,
    num_warmup: int,
) -> None:
    if num_warmup <= 0 or len(frames) == 0:
        return
    for i in range(num_warmup):
        _run_forward(forward_fn, frames[i % len(frames)], device)
    _cuda_sync(device)


@torch.no_grad()
def time_frames(
    forward_fn: ForwardFn,
    frames: Sequence[np.ndarray],
    device: str,
    timing_runs: int,
) -> List[float]:
    latencies_ms: List[float] = []
    for _ in range(max(1, timing_runs)):
        for rgb_uint8 in frames:
            _cuda_sync(device)
            t0 = time.perf_counter()
            _run_forward(forward_fn, rgb_uint8, device)
            _cuda_sync(device)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    return latencies_ms


def _stats_from_latencies(latencies: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(latencies, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def _free_gpu(device: str, obj) -> None:
    del obj
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def benchmark_with_forward_fn(
    family: str,
    label: str,
    backend: str,
    size_label: str,
    forward_fn: ForwardFn,
    frames: Sequence[np.ndarray],
    device: str,
    warmup_frames_n: int,
    timing_runs: int,
) -> BenchmarkResult:
    warmup_frames(forward_fn, frames, device, warmup_frames_n)
    latencies = time_frames(forward_fn, frames, device, timing_runs)
    mean_ms, std_ms = _stats_from_latencies(latencies)
    return BenchmarkResult(
        family=family,
        label=label,
        backend=backend,
        size_label=size_label,
        mean_ms=mean_ms,
        std_ms=std_ms,
        num_frames=len(latencies),
    )


def make_dinov2_forward_fn(
    hf_model: str,
    output_dim: int,
    size_hw: Tuple[int, int],
    device: str,
) -> Tuple[ForwardFn, DINOv2VisualEncoder]:
    transform = DINOv2Transform(augmentations_name="dinov2", size_hw=size_hw)
    encoder = DINOv2VisualEncoder(
        model_name=hf_model,
        resize_hw=size_hw,
        output_dim=output_dim,
    )
    encoder = encoder.to(device)
    encoder.eval()

    @torch.no_grad()
    def forward_fn(rgb_uint8: np.ndarray) -> torch.Tensor:
        rgb = torch.from_numpy(rgb_uint8).to(device)
        rgb = rgb.unsqueeze(0)
        rgb = transform(rgb)
        return encoder(rgb)

    return forward_fn, encoder


def benchmark_dinov2_hf_configuration(
    label: str,
    hf_model: str,
    output_dim: int,
    size_label: str,
    size_hw: Tuple[int, int],
    frames: Sequence[np.ndarray],
    device: str,
    warmup_frames_n: int,
    timing_runs: int,
) -> BenchmarkResult:
    forward_fn, encoder = make_dinov2_forward_fn(
        hf_model, output_dim, size_hw, device
    )
    try:
        return benchmark_with_forward_fn(
            family="DINOv2",
            label=label,
            backend="HF",
            size_label=size_label,
            forward_fn=forward_fn,
            frames=frames,
            device=device,
            warmup_frames_n=warmup_frames_n,
            timing_runs=timing_runs,
        )
    finally:
        _free_gpu(device, encoder)


def load_dinov2_hub_model(hub_name: str, device: str) -> torch.nn.Module:
    """Meta DINOv2 via torch.hub (MemEffAttention when xformers is available)."""
    cache_dir = "facebookresearch_dinov2_main"
    try:
        model = torch.hub.load(
            DINOV2_HUB_REPO,
            hub_name,
            source="github",
            pretrained=True,
            trust_repo=True,
        )
    except TypeError:
        _patch_hub_cache_for_py39(cache_dir)
        model = torch.hub.load(
            DINOV2_HUB_REPO,
            hub_name,
            source="github",
            pretrained=True,
            trust_repo=True,
        )
    model = model.to(device)
    model.eval()
    return model


def make_dinov2_hub_forward_fn(
    model: torch.nn.Module,
    size_hw: Tuple[int, int],
    device: str,
) -> ForwardFn:
    """Same crop/aug as HF path; ImageNet norm applied before hub forward_features."""
    transform = DINOv2Transform(augmentations_name="dinov2", size_hw=size_hw)
    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def forward_fn(rgb_uint8: np.ndarray) -> torch.Tensor:
        rgb = torch.from_numpy(rgb_uint8).to(device)
        rgb = rgb.unsqueeze(0)
        x = transform(rgb)
        x = (x - mean) / std
        feats = model.forward_features(x)
        return feats["x_norm_clstoken"]

    return forward_fn


def benchmark_dinov2_hub_configuration(
    label: str,
    hub_name: str,
    size_label: str,
    size_hw: Tuple[int, int],
    frames: Sequence[np.ndarray],
    device: str,
    warmup_frames_n: int,
    timing_runs: int,
) -> BenchmarkResult:
    model = load_dinov2_hub_model(hub_name, device)
    forward_fn = make_dinov2_hub_forward_fn(model, size_hw, device)
    try:
        return benchmark_with_forward_fn(
            family="DINOv2",
            label=label,
            backend="hub",
            size_label=size_label,
            forward_fn=forward_fn,
            frames=frames,
            device=device,
            warmup_frames_n=warmup_frames_n,
            timing_runs=timing_runs,
        )
    finally:
        _free_gpu(device, model)


def load_eupe_model(
    hub_name: str,
    hf_repo_id: str,
    weight_file: str,
    device: str,
) -> torch.nn.Module:
    """Load EUPE ViT: weights from HF Hub, architecture from torch.hub (GitHub cache)."""
    weights_path = download_eupe_weights(hf_repo_id, weight_file)
    print(f"    weights: {weights_path}", flush=True)
    try:
        model = torch.hub.load(
            EUPE_HUB_REPO,
            hub_name,
            source="github",
            pretrained=True,
            weights=weights_path,
            trust_repo=True,
        )
    except TypeError:
        # First load on Python 3.9: hub zip uses 3.10+ type syntax; patch and retry.
        _patch_hub_cache_for_py39("facebookresearch_EUPE_main")
        model = torch.hub.load(
            EUPE_HUB_REPO,
            hub_name,
            source="github",
            pretrained=True,
            weights=weights_path,
            trust_repo=True,
        )
    model = model.to(device)
    model.eval()
    return model


def make_eupe_forward_fn(
    model: torch.nn.Module,
    size_hw: Tuple[int, int],
    device: str,
) -> ForwardFn:
    transform = CenterCropImageNetTransform(size_hw=size_hw)

    @torch.no_grad()
    def forward_fn(rgb_uint8: np.ndarray) -> torch.Tensor:
        rgb = torch.from_numpy(rgb_uint8).to(device)
        rgb = rgb.unsqueeze(0)
        x = transform(rgb)
        feats = model.forward_features(x)
        return feats["x_norm_clstoken"]

    return forward_fn


def benchmark_eupe_configuration(
    label: str,
    hub_name: str,
    hf_repo_id: str,
    weight_file: str,
    size_label: str,
    size_hw: Tuple[int, int],
    frames: Sequence[np.ndarray],
    device: str,
    warmup_frames_n: int,
    timing_runs: int,
) -> BenchmarkResult:
    model = load_eupe_model(hub_name, hf_repo_id, weight_file, device)
    forward_fn = make_eupe_forward_fn(model, size_hw, device)
    try:
        return benchmark_with_forward_fn(
            family="EUPE-ViT",
            label=label,
            backend="hub",
            size_label=size_label,
            forward_fn=forward_fn,
            frames=frames,
            device=device,
            warmup_frames_n=warmup_frames_n,
            timing_runs=timing_runs,
        )
    finally:
        _free_gpu(device, model)


def print_results_table(
    title: str,
    results: Sequence[BenchmarkResult],
    footnote: str = "",
) -> None:
    print(title)
    if footnote:
        print(footnote)
    header = (
        f"{'Model':<14} {'Backend':<6} {'Size':<10} "
        f"{'mean_ms':>10} {'std_ms':>10} {'frames':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.label:<14} {r.backend:<6} {r.size_label:<10} "
            f"{r.mean_ms:>10.3f} {r.std_ms:>10.3f} {r.num_frames:>8}"
        )
    print()


def run_dinov2_benchmarks(
    frames: Sequence[np.ndarray],
    device: str,
    warmup_frames_n: int,
    timing_runs: int,
    backend: str,
) -> List[BenchmarkResult]:
    results: List[BenchmarkResult] = []
    run_hf = backend in ("hf", "both")
    run_hub = backend in ("hub", "both")

    for label, hf_model, hub_name, output_dim in DINOV2_MODELS:
        for size_label, size_hw in DINOV2_SIZES:
            if run_hf:
                print(f"  [DINOv2/HF] {label} @ {size_label} ...", flush=True)
                results.append(
                    benchmark_dinov2_hf_configuration(
                        label=label,
                        hf_model=hf_model,
                        output_dim=output_dim,
                        size_label=size_label,
                        size_hw=size_hw,
                        frames=frames,
                        device=device,
                        warmup_frames_n=warmup_frames_n,
                        timing_runs=timing_runs,
                    )
                )
            if run_hub:
                print(f"  [DINOv2/hub] {label} @ {size_label} ...", flush=True)
                results.append(
                    benchmark_dinov2_hub_configuration(
                        label=label,
                        hub_name=hub_name,
                        size_label=size_label,
                        size_hw=size_hw,
                        frames=frames,
                        device=device,
                        warmup_frames_n=warmup_frames_n,
                        timing_runs=timing_runs,
                    )
                )
    return results


def run_eupe_benchmarks(
    frames: Sequence[np.ndarray],
    device: str,
    warmup_frames_n: int,
    timing_runs: int,
) -> List[BenchmarkResult]:
    results: List[BenchmarkResult] = []
    for label, hub_name, hf_repo_id, weight_file in EUPE_MODELS:
        for size_label, size_hw in EUPE_SIZES:
            print(f"  [EUPE] {label} @ {size_label} ...", flush=True)
            results.append(
                benchmark_eupe_configuration(
                    label=label,
                    hub_name=hub_name,
                    hf_repo_id=hf_repo_id,
                    weight_file=weight_file,
                    size_label=size_label,
                    size_hw=size_hw,
                    frames=frames,
                    device=device,
                    warmup_frames_n=warmup_frames_n,
                    timing_runs=timing_runs,
                )
            )
    return results


def main() -> None:
    args = parse_args()
    print(f"Device: {args.device}")
    print(f"Collecting RGB frames (split={args.split}, scene={args.scene}) ...")

    task_cfg = build_task_config_for_split(args.config, args.split)
    env = habitat.Env(config=task_cfg)
    try:
        episodes = list(env.episodes)
    finally:
        env.close()

    ep = select_episode(episodes, args.scene, args.episode_id)
    episode_id, frames = collect_episode_rgb_frames(
        args.config, args.split, ep
    )
    print(
        f"Episode: {episode_id}  scene: {args.scene}  "
        f"frames: {len(frames)}  warmup: {args.warmup_frames}  "
        f"timing_runs: {args.timing_runs}"
    )
    print("\nBenchmarking preprocess + encoder forward (batch=1, per-frame ms) ...\n")

    dinov2_results = run_dinov2_benchmarks(
        frames=frames,
        device=args.device,
        warmup_frames_n=args.warmup_frames,
        timing_runs=args.timing_runs,
        backend=args.dinov2_backend,
    )
    print_results_table(
        "=== DINOv2 (patch 14) ===",
        dinov2_results,
        footnote=(
            "HF = HuggingFace transformers (pirlnav training path). "
            "hub = Meta torch.hub inference backbone (MemEffAttention if xformers). "
            "Same sizes and weights tier; compare HF vs hub rows per model."
        ),
    )

    if args.skip_eupe:
        print("EUPE table skipped (--skip-eupe).")
        print("\nDone.")
        return

    print(
        f"EUPE: weights from Hugging Face ({', '.join(m[2] for m in EUPE_MODELS)}); "
        f"architecture via torch.hub {EUPE_HUB_REPO!r} (cached on first run).\n"
    )
    eupe_results = run_eupe_benchmarks(
        frames=frames,
        device=args.device,
        warmup_frames_n=args.warmup_frames,
        timing_runs=args.timing_runs,
    )
    print_results_table(
        "=== EUPE ViT (patch 16) ===",
        eupe_results,
        footnote=(
            "Compact row: 256x256 (16x16 patches). Large row: 480x624 vs "
            "DINOv2 476x630 (34x45 vs 30x39 patches)."
        ),
    )
    print("Done.")


if __name__ == "__main__":
    main()
