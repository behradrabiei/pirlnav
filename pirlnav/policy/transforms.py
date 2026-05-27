from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter, RandomApply


SizeT = Union[int, Tuple[int, int]]

# DINOv2 ViT patch size (facebook/dinov2-*).
DINOV2_PATCH_SIZE = 14


def _as_hw(size: SizeT) -> Tuple[int, int]:
    if isinstance(size, int):
        return size, size
    h, w = size
    return int(h), int(w)


def patch_aligned_edge(
    target_edge: int,
    patch_size: int = DINOV2_PATCH_SIZE,
) -> int:
    """Largest multiple of ``patch_size`` not exceeding ``target_edge``."""
    if target_edge < patch_size:
        return patch_size
    return (target_edge // patch_size) * patch_size


def default_dinov2_size_hw(
    resnet_image_size: int = 256,
    patch_size: int = DINOV2_PATCH_SIZE,
) -> Tuple[int, int]:
    """Square DINOv2 input size mirroring the ResNet ``image_size`` pipeline."""
    edge = patch_aligned_edge(resnet_image_size, patch_size)
    return edge, edge


def resize_and_center_crop(
    x: torch.Tensor, size_hw: Tuple[int, int]
) -> torch.Tensor:
    """Resize shorter edge to ``min(H, W)``, then center-crop to ``(H, W)``.

    Matches ``ResizeTransform`` / ``ShiftAndJitterTransform`` used by the
    original ResNet visual encoder (``image_size=256``), but allows a
    rectangular crop when ``size_hw`` is not square.
    """
    h, w = _as_hw(size_hw)
    x = TF.resize(x, min(h, w))
    x = TF.center_crop(x, output_size=[h, w])
    return x


class RandomShiftsAug(nn.Module):
    """Random pixel-level shift augmentation.

    Originally required square inputs; generalized to rectangular so it can
    be used with any patch-aligned DINOv2 input size.
    """

    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, _, h, w = x.size()
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")

        eps_h = 1.0 / (h + 2 * self.pad)
        eps_w = 1.0 / (w + 2 * self.pad)
        arange_h = torch.linspace(
            -1.0 + eps_h, 1.0 - eps_h, h + 2 * self.pad,
            device=x.device, dtype=x.dtype,
        )[:h]
        arange_w = torch.linspace(
            -1.0 + eps_w, 1.0 - eps_w, w + 2 * self.pad,
            device=x.device, dtype=x.dtype,
        )[:w]

        grid_y = arange_h.view(h, 1).expand(h, w).unsqueeze(-1)
        grid_x = arange_w.view(1, w).expand(h, w).unsqueeze(-1)
        base_grid = torch.cat([grid_x, grid_y], dim=-1)
        base_grid = base_grid.unsqueeze(0).expand(n, -1, -1, -1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift_scale = shift.clone()
        shift_scale[..., 0] = shift[..., 0] * (2.0 / (w + 2 * self.pad))
        shift_scale[..., 1] = shift[..., 1] * (2.0 / (h + 2 * self.pad))

        grid = base_grid + shift_scale
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class Transform:
    randomize_environments: bool = False

    def apply(self, x: torch.Tensor):
        raise NotImplementedError

    def __call__(
        self,
        x: torch.Tensor,
        N: Optional[int] = None,
    ):
        if not self.randomize_environments or N is None:
            return self.apply(x)

        # shapes
        TN = x.size(0)
        T = TN // N

        # apply the same augmentation when t == 1 for speed
        # typically, t == 1 during policy rollout
        if T == 1:
            return self.apply(x)

        # put environment (n) first
        _, A, B, C = x.shape
        x = torch.einsum("tnabc->ntabc", x.view(T, N, A, B, C))

        # apply the same transform within each environment
        x = torch.cat([self.apply(imgs) for imgs in x])

        # put timestep (t) first
        _, A, B, C = x.shape
        x = torch.einsum("ntabc->tnabc", x.view(N, T, A, B, C)).flatten(0, 1)

        return x


class ResizeTransform(Transform):
    def __init__(self, size):
        self.size = size

    def apply(self, x):
        x = x.permute(0, 3, 1, 2)
        x = resize_and_center_crop(x, self.size)
        x = x.float() / 255.0
        return x


class ShiftAndJitterTransform(Transform):
    def __init__(self, augmentations_name, size):
        self.size = size
        self.augmentations_name = augmentations_name

    def apply(self, x):
        x = x.permute(0, 3, 1, 2)
        x = resize_and_center_crop(x, self.size)
        x = x.float() / 255.0
        if "jitter" in self.augmentations_name:
            x = RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.4)], p=1.0)(x)
        if "shift" in self.augmentations_name:
            x = RandomShiftsAug(16)(x)
        return x


class DINOv2Transform(Transform):
    """Preprocessing for the frozen DINOv2 visual encoder.

    Uses the same resize-then-center-crop pipeline as the ResNet visual
    encoder (``ResizeTransform``): resize the shorter edge to ``min(H, W)``,
    then center-crop to ``(H, W)``. Default ``(252, 252)`` is the largest
    multiple of DINOv2's patch size (14) not exceeding the ResNet
    ``image_size`` of 256. Override via ``dinov2_resize_h/w`` in config
    (e.g. 476x630 for a higher-res rectangular crop). ImageNet normalization
    is applied inside the encoder; this transform outputs ``[0, 1]`` floats.
    """

    def __init__(self, augmentations_name: str, size_hw: Tuple[int, int]):
        self.augmentations_name = augmentations_name
        self.size_hw = _as_hw(size_hw)
        h, w = self.size_hw
        if h % DINOV2_PATCH_SIZE != 0 or w % DINOV2_PATCH_SIZE != 0:
            raise ValueError(
                f"DINOv2 input size {self.size_hw} must be a multiple of "
                f"patch size {DINOV2_PATCH_SIZE}."
            )

    def apply(self, x):
        x = x.permute(0, 3, 1, 2)
        x = resize_and_center_crop(x, self.size_hw)
        x = x.float() / 255.0
        if "jitter" in self.augmentations_name:
            x = RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.4)], p=1.0)(x)
        if "shift" in self.augmentations_name:
            x = RandomShiftsAug(16)(x)
        return x


def get_transform(name, size=None, size_hw=None):
    """Return a preprocessing transform by name.

    ``size`` is used for square-input transforms (``resize``, legacy
    ``jitter+shift`` path). ``size_hw`` is used for the DINOv2 transform,
    which needs a rectangular ``(H, W)``.
    """
    if name.startswith("dinov2"):
        if size_hw is None:
            raise ValueError("DINOv2 transform requires size_hw=(H, W).")
        return DINOv2Transform(name, size_hw=size_hw)
    if name == "resize":
        return ResizeTransform(size)
    elif "shift" in name or "jitter" in name:
        return ShiftAndJitterTransform(name, size)
    else:
        raise ValueError(f"Unknown transform {name}")
