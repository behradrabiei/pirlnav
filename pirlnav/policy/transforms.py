from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter, RandomApply


SizeT = Union[int, Tuple[int, int]]


def _as_hw(size: SizeT) -> Tuple[int, int]:
    if isinstance(size, int):
        return size, size
    h, w = size
    return int(h), int(w)


class RandomShiftsAug(nn.Module):
    """Random pixel-level shift augmentation.

    Originally required square inputs; generalized to rectangular so it can be
    used with DINOv2's 476x630 pre-processing.
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
        x = TF.resize(x, self.size)
        x = TF.center_crop(x, output_size=self.size)
        x = x.float() / 255.0
        return x


class ShiftAndJitterTransform(Transform):
    def __init__(self, augmentations_name, size):
        self.size = size
        self.augmentations_name = augmentations_name

    def apply(self, x):
        x = x.permute(0, 3, 1, 2)
        x = TF.resize(x, self.size)
        x = TF.center_crop(x, output_size=self.size)
        x = x.float() / 255.0
        if "jitter" in self.augmentations_name:
            x = RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.4)], p=1.0)(x)
        if "shift" in self.augmentations_name:
            x = RandomShiftsAug(16)(x)
        return x


class DINOv2Transform(Transform):
    """Preprocessing for the frozen DINOv2 visual encoder.

    Center-crops RGB to a rectangular ``(H, W)`` (default 476x630 = nearest
    multiple of patch_size=14 below the habitat 480x640 sensor), divides by
    255, then optionally applies jitter and/or shift augmentations. This
    matches ``compute_dino_cls`` in the reference
    ``collect_demonstrations.py`` pipeline. ImageNet normalization is
    applied inside the encoder itself so this stays symmetric with the
    other transforms (they all emit tensors in ``[0, 1]``).
    """

    def __init__(self, augmentations_name: str, size_hw: Tuple[int, int]):
        self.augmentations_name = augmentations_name
        self.size_hw = _as_hw(size_hw)

    def apply(self, x):
        x = x.permute(0, 3, 1, 2)
        x = TF.center_crop(x, output_size=list(self.size_hw))
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
