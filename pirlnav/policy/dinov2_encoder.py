"""Frozen DINOv2 visual encoder.

Wraps a HuggingFace DINOv2 model so it plugs into ``ObjectNavILMAENet`` as a
drop-in replacement for the trainable OVRL ResNet-50. Matches the preprocessing
used in ``collect_demonstrations.py`` (ImageNet normalize + DINOv2 forward,
return ``last_hidden_state[:, 0]``).

Design notes
------------
* The backbone is always frozen (``requires_grad_(False)`` + ``.eval()``).
* The backbone is attached via ``object.__setattr__`` rather than the usual
  ``nn.Module`` setter. This keeps it functional (forward, device moves) but
  excludes its ~86 M parameters from the parent ``state_dict``, so each saved
  checkpoint only stores the small trainable heads.
* ``_apply`` is overridden so ``.to()`` / ``.cuda()`` / dtype casts propagate
  to the hidden backbone.
* ``train()`` is overridden so the backbone stays in eval mode even when the
  surrounding policy is switched to train mode (no dropout / BN drift).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class DINOv2VisualEncoder(nn.Module):
    """Frozen DINOv2 encoder that returns the CLS token."""

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        resize_hw: Tuple[int, int] = (476, 630),
        output_dim: int = 768,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.resize_hw = tuple(resize_hw)
        self.output_size = int(output_dim)
        self.is_blind = False

        from transformers import AutoModel

        backbone = AutoModel.from_pretrained(model_name)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)

        object.__setattr__(self, "backbone", backbone)

        self.register_buffer(
            "mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    def _apply(self, fn):
        out = super()._apply(fn)
        self.backbone._apply(fn)
        return out

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """Run DINOv2 on a preprocessed RGB batch.

        Expects ``rgb`` as ``(B, 3, H, W)`` floats in ``[0, 1]`` at the
        encoder's resize size. ImageNet normalization is applied here so the
        upstream transform only has to handle resize + augmentations.
        Returns the CLS token of shape ``(B, output_size)``.
        """
        if rgb.dtype != self.mean.dtype:
            rgb = rgb.to(self.mean.dtype)
        x = (rgb - self.mean) / self.std
        out = self.backbone(pixel_values=x)
        return out.last_hidden_state[:, 0]
