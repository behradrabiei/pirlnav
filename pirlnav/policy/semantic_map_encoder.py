"""CNN encoder for the egocentric semantic + occupancy map.

Wraps the repo's ``VisualEncoder`` (resnet-with-GroupNorm family) with
``input_channels = NUM_CHANNELS + 1 = 24`` so it can ingest a one-hot encoding
of the (H, W) int8 label map produced by :class:`SemanticMapSensor` (UNKNOWN
included as an explicit channel). Trained from scratch.

The sensor stores ``(H, W)`` int8 in the rollout buffer (~17 MB at 256x256,
65 steps, 4 envs) and the one-hot conversion is done here in ``forward`` so we
don't pay the 100x memory cost of putting a (24, H, W) float tensor in the
buffer.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from pirlnav.policy.visual_encoder import VisualEncoder
from pirlnav.task.semantic_map import NUM_CHANNELS


class SemanticMapEncoder(nn.Module):
    """``(B, H, W) int8`` label map -> ``(B, output_dim)`` embedding.

    ``label_map`` values: ``-1`` UNKNOWN, ``0`` FREE, ``1`` OCCUPIED, ``2..22``
    goal classes. We map ``-1 -> 0``, ``0..22 -> 1..23`` and one-hot to 24
    channels so UNKNOWN is an explicit input feature.
    """

    NUM_INPUT_CHANNELS: int = NUM_CHANNELS + 1  # 24 (UNKNOWN + 23 labels)

    def __init__(
        self,
        image_size: int = 256,
        output_dim: int = 32,
        backbone: str = "resnet18",
        resnet_baseplanes: int = 32,
        resnet_ngroups: int = 16,
    ) -> None:
        super().__init__()
        self.backbone = VisualEncoder(
            image_size=image_size,
            backbone=backbone,
            input_channels=self.NUM_INPUT_CHANNELS,
            resnet_baseplanes=resnet_baseplanes,
            resnet_ngroups=resnet_ngroups,
            normalize_visual_inputs=False,
            avgpooled_image=False,
            drop_path_rate=0.0,
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(self.backbone.output_size), output_dim),
        )
        self.output_dim = output_dim

    def forward(self, label_map):  # (B, H, W) int8 in [-1, NUM_CHANNELS - 1]
        x = (label_map.long() + 1).clamp_(0, self.NUM_INPUT_CHANNELS - 1)
        x = F.one_hot(x, num_classes=self.NUM_INPUT_CHANNELS)
        x = x.permute(0, 3, 1, 2).float()
        return self.fc(self.backbone(x))
