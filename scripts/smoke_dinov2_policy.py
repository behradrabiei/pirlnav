"""Smoke test for the DINOv2 policy variant.

Exercises:
 * module construction via ``baseline_registry`` and ``from_config``
 * transform + encoder shape plumbing on a dummy RGB batch
 * param-count delta (trainable params should be dominated by the GRU, not by
   the visual encoder)
 * checkpoint round-trip: frozen DINOv2 weights must not enter state_dict,
   and a fresh model must be able to load a saved state_dict cleanly
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from gym import spaces
import numpy as np

import pirlnav.policy.visual_policy  # noqa: F401 - registers ObjectNavILMAEPolicy
from habitat_baselines.common.baseline_registry import baseline_registry
from pirlnav.config import get_config


def build_dummy_observation_space():
    return spaces.Dict(
        {
            "rgb": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
            "gps": spaces.Box(low=-1e5, high=1e5, shape=(2,), dtype=np.float32),
            "compass": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            "objectgoal": spaces.Box(low=0, high=5, shape=(1,), dtype=np.int64),
        }
    )


def build_dummy_action_space(n: int = 6):
    return spaces.Discrete(n)


def count_params(mod: torch.nn.Module):
    total = sum(p.numel() for p in mod.parameters())
    trainable = sum(p.numel() for p in mod.parameters() if p.requires_grad)
    return total, trainable


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device = {device}")

    cfg = get_config("configs/experiments/il_objectnav_mp3d_dinov2.yaml")
    cfg.defrost()
    cfg.RUN_TYPE = "train"
    cfg.freeze()

    obs_space = build_dummy_observation_space()
    act_space = build_dummy_action_space(n=6)

    policy_cls = baseline_registry.get_policy(cfg.IL.POLICY.name)
    assert policy_cls is not None, f"Policy not registered: {cfg.IL.POLICY.name}"
    policy = policy_cls.from_config(cfg, obs_space, act_space)
    policy = policy.to(device)

    net = policy.net
    total, trainable = count_params(policy)
    visual_total, visual_trainable = count_params(net.visual_encoder)
    print(f"[smoke] policy params: total={total/1e6:.2f}M trainable={trainable/1e6:.2f}M")
    print(f"[smoke] visual_encoder params (includes frozen DINOv2 backbone): "
          f"total={visual_total/1e6:.2f}M trainable={visual_trainable/1e6:.2f}M")

    state_dict = policy.state_dict()
    state_dict_numel = sum(v.numel() for v in state_dict.values())
    print(f"[smoke] policy.state_dict() total tensors numel = {state_dict_numel/1e6:.2f}M")
    backbone_keys = [k for k in state_dict if "backbone" in k and "visual_encoder" in k]
    assert not backbone_keys, (
        f"DINOv2 backbone params leaked into state_dict: {backbone_keys[:5]}"
    )
    print(f"[smoke] DINOv2 backbone correctly excluded from state_dict.")

    # Exercise the pieces we actually changed: transform + DINOv2 encoder +
    # visual_fc. The rest of the forward path (GRU, action head) is unchanged
    # relative to the OVRL variant, so we don't need to round-trip through the
    # episode-packing seq_forward path here.
    B = 3
    rgb = torch.randint(0, 256, (B, 480, 640, 3), dtype=torch.uint8, device=device)

    policy.eval()
    net.visual_encoder.eval()
    with torch.no_grad():
        rgb_t = net.visual_transform(rgb, N=B)
    expected_hw = (cfg.POLICY.RGB_ENCODER.dinov2_resize_h,
                   cfg.POLICY.RGB_ENCODER.dinov2_resize_w)
    print(f"[smoke] post-transform shape: {tuple(rgb_t.shape)} (expected ({B}, 3, {expected_hw[0]}, {expected_hw[1]}))")
    assert rgb_t.shape == (B, 3, expected_hw[0], expected_hw[1])
    assert rgb_t.dtype == torch.float32
    assert 0.0 <= rgb_t.min().item() and rgb_t.max().item() <= 1.0 + 1e-3

    with torch.no_grad():
        cls = net.visual_encoder(rgb_t)
    print(f"[smoke] CLS shape: {tuple(cls.shape)} (expected ({B}, {cfg.POLICY.RGB_ENCODER.dinov2_output_dim}))")
    assert cls.shape == (B, cfg.POLICY.RGB_ENCODER.dinov2_output_dim)

    with torch.no_grad():
        proj = net.visual_fc(cls)
    print(f"[smoke] visual_fc output shape: {tuple(proj.shape)} (expected ({B}, {cfg.POLICY.RGB_ENCODER.hidden_size}))")
    assert proj.shape == (B, cfg.POLICY.RGB_ENCODER.hidden_size)

    tmp_ckpt = Path("/tmp/dinov2_smoke_ckpt.pth")
    torch.save(state_dict, tmp_ckpt)
    print(f"[smoke] saved checkpoint -> {tmp_ckpt} ({tmp_ckpt.stat().st_size/1e6:.2f} MB)")

    policy2 = policy_cls.from_config(cfg, obs_space, act_space).to(device)
    missing, unexpected = policy2.load_state_dict(state_dict, strict=False)
    print(f"[smoke] load_state_dict missing keys: {len(missing)}  unexpected: {len(unexpected)}")
    if missing:
        print(f"  first missing: {missing[:5]}")
    if unexpected:
        print(f"  first unexpected: {unexpected[:5]}")
    assert not unexpected, "Unexpected keys when loading checkpoint"

    # Reuse the already-augmented rgb_t to isolate encoder/fc determinism from
    # the stochastic (jitter+shift) transform.
    policy2.eval()
    policy2.net.visual_encoder.eval()
    with torch.no_grad():
        cls2 = policy2.net.visual_encoder(rgb_t)
        proj2 = policy2.net.visual_fc(cls2)
    max_diff_cls = (cls2 - cls).abs().max().item()
    max_diff_proj = (proj2 - proj).abs().max().item()
    print(f"[smoke] reload diff (same pre-aug input): CLS={max_diff_cls:.3e}  visual_fc={max_diff_proj:.3e}")
    assert max_diff_cls < 1e-4, f"DINOv2 forward diverges after reload: {max_diff_cls}"
    assert max_diff_proj < 1e-4, f"visual_fc forward diverges after reload: {max_diff_proj}"

    # Confirm the DINOv2 backbone is still frozen (all params require_grad=False).
    backbone = net.visual_encoder.backbone
    n_trainable_backbone = sum(int(p.requires_grad) for p in backbone.parameters())
    print(f"[smoke] DINOv2 backbone trainable params: {n_trainable_backbone} (expected 0)")
    assert n_trainable_backbone == 0

    print("[smoke] OK")


if __name__ == "__main__":
    main()
