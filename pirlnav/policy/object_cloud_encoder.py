"""PTv1-style encoder for the egocentric object cloud.

Ports the ``SimpleObjectCloudEncoder`` from
``simple_object_cloud_encoder.ipynb`` and wraps it for the policy: input is a
single packed ``(B, MAX_OBJECTS, 4) float32`` tensor where each row is
``[class_idx, x, y, z]`` and rows with ``class_idx < 0`` are padding. The
wrapper handles padding-mask derivation and the empty-cloud edge case
(returns a literal-zero embedding without polluting any class embedding;
gradient through the multiplicative gate is also zero, so empty samples
contribute no parameter updates inside the encoder).
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


RPEMode = Literal["raw", "normalized", "direction_only", "log_distance"]


def compute_relative_positions(
    pos_i: torch.Tensor,                          # (B, N, 3)
    pos_j: torch.Tensor,                          # (B, M, 3)
    mode: RPEMode = "log_distance",
    scene_scale: Optional[torch.Tensor] = None,   # (B,) only for "normalized"
) -> torch.Tensor:
    """Relative offsets from every point in ``pos_i`` to every point in ``pos_j``.

    Returns ``(B, N, M, 3)``. Modes::

        raw            : metric offset p_i - p_j.
        direction_only : unit vector, no distance.
        log_distance   : unit vector scaled by log(1 + ||offset||).  (default)
        normalized     : offset divided by per-batch scene scale.
    """
    delta = pos_i.unsqueeze(2) - pos_j.unsqueeze(1)         # (B, N, M, 3)

    if mode == "raw":
        return delta

    dist = delta.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    direction = delta / dist

    if mode == "direction_only":
        return direction

    if mode == "log_distance":
        return direction * torch.log1p(dist)

    if mode == "normalized":
        if scene_scale is None:
            scene_scale = dist.flatten(1).max(dim=1).values.clamp(min=1e-6)
        return delta / scene_scale.view(-1, 1, 1, 1)

    raise ValueError(f"Unknown RPE mode: {mode}")


class VectorAttention(nn.Module):
    """PTv1-style full vector attention with additive RPE and pre-norm.

    For each query ``i`` and key ``j``::

        delta_ij  = MLP_pos(p_i - p_j)
        r_ij      = gamma(Q_i - K_j + delta_ij)
        alpha_ij  = softmax_j(r_ij)                # per-channel softmax
        y_i       = sum_j(alpha_ij * (V_j + delta_ij))
    """

    def __init__(
        self,
        d_model: int,
        rpe_mode: RPEMode = "log_distance",
        dropout: float = 0.0,
        use_separate_norms: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.rpe_mode = rpe_mode

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model) if use_separate_norms else self.norm_q

        self.proj_q = nn.Linear(d_model, d_model, bias=False)
        self.proj_k = nn.Linear(d_model, d_model, bias=False)
        self.proj_v = nn.Linear(d_model, d_model, bias=False)
        self.proj_out = nn.Linear(d_model, d_model)

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                              # (B, N, d)
        pos: torch.Tensor,                            # (B, N, 3)
        key_x: torch.Tensor,                          # (B, M, d)
        key_pos: torch.Tensor,                        # (B, M, 3)
        mask: Optional[torch.Tensor] = None,          # (B, M) bool, True=valid
        scene_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:                                # (B, N, d)
        x_norm = self.norm_q(x)
        kv_norm = self.norm_kv(key_x)

        Q = self.proj_q(x_norm)
        K = self.proj_k(kv_norm)
        V = self.proj_v(kv_norm)

        rel = compute_relative_positions(pos, key_pos, self.rpe_mode, scene_scale)
        delta = self.pos_mlp(rel)                     # (B, N, M, d)

        relation = self.gamma(Q.unsqueeze(2) - K.unsqueeze(1) + delta)

        if mask is not None:
            relation = relation.masked_fill(
                ~mask.unsqueeze(1).unsqueeze(-1), float("-inf")
            )

        attn = F.softmax(relation, dim=2)             # per-channel softmax over keys
        attn = self.attn_dropout(attn)

        Vp = V.unsqueeze(1) + delta                   # (B, N, M, d)
        out = (attn * Vp).sum(dim=2)                  # (B, N, d)
        out = self.out_dropout(self.proj_out(out))

        return x + out


class FFN(nn.Module):
    """Pre-norm feed-forward with residual."""

    def __init__(self, d_model: int, expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class ClassEmbedding(nn.Module):
    """Categorical class index -> ``d_model`` vector with LayerNorm."""

    def __init__(self, num_classes: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(num_classes, d_model)
        self.norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, class_idx: torch.Tensor) -> torch.Tensor:
        return self.norm(self.embed(class_idx))


class SimpleObjectCloudLayer(nn.Module):
    """Object self-attention + read-only CLS cross-attention + 2 FFNs."""

    def __init__(
        self,
        d_model: int,
        rpe_mode: RPEMode = "log_distance",
        ffn_expansion: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.obj_self_attn = VectorAttention(
            d_model, rpe_mode, dropout, use_separate_norms=False,
        )
        self.cls_cross_attn = VectorAttention(
            d_model, rpe_mode, dropout, use_separate_norms=True,
        )
        self.obj_ffn = FFN(d_model, ffn_expansion, dropout)
        self.cls_ffn = FFN(d_model, ffn_expansion, dropout)

    def forward(
        self,
        obj_feat: torch.Tensor,
        obj_pos: torch.Tensor,
        cls_feat: torch.Tensor,
        cls_pos: torch.Tensor,
        obj_mask: Optional[torch.Tensor] = None,
        scene_scale: Optional[torch.Tensor] = None,
    ):
        obj_feat = self.obj_self_attn(
            x=obj_feat, pos=obj_pos,
            key_x=obj_feat, key_pos=obj_pos,
            mask=obj_mask, scene_scale=scene_scale,
        )
        cls_feat = self.cls_cross_attn(
            x=cls_feat, pos=cls_pos,
            key_x=obj_feat, key_pos=obj_pos,
            mask=obj_mask, scene_scale=scene_scale,
        )
        obj_feat = self.obj_ffn(obj_feat)
        cls_feat = self.cls_ffn(cls_feat)
        return obj_feat, cls_feat


class SimpleObjectCloudEncoder(nn.Module):
    """Stripped-down PTv1-style encoder that summarises a set of categorically
    -labeled, positioned objects into a fixed CLS embedding.

    Inputs
    ------
    class_idx : (B, N) int64 in [0, num_classes).
    obj_pos   : (B, N, 3) float.
    agent_pos : (B, 3)    float (origin in agent-frame coords).
    obj_mask  : (B, N)    bool, True = valid object. Each sample MUST contain
                at least one valid object (the wrapper below enforces this).
    goal_class : (B,) int64 in [0, num_classes), optional. When provided
                *and* ``use_goal_conditioning`` was True at construction, the
                CLS token is biased by ``ClassEmbedding(goal_class)`` (option
                A) and each object feature receives a learned bias when its
                class matches the goal (option B). Otherwise ignored.

    Output
    ------
    cls_out : (B, out_dim).
    """

    def __init__(
        self,
        num_classes: int = 21,
        d_model: int = 64,
        out_dim: int = 32,
        num_layers: int = 2,
        rpe_mode: RPEMode = "log_distance",
        ffn_expansion: int = 2,
        dropout: float = 0.0,
        use_goal_conditioning: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.out_dim = out_dim
        self.rpe_mode = rpe_mode

        self.input_embed = ClassEmbedding(num_classes, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.layers = nn.ModuleList([
            SimpleObjectCloudLayer(
                d_model=d_model,
                rpe_mode=rpe_mode,
                ffn_expansion=ffn_expansion,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.cls_norm = nn.LayerNorm(d_model)
        self.cls_head = nn.Linear(d_model, out_dim)

        # Option B: per-object is_goal flag embedding.  Gated so that when
        # use_goal_conditioning is False the module is bit-identical to the
        # pre-goal-conditioning version (same RNG draws, same state_dict
        # keys, old checkpoints still load).
        if use_goal_conditioning:
            self.goal_flag_embed = nn.Embedding(2, d_model)
            nn.init.normal_(self.goal_flag_embed.weight, std=0.02)

    def forward(
        self,
        class_idx: torch.Tensor,
        obj_pos: torch.Tensor,
        agent_pos: torch.Tensor,
        obj_mask: torch.Tensor,
        goal_class: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = class_idx.shape[0]

        obj_feat = self.input_embed(class_idx)

        use_goal = goal_class is not None and hasattr(self, "goal_flag_embed")
        if use_goal:
            # Option B: tag goal-class objects with a learned bias.
            is_goal = (class_idx == goal_class.unsqueeze(1)) & obj_mask
            obj_feat = obj_feat + self.goal_flag_embed(is_goal.long())
            # Option A: bias the CLS query with the goal-class embedding so
            # the cross-attention is goal-aware from layer 1.
            cls_feat = (
                self.cls_token.expand(B, 1, self.d_model)
                + self.input_embed(goal_class).unsqueeze(1)
            )
        else:
            cls_feat = self.cls_token.expand(B, 1, self.d_model)

        cls_pos = agent_pos.unsqueeze(1)

        scene_scale = None
        if self.rpe_mode == "normalized":
            inv = ~obj_mask.unsqueeze(-1)
            pos_max = obj_pos.masked_fill(inv, float("-inf")).max(dim=1).values
            pos_min = obj_pos.masked_fill(inv, float("inf")).min(dim=1).values
            scene_scale = (pos_max - pos_min).norm(dim=-1).clamp(min=1e-6)

        for layer in self.layers:
            obj_feat, cls_feat = layer(
                obj_feat=obj_feat, obj_pos=obj_pos,
                cls_feat=cls_feat, cls_pos=cls_pos,
                obj_mask=obj_mask, scene_scale=scene_scale,
            )

        return self.cls_head(self.cls_norm(cls_feat.squeeze(1)))


class ObjectCloudEncoder(nn.Module):
    """Policy-facing wrapper around :class:`SimpleObjectCloudEncoder`.

    Consumes a single packed ``(B, MAX_OBJECTS, 4) float32`` tensor:
    each row is ``[class_idx, x, y, z]`` in agent-frame coordinates, with
    padding rows having ``class_idx < 0``. Produces ``(B, out_dim)``.

    Optional goal conditioning (enabled with ``use_goal_conditioning=True``
    at construction): callers may pass a ``goal_class: (B,) int64`` tensor
    to ``forward`` to bias the encoder toward goal-relevant objects. See
    :class:`SimpleObjectCloudEncoder` for details. When the flag is off (or
    no ``goal_class`` is supplied) the encoder behaves bit-identically to
    the pre-goal-conditioning version.

    Empty-cloud handling: when a sample has no valid objects, we make slot 0
    "valid" with a clamped class id (0) only so the per-channel softmax
    inside ``VectorAttention`` is well-defined, then multiply the encoder
    output by a 0/1 ``has_any`` mask. The forward output is a literal zero
    embedding for empty samples; the multiply also zeros the backward
    gradient through the encoder, so the class-0 embedding (and every other
    encoder param) sees no gradient from those samples.
    """

    def __init__(
        self,
        num_classes: int = 21,
        d_model: int = 64,
        out_dim: int = 32,
        num_layers: int = 2,
        rpe_mode: RPEMode = "log_distance",
        ffn_expansion: int = 2,
        dropout: float = 0.0,
        use_goal_conditioning: bool = False,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.inner = SimpleObjectCloudEncoder(
            num_classes=num_classes,
            d_model=d_model,
            out_dim=out_dim,
            num_layers=num_layers,
            rpe_mode=rpe_mode,
            ffn_expansion=ffn_expansion,
            dropout=dropout,
            use_goal_conditioning=use_goal_conditioning,
        )

    def forward(
        self,
        packed: torch.Tensor,
        goal_class: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``packed``: ``(B, MAX_OBJECTS, 4) float32``.

        ``goal_class``: optional ``(B,) int64`` tensor of the active goal
        category per sample. Only used when the encoder was constructed
        with ``use_goal_conditioning=True``.
        """
        if packed.dim() != 3 or packed.size(-1) != 4:
            raise ValueError(
                f"ObjectCloudEncoder expects (B, MAX_OBJECTS, 4); got {tuple(packed.shape)}"
            )

        class_idx_f = packed[..., 0]
        pos = packed[..., 1:4]

        mask = class_idx_f >= 0                      # (B, N) bool
        has_any = mask.any(dim=1, keepdim=True)      # (B, 1) bool

        safe_class = class_idx_f.clamp_min(0).long()
        safe_mask = mask.clone()
        # For samples with no valid objects, mark slot 0 as valid so the
        # per-channel softmax over keys never sees an all-(-inf) row.
        empty_rows = ~has_any.squeeze(1)
        if empty_rows.any():
            safe_mask[empty_rows, 0] = True

        agent_pos = pos.new_zeros(pos.size(0), 3)

        cls_out = self.inner(
            class_idx=safe_class,
            obj_pos=pos,
            agent_pos=agent_pos,
            obj_mask=safe_mask,
            goal_class=goal_class.long() if goal_class is not None else None,
        )

        # Zero out empty samples (and their gradient through this multiply).
        return cls_out * has_any.to(cls_out.dtype)
