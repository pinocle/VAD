"""Memory-guided RGB patch DiT and continuous flow-matching sampler."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.pipelines.rgb_patch_pipeline import RGBPatchModelConfig, RGBPatchShape


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Build sinusoidal embeddings for continuous flow time values."""

    half_dim = dim // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
        / max(half_dim - 1, 1)
    )
    values = timesteps.float()[:, None] * frequencies[None]
    embedding = torch.cat((values.sin(), values.cos()), dim=-1)
    return torch.nn.functional.pad(embedding, (0, dim % 2))


def modulate(tokens: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer-normalization modulation."""

    return tokens * (1.0 + scale[:, None]) + shift[:, None]


class MemoryGuidedAppearanceMotionCondition(nn.Module):
    """Turn past RGB patches into appearance-motion tokens guided by learned memory."""

    def __init__(
        self,
        *,
        shape: RGBPatchShape,
        hidden_size: int,
        memory_size: int,
        temperature: float,
    ) -> None:
        super().__init__()
        self.shape = shape
        self.hidden_size = hidden_size
        self.temperature = temperature
        self.appearance_proj = nn.Linear(shape.patch_dim, hidden_size)
        self.motion_proj = nn.Linear(shape.patch_dim, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.temporal_pos = nn.Parameter(torch.zeros(1, shape.context_frames, 1, hidden_size))
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, shape.num_patches, hidden_size))
        self.memory_keys = nn.Parameter(torch.randn(memory_size, hidden_size) * 0.02)
        self.memory_values = nn.Parameter(torch.randn(memory_size, hidden_size) * 0.02)
        self.memory_norm = nn.LayerNorm(hidden_size)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cross-attention condition tokens and memory retrieval distance."""

        expected = (
            self.shape.context_frames,
            self.shape.num_patches,
            self.shape.patch_dim,
        )
        if context.ndim != 4 or tuple(context.shape[1:]) != expected:
            raise ValueError(
                "context must be shaped [B, T_context, P, patch_dim]: "
                f"got {tuple(context.shape)}, expected [B, {expected[0]}, {expected[1]}, {expected[2]}]"
            )

        motion = torch.zeros_like(context)
        motion[:, 1:] = context[:, 1:] - context[:, :-1]
        tokens = self.appearance_proj(context) + self.motion_proj(motion)
        tokens = self.norm(tokens + self.temporal_pos + self.spatial_pos)

        descriptor = tokens.mean(dim=(1, 2))
        normalized_descriptor = torch.nn.functional.normalize(descriptor, dim=-1)
        normalized_keys = torch.nn.functional.normalize(self.memory_keys, dim=-1)
        logits = normalized_descriptor @ normalized_keys.T / self.temperature
        weights = logits.softmax(dim=-1)
        retrieved = weights @ self.memory_values
        memory_distance = (
            normalized_descriptor - torch.nn.functional.normalize(retrieved, dim=-1)
        ).square()
        memory_distance = memory_distance.mean(dim=-1)

        guided_tokens = tokens + self.memory_norm(retrieved)[:, None, None]
        return guided_tokens.flatten(1, 2), memory_distance


class DiTBlock(nn.Module):
    """Self-attention, cross-attention, and adaptive-LN MLP DiT block."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm_cross = nn.LayerNorm(hidden_size)
        self.norm_mlp = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6))

    def forward(
        self,
        tokens: torch.Tensor,
        condition: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one conditioned DiT block."""

        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.ada_ln(
            time_embedding
        ).chunk(6, dim=-1)
        attention_input = modulate(self.norm_self(tokens), shift_attn, scale_attn)
        self_output, _ = self.self_attn(
            attention_input, attention_input, attention_input, need_weights=False
        )
        tokens = tokens + gate_attn[:, None] * self_output
        cross_output, _ = self.cross_attn(
            self.norm_cross(tokens), condition, condition, need_weights=False
        )
        tokens = tokens + cross_output
        mlp_input = modulate(self.norm_mlp(tokens), shift_mlp, scale_mlp)
        return tokens + gate_mlp[:, None] * self.mlp(mlp_input)


class RGBPatchDiT(nn.Module):
    """DiT that predicts flow velocity for future fixed-grid RGB patches."""

    def __init__(self, *, shape: RGBPatchShape, config: RGBPatchModelConfig) -> None:
        super().__init__()
        self.shape = shape
        self.config = config
        self.hidden_size = config.hidden_size
        self.gradient_checkpointing = False
        self.input_proj = nn.Linear(shape.patch_dim, config.hidden_size)
        self.output_norm = nn.LayerNorm(config.hidden_size)
        self.output_proj = nn.Linear(config.hidden_size, shape.patch_dim)
        self.future_temporal_pos = nn.Parameter(
            torch.zeros(1, shape.future_frames, 1, config.hidden_size)
        )
        self.future_spatial_pos = nn.Parameter(
            torch.zeros(1, 1, shape.num_patches, config.hidden_size)
        )
        self.condition_encoder = MemoryGuidedAppearanceMotionCondition(
            shape=shape,
            hidden_size=config.hidden_size,
            memory_size=config.memory_size,
            temperature=config.memory_temperature,
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.SiLU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(config.hidden_size, config.num_heads, config.mlp_ratio, config.dropout)
                for _ in range(config.num_layers)
            ]
        )
        nn.init.trunc_normal_(self.future_temporal_pos, std=0.02)
        nn.init.trunc_normal_(self.future_spatial_pos, std=0.02)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        """Enable activation checkpointing for DiT blocks during training."""

        self.gradient_checkpointing = bool(enabled)

    def forward(
        self,
        noisy_target: torch.Tensor,
        time_values: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict future patch velocity and return per-sample memory distance."""

        expected = (
            self.shape.future_frames,
            self.shape.num_patches,
            self.shape.patch_dim,
        )
        if noisy_target.ndim != 4 or tuple(noisy_target.shape[1:]) != expected:
            raise ValueError(
                "noisy_target must be shaped [B, T_future, P, patch_dim]: "
                f"got {tuple(noisy_target.shape)}, expected [B, {expected[0]}, {expected[1]}, {expected[2]}]"
            )
        if time_values.ndim != 1 or time_values.shape[0] != noisy_target.shape[0]:
            raise ValueError("time_values must be shaped [B]")

        condition, memory_distance = self.condition_encoder(context)
        tokens = self.input_proj(noisy_target)
        tokens = tokens + self.future_temporal_pos + self.future_spatial_pos
        tokens = tokens.flatten(1, 2)
        embedded_time = time_values * self.config.time_embedding_scale
        time_embedding = self.time_mlp(timestep_embedding(embedded_time, self.hidden_size))
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                tokens = checkpoint(block, tokens, condition, time_embedding, use_reentrant=False)
            else:
                tokens = block(tokens, condition, time_embedding)
        velocity = self.output_proj(self.output_norm(tokens))
        velocity = velocity.reshape(
            noisy_target.shape[0],
            self.shape.future_frames,
            self.shape.num_patches,
            self.shape.patch_dim,
        )
        return velocity, memory_distance


class FlowMatcher:
    """Train and sample a velocity field between Gaussian noise and RGB patches."""

    def __init__(self, *, inference_steps: int) -> None:
        if inference_steps <= 0:
            raise ValueError("inference_steps must be positive")
        self.inference_steps = inference_steps

    def prepare_training_pair(
        self,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a flow time and produce noisy patches plus target velocity."""

        if target.ndim != 4:
            raise ValueError("target must be shaped [B, T_future, P, patch_dim]")
        noise = torch.randn_like(target)
        time_values = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
        noisy_target = (1.0 - time_values[:, None, None, None]) * noise + time_values[
            :, None, None, None
        ] * target
        velocity_target = target - noise
        return noisy_target, time_values, velocity_target

    @torch.inference_mode()
    def sample(
        self,
        model: RGBPatchDiT,
        *,
        context: torch.Tensor,
        target_shape: tuple[int, int, int, int],
        inference_steps: int | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate the learned velocity field from noise to future RGB patches."""

        steps = inference_steps or self.inference_steps
        if steps <= 0:
            raise ValueError("inference_steps must be positive")
        if initial_noise is None:
            generated = torch.randn(target_shape, device=context.device, dtype=context.dtype)
        else:
            if tuple(initial_noise.shape) != target_shape:
                raise ValueError(
                    "initial_noise shape must match target_shape: "
                    f"got {tuple(initial_noise.shape)}, expected {target_shape}"
                )
            generated = initial_noise.to(device=context.device, dtype=context.dtype).clone()
        memory_distance = torch.zeros(target_shape[0], device=context.device, dtype=context.dtype)
        step_size = 1.0 / steps
        for step in range(steps):
            time_values = torch.full(
                (target_shape[0],),
                step / steps,
                device=context.device,
                dtype=context.dtype,
            )
            velocity, memory_distance = model(generated, time_values, context)
            generated = generated + step_size * velocity
        return generated, memory_distance
