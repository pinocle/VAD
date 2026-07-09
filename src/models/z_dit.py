"""C_high/C_low-conditioned Z flow transformer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

CONDITION_MODE_BASELINE = "baseline"
CONDITION_MODE_APPEARANCE_ONLY = "appearance_only"
CONDITION_MODE_TRACK_ONLY = "track_only"
CONDITION_MODE_APPEARANCE_TRACK = "appearance_track"
CONDITION_MODES = {
    CONDITION_MODE_BASELINE,
    CONDITION_MODE_APPEARANCE_ONLY,
    CONDITION_MODE_TRACK_ONLY,
    CONDITION_MODE_APPEARANCE_TRACK,
}
PATCH_CONDITION_MODES = {
    CONDITION_MODE_APPEARANCE_ONLY,
    CONDITION_MODE_TRACK_ONLY,
    CONDITION_MODE_APPEARANCE_TRACK,
}


def normalize_condition_mode(mode: str) -> str:
    """Return a normalized patch-condition mode."""

    value = str(mode).strip().lower()
    if value not in CONDITION_MODES:
        raise ValueError(
            "condition.mode must be baseline, appearance_only, track_only, or appearance_track"
        )
    return value


def condition_mode_uses_appearance(mode: str) -> bool:
    """Return whether a condition mode consumes appearance patch grids."""

    return normalize_condition_mode(mode) in {
        CONDITION_MODE_APPEARANCE_ONLY,
        CONDITION_MODE_APPEARANCE_TRACK,
    }


def condition_mode_uses_track(mode: str) -> bool:
    """Return whether a condition mode consumes dense track grids."""

    return normalize_condition_mode(mode) in {
        CONDITION_MODE_TRACK_ONLY,
        CONDITION_MODE_APPEARANCE_TRACK,
    }


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Return sinusoidal timestep embeddings."""

    half = dim // 2
    exponent = -math.log(10000.0) * torch.arange(half, device=timesteps.device) / max(half - 1, 1)
    embedding = timesteps.float()[:, None] * torch.exp(exponent)[None]
    embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=-1)
    if dim % 2:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply AdaLN scale and shift."""

    return x * (1 + scale[:, None]) + shift[:, None]


def high_token_grid(num_tokens: int) -> tuple[int, int]:
    """Return a spatial grid for C_high patch tokens."""

    side = int(math.sqrt(num_tokens))
    if side * side == num_tokens:
        return side, side
    return 1, num_tokens


class HighAdapter(nn.Module):
    """Project C_high encoder tokens into DiT hidden tokens."""

    def __init__(
        self,
        *,
        input_dim: int,
        high_frames: int,
        high_tokens: int,
        hidden_size: int,
        max_tokens: int | None,
        use_pos_embedding: bool,
        token_reduction: str = "uniform",
        foreground_ratio: float = 0.75,
        aligned_frames: int | None = None,
        aligned_grid_h: int | None = None,
        aligned_grid_w: int | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.high_frames = high_frames
        self.high_tokens = high_tokens
        self.hidden_size = hidden_size
        self.max_tokens = max_tokens
        self.token_reduction = token_reduction
        self.foreground_ratio = foreground_ratio
        self.use_pos_embedding = use_pos_embedding
        self.aligned_frames = aligned_frames
        self.aligned_grid_h = aligned_grid_h
        self.aligned_grid_w = aligned_grid_w
        if self.token_reduction == "aligned_grid" and (
            aligned_frames is None or aligned_grid_h is None or aligned_grid_w is None
        ):
            raise ValueError(
                "token_reduction='aligned_grid' requires future frame and grid metadata"
            )

        self.proj = nn.Linear(input_dim, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        if use_pos_embedding:
            self.temporal_pos = nn.Parameter(torch.zeros(1, high_frames, 1, hidden_size))
            self.token_pos = nn.Parameter(torch.zeros(1, 1, high_tokens, hidden_size))
            nn.init.trunc_normal_(self.temporal_pos, std=0.02)
            nn.init.trunc_normal_(self.token_pos, std=0.02)

    def forward(
        self,
        high: torch.Tensor,
        token_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return C_high condition tokens shaped ``[B, Lc, D]``."""

        x = self.project_high(high)
        if self.token_reduction == "aligned_grid":
            return self.align_to_z_grid(x)
        if self.token_reduction == "foreground_background" and token_scores is not None:
            return self.select_foreground_background_tokens(x, token_scores)
        x = x.flatten(1, 2)
        if self.max_tokens is not None and x.shape[1] > self.max_tokens:
            indices = torch.linspace(0, x.shape[1] - 1, self.max_tokens, device=x.device).round()
            x = x.index_select(1, indices.long())
        return x

    def project_high(self, high: torch.Tensor) -> torch.Tensor:
        """Project cached C_high features while preserving ``[B, T, N]`` layout."""

        if high.ndim != 4:
            raise ValueError("high must be shaped [B, T, N, D]")
        _, frames, tokens, dim = high.shape
        if (frames, tokens, dim) != (self.high_frames, self.high_tokens, self.input_dim):
            raise ValueError(
                "C_high shape does not match model initialization: "
                f"got {(frames, tokens, dim)}, expected "
                f"{(self.high_frames, self.high_tokens, self.input_dim)}"
            )

        x = self.proj(high)
        if self.use_pos_embedding:
            x = x + self.temporal_pos + self.token_pos
        x = self.norm(x)
        return x

    def align_to_z_grid(self, tokens: torch.Tensor) -> torch.Tensor:
        """Pool C_high patch tokens onto the Z token grid."""

        batch, frames, num_tokens, hidden = tokens.shape
        grid_h, grid_w = high_token_grid(num_tokens)
        if grid_h * grid_w != num_tokens or grid_h == 1:
            raise ValueError(
                f"token_reduction='aligned_grid' requires square patch tokens, got {num_tokens}"
            )
        target_h = int(self.aligned_grid_h)
        target_w = int(self.aligned_grid_w)
        x = tokens.reshape(batch, frames, grid_h, grid_w, hidden)
        x = x.permute(0, 1, 4, 2, 3).reshape(batch * frames, hidden, grid_h, grid_w)
        x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(target_h, target_w))
        x = x.reshape(batch, frames, hidden, target_h * target_w)
        x = x.permute(0, 1, 3, 2).contiguous()
        return x.flatten(1, 2)

    def future_aligned_tokens(self, aligned_high_tokens: torch.Tensor) -> torch.Tensor:
        """Reduce aligned C_high tokens over time and repeat them for future frames."""

        if self.token_reduction != "aligned_grid":
            raise ValueError("future_aligned_tokens is only valid for aligned_grid mode")
        if aligned_high_tokens.ndim != 3:
            raise ValueError("aligned_high_tokens must be shaped [B, T*N, D]")
        batch, length, hidden = aligned_high_tokens.shape
        target_tokens = int(self.aligned_grid_h) * int(self.aligned_grid_w)
        expected_length = self.high_frames * target_tokens
        if (length, hidden) != (expected_length, self.hidden_size):
            raise ValueError(
                "aligned C_high token shape does not match model grid: "
                f"got {(length, hidden)}, expected {(expected_length, self.hidden_size)}"
            )
        x = aligned_high_tokens.reshape(batch, self.high_frames, target_tokens, hidden)
        x = x.mean(dim=1, keepdim=True)
        x = x.expand(batch, int(self.aligned_frames), target_tokens, hidden)
        return x.reshape(batch, int(self.aligned_frames) * target_tokens, hidden)

    def select_foreground_background_tokens(
        self,
        tokens: torch.Tensor,
        token_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Select high-C_low foreground tokens plus low-C_low background tokens per frame."""

        if token_scores.shape != tokens.shape[:3]:
            raise ValueError(
                "C_high token_scores must be shaped [B, T, N] and match C_high tokens: "
                f"got {tuple(token_scores.shape)}, expected {tuple(tokens.shape[:3])}"
            )
        batch, frames, num_tokens, hidden = tokens.shape
        if self.max_tokens is None or frames * num_tokens <= self.max_tokens:
            return tokens.flatten(1, 2)
        if self.max_tokens < frames:
            raise ValueError(
                "high_adapter.max_tokens must be at least high_frames when "
                "token_reduction='foreground_background'"
            )

        per_frame_tokens = max(1, min(num_tokens, self.max_tokens // frames))
        if per_frame_tokens == 1:
            foreground_tokens = 1
        else:
            foreground_tokens = int(round(per_frame_tokens * self.foreground_ratio))
            foreground_tokens = min(max(foreground_tokens, 1), per_frame_tokens - 1)
        background_tokens = per_frame_tokens - foreground_tokens

        foreground_indices = token_scores.topk(foreground_tokens, dim=-1).indices
        if background_tokens > 0:
            background_indices = (-token_scores).topk(background_tokens, dim=-1).indices
            indices = torch.cat([foreground_indices, background_indices], dim=-1)
        else:
            indices = foreground_indices

        gather_indices = indices.unsqueeze(-1).expand(batch, frames, per_frame_tokens, hidden)
        return tokens.gather(dim=2, index=gather_indices).flatten(1, 2)


class ZPatchEmbed(nn.Module):
    """Patchify VAE Z sequences into DiT tokens and invert them."""

    def __init__(
        self,
        *,
        future_frames: int,
        z_channels: int,
        z_height: int,
        z_width: int,
        patch_size: int,
        hidden_size: int,
        use_temporal_pos: bool,
        use_spatial_pos: bool,
    ) -> None:
        super().__init__()
        if z_height % patch_size or z_width % patch_size:
            raise ValueError("z_adapter.patch_size must divide Z height and width")

        self.future_frames = future_frames
        self.z_channels = z_channels
        self.z_height = z_height
        self.z_width = z_width
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.grid_h = z_height // patch_size
        self.grid_w = z_width // patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.patch_dim = z_channels * patch_size * patch_size
        self.use_temporal_pos = use_temporal_pos
        self.use_spatial_pos = use_spatial_pos

        self.proj = nn.Linear(self.patch_dim, hidden_size)
        self.unproj = nn.Linear(hidden_size, self.patch_dim)
        if use_temporal_pos:
            self.temporal_pos = nn.Parameter(torch.zeros(1, future_frames, 1, hidden_size))
            nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        if use_spatial_pos:
            self.spatial_pos = nn.Parameter(torch.zeros(1, 1, self.num_patches, hidden_size))
            nn.init.trunc_normal_(self.spatial_pos, std=0.02)

    def patchify(self, z: torch.Tensor) -> torch.Tensor:
        """Return flattened Z patches shaped ``[B, T*N, patch_dim]``."""

        if z.ndim != 5:
            raise ValueError("z must be shaped [B, T, C, H, W]")
        batch, frames, channels, height, width = z.shape
        expected = (
            self.future_frames,
            self.z_channels,
            self.z_height,
            self.z_width,
        )
        if (frames, channels, height, width) != expected:
            raise ValueError(
                "Z shape does not match model initialization: "
                f"got {(frames, channels, height, width)}, expected {expected}"
            )

        patch = self.patch_size
        x = z.reshape(batch * frames, channels, height, width)
        x = x.unfold(2, patch, patch).unfold(3, patch, patch)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.reshape(batch, frames, self.num_patches, self.patch_dim)
        return x.flatten(1, 2)

    def unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return Z sequence shaped ``[B, T, C, H, W]``."""

        if tokens.ndim != 3:
            raise ValueError("tokens must be shaped [B, T*N, patch_dim]")
        batch, length, patch_dim = tokens.shape
        expected_length = self.future_frames * self.num_patches
        if (length, patch_dim) != (expected_length, self.patch_dim):
            raise ValueError(
                "token shape does not match Z grid: "
                f"got {(length, patch_dim)}, expected {(expected_length, self.patch_dim)}"
            )

        patch = self.patch_size
        x = tokens.reshape(
            batch,
            self.future_frames,
            self.grid_h,
            self.grid_w,
            self.z_channels,
            patch,
            patch,
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        return x.reshape(
            batch,
            self.future_frames,
            self.z_channels,
            self.z_height,
            self.z_width,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return Z tokens shaped ``[B, Lz, D]``."""

        batch = z.shape[0]
        x = self.proj(self.patchify(z))
        x = x.reshape(batch, self.future_frames, self.num_patches, self.hidden_size)
        if self.use_temporal_pos:
            x = x + self.temporal_pos
        if self.use_spatial_pos:
            x = x + self.spatial_pos
        return x.flatten(1, 2)

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Project DiT output tokens back to Z space."""

        return self.unpatchify(self.unproj(tokens))


class LowPatchAdapter(nn.Module):
    """Patchify simple C_low tensors into DiT condition tokens."""

    def __init__(
        self,
        *,
        low_frames: int,
        low_channels: int,
        low_height: int,
        low_width: int,
        patch_size: int,
        hidden_size: int,
        use_temporal_pos: bool,
        use_spatial_pos: bool,
    ) -> None:
        super().__init__()
        if low_height % patch_size or low_width % patch_size:
            raise ValueError("low_adapter.patch_size must divide C_low height and width")

        self.low_frames = low_frames
        self.low_channels = low_channels
        self.low_height = low_height
        self.low_width = low_width
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.grid_h = low_height // patch_size
        self.grid_w = low_width // patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.patch_dim = low_channels * patch_size * patch_size
        self.use_temporal_pos = use_temporal_pos
        self.use_spatial_pos = use_spatial_pos

        self.proj = nn.Linear(self.patch_dim, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        if use_temporal_pos:
            self.temporal_pos = nn.Parameter(torch.zeros(1, low_frames, 1, hidden_size))
            nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        if use_spatial_pos:
            self.spatial_pos = nn.Parameter(torch.zeros(1, 1, self.num_patches, hidden_size))
            nn.init.trunc_normal_(self.spatial_pos, std=0.02)

    def patchify(self, low: torch.Tensor) -> torch.Tensor:
        """Return flattened C_low patches shaped ``[B, T*N, patch_dim]``."""

        if low.ndim != 5:
            raise ValueError("C_low must be shaped [B, T, C, H, W]")
        batch, frames, channels, height, width = low.shape
        expected = (
            self.low_frames,
            self.low_channels,
            self.low_height,
            self.low_width,
        )
        if (frames, channels, height, width) != expected:
            raise ValueError(
                "C_low shape does not match model initialization: "
                f"got {(frames, channels, height, width)}, expected {expected}"
            )

        patch = self.patch_size
        x = low.reshape(batch * frames, channels, height, width)
        x = x.unfold(2, patch, patch).unfold(3, patch, patch)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.reshape(batch, frames, self.num_patches, self.patch_dim)
        return x.flatten(1, 2)

    def forward(self, low: torch.Tensor) -> torch.Tensor:
        """Return C_low condition tokens shaped ``[B, Lm, D]``."""

        batch = low.shape[0]
        x = self.proj(self.patchify(low))
        x = x.reshape(batch, self.low_frames, self.num_patches, self.hidden_size)
        if self.use_temporal_pos:
            x = x + self.temporal_pos
        if self.use_spatial_pos:
            x = x + self.spatial_pos
        x = self.norm(x).flatten(1, 2)
        return x


class TrackGridAdapter(nn.Module):
    """Project dense C_track grids into DiT condition tokens."""

    def __init__(
        self,
        *,
        track_frames: int,
        track_channels: int,
        grid_h: int,
        grid_w: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        if track_frames <= 0:
            raise ValueError("track_frames must be positive")
        if track_channels <= 0:
            raise ValueError("track_channels must be positive")
        if grid_h <= 0 or grid_w <= 0:
            raise ValueError("track grid spatial dimensions must be positive")

        self.track_frames = track_frames
        self.track_channels = track_channels
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.hidden_size = hidden_size
        self.num_patches = grid_h * grid_w

        self.value_proj = nn.Linear(track_channels, hidden_size)
        self.type_embedding = nn.Parameter(torch.zeros(track_channels, hidden_size))
        self.temporal_pos = nn.Parameter(torch.zeros(1, track_frames, 1, 1, hidden_size))
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, grid_h, grid_w, hidden_size))
        self.norm = nn.LayerNorm(hidden_size)

        nn.init.trunc_normal_(self.type_embedding, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)

    def forward(self, track_grid: torch.Tensor) -> torch.Tensor:
        """Return C_track condition tokens shaped ``[B, T*H*W, D]``."""

        if track_grid.ndim != 5:
            raise ValueError("C_track must be shaped [B, T, C, H, W]")
        batch, frames, channels, height, width = track_grid.shape
        expected = (self.track_frames, self.track_channels, self.grid_h, self.grid_w)
        if (frames, channels, height, width) != expected:
            raise ValueError(
                "C_track shape does not match model initialization: "
                f"got {(frames, channels, height, width)}, expected {expected}"
            )

        grid = track_grid.float()
        x = grid.permute(0, 1, 3, 4, 2).contiguous()
        x = self.value_proj(x)
        type_tokens = torch.einsum("btchw,cd->bthwd", grid, self.type_embedding)
        x = x + type_tokens + self.temporal_pos + self.spatial_pos
        x = self.norm(x)
        return x.reshape(batch, frames * self.num_patches, self.hidden_size)


class PatchConditionAdapter(nn.Module):
    """Project dense appearance/track condition grids into DiT condition tokens."""

    def __init__(
        self,
        *,
        condition_frames: int,
        input_channels: int,
        grid_h: int,
        grid_w: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        if condition_frames <= 0:
            raise ValueError("condition_frames must be positive")
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if grid_h <= 0 or grid_w <= 0:
            raise ValueError("condition grid spatial dimensions must be positive")

        self.condition_frames = condition_frames
        self.input_channels = input_channels
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.hidden_size = hidden_size
        self.num_patches = grid_h * grid_w

        self.proj = nn.Linear(input_channels, hidden_size)
        self.temporal_pos = nn.Parameter(torch.zeros(1, condition_frames, 1, 1, hidden_size))
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, grid_h, grid_w, hidden_size))
        self.norm = nn.LayerNorm(hidden_size)

        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)

    def forward(self, condition_grid: torch.Tensor) -> torch.Tensor:
        """Return condition tokens shaped ``[B, T*H*W, D]`` using row-major order."""

        if condition_grid.ndim != 5:
            raise ValueError("condition_grid must be shaped [B, T, C, H, W]")
        batch, frames, channels, height, width = condition_grid.shape
        expected = (self.condition_frames, self.input_channels, self.grid_h, self.grid_w)
        if (frames, channels, height, width) != expected:
            raise ValueError(
                "condition_grid shape does not match model initialization: "
                f"got {(frames, channels, height, width)}, expected {expected}"
            )

        x = condition_grid.float().permute(0, 1, 3, 4, 2).contiguous()
        x = self.proj(x) + self.temporal_pos + self.spatial_pos
        x = self.norm(x)
        return x.reshape(batch, frames * self.num_patches, self.hidden_size)


class DiTBlock(nn.Module):
    """Transformer block with Z self-attention and condition cross-attention."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_cross = nn.LayerNorm(hidden_size)
        self.high_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_mlp = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )
        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6),
        )
        nn.init.zeros_(self.ada_ln[-1].weight)
        nn.init.zeros_(self.ada_ln[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        high: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Run one DiT block."""

        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.ada_ln(
            time_embedding
        ).chunk(6, dim=-1)

        attn_input = modulate(self.norm_self(x), shift_attn, scale_attn)
        attn_output, _ = self.self_attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + gate_attn[:, None] * attn_output

        normalized_high = self.high_norm(high)
        cross_output, _ = self.cross_attn(
            self.norm_cross(x),
            normalized_high,
            normalized_high,
            need_weights=False,
        )
        x = x + cross_output

        mlp_input = modulate(self.norm_mlp(x), shift_mlp, scale_mlp)
        x = x + gate_mlp[:, None] * self.mlp(mlp_input)
        return x


@dataclass(frozen=True)
class ZDiTShape:
    """Input tensor shapes used to initialize the Z DiT."""

    high_frames: int
    high_tokens: int
    high_dim: int
    future_frames: int
    z_channels: int
    z_height: int
    z_width: int
    low_frames: int | None = None
    low_channels: int | None = None
    low_height: int | None = None
    low_width: int | None = None
    track_frames: int | None = None
    track_channels: int | None = None
    track_grid_h: int | None = None
    track_grid_w: int | None = None


class ConditionedZDiT(nn.Module):
    """Velocity-prediction DiT conditioned on C_high and optional C_low features."""

    def __init__(
        self,
        *,
        shape: ZDiTShape,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        high_max_tokens: int | None,
        high_use_pos_embedding: bool,
        z_patch_size: int,
        z_use_temporal_pos: bool,
        z_use_spatial_pos: bool,
        high_token_reduction: str = "uniform",
        high_foreground_ratio: float = 0.75,
        low_patch_size: int = 16,
        low_use_temporal_pos: bool = True,
        low_use_spatial_pos: bool = True,
        condition_mode: str = CONDITION_MODE_BASELINE,
        use_track_grid: bool = False,
        track_grid_gate_init: float = 0.1,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("model.dit.hidden_size must be divisible by model.dit.num_heads")

        self.shape = shape
        self.hidden_size = hidden_size
        self.gradient_checkpointing = gradient_checkpointing
        self.condition_mode = normalize_condition_mode(condition_mode)
        self.uses_patch_condition = self.condition_mode in PATCH_CONDITION_MODES
        if shape.z_height % z_patch_size or shape.z_width % z_patch_size:
            raise ValueError("z_adapter.patch_size must divide Z height and width")
        z_grid_h = shape.z_height // z_patch_size
        z_grid_w = shape.z_width // z_patch_size

        self.high_adapter = None
        self.low_adapter = None
        if not self.uses_patch_condition:
            self.high_adapter = HighAdapter(
                input_dim=shape.high_dim,
                high_frames=shape.high_frames,
                high_tokens=shape.high_tokens,
                hidden_size=hidden_size,
                max_tokens=high_max_tokens,
                use_pos_embedding=high_use_pos_embedding,
                token_reduction=high_token_reduction,
                foreground_ratio=high_foreground_ratio,
                aligned_frames=shape.future_frames,
                aligned_grid_h=z_grid_h,
                aligned_grid_w=z_grid_w,
            )
            self.low_adapter = self.build_low_adapter(
                shape=shape,
                hidden_size=hidden_size,
                patch_size=low_patch_size,
                z_grid_h=z_grid_h,
                z_grid_w=z_grid_w,
                use_temporal_pos=low_use_temporal_pos,
                use_spatial_pos=low_use_spatial_pos,
            )

        legacy_track_grid = bool(use_track_grid) and not self.uses_patch_condition
        self.track_grid_adapter = self.build_track_grid_adapter(
            shape=shape,
            hidden_size=hidden_size,
            z_grid_h=z_grid_h,
            z_grid_w=z_grid_w,
            enabled=legacy_track_grid,
        )
        if self.track_grid_adapter is not None:
            self.track_grid_gate = nn.Parameter(torch.tensor(float(track_grid_gate_init)))
        self.patch_condition_adapter = self.build_patch_condition_adapter(
            shape=shape,
            hidden_size=hidden_size,
            z_grid_h=z_grid_h,
            z_grid_w=z_grid_w,
        )
        self.z_adapter = ZPatchEmbed(
            future_frames=shape.future_frames,
            z_channels=shape.z_channels,
            z_height=shape.z_height,
            z_width=shape.z_width,
            patch_size=z_patch_size,
            hidden_size=hidden_size,
            use_temporal_pos=z_use_temporal_pos,
            use_spatial_pos=z_use_spatial_pos,
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        if not self.uses_patch_condition and high_token_reduction == "aligned_grid":
            self.aligned_high_norm = nn.LayerNorm(hidden_size)
            self.aligned_high_gate = nn.Parameter(torch.ones(()))

    def build_low_adapter(
        self,
        *,
        shape: ZDiTShape,
        hidden_size: int,
        patch_size: int,
        z_grid_h: int,
        z_grid_w: int,
        use_temporal_pos: bool,
        use_spatial_pos: bool,
    ) -> LowPatchAdapter | None:
        """Build the optional C_low condition adapter when shape metadata exists."""

        low_shape = (
            shape.low_frames,
            shape.low_channels,
            shape.low_height,
            shape.low_width,
        )
        if all(value is None for value in low_shape):
            return None
        if any(value is None for value in low_shape):
            raise ValueError("C_low shape metadata must be complete when C_low is enabled")
        if int(shape.low_height) % patch_size or int(shape.low_width) % patch_size:
            raise ValueError("low_adapter.patch_size must divide C_low height and width")
        low_grid_h = int(shape.low_height) // patch_size
        low_grid_w = int(shape.low_width) // patch_size
        if (low_grid_h, low_grid_w) != (z_grid_h, z_grid_w):
            raise ValueError(
                "C_low patch grid must match Z patch grid: "
                f"got {(low_grid_h, low_grid_w)}, expected {(z_grid_h, z_grid_w)}"
            )
        return LowPatchAdapter(
            low_frames=int(shape.low_frames),
            low_channels=int(shape.low_channels),
            low_height=int(shape.low_height),
            low_width=int(shape.low_width),
            patch_size=patch_size,
            hidden_size=hidden_size,
            use_temporal_pos=use_temporal_pos,
            use_spatial_pos=use_spatial_pos,
        )

    def build_track_grid_adapter(
        self,
        *,
        shape: ZDiTShape,
        hidden_size: int,
        z_grid_h: int,
        z_grid_w: int,
        enabled: bool,
    ) -> TrackGridAdapter | None:
        """Build the optional C_track grid condition adapter."""

        track_shape = (
            shape.track_frames,
            shape.track_channels,
            shape.track_grid_h,
            shape.track_grid_w,
        )
        if not enabled:
            return None
        if any(value is None for value in track_shape):
            raise ValueError("C_track shape metadata must be complete when track grid is enabled")
        if (int(shape.track_grid_h), int(shape.track_grid_w)) != (z_grid_h, z_grid_w):
            raise ValueError(
                "C_track grid must match Z patch grid: "
                f"got {(shape.track_grid_h, shape.track_grid_w)}, expected {(z_grid_h, z_grid_w)}"
            )
        return TrackGridAdapter(
            track_frames=int(shape.track_frames),
            track_channels=int(shape.track_channels),
            grid_h=int(shape.track_grid_h),
            grid_w=int(shape.track_grid_w),
            hidden_size=hidden_size,
        )

    def build_patch_condition_adapter(
        self,
        *,
        shape: ZDiTShape,
        hidden_size: int,
        z_grid_h: int,
        z_grid_w: int,
    ) -> PatchConditionAdapter | None:
        """Build the single dense patch-condition adapter for simplified modes."""

        if not self.uses_patch_condition:
            return None
        input_channels = 0
        if condition_mode_uses_appearance(self.condition_mode):
            low_shape = (
                shape.low_frames,
                shape.low_channels,
                shape.low_height,
                shape.low_width,
            )
            if any(value is None for value in low_shape):
                raise ValueError(
                    f"condition.mode={self.condition_mode} requires complete C_low/appearance "
                    "shape metadata"
                )
            input_channels += int(shape.low_channels)
        if condition_mode_uses_track(self.condition_mode):
            track_shape = (
                shape.track_frames,
                shape.track_channels,
                shape.track_grid_h,
                shape.track_grid_w,
            )
            if any(value is None for value in track_shape):
                raise ValueError(
                    f"condition.mode={self.condition_mode} requires complete C_track shape metadata"
                )
            if (int(shape.track_grid_h), int(shape.track_grid_w)) != (z_grid_h, z_grid_w):
                raise ValueError(
                    "C_track grid must match Z patch grid: "
                    f"got {(shape.track_grid_h, shape.track_grid_w)}, expected {(z_grid_h, z_grid_w)}"
                )
            input_channels += int(shape.track_channels)

        target_frames = self.patch_condition_target_frames(shape)
        return PatchConditionAdapter(
            condition_frames=target_frames,
            input_channels=input_channels,
            grid_h=z_grid_h,
            grid_w=z_grid_w,
            hidden_size=hidden_size,
        )

    def patch_condition_target_frames(self, shape: ZDiTShape) -> int:
        """Return the temporal length used by the simplified patch-condition branch."""

        if condition_mode_uses_track(self.condition_mode):
            if shape.track_frames is None:
                raise ValueError(f"condition.mode={self.condition_mode} requires C_track frames")
            return int(shape.track_frames)
        if condition_mode_uses_appearance(self.condition_mode):
            if shape.low_frames is None:
                raise ValueError(f"condition.mode={self.condition_mode} requires C_low frames")
            return int(shape.low_frames)
        raise ValueError(f"condition.mode={self.condition_mode} does not use patch conditions")

    def build_patch_condition_grid(
        self,
        low_features: torch.Tensor | None,
        track_grid: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return concatenated ``[appearance_patch, track_grid]`` condition grids."""

        if self.patch_condition_adapter is None:
            raise RuntimeError("patch condition adapter is not enabled")
        target_frames = self.patch_condition_adapter.condition_frames
        target_grid = (self.patch_condition_adapter.grid_h, self.patch_condition_adapter.grid_w)
        parts = []
        if condition_mode_uses_appearance(self.condition_mode):
            if low_features is None:
                raise ValueError(
                    f"condition.mode={self.condition_mode} requires C_low/appearance features"
                )
            parts.append(self.align_appearance_grid(low_features, target_frames, target_grid))
        if condition_mode_uses_track(self.condition_mode):
            if track_grid is None:
                raise ValueError(f"condition.mode={self.condition_mode} requires C_track grid")
            parts.append(self.validate_patch_track_grid(track_grid, target_frames, target_grid))
        if not parts:
            raise RuntimeError(f"condition.mode={self.condition_mode} produced no condition inputs")
        return torch.cat(parts, dim=2)

    def align_appearance_grid(
        self,
        low: torch.Tensor,
        target_frames: int,
        target_grid: tuple[int, int],
    ) -> torch.Tensor:
        """Align cached appearance/C_low features to the current condition grid."""

        if low.ndim != 5:
            raise ValueError("C_low/appearance features must be shaped [B, T, C, H, W]")
        batch, frames, channels, height, width = low.shape
        expected = (
            self.shape.low_frames,
            self.shape.low_channels,
            self.shape.low_height,
            self.shape.low_width,
        )
        if (frames, channels, height, width) != expected:
            raise ValueError(
                "C_low/appearance shape does not match model initialization: "
                f"got {(frames, channels, height, width)}, expected {expected}"
            )
        x = self.align_condition_frames(low.float(), target_frames)
        if tuple(x.shape[-2:]) == target_grid:
            return x
        x = x.reshape(batch * target_frames, channels, x.shape[-2], x.shape[-1])
        x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=target_grid)
        return x.reshape(batch, target_frames, channels, target_grid[0], target_grid[1])

    def validate_patch_track_grid(
        self,
        track_grid: torch.Tensor,
        target_frames: int,
        target_grid: tuple[int, int],
    ) -> torch.Tensor:
        """Validate dense C_track grids for the simplified condition branch."""

        if track_grid.ndim != 5:
            raise ValueError("C_track must be shaped [B, T, C, H, W]")
        _, frames, channels, height, width = track_grid.shape
        expected = (target_frames, self.shape.track_channels, target_grid[0], target_grid[1])
        if (frames, channels, height, width) != expected:
            raise ValueError(
                "C_track shape does not match model initialization: "
                f"got {(frames, channels, height, width)}, expected {expected}"
            )
        return track_grid.float()

    def align_condition_frames(self, grid: torch.Tensor, target_frames: int) -> torch.Tensor:
        """Align a condition grid along time with deterministic nearest-style sampling."""

        frames = grid.shape[1]
        if frames <= 0 or target_frames <= 0:
            raise ValueError("condition grids must contain at least one frame")
        if frames == target_frames:
            return grid
        if frames == target_frames - 1:
            first = grid[:, :1]
            middle = 0.5 * (grid[:, :-1] + grid[:, 1:]) if frames > 1 else grid[:, :0]
            last = grid[:, -1:]
            return torch.cat([first, middle, last], dim=1)
        indices = torch.linspace(
            0,
            frames - 1,
            target_frames,
            device=grid.device,
        ).round()
        return grid.index_select(1, indices.long())

    def forward(
        self,
        z_at_time: torch.Tensor,
        time_values: torch.Tensor,
        high_features: torch.Tensor,
        low_features: torch.Tensor | None = None,
        track_grid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the flow velocity for Z at a continuous time."""

        if self.patch_condition_adapter is not None:
            condition_grid = self.build_patch_condition_grid(low_features, track_grid)
            condition_tokens = self.patch_condition_adapter(condition_grid)
            x = self.z_adapter(z_at_time)
            time_tokens = self.time_mlp(timestep_embedding(time_values, self.hidden_size))
            for block in self.blocks:
                if self.gradient_checkpointing and self.training:
                    x = checkpoint(block, x, condition_tokens, time_tokens, use_reentrant=False)
                else:
                    x = block(x, condition_tokens, time_tokens)
            return self.z_adapter.decode_tokens(self.output_norm(x))

        if self.high_adapter is None:
            raise RuntimeError("baseline condition mode requires the high adapter")
        high_scores = None
        if (
            self.high_adapter.token_reduction == "foreground_background"
            and low_features is not None
        ):
            high_scores = self.low_to_high_scores(low_features)

        high_tokens = self.high_adapter(high_features, token_scores=high_scores)
        x = self.z_adapter(z_at_time)
        if self.high_adapter.token_reduction == "aligned_grid":
            aligned_tokens = self.high_adapter.future_aligned_tokens(high_tokens)
            if aligned_tokens.shape != x.shape:
                raise ValueError(
                    "aligned C_high tokens must match Z tokens: "
                    f"got {tuple(aligned_tokens.shape)}, expected {tuple(x.shape)}"
                )
            x = x + self.aligned_high_gate * self.aligned_high_norm(aligned_tokens)

        if self.low_adapter is not None:
            if low_features is None:
                raise ValueError("model was initialized with C_low condition but C_low is missing")
            low_tokens = self.low_adapter(low_features)
            high_tokens = torch.cat([high_tokens, low_tokens], dim=1)
        elif low_features is not None:
            raise ValueError("model was initialized without C_low condition but C_low was provided")

        if self.track_grid_adapter is not None:
            if track_grid is None:
                raise ValueError(
                    "model was initialized with C_track condition but C_track is missing"
                )
            track_tokens = self.track_grid_gate * self.track_grid_adapter(track_grid)
            high_tokens = torch.cat([high_tokens, track_tokens], dim=1)
        elif track_grid is not None:
            raise ValueError(
                "model was initialized without C_track condition but C_track was provided"
            )

        time_tokens = self.time_mlp(timestep_embedding(time_values, self.hidden_size))
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, high_tokens, time_tokens, use_reentrant=False)
            else:
                x = block(x, high_tokens, time_tokens)
        return self.z_adapter.decode_tokens(self.output_norm(x))

    def low_to_high_scores(self, low: torch.Tensor) -> torch.Tensor:
        """Convert raw C_low tensors into patch scores aligned with C_high tokens."""

        if low.ndim != 5:
            raise ValueError("C_low must be shaped [B, T, C, H, W]")
        batch, low_frames, _, height, width = low.shape
        if low_frames <= 0:
            raise ValueError("C_low must contain at least one frame difference")

        magnitude = low.float().abs().mean(dim=2)
        high_frames = self.shape.high_frames
        if low_frames == high_frames:
            frame_scores = magnitude
        elif low_frames == high_frames - 1:
            first = magnitude[:, :1]
            middle = (
                0.5 * (magnitude[:, :-1] + magnitude[:, 1:]) if low_frames > 1 else magnitude[:, :0]
            )
            last = magnitude[:, -1:]
            frame_scores = torch.cat([first, middle, last], dim=1)
        else:
            indices = torch.linspace(
                0,
                low_frames - 1,
                high_frames,
                device=low.device,
            ).round()
            frame_scores = magnitude.index_select(1, indices.long())

        grid_h, grid_w = high_token_grid(self.shape.high_tokens)
        pooled = torch.nn.functional.adaptive_avg_pool2d(
            frame_scores.reshape(batch * high_frames, 1, height, width),
            output_size=(grid_h, grid_w),
        )
        return pooled.reshape(batch, high_frames, self.shape.high_tokens)


class FlowMatchingSampler:
    """Linear-path conditional flow matching trainer and Euler sampler.

    The project uses the noise-to-data convention:

    ``z_tau = (1 - tau) * sigma + tau * z``
    ``v_target = z - sigma``

    Inference starts from ``sigma`` and integrates forward with Euler steps.
    """

    def __init__(
        self,
        *,
        inference_steps: int,
        timestep_distribution: str,
        time_embedding_scale: float,
        normalize_z: bool,
        beta_alpha: float = 1.5,
        beta_beta: float = 1.0,
        beta_s: float = 0.999,
    ) -> None:
        if inference_steps <= 0:
            raise ValueError("flow_matching.inference_steps must be positive")
        if time_embedding_scale <= 0:
            raise ValueError("flow_matching.time_embedding_scale must be positive")
        if timestep_distribution not in {"uniform", "gr00t_beta"}:
            raise ValueError("flow_matching.timestep_distribution must be uniform or gr00t_beta")
        if beta_alpha <= 0 or beta_beta <= 0 or not 0 < beta_s <= 1:
            raise ValueError("flow_matching beta parameters must be positive with beta_s in (0, 1]")

        self.inference_steps = inference_steps
        self.timestep_distribution = timestep_distribution
        self.time_embedding_scale = time_embedding_scale
        self.normalize_z = normalize_z
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta
        self.beta_s = beta_s

    def sample_timesteps(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample continuous flow-matching times in ``[0, 1]``."""

        if self.timestep_distribution == "uniform":
            return torch.rand(batch_size, device=device, dtype=dtype, generator=generator)

        if self.beta_beta == 1.0:
            uniform = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
            beta_value = uniform.clamp_min(1e-12).pow(1.0 / self.beta_alpha)
        else:
            distribution = torch.distributions.Beta(
                torch.tensor(self.beta_alpha, device=device, dtype=dtype),
                torch.tensor(self.beta_beta, device=device, dtype=dtype),
            )
            beta_value = distribution.sample((batch_size,))
        return (self.beta_s * (1.0 - beta_value)).clamp(0.0, 1.0)

    def prepare_training_pair(
        self,
        clean_z: torch.Tensor,
        *,
        z_stats: dict[str, torch.Tensor] | None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``z_tau``, scaled model times, velocity target, and normalized clean Z."""

        clean = self.normalize(clean_z, z_stats)
        noise = torch.randn(
            clean.shape,
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
        tau = self.sample_timesteps(
            clean.shape[0],
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
        tau_view = tau.reshape(clean.shape[0], *((1,) * (clean.ndim - 1)))
        z_at_tau = (1.0 - tau_view) * noise + tau_view * clean
        target_velocity = clean - noise
        return z_at_tau, self.scale_time(tau), target_velocity, clean

    @torch.inference_mode()
    def euler_sample(
        self,
        model: ConditionedZDiT,
        high_features: torch.Tensor,
        output_shape: tuple[int, int, int, int, int],
        *,
        low_features: torch.Tensor | None = None,
        track_features: torch.Tensor | None = None,
        z_stats: dict[str, torch.Tensor] | None,
        inference_steps: int | None = None,
        generator: torch.Generator | None = None,
        return_normalized: bool = False,
    ) -> torch.Tensor:
        """Generate Z_hat by forward Euler integration from Gaussian noise."""

        steps = self.inference_steps if inference_steps is None else inference_steps
        if steps <= 0:
            raise ValueError("flow_matching.inference_steps must be positive")

        device = high_features.device
        z = torch.randn(
            output_shape,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        high = high_features.float()
        low = low_features.float() if low_features is not None else None
        track = track_features.float() if track_features is not None else None
        step_size = 1.0 / steps
        for index in range(steps):
            tau = torch.full(
                (output_shape[0],),
                index / steps,
                device=device,
                dtype=torch.float32,
            )
            velocity = model(z, self.scale_time(tau), high, low, track).float()
            z = z + step_size * velocity

        if return_normalized:
            return z
        return self.denormalize(z, z_stats)

    def scale_time(self, tau: torch.Tensor) -> torch.Tensor:
        """Scale ``[0, 1]`` flow times into the sinusoidal embedding range."""

        return tau.float() * self.time_embedding_scale

    def normalize(
        self,
        z: torch.Tensor,
        z_stats: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        """Normalize target Z when configured."""

        if not self.normalize_z:
            return z.float()
        if z_stats is None:
            raise ValueError("flow_matching.normalize_z requires z_stats")
        mean = z_stats["mean"].to(z.device).float()
        std = z_stats["std"].to(z.device).float().clamp_min(1e-6)
        return (z.float() - mean) / std

    def denormalize(
        self,
        z: torch.Tensor,
        z_stats: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        """Map normalized generated Z_hat back to the cached VAE Z scale."""

        if not self.normalize_z:
            return z.float()
        if z_stats is None:
            raise ValueError("flow_matching.normalize_z requires z_stats")
        mean = z_stats["mean"].to(z.device).float()
        std = z_stats["std"].to(z.device).float().clamp_min(1e-6)
        return z.float() * std + mean
