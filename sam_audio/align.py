import math
from typing import Optional
import torch
import torch.nn as nn
import comfy.ops


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0, device=None, dtype=None):
        super().__init__()
        assert dim % 2 == 0
        half_dim = dim // 2
        inv_freq = torch.exp(-math.log(theta) * torch.arange(half_dim, dtype=torch.float32) / half_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        # pos: [B] or [B, T]
        inv_freq = self.inv_freq.to(device=pos.device, dtype=torch.float32)
        if pos.ndim == 1:
            args = pos.unsqueeze(1).float() * inv_freq.unsqueeze(0)
        else:
            args = pos.unsqueeze(-1).float() * inv_freq
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class AlignModalities(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True, device=None, dtype=None, operations=None):
        super().__init__()

        self.conv = operations.Conv1d(in_channels, out_channels, kernel_size=1, device=device, dtype=dtype)
        self.normalize = normalize
        if normalize:
            self.layer_norm = operations.LayerNorm(out_channels, device=device, dtype=dtype)
        else:
            self.layer_norm = None
        self.gate = nn.Parameter(torch.empty(1, device=device, dtype=dtype))

    def forward(self, anchor: torch.Tensor, tgt: Optional[torch.Tensor] = None) -> torch.Tensor:
        # anchor: [B, T, out_channels]
        # tgt: [B, in_channels, T]
        if tgt is None:
            return anchor

        post_conv = self.conv(tgt).permute(0, 2, 1)  # [B, T, out_channels]
        if self.layer_norm is not None:
            post_conv = self.layer_norm(post_conv)

        gate = comfy.ops.cast_to_input(self.gate, anchor)
        return anchor + gate.tanh() * post_conv


class EmbedAnchors(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, out_dim: int, device=None, dtype=None, operations=None):
        super().__init__()

        # num_embeddings + 1 for padding token
        self.embed = operations.Embedding(num_embeddings + 1, embedding_dim, padding_idx=num_embeddings, device=device, dtype=dtype)
        self.gate = nn.Parameter(torch.empty(1, device=device, dtype=dtype))
        self.proj = operations.Linear(embedding_dim, out_dim, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, anchor_ids: Optional[torch.Tensor] = None, anchor_alignment: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, T, out_dim]
        # anchor_ids: [B, num_anchors]
        # anchor_alignment: [B, T]
        if anchor_ids is None or anchor_alignment is None:
            return x

        aligned_ids = anchor_ids.gather(1, anchor_alignment)
        embs = self.embed(aligned_ids)
        proj = self.proj(embs)
        gate = comfy.ops.cast_to_input(self.gate, x)
        return x + gate.tanh() * proj
