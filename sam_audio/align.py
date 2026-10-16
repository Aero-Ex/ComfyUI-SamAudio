import torch
import torch.nn as nn
import comfy.ops


class AlignModalities(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, device=None, dtype=None, operations=None):
        super().__init__()
        self.conv = operations.Conv1d(in_channels, out_channels, kernel_size=1, device=device, dtype=dtype)
        self.layer_norm = operations.LayerNorm(out_channels, device=device, dtype=dtype)
        self.gate = nn.Parameter(torch.empty(1, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        # x: [B, T, out_channels], features: [B, in_channels, T]
        h = self.layer_norm(self.conv(features).movedim(1, 2))
        return torch.addcmul(x, comfy.ops.cast_to_input(self.gate, x).tanh(), h)


class EmbedAnchors(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, out_dim: int, device=None, dtype=None, operations=None):
        super().__init__()
        # extra slot is the <pad> token
        self.embed = operations.Embedding(num_embeddings + 1, embedding_dim, padding_idx=num_embeddings, device=device, dtype=dtype)
        self.gate = nn.Parameter(torch.empty(1, device=device, dtype=dtype))
        self.proj = operations.Linear(embedding_dim, out_dim, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, anchor_ids: torch.Tensor, anchor_alignment: torch.Tensor) -> torch.Tensor:
        h = self.proj(self.embed(anchor_ids.gather(1, anchor_alignment), out_dtype=x.dtype))
        return torch.addcmul(x, comfy.ops.cast_to_input(self.gate, x).tanh(), h)
