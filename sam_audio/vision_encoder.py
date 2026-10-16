import torch
import torch.nn as nn
import torch.nn.functional as F
import comfy.ops
from comfy.ldm.flux.math import apply_rope1, rope
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention


def rope_2d(grid_h: int, grid_w: int, head_dim: int, device) -> torch.Tensor:
    # PE Rope2D: first half of each head rotates by x, second half by y, 1-based positions; the cls token is not rotated
    y, x = torch.meshgrid(torch.arange(1, grid_h + 1, device=device), torch.arange(1, grid_w + 1, device=device), indexing="ij")
    patches = torch.cat([rope(x.reshape(1, -1), head_dim // 2, 10000), rope(y.reshape(1, -1), head_dim // 2, 10000)], dim=-3)
    cls = rope(torch.zeros(1, 1, device=device), head_dim, 10000)
    return torch.cat([cls, patches], dim=1).unsqueeze(1)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, device=None, dtype=None, operations=None):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.comfy_attention = ComfyAttention()

        self.ln_1 = operations.LayerNorm(d_model, device=device, dtype=dtype)
        self.in_proj = operations.Linear(d_model, 3 * d_model, bias=True, device=device, dtype=dtype)
        self.out_proj = operations.Linear(d_model, d_model, bias=True, device=device, dtype=dtype)

        self.ln_2 = operations.LayerNorm(d_model, device=device, dtype=dtype)
        self.c_fc = operations.Linear(d_model, d_model * 4, bias=True, device=device, dtype=dtype)
        self.c_proj = operations.Linear(d_model * 4, d_model, bias=True, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = self.in_proj(self.ln_1(x)).reshape(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q = apply_rope1(q, freqs)
        k = apply_rope1(k, freqs)
        attn_out = optimized_attention(AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v), self.n_head, skip_reshape=True, preferred_attention=self.comfy_attention)
        x = x + self.out_proj(attn_out)
        return x + self.c_proj(F.gelu(self.c_fc(self.ln_2(x))))


class AttentionPool(nn.Module):
    def __init__(self, embed_dim: int = 1024, num_heads: int = 8, device=None, dtype=None, operations=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.comfy_attention = ComfyAttention()

        self.probe = nn.Parameter(torch.empty(1, 1, embed_dim, device=device, dtype=dtype))
        self.in_proj = operations.Linear(embed_dim, 3 * embed_dim, bias=True, device=device, dtype=dtype)
        self.out_proj = operations.Linear(embed_dim, embed_dim, bias=True, device=device, dtype=dtype)

        self.layernorm = operations.LayerNorm(embed_dim, device=device, dtype=dtype)
        self.c_fc = operations.Linear(embed_dim, embed_dim * 4, bias=True, device=device, dtype=dtype)
        self.c_proj = operations.Linear(embed_dim * 4, embed_dim, bias=True, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        probe = comfy.ops.cast_to_input(self.probe, x).expand(B, 1, C)

        q = self.in_proj(probe)[..., :self.embed_dim].reshape(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.in_proj(x)
        k = kv[..., self.embed_dim : 2 * self.embed_dim].reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = kv[..., 2 * self.embed_dim :].reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn_out = optimized_attention(AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v), self.num_heads, skip_reshape=True, preferred_attention=self.comfy_attention)

        out = probe + self.out_proj(attn_out)
        out = out + self.c_proj(F.gelu(self.c_fc(self.layernorm(out))))
        return out.squeeze(1)


class VisualTransformer(nn.Module):
    def __init__(self, image_size: int = 336, patch_size: int = 14, width: int = 1024, layers: int = 24, heads: int = 16, device=None, dtype=None, operations=None):
        super().__init__()
        self.conv1 = operations.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False, device=device, dtype=dtype)
        self.class_embedding = nn.Parameter(torch.empty(width, device=device, dtype=dtype))
        grid_size = image_size // patch_size
        self.positional_embedding = nn.Parameter(torch.empty(grid_size * grid_size + 1, width, device=device, dtype=dtype))
        self.ln_pre = operations.LayerNorm(width, device=device, dtype=dtype)

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, device=device, dtype=dtype, operations=operations)
            for _ in range(layers)
        ])

        self.ln_post = operations.LayerNorm(width, device=device, dtype=dtype)
        self.attn_pool = AttentionPool(embed_dim=width, num_heads=8, device=device, dtype=dtype, operations=operations)
        self.proj = nn.Parameter(torch.empty(width, width, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, image_size, image_size] in [-1, 1]
        x = self.conv1(x)
        B, C, H, W = x.shape
        x = x.flatten(2).movedim(1, 2)

        cls_emb = comfy.ops.cast_to_input(self.class_embedding, x).expand(B, 1, C)
        x = torch.cat([cls_emb, x], dim=1) + comfy.ops.cast_to_input(self.positional_embedding, x)
        x = self.ln_pre(x)

        freqs = rope_2d(H, W, self.resblocks[0].head_dim, x.device)
        for block in self.resblocks:
            x = block(x, freqs)

        x = self.attn_pool(self.ln_post(x))
        return x @ comfy.ops.cast_to_input(self.proj, x)


class PerceptionEncoder(nn.Module):
    def __init__(self, config: dict, device=None, dtype=None, operations=None):
        super().__init__()
        self.image_size = config.get("image_size", 336)
        self.dim = config.get("dim", 1024)
        self.normalize_feature = config.get("normalize_feature", True)
        self.visual = VisualTransformer(image_size=self.image_size, width=self.dim, device=device, dtype=dtype, operations=operations)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [N, 3, image_size, image_size] in [-1, 1] -> [N, dim]
        x = self.visual(frames)
        return F.normalize(x, dim=-1) if self.normalize_feature else x
