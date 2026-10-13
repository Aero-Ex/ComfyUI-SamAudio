import torch
import torch.nn as nn
import torch.nn.functional as F
import comfy.ops
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, device=None, dtype=None, operations=None):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.comfy_attention = ComfyAttention()

        self.ln_1 = operations.LayerNorm(d_model, device=device, dtype=dtype)
        self.in_proj = operations.Linear(d_model, 3 * d_model, bias=True, device=device, dtype=dtype)
        self.out_proj = operations.Linear(d_model, d_model, bias=True, device=device, dtype=dtype)

        self.ln_2 = operations.LayerNorm(d_model, device=device, dtype=dtype)
        self.c_fc = operations.Linear(d_model, d_model * 4, bias=True, device=device, dtype=dtype)
        self.act = QuickGELU()
        self.c_proj = operations.Linear(d_model * 4, d_model, bias=True, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        nx = self.ln_1(x)
        B, T, C = nx.shape
        qkv = self.in_proj(nx).reshape(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        qc = AttentionTensorContainer(q)
        kc = AttentionTensorContainer(k)
        vc = AttentionTensorContainer(v)
        attn_out = optimized_attention(qc, kc, vc, self.n_head, skip_reshape=True, preferred_attention=self.comfy_attention)
        x = x + self.out_proj(attn_out)

        # MLP
        x = x + self.c_proj(self.act(self.c_fc(self.ln_2(x))))
        return x


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
        self.act = QuickGELU()
        self.c_proj = operations.Linear(embed_dim * 4, embed_dim, bias=True, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        probe = comfy.ops.cast_to_input(self.probe, x).expand(B, 1, C)

        q = self.in_proj(probe)[..., :self.embed_dim].reshape(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.in_proj(x)
        k = kv[..., self.embed_dim : 2 * self.embed_dim].reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = kv[..., 2 * self.embed_dim :].reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        qc = AttentionTensorContainer(q)
        kc = AttentionTensorContainer(k)
        vc = AttentionTensorContainer(v)
        attn_out = optimized_attention(qc, kc, vc, self.num_heads, skip_reshape=True, preferred_attention=self.comfy_attention)
        attn_out = self.out_proj(attn_out)

        out = probe + attn_out
        out = out + self.c_proj(self.act(self.c_fc(self.layernorm(out))))
        return out.squeeze(1)


class VisualTransformer(nn.Module):
    def __init__(self, image_size: int = 336, patch_size: int = 14, width: int = 1024, layers: int = 24, heads: int = 16, device=None, dtype=None, operations=None):
        super().__init__()

        self.conv1 = operations.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False, device=device, dtype=dtype)
        self.class_embedding = nn.Parameter(torch.empty(width, device=device, dtype=dtype))
        grid_size = image_size // patch_size
        num_patches = grid_size * grid_size
        self.positional_embedding = nn.Parameter(torch.empty(num_patches + 1, width, device=device, dtype=dtype))
        self.ln_pre = operations.LayerNorm(width, device=device, dtype=dtype)

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, device=device, dtype=dtype, operations=operations)
            for _ in range(layers)
        ])

        self.ln_post = operations.LayerNorm(width, device=device, dtype=dtype)
        self.attn_pool = AttentionPool(embed_dim=width, num_heads=8, device=device, dtype=dtype, operations=operations)
        self.proj = nn.Parameter(torch.empty(width, width, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        # x: [B, 3, H, W] in [-1, 1]
        x = self.conv1(x)  # [B, width, grid_h, grid_w]
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)  # [B, num_patches, width]

        cls_emb = comfy.ops.cast_to_input(self.class_embedding, x).unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        x = torch.cat([cls_emb, x], dim=1)

        pos_emb = comfy.ops.cast_to_input(self.positional_embedding, x)
        x = x + pos_emb.unsqueeze(0)
        x = self.ln_pre(x)

        for block in self.resblocks:
            x = block(x)

        x = self.ln_post(x)
        x = self.attn_pool(x)  # [B, width]

        proj = comfy.ops.cast_to_input(self.proj, x)
        x = x @ proj

        if normalize:
            x = F.normalize(x, dim=-1)
        return x


class PerceptionEncoder(nn.Module):
    def __init__(self, config: dict, device=None, dtype=None, operations=None):
        super().__init__()
        self.image_size = config.get("image_size", 336)
        self.dim = config.get("dim", 1024)
        self.normalize_feature = config.get("normalize_feature", True)
        self.visual = VisualTransformer(image_size=self.image_size, width=self.dim, device=device, dtype=dtype, operations=operations)

    def preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [T, C, H, W] or [T, H, W, C] in [0, 1]
        if frames.shape[-1] == 3 or frames.shape[-1] == 4:
            frames = frames[..., :3].permute(0, 3, 1, 2)
        if frames.shape[-2] != self.image_size or frames.shape[-1] != self.image_size:
            frames = F.interpolate(frames.float(), size=(self.image_size, self.image_size), mode="bicubic", align_corners=False)
        # Normalize: [0, 1] -> [-1, 1] via mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        return (frames - 0.5) / 0.5

    def encode_video(self, frames: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
        # frames: [T, 3, H, W]
        frames = self.preprocess(frames).to(dtype=self.visual.conv1.weight.dtype)
        T = frames.shape[0]
        feats = []
        for i in range(0, T, batch_size):
            chunk = frames[i : i + batch_size]
            feat = self.visual(chunk, normalize=self.normalize_feature)
            feats.append(feat)
        return torch.cat(feats, dim=0)  # [T, dim]
