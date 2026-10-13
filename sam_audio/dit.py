import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import comfy.ops
import comfy.model_prefetch
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift


def gate(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    return x * g


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seqlen: int = 10000, theta: float = 20000.0, device=None, dtype=None):
        super().__init__()
        self.dim = dim
        self.max_seqlen = max_seqlen
        self.theta = theta

        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32)[: (dim // 2)] / dim))
        t = torch.arange(max_seqlen, device=device, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        cos = freqs.cos()
        sin = freqs.sin()
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, T, D]
        seqlen = x.shape[2]
        cos = self.cos[:seqlen].to(dtype=x.dtype, device=x.device)  # [T, D//2]
        sin = self.sin[:seqlen].to(dtype=x.dtype, device=x.device)  # [T, D//2]
        # Paired rotation: x_ [B, H, T, D//2, 2]
        x_paired = x.reshape(x.shape[0], x.shape[1], seqlen, self.dim // 2, 2)
        x1 = x_paired[..., 0]
        x2 = x_paired[..., 1]
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        return torch.stack([x1_rot, x2_rot], dim=-1).flatten(3)


class Attention(nn.Module):
    def __init__(self, dim: int, head_dim: int, n_heads: int, norm_eps: float = 1e-5, device=None, dtype=None, operations=None):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.comfy_attention = ComfyAttention()

        self.wq = operations.Linear(dim, n_heads * head_dim, bias=False, device=device, dtype=dtype)
        self.wk = operations.Linear(dim, n_heads * head_dim, bias=False, device=device, dtype=dtype)
        self.wv = operations.Linear(dim, n_heads * head_dim, bias=False, device=device, dtype=dtype)
        self.wo = operations.Linear(n_heads * head_dim, dim, bias=False, device=device, dtype=dtype)

        self.q_norm = operations.RMSNorm(head_dim, eps=norm_eps, device=device, dtype=dtype)
        self.k_norm = operations.RMSNorm(head_dim, eps=norm_eps, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        cross_x: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope: Optional[RotaryEmbedding] = None,
        transformer_options: dict = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        xq = self.wq(x)
        if cross_x is not None:
            xk = self.wk(cross_x)
            xv = self.wv(cross_x)
            Tk = cross_x.shape[1]
        else:
            xk = self.wk(x)
            xv = self.wv(x)
            Tk = T

        # Reshape to [B, n_heads, T, head_dim] matching upstream: B x T x D x H -> B x H x T x D
        xq = xq.reshape(B, T, self.head_dim, self.n_heads).permute(0, 3, 1, 2)
        xk = xk.reshape(B, Tk, self.head_dim, self.n_heads).permute(0, 3, 1, 2)
        xv = xv.reshape(B, Tk, self.head_dim, self.n_heads).permute(0, 3, 1, 2)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        if rope is not None:
            xq = rope(xq)
            xk = rope(xk)

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)

        qc = AttentionTensorContainer(xq)
        kc = AttentionTensorContainer(xk)
        vc = AttentionTensorContainer(xv)

        out = optimized_attention(
            qc, kc, vc, self.n_heads,
            mask=attn_mask,
            skip_reshape=True,
            preferred_attention=self.comfy_attention,
            transformer_options=transformer_options or {},
        )
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, device=None, dtype=None, operations=None):
        super().__init__()
        self.w1 = operations.Linear(dim, hidden_dim, bias=False, device=device, dtype=dtype)
        self.w2 = operations.Linear(hidden_dim, dim, bias=False, device=device, dtype=dtype)
        self.w3 = operations.Linear(dim, hidden_dim, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DiTBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, norm_eps: float = 1e-5, ffn_exp: int = 4, ffn_dim_multiplier: float = 1.0, multiple_of: int = 64, device=None, dtype=None, operations=None):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.n_heads = n_heads

        hidden_dim = int(2 * ffn_exp * dim / 3)
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.attention = Attention(dim=dim, head_dim=self.head_dim, n_heads=n_heads, norm_eps=norm_eps, device=device, dtype=dtype, operations=operations)
        self.cross_attention = Attention(dim=dim, head_dim=self.head_dim, n_heads=n_heads, norm_eps=norm_eps, device=device, dtype=dtype, operations=operations)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=hidden_dim, device=device, dtype=dtype, operations=operations)

        self.attention_norm = operations.RMSNorm(dim, eps=norm_eps, device=device, dtype=dtype)
        self.ffn_norm = operations.RMSNorm(dim, eps=norm_eps, device=device, dtype=dtype)

        self.scale_shift_table = nn.Parameter(torch.empty(6, dim, device=device, dtype=dtype))

    def forward(
        self,
        x: torch.Tensor,
        cross_x: Optional[torch.Tensor],
        t0: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
        rope: Optional[RotaryEmbedding] = None,
        transformer_options: dict = None,
    ) -> torch.Tensor:
        scale_shift = comfy.ops.cast_to_input(self.scale_shift_table, x)
        biases = scale_shift.unsqueeze(0) + t0.reshape(x.shape[0], 6, -1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = biases.chunk(6, dim=1)

        norm_x = modulate(self.attention_norm(x), shift_msa, scale_msa)
        h_attn = self.attention(norm_x, key_padding_mask=padding_mask, rope=rope, transformer_options=transformer_options)
        h = x + gate(h_attn, gate_msa)

        if cross_x is not None:
            h_cross = self.cross_attention(h, cross_x=cross_x, key_padding_mask=memory_padding_mask, transformer_options=transformer_options)
            h = h + h_cross

        norm_h = modulate(self.ffn_norm(h), shift_mlp, scale_mlp)
        h_ff = self.feed_forward(norm_h)
        return h + gate(h_ff, gate_mlp)


class Patcher(nn.Module):
    def __init__(self, dim: int, device=None, dtype=None, operations=None):
        super().__init__()
        # block1: GroupNorm(1, dim) -> SiLU -> Conv1d(dim, dim, 3, padding=1)
        self.block1_gn = operations.GroupNorm(1, dim, device=device, dtype=dtype)
        self.block1_conv = operations.Conv1d(dim, dim, kernel_size=3, padding=1, device=device, dtype=dtype)
        # block2: GroupNorm(1, dim) -> SiLU -> Conv1d(dim, dim, 3, padding=1)
        self.block2_gn = operations.GroupNorm(1, dim, device=device, dtype=dtype)
        self.block2_conv = operations.Conv1d(dim, dim, kernel_size=3, padding=1, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        h = self.block1_conv(F.silu(self.block1_gn(x)))
        h = self.block2_conv(F.silu(self.block2_gn(h)))
        return x + h


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int, freq_dim: int = 256, max_period: int = 10000, device=None, dtype=None, operations=None):
        super().__init__()
        self.freq_dim = freq_dim
        half = freq_dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)

        self.w1 = operations.Linear(freq_dim, dim, bias=False, device=device, dtype=dtype)
        self.w2 = operations.Linear(dim, dim, bias=False, device=device, dtype=dtype)
        self.w3 = operations.Linear(freq_dim, dim, bias=False, device=device, dtype=dtype)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        freqs = self.freqs.to(device=t.device, dtype=torch.float32)
        args = t.unsqueeze(1).float() * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(dtype=self.w1.weight.dtype)
        return self.w2(F.silu(self.w1(emb)) * self.w3(emb))


class ContextEmbedder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, device=None, dtype=None, operations=None):
        super().__init__()
        self.w1 = operations.Linear(in_dim, out_dim, bias=False, device=device, dtype=dtype)
        self.w2 = operations.Linear(out_dim, out_dim, bias=False, device=device, dtype=dtype)
        self.w3 = operations.Linear(in_dim, out_dim, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DiT(nn.Module):
    def __init__(self, config: dict, device=None, dtype=None, operations=None):
        super().__init__()

        dim = config.get("dim", 1536)
        n_heads = config.get("n_heads", 12)
        n_layers = config.get("n_layers", 12)
        norm_eps = config.get("norm_eps", 1e-5)
        ffn_exp = config.get("ffn_exp", 4)
        ffn_dim_multiplier = config.get("ffn_dim_multiplier", 1)
        multiple_of = config.get("multiple_of", 64)
        max_positions = config.get("max_positions", 10000)
        freq_dim = config.get("frequency_embedding_dim", 256)
        context_dim = config.get("context_dim", 1536)
        out_channels = config.get("out_channels", 256)

        self.dim = dim
        self.head_dim = dim // n_heads

        self.rope = RotaryEmbedding(dim=self.head_dim, max_seqlen=max_positions, theta=max(10000.0, 2.0 * max_positions), device=device, dtype=dtype)

        self.x_embedder = Patcher(dim=dim, device=device, dtype=dtype, operations=operations)
        self.t_embedder = TimestepEmbedder(dim=dim, freq_dim=freq_dim, device=device, dtype=dtype, operations=operations)
        self.t_block = operations.Linear(dim, dim * 6, bias=True, device=device, dtype=dtype)
        self.y_embedder = ContextEmbedder(in_dim=context_dim, out_dim=dim, device=device, dtype=dtype, operations=operations)

        layers = []
        for _ in range(n_layers):
            layers.append(DiTBlock(
                dim=dim,
                n_heads=n_heads,
                norm_eps=norm_eps,
                ffn_exp=ffn_exp,
                ffn_dim_multiplier=ffn_dim_multiplier,
                multiple_of=multiple_of,
                device=device,
                dtype=dtype,
                operations=operations,
            ))
        self.layers = nn.ModuleList(layers)

        self.norm = operations.RMSNorm(dim, eps=norm_eps, device=device, dtype=dtype)
        self.output = operations.Linear(dim, out_channels, bias=False, device=device, dtype=dtype)
        self.final_layer_scale_shift_table = nn.Parameter(torch.empty(2, dim, device=device, dtype=dtype))

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
        transformer_options: dict = None,
    ) -> torch.Tensor:
        # x: [B, T, C]
        h = self.x_embedder(x.permute(0, 2, 1)).permute(0, 2, 1)

        t = self.t_embedder(time)
        t0 = self.t_block(F.silu(t))

        y = self.y_embedder(memory) if memory is not None else None

        transformer_options = transformer_options or {}
        prefetch = comfy.model_prefetch.make_prefetch_queue(list(self.layers), x.device, transformer_options)
        for i, layer in enumerate(self.layers):
            comfy.model_prefetch.prefetch_queue_pop(prefetch, x.device, layer)
            transformer_options["block_index"] = i
            h = layer(
                h,
                cross_x=y,
                t0=t0,
                padding_mask=padding_mask,
                memory_padding_mask=memory_padding_mask,
                rope=self.rope,
                transformer_options=transformer_options,
            )
        comfy.model_prefetch.prefetch_queue_pop(prefetch, x.device, None)

        scale_shift = comfy.ops.cast_to_input(self.final_layer_scale_shift_table, h)
        shift, scale = (scale_shift.unsqueeze(0) + t.unsqueeze(1)).chunk(2, dim=1)

        h = modulate(self.norm(h), shift, scale)
        return self.output(h)
