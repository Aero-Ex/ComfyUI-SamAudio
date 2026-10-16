import torch
import torch.nn as nn
import torch.nn.functional as F
import comfy.ops
import comfy.model_prefetch
from comfy.ldm.flux.math import apply_rope1, rope
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention
from comfy.ldm.modules.diffusionmodules.util import timestep_embedding


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.addcmul(shift, x, 1.0 + scale)


class Attention(nn.Module):
    def __init__(self, dim: int, head_dim: int, n_heads: int, norm_eps: float = 1e-5, device=None, dtype=None, operations=None):
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.comfy_attention = ComfyAttention()

        self.wq = operations.Linear(dim, n_heads * head_dim, bias=False, device=device, dtype=dtype)
        self.wk = operations.Linear(dim, n_heads * head_dim, bias=False, device=device, dtype=dtype)
        self.wv = operations.Linear(dim, n_heads * head_dim, bias=False, device=device, dtype=dtype)
        self.wo = operations.Linear(n_heads * head_dim, dim, bias=False, device=device, dtype=dtype)

        self.q_norm = operations.RMSNorm(head_dim, eps=norm_eps, device=device, dtype=dtype)
        self.k_norm = operations.RMSNorm(head_dim, eps=norm_eps, device=device, dtype=dtype)

    def heads(self, x: torch.Tensor) -> torch.Tensor:
        # upstream splits channels as (head_dim, n_heads); copy to [B, H, T, head_dim] so attention gets a unit-stride head dim
        return x.unflatten(-1, (self.head_dim, self.n_heads)).movedim(-1, 1).contiguous()

    def forward(self, x: torch.Tensor, context: torch.Tensor = None, freqs: torch.Tensor = None, transformer_options={}) -> torch.Tensor:
        if context is None:
            context = x
        q = self.q_norm(self.heads(self.wq(x)))
        k = self.k_norm(self.heads(self.wk(context)))
        v = self.heads(self.wv(context))

        if freqs is not None:
            q = apply_rope1(q, freqs)
            k = apply_rope1(k, freqs)

        out = optimized_attention(
            AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v), self.n_heads,
            skip_reshape=True, preferred_attention=self.comfy_attention, transformer_options=transformer_options,
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
        head_dim = dim // n_heads

        hidden_dim = int(2 * ffn_exp * dim / 3)
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.attention = Attention(dim=dim, head_dim=head_dim, n_heads=n_heads, norm_eps=norm_eps, device=device, dtype=dtype, operations=operations)
        self.cross_attention = Attention(dim=dim, head_dim=head_dim, n_heads=n_heads, norm_eps=norm_eps, device=device, dtype=dtype, operations=operations)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=hidden_dim, device=device, dtype=dtype, operations=operations)

        self.attention_norm = operations.RMSNorm(dim, eps=norm_eps, device=device, dtype=dtype)
        self.ffn_norm = operations.RMSNorm(dim, eps=norm_eps, device=device, dtype=dtype)

        self.scale_shift_table = nn.Parameter(torch.empty(6, dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor, context: torch.Tensor, t0: torch.Tensor, freqs: torch.Tensor, transformer_options={}) -> torch.Tensor:
        biases = comfy.ops.cast_to_input(self.scale_shift_table, x).unsqueeze(0) + t0.reshape(x.shape[0], 6, -1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = biases.chunk(6, dim=1)

        h = self.attention(modulate(self.attention_norm(x), shift_msa, scale_msa), freqs=freqs, transformer_options=transformer_options)
        x = torch.addcmul(x, h, gate_msa)
        x = x + self.cross_attention(x, context=context, transformer_options=transformer_options)
        h = self.feed_forward(modulate(self.ffn_norm(x), shift_mlp, scale_mlp))
        return torch.addcmul(x, h, gate_mlp)


class Patcher(nn.Module):
    def __init__(self, dim: int, device=None, dtype=None, operations=None):
        super().__init__()
        self.block1_gn = operations.GroupNorm(1, dim, device=device, dtype=dtype)
        self.block1_conv = operations.Conv1d(dim, dim, kernel_size=3, padding=1, device=device, dtype=dtype)
        self.block2_gn = operations.GroupNorm(1, dim, device=device, dtype=dtype)
        self.block2_conv = operations.Conv1d(dim, dim, kernel_size=3, padding=1, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block1_conv(F.silu(self.block1_gn(x)))
        h = self.block2_conv(F.silu(self.block2_gn(h)))
        return x + h


class SwiGLUProjection(nn.Module):
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
        self.dim = dim
        self.head_dim = dim // n_heads
        self.freq_dim = config.get("frequency_embedding_dim", 256)
        self.rope_theta = max(10000, 2 * config.get("max_positions", 10000))

        self.x_embedder = Patcher(dim=dim, device=device, dtype=dtype, operations=operations)
        self.t_embedder = SwiGLUProjection(self.freq_dim, dim, device=device, dtype=dtype, operations=operations)
        self.t_block = operations.Linear(dim, dim * 6, bias=True, device=device, dtype=dtype)
        self.y_embedder = SwiGLUProjection(config.get("context_dim", 1536), dim, device=device, dtype=dtype, operations=operations)

        self.layers = nn.ModuleList([
            DiTBlock(
                dim=dim,
                n_heads=n_heads,
                norm_eps=norm_eps,
                ffn_exp=config.get("ffn_exp", 4),
                ffn_dim_multiplier=config.get("ffn_dim_multiplier", 1),
                multiple_of=config.get("multiple_of", 64),
                device=device,
                dtype=dtype,
                operations=operations,
            )
            for _ in range(n_layers)
        ])

        self.norm = operations.RMSNorm(dim, eps=norm_eps, device=device, dtype=dtype)
        self.output = operations.Linear(dim, config.get("out_channels", 256), bias=False, device=device, dtype=dtype)
        self.final_layer_scale_shift_table = nn.Parameter(torch.empty(2, dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor, time: torch.Tensor, context: torch.Tensor, transformer_options: dict) -> torch.Tensor:
        # x: [B, T, C], context: [B, L, context_dim]
        transformer_options = transformer_options.copy()
        h = self.x_embedder(x.movedim(1, 2)).movedim(1, 2)

        t = self.t_embedder(timestep_embedding(time, self.freq_dim).to(x.dtype))
        t0 = self.t_block(F.silu(t))
        y = self.y_embedder(context)
        freqs = rope(torch.arange(x.shape[1], device=x.device).unsqueeze(0), self.head_dim, self.rope_theta).unsqueeze(1)

        prefetch = comfy.model_prefetch.make_prefetch_queue(list(self.layers), x.device, transformer_options)
        for i, layer in enumerate(self.layers):
            comfy.model_prefetch.prefetch_queue_pop(prefetch, x.device, layer)
            transformer_options["block_index"] = i
            h = layer(h, y, t0, freqs, transformer_options=transformer_options)
        comfy.model_prefetch.prefetch_queue_pop(prefetch, x.device, None)

        shift, scale = (comfy.ops.cast_to_input(self.final_layer_scale_shift_table, h).unsqueeze(0) + t.unsqueeze(1)).chunk(2, dim=1)
        return self.output(modulate(self.norm(h), shift, scale))
