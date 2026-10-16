import math
import torch
import torch.nn as nn
from comfy.ldm.modules.diffusionmodules.util import timestep_embedding
from .codec import DACVAE
from .dit import DiT
from .align import AlignModalities, EmbedAnchors
from .vision_encoder import PerceptionEncoder

# anchor vocabulary: 0 <null>, 1 "+", 2 "-", 3 <pad>
ANCHOR_IDS = {"+": 1, "-": 2}


class SAMAudioModel(nn.Module):
    def __init__(self, config: dict, device=None, dtype=None, manual_cast_dtype=None, operations=None):
        super().__init__()
        self.manual_cast_dtype = manual_cast_dtype
        vision_dim = config.get("vision_encoder", {}).get("dim", 1024)

        self.audio_codec = DACVAE(config.get("audio_codec", {}), device=device, dtype=dtype, operations=operations)
        self.transformer = DiT(config.get("transformer", {}), device=device, dtype=dtype, operations=operations)

        dim = self.transformer.dim
        self.proj = operations.Linear(config.get("in_channels", 768), dim, bias=True, device=device, dtype=dtype)
        self.memory_proj = operations.Linear(config.get("text_encoder", {}).get("dim", 768), dim, bias=True, device=device, dtype=dtype)
        self.align_masked_video = AlignModalities(vision_dim, dim, device=device, dtype=dtype, operations=operations)
        self.embed_anchors = EmbedAnchors(config.get("num_anchors", 3), config.get("anchor_embedding_dim", 128), dim, device=device, dtype=dtype, operations=operations)
        self.vision_encoder = PerceptionEncoder(config.get("vision_encoder", {}), device=device, dtype=dtype, operations=operations)

    @property
    def sample_rate(self) -> int:
        return self.audio_codec.sample_rate

    @property
    def hop_length(self) -> int:
        return self.audio_codec.hop_length

    def get_dtype_inference(self):
        return self.manual_cast_dtype or self.proj.weight.dtype

    def forward(self, x, time, audio_features, text_features, video_features, anchor_ids, anchor_alignment, transformer_options):
        # x, audio_features: [B, T, 2 * latent], text_features: [B or 1, L, text_dim], video_features: [B, vision_dim, T]
        h = self.proj(torch.cat([x, torch.zeros_like(audio_features), audio_features], dim=2))
        h = self.align_masked_video(h, video_features)
        h = self.embed_anchors(h, anchor_ids, anchor_alignment)
        memory = self.memory_proj(text_features) + timestep_embedding(time, self.transformer.dim).to(h.dtype).unsqueeze(1)
        return self.transformer(h, time, memory, transformer_options=transformer_options)

    def prepare_anchors(self, anchors, batch_size: int, length: int, device):
        # anchors: [(token, start_seconds, end_seconds)]; every latent frame points at a slot of the id list
        ids = [0, 3]
        alignment = [0] * length
        for token, start_time, end_time in anchors:
            start = math.ceil(start_time * self.sample_rate / self.hop_length)
            end = min(math.ceil(end_time * self.sample_rate / self.hop_length), length)
            alignment[start:end] = [len(ids)] * max(end - start, 0)
            ids.append(ANCHOR_IDS[token])
        return torch.tensor([ids] * batch_size, device=device), torch.tensor([alignment] * batch_size, device=device)
