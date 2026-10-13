from typing import Optional, List, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import comfy.ops
from .codec import DACVAE
from .dit import DiT
from .align import SinusoidalEmbedding, AlignModalities, EmbedAnchors
from .vision_encoder import PerceptionEncoder
from .sampler import solve_ode


class SAMAudioModel(nn.Module):
    def __init__(self, config: dict, device=None, dtype=None, operations=None):
        super().__init__()

        self.config = config
        in_channels = config.get("in_channels", 768)
        text_dim = config.get("text_encoder", {}).get("dim", 768)
        num_anchors = config.get("num_anchors", 3)
        anchor_emb_dim = config.get("anchor_embedding_dim", 128)

        self.audio_codec = DACVAE(config.get("audio_codec", {}), device=device, dtype=dtype, operations=operations)
        self.transformer = DiT(config.get("transformer", {}), device=device, dtype=dtype, operations=operations)

        dim = self.transformer.dim
        self.proj = operations.Linear(in_channels, dim, bias=True, device=device, dtype=dtype)
        self.memory_proj = operations.Linear(text_dim, dim, bias=True, device=device, dtype=dtype)
        self.align_masked_video = AlignModalities(config.get("vision_encoder", {}).get("dim", 1024), dim, device=device, dtype=dtype, operations=operations)
        self.embed_anchors = EmbedAnchors(num_anchors, anchor_emb_dim, dim, device=device, dtype=dtype, operations=operations)
        self.timestep_emb = SinusoidalEmbedding(dim, device=device, dtype=dtype)
        self.vision_encoder = PerceptionEncoder(config.get("vision_encoder", {}), device=device, dtype=dtype, operations=operations)

    @property
    def sample_rate(self) -> int:
        return self.audio_codec.sample_rate

    @property
    def hop_length(self) -> int:
        return self.audio_codec.hop_length

    def forward_vector_field(
        self,
        noisy_audio: torch.Tensor,
        time: torch.Tensor,
        audio_features: torch.Tensor,
        text_features: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        masked_video_features: Optional[torch.Tensor] = None,
        anchor_ids: Optional[torch.Tensor] = None,
        anchor_alignment: Optional[torch.Tensor] = None,
        audio_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # noisy_audio: [B, T, 256]
        # audio_features: [B, T, 256]
        # Concatenate noisy target/residual, zero features, and clean mixture features
        zeros = torch.zeros_like(audio_features)
        x = torch.cat([noisy_audio, zeros, audio_features], dim=2)  # [B, T, 768]

        projected = self.proj(x)
        aligned = self.align_masked_video(projected, masked_video_features)
        aligned = self.embed_anchors(aligned, anchor_ids, anchor_alignment)

        t_emb = self.timestep_emb(time).unsqueeze(1).to(dtype=x.dtype)
        if text_features is not None:
            memory = self.memory_proj(text_features) + t_emb
        else:
            memory = t_emb

        return self.transformer(
            aligned,
            time,
            padding_mask=audio_pad_mask,
            memory=memory,
            memory_padding_mask=text_mask,
        )

    def prepare_anchors(
        self,
        batch_size: int,
        seq_len: int,
        audio_pad_mask: Optional[torch.Tensor] = None,
        anchors: Optional[List[List[Tuple[str, float, float]]]] = None,
        device=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        anchor_dict = {"<null>": 0, "+": 1, "-": 2, "<pad>": 3}
        if anchors is None:
            anchor_ids = torch.full((batch_size, 2), anchor_dict["<null>"], dtype=torch.long, device=device)
            anchor_ids[:, 1] = anchor_dict["<pad>"]
            anchor_alignment = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
            if audio_pad_mask is not None:
                anchor_alignment[~audio_pad_mask] = 1  # point to pad
            return anchor_ids, anchor_alignment

        anchor_alignment = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        if audio_pad_mask is not None:
            anchor_alignment[~audio_pad_mask] = 1

        ids = []
        for i, sample_anchors in enumerate(anchors):
            current = [anchor_dict["<null>"], anchor_dict["<pad>"]]
            for token, start_time, end_time in sample_anchors:
                start_idx = math.ceil(start_time * self.sample_rate / self.hop_length)
                end_idx = math.ceil(end_time * self.sample_rate / self.hop_length)
                start_idx = max(0, min(seq_len, start_idx))
                end_idx = max(0, min(seq_len, end_idx))
                anchor_alignment[i, start_idx:end_idx] = len(current)
                current.append(anchor_dict.get(token, anchor_dict["<null>"]))
            ids.append(torch.tensor(current, dtype=torch.long, device=device))

        # Pad ids to same length
        max_id_len = max(len(id_tensor) for id_tensor in ids)
        padded_ids = torch.full((batch_size, max_id_len), anchor_dict["<pad>"], dtype=torch.long, device=device)
        for i, id_tensor in enumerate(ids):
            padded_ids[i, :len(id_tensor)] = id_tensor

        return padded_ids, anchor_alignment

    def separate(
        self,
        waveform: torch.Tensor,
        text_features: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        anchors: Optional[List[List[Tuple[str, float, float]]]] = None,
        video_frames: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
        steps: int = 16,
        method: str = "midpoint",
        seed: Optional[int] = None,
        cfg: float = 1.0,
        uncond_text_features: Optional[torch.Tensor] = None,
        uncond_text_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = comfy.model_management.get_torch_device()
        dtype = self.proj.weight.dtype

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        elif waveform.ndim == 3 and waveform.shape[1] > 1:
            # Downmix multi-channel to mono
            waveform = waveform.mean(1, keepdim=True)

        B, _, orig_samples = waveform.shape
        waveform = waveform.to(device=device, dtype=dtype)
        # 1. Encode audio with DACVAE
        encoded = self.audio_codec.encode(waveform)  # [B, 128, T]
        T = encoded.shape[-1]

        # Audio features: [B, T, 256]
        audio_features = torch.cat([encoded.permute(0, 2, 1), encoded.permute(0, 2, 1)], dim=2).to(dtype=dtype)

        # 2. Prepare masks and anchors
        audio_pad_mask = torch.ones((B, T), dtype=torch.bool, device=device)
        anchor_ids, anchor_alignment = self.prepare_anchors(
            B, T, audio_pad_mask=audio_pad_mask, anchors=anchors, device=device
        )

        # 3. Prepare visual conditioning
        masked_video_features = None
        if video_frames is not None:
            if video_frames.ndim == 4 and video_frames.shape[-1] in (3, 4):
                video_frames = video_frames[..., :3].permute(0, 3, 1, 2)
            if video_mask is not None:
                if video_mask.ndim == 3:
                    video_mask = video_mask.unsqueeze(1)
                elif video_mask.ndim == 4 and video_mask.shape[-1] == 1:
                    video_mask = video_mask.permute(0, 3, 1, 2)
                v_frames = video_frames * (video_mask > 0).to(dtype=video_frames.dtype)
            else:
                v_frames = video_frames
            v_feats = self.vision_encoder.encode_video(v_frames)  # [T_v, 1024]
            # Interpolate video features to audio sequence length T
            v_feats = v_feats.unsqueeze(0).permute(0, 2, 1)  # [1, 1024, T_v]
            if v_feats.shape[-1] != T:
                v_feats = F.interpolate(v_feats, size=T, mode="linear", align_corners=False)
            masked_video_features = v_feats.expand(B, -1, -1).to(dtype=dtype)

        # 4. Prepare text features
        if text_features is not None:
            text_features = text_features.to(device=device, dtype=dtype)
        if text_mask is not None:
            text_mask = text_mask.to(device=device)

        # 5. Initialize noise
        if noise is None:
            if seed is not None:
                generator = torch.Generator(device=device).manual_seed(seed)
                noise = torch.randn(audio_features.shape, generator=generator, device=device, dtype=dtype)
            else:
                noise = torch.randn_like(audio_features)
        else:
            noise = noise.to(device=device, dtype=dtype)

        # 6. Define vector field
        def vector_field_fn(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            vf = self.forward_vector_field(
                noisy_audio=y,
                time=t,
                audio_features=audio_features,
                text_features=text_features,
                text_mask=text_mask,
                masked_video_features=masked_video_features,
                anchor_ids=anchor_ids,
                anchor_alignment=anchor_alignment,
                audio_pad_mask=audio_pad_mask,
            )
            if cfg > 1.0 and (uncond_text_features is not None or text_features is not None):
                u_text = uncond_text_features.to(device=device, dtype=dtype) if uncond_text_features is not None else None
                u_mask = uncond_text_mask.to(device=device) if uncond_text_mask is not None else None
                vf_uncond = self.forward_vector_field(
                    noisy_audio=y,
                    time=t,
                    audio_features=audio_features,
                    text_features=u_text,
                    text_mask=u_mask,
                    masked_video_features=masked_video_features,
                    anchor_ids=anchor_ids,
                    anchor_alignment=anchor_alignment,
                    audio_pad_mask=audio_pad_mask,
                )
                vf = vf_uncond + cfg * (vf - vf_uncond)
            return vf

        # 7. Integrate ODE from t=0.0 to t=1.0
        generated = solve_ode(vector_field_fn, noise, steps=steps, method=method)

        # 8. Decode target and residual
        target_lat = generated[:, :, :128].permute(0, 2, 1).to(dtype=dtype)
        residual_lat = generated[:, :, 128:256].permute(0, 2, 1).to(dtype=dtype)
        target_wav = self.audio_codec.decode(target_lat)[:, :, :orig_samples]
        residual_wav = self.audio_codec.decode(residual_lat)[:, :, :orig_samples]

        return target_wav, residual_wav
