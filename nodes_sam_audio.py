import json
import logging
import os
import torch
import torch.nn.functional as F
from typing_extensions import override
import folder_paths
import comfy.audio
import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.utils
from comfy_api.latest import ComfyExtension, io
from .sam_audio.model import SAMAudioModel
from .sam_audio.sampler import solve_ode


def _load_sam_audio(model_path: str):
    sd, metadata = comfy.utils.load_torch_file(model_path, safe_load=True, return_metadata=True)
    if not metadata or "sam_audio.config" not in metadata:
        raise RuntimeError(f"{os.path.basename(model_path)} is not a ComfyUI SAM-Audio checkpoint; download sam_audio_base/large.safetensors from huggingface.co/Aero-Ex/ComfyUI-SamAudio.")
    config = json.loads(metadata["sam_audio.config"])

    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()
    supported_dtypes = [torch.bfloat16, torch.float16, torch.float32]
    weight_dtype = comfy.utils.weight_dtype(sd)
    dtype = comfy.model_management.unet_dtype(model_params=comfy.utils.calculate_parameters(sd), supported_dtypes=supported_dtypes, weight_dtype=weight_dtype)
    manual_cast_dtype = comfy.model_management.unet_manual_cast(dtype, load_device, supported_dtypes=supported_dtypes)
    operations = comfy.ops.pick_operations(dtype, manual_cast_dtype, load_device=load_device)
    if weight_dtype != dtype:
        # lazily loaded layers keep the checkpoint tensors, so convert them to the model dtype once here
        sd = {k: v.to(dtype) if v.dtype == weight_dtype else v for k, v in sd.items()}

    model = SAMAudioModel(config, device=offload_device, dtype=dtype, manual_cast_dtype=manual_cast_dtype, operations=operations)
    model.load_state_dict(sd, strict=True)
    return comfy.model_patcher.CoreModelPatcher(model, load_device=load_device, offload_device=offload_device)


def _encode_text(clip, text: str) -> torch.Tensor:
    tokens = clip.tokenize(text)
    cond = clip.encode_from_tokens(tokens)
    # ComfyUI pads T5 to a minimum length; keep up to the end token like upstream padding="longest"
    ids = [token for token, _ in tokens["t5base"][0]]
    return cond[:, :ids.index(1) + 1]


def _codec_apply(fn, x: torch.Tensor, scale: float, out_channels: int, tile: int, overlap: int) -> torch.Tensor:
    try:
        return fn(x)
    except Exception as e:
        comfy.model_management.raise_non_oom(e)
    logging.warning("Ran out of memory in the SAM-Audio codec, retrying with tiling.")
    comfy.model_management.soft_empty_cache()
    return comfy.utils.tiled_scale_multidim(x, fn, tile=(tile,), overlap=overlap, upscale_amount=scale, out_channels=out_channels, output_device=x.device)


def _encode_video(model: SAMAudioModel, images: torch.Tensor, mask, length: int, device, dtype) -> torch.Tensor:
    # one frame per latent frame, spread over the clip like upstream; each distinct frame is encoded once
    count = images.shape[0]
    index = [round(i * (count - 1) / max(length - 1, 1)) for i in range(length)]
    frames_used = sorted(set(index))
    frames = images[frames_used].movedim(-1, 1)
    if mask is not None:
        mask = mask[[round(i * (mask.shape[0] - 1) / max(count - 1, 1)) for i in frames_used]].unsqueeze(1)
        if mask.shape[-2:] != frames.shape[-2:]:
            mask = F.interpolate(mask, size=frames.shape[-2:])
        # like upstream's mask_videos: the model sees the scene with the tracked object blacked out
        frames = frames * (1 - mask)

    encoder = model.vision_encoder
    size = (encoder.image_size, encoder.image_size)
    features = []
    for i in range(0, len(frames_used), 32):
        x = F.interpolate(frames[i:i + 32].to(device), size=size, mode="bicubic", antialias=True)
        features.append(encoder((x * 2 - 1).to(dtype)))
    features = torch.cat(features)
    slot = {frame: i for i, frame in enumerate(frames_used)}
    return features[[slot[i] for i in index]].movedim(0, 1).unsqueeze(0)


# SAM-Audio stops separating on long clips (the target fades to silence beyond ~30 s, upstream too), so long audio is split into
# crossfaded chunks; 15 s chunks lost the target less often than 30 s ones on long speech-over-music tests
CHUNK_SECONDS = 15
OVERLAP_SECONDS = 2


def _separate_chunk(model, waveform, text, uncond_text, anchors, images, mask, steps, method, cfg, generator, pbar):
    inner: SAMAudioModel = model.model
    codec = inner.audio_codec
    device = model.load_device
    dtype = inner.get_dtype_inference()
    hop = codec.hop_length

    latents = _codec_apply(codec.encode, codec.pad(waveform), 1 / hop, codec.latent_channels, 256 * hop, 32 * hop).to(dtype)
    audio_features = latents.movedim(1, 2).repeat(1, 1, 2)
    B, T, _ = audio_features.shape

    if images is None:
        # upstream feeds zero video features when there is no video
        video_features = audio_features.new_zeros(B, inner.vision_encoder.dim, T)
    else:
        video_features = _encode_video(inner, images, mask, T, device, dtype).expand(B, -1, -1)
    anchor_ids, anchor_alignment = inner.prepare_anchors(anchors, B, T, device)
    transformer_options = {"prefetch_dynamic_vbars": model.is_dynamic()}

    def velocity(x, t):
        v = inner(x, t, audio_features, text, video_features, anchor_ids, anchor_alignment, transformer_options)
        if uncond_text is None:
            return v
        v_uncond = inner(x, t, audio_features, uncond_text, video_features, anchor_ids, anchor_alignment, transformer_options)
        return torch.lerp(v_uncond, v, cfg)

    noise = torch.randn(audio_features.shape, generator=generator).to(device=device, dtype=dtype)
    generated = solve_ode(velocity, noise, steps, method, pbar)
    return [_codec_apply(codec.decode, x.movedim(1, 2), hop, 1, 256, 32)[..., :waveform.shape[-1]] for x in generated.chunk(2, dim=2)]


def _time_slice(frames, start: int, end: int, samples: int):
    # the frames of a video spanning the whole audio that fall in samples [start, end)
    count = frames.shape[0]
    first = min(start * count // samples, count - 1)
    return frames[first:max(first + 1, -(-end * count // samples))]


def _separate(model, audio, clip, prompt: str, negative_prompt: str, anchors, images, mask, steps: int, method: str, cfg: float, seed: int, chunk_long_audio: bool) -> io.NodeOutput:
    text = _encode_text(clip, prompt)
    uncond_text = _encode_text(clip, negative_prompt) if cfg != 1.0 else None

    comfy.model_management.load_models_gpu([model])
    inner: SAMAudioModel = model.model
    device = model.load_device
    dtype = inner.get_dtype_inference()
    sr = inner.sample_rate
    text = text.to(device=device, dtype=dtype)
    if uncond_text is not None:
        uncond_text = uncond_text.to(device=device, dtype=dtype)

    waveform = audio["waveform"].to(device).mean(1, keepdim=True)
    waveform = comfy.audio.resample(waveform, audio["sample_rate"], sr).to(dtype)
    samples = waveform.shape[-1]
    chunk, overlap = CHUNK_SECONDS * sr, OVERLAP_SECONDS * sr
    starts = list(range(0, samples - overlap, chunk - overlap)) if chunk_long_audio and samples > chunk else [0]

    generator = torch.Generator().manual_seed(seed)
    pbar = comfy.utils.ProgressBar(steps * len(starts))
    target = waveform.new_zeros(waveform.shape, dtype=torch.float32)
    residual = torch.zeros_like(target)
    for i, start in enumerate(starts):
        end = samples if i == len(starts) - 1 else start + chunk
        offset = start / sr
        chunk_anchors = [(token, max(s - offset, 0.0), max(e - offset, 0.0)) for token, s, e in anchors]
        chunk_images = None if images is None else _time_slice(images, start, end, samples)
        chunk_mask = None if mask is None else _time_slice(mask, start, end, samples)
        outs = _separate_chunk(model, waveform[..., start:end], text, uncond_text, chunk_anchors, chunk_images, chunk_mask, steps, method, cfg, generator, pbar)
        # linear crossfade: fade-ins and fade-outs over each overlap sum to one
        fade = torch.ones(end - start, device=device)
        if i > 0:
            fade[:overlap] = torch.linspace(0, 1, overlap, device=device)
        if i < len(starts) - 1:
            fade[-overlap:] = torch.linspace(1, 0, overlap, device=device)
        for acc, out in zip((target, residual), outs):
            acc[..., start:end] += out.float() * fade
    return io.NodeOutput(
        {"waveform": target.cpu(), "sample_rate": sr},
        {"waveform": residual.cpu(), "sample_rate": sr},
    )


def _common_inputs():
    return [
        io.Int.Input("steps", default=16, min=1, max=128, step=1, tooltip="ODE integration steps (16 midpoint steps matches upstream)."),
        io.Combo.Input("method", options=["midpoint", "euler"], default="midpoint", tooltip="ODE integration numerical solver."),
        io.Float.Input("cfg", default=1.0, min=0.0, max=10.0, step=0.1, tooltip="Text guidance strength. 1.0 matches upstream (no guidance); other values add a second pass per step. Not part of upstream SAM-Audio."),
        io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True, tooltip="Random seed for the initial noise."),
        io.Boolean.Input("chunk_long_audio", default=True, tooltip=f"Separate audio longer than {CHUNK_SECONDS} s in crossfaded {CHUNK_SECONDS} s chunks. SAM-Audio stops separating on long clips (the target fades to silence beyond ~30 s); turn off to process the whole clip at once like upstream."),
    ]


SEPARATE_OUTPUTS = [
    io.Audio.Output(display_name="target_audio"),
    io.Audio.Output(display_name="residual_audio"),
]


class SAMAudioModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SAMAudioModelLoader",
            display_name="Load SAM-Audio Model",
            category="audio/loaders",
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=folder_paths.get_filename_list("sam_audio"),
                    tooltip="SAM-Audio .safetensors checkpoint from models/sam-audio.",
                ),
            ],
            outputs=[
                io.Custom("SAM_AUDIO").Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, model_name: str) -> io.NodeOutput:
        return io.NodeOutput(_load_sam_audio(folder_paths.get_full_path_or_raise("sam_audio", model_name)))


class SAMAudioTextSeparate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SAMAudioTextSeparate",
            display_name="SAM-Audio Text Separate",
            category="audio/separation",
            description="Isolate or separate a sound from audio using a natural language description.",
            inputs=[
                io.Custom("SAM_AUDIO").Input("model", tooltip="The loaded SAM-Audio model."),
                io.Audio.Input("audio", tooltip="Input audio mixture to separate."),
                io.String.Input("prompt", default="A person speaking", multiline=True, tooltip="Description of the sound to isolate (e.g. 'A dog barking', 'Piano playing a melody')."),
                io.String.Input("negative_prompt", default="", multiline=True, tooltip="Sound to steer away from when cfg is not 1.0 (e.g. 'music', 'crowd noise')."),
                io.Clip.Input("clip", tooltip="T5-base text encoder (Load CLIP with t5-base.safetensors, type stable_audio)."),
                *_common_inputs(),
            ],
            outputs=SEPARATE_OUTPUTS,
        )

    @classmethod
    def execute(cls, model, audio, prompt: str, negative_prompt: str, clip, steps: int, method: str, cfg: float, seed: int, chunk_long_audio: bool) -> io.NodeOutput:
        return _separate(model, audio, clip, prompt, negative_prompt, [], None, None, steps, method, cfg, seed, chunk_long_audio)


class SAMAudioSpanSeparate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SAMAudioSpanSeparate",
            display_name="SAM-Audio Span Separate",
            category="audio/separation",
            description="Isolate sounds by specifying a time range where the target sound occurs (+) or does not occur (-).",
            inputs=[
                io.Custom("SAM_AUDIO").Input("model", tooltip="The loaded SAM-Audio model."),
                io.Audio.Input("audio", tooltip="Input audio mixture."),
                io.Float.Input("start_time", default=0.0, min=0.0, max=3600.0, step=0.1, tooltip="Anchor span start time in seconds."),
                io.Float.Input("end_time", default=1.0, min=0.0, max=3600.0, step=0.1, tooltip="Anchor span end time in seconds."),
                io.Combo.Input("mode", options=["present (+)", "absent (-)"], default="present (+)", tooltip="Whether the sound IS present or NOT present in this span."),
                io.String.Input("prompt", default="", multiline=True, tooltip="Optional text description to assist separation."),
                io.Clip.Input("clip", tooltip="T5-base text encoder (Load CLIP with t5-base.safetensors, type stable_audio)."),
                *_common_inputs(),
            ],
            outputs=SEPARATE_OUTPUTS,
        )

    @classmethod
    def execute(cls, model, audio, start_time: float, end_time: float, mode: str, prompt: str, clip, steps: int, method: str, cfg: float, seed: int, chunk_long_audio: bool) -> io.NodeOutput:
        anchors = [("+" if mode == "present (+)" else "-", start_time, end_time)]
        return _separate(model, audio, clip, prompt, "", anchors, None, None, steps, method, cfg, seed, chunk_long_audio)


class SAMAudioVisualSeparate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SAMAudioVisualSeparate",
            display_name="SAM-Audio Visual Separate",
            category="audio/separation",
            description="Isolate sounds associated with a visual object in video frames using an optional object mask (e.g. from a SAM3 video tracker).",
            inputs=[
                io.Custom("SAM_AUDIO").Input("model", tooltip="The loaded SAM-Audio model."),
                io.Audio.Input("audio", tooltip="Input audio mixture."),
                io.Image.Input("images", tooltip="Video frames spanning the same duration as the audio."),
                io.Mask.Input("mask", optional=True, tooltip="Mask of the object whose sound to isolate (e.g. from a SAM3 video tracker); resized to the frames if needed."),
                io.String.Input("prompt", default="", multiline=True, tooltip="Optional text description."),
                io.Clip.Input("clip", tooltip="T5-base text encoder (Load CLIP with t5-base.safetensors, type stable_audio)."),
                *_common_inputs(),
            ],
            outputs=SEPARATE_OUTPUTS,
        )

    @classmethod
    def execute(cls, model, audio, images, prompt: str, clip, steps: int, method: str, cfg: float, seed: int, chunk_long_audio: bool, mask=None) -> io.NodeOutput:
        return _separate(model, audio, clip, prompt, "", [], images, mask, steps, method, cfg, seed, chunk_long_audio)


class SAMAudioExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SAMAudioModelLoader,
            SAMAudioTextSeparate,
            SAMAudioSpanSeparate,
            SAMAudioVisualSeparate,
        ]


async def comfy_entrypoint() -> SAMAudioExtension:
    return SAMAudioExtension()
