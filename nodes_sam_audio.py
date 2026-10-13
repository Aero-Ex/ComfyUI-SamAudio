import json
import logging
import os
import torch
from typing_extensions import override
import folder_paths
import comfy.audio
import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.sd
import comfy.utils
from comfy_api.latest import ComfyExtension, io
from .sam_audio.convert import convert_sam_audio_state_dict, is_raw_sam_audio_checkpoint
from .sam_audio.model import SAMAudioModel


logger = logging.getLogger(__name__)


def _load_sam_audio(model_path: str):
    sd, metadata = comfy.utils.load_torch_file(model_path, safe_load=True, return_metadata=True)
    metadata = metadata or {}

    config = None
    if "sam_audio.config" in metadata:
        config = json.loads(metadata["sam_audio.config"])
    else:
        # Check adjacent config files
        model_dir = os.path.dirname(model_path)
        base_name = os.path.basename(model_path).lower()
        if "large" in base_name:
            cfg_path = os.path.join(model_dir, "configlarge.json")
        else:
            cfg_path = os.path.join(model_dir, "config.json")

        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            # Fallback to base config
            cfg_path = os.path.join(model_dir, "config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    config = json.load(f)

    if config is None:
        raise RuntimeError(f"Could not find configuration for SAM-Audio checkpoint {model_path}. Please provide config.json alongside the weights.")

    if is_raw_sam_audio_checkpoint(sd):
        num_layers = config.get("transformer", {}).get("n_layers", 12)
        logger.info(f"Detected raw upstream SAM-Audio checkpoint, converting state dict on the fly (layers: {num_layers})...")
        sd = convert_sam_audio_state_dict(sd, num_layers=num_layers)

    supported_dtypes = [torch.bfloat16, torch.float16, torch.float32]
    dtype = comfy.model_management.unet_dtype(model_params=-1, supported_dtypes=supported_dtypes)
    manual_cast_dtype = comfy.model_management.unet_manual_cast(dtype, comfy.model_management.get_torch_device(), supported_dtypes=supported_dtypes)
    operations = comfy.ops.pick_operations(dtype, manual_cast_dtype, load_device=comfy.model_management.get_torch_device())

    model = SAMAudioModel(
        config,
        device=comfy.model_management.unet_offload_device(),
        dtype=dtype,
        operations=operations,
    )
    model.load_state_dict(sd, strict=True)

    patcher = comfy.model_patcher.CoreModelPatcher(
        model,
        load_device=comfy.model_management.get_torch_device(),
        offload_device=comfy.model_management.unet_offload_device(),
    )
    return patcher


def _encode_text_prompt(prompt: str, clip=None, device=None, dtype=None):
    if not prompt or not prompt.strip():
        return None, None

    if clip is None:
        raise ValueError("SAMAudioTextSeparate: A CLIP text encoder must be connected to the 'clip' socket.")
    tokens = clip.tokenize(prompt)
    cond, _ = clip.encode_from_tokens(tokens, return_pooled=True)
    if "t5base" in tokens and len(tokens["t5base"]) > 0:
        raw_ids = [tok_id for tok_id, _ in tokens["t5base"][0]]
        real_len = next((i + 1 for i, tok_id in enumerate(raw_ids) if tok_id == 1), len(raw_ids))
        cond = cond[:, :real_len, :]
    if device is not None:
        cond = cond.to(device=device)
    if dtype is not None:
        cond = cond.to(dtype=dtype)

    mask = torch.ones((cond.shape[0], cond.shape[1]), dtype=torch.bool, device=cond.device)
    return cond, mask


def _resample_audio(audio_dict, target_sr=48000):
    waveform = audio_dict["waveform"]
    sr = audio_dict.get("sample_rate", target_sr)
    if sr != target_sr:
        waveform = comfy.audio.resample(waveform, sr, target_sr)
    return waveform



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
                    tooltip="Select SAM-Audio checkpoint (.safetensors or .pt) from models/sam-audio.",
                ),
            ],
            outputs=[
                io.Custom("SAM_AUDIO").Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, model_name: str) -> io.NodeOutput:
        model_path = folder_paths.get_full_path_or_raise("sam_audio", model_name)
        patcher = _load_sam_audio(model_path)
        return io.NodeOutput(patcher)


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
                io.String.Input(
                    "prompt",
                    default="A person speaking",
                    multiline=True,
                    tooltip="Description of the sound to isolate (e.g. 'A dog barking', 'Piano playing a melody').",
                ),
                io.String.Input(
                    "negative_prompt",
                    default="",
                    multiline=True,
                    tooltip="Optional sound description to explicitly suppress/subtract (e.g. 'music', 'drums', 'crowd noise'). Active when cfg > 1.0.",
                ),
                io.Clip.Input("clip", tooltip="Connect T5-base CLIP model (e.g. from Load CLIP with t5-base.safetensors)."),
                io.Int.Input("steps", default=16, min=1, max=128, step=1, tooltip="ODE integration steps (default 16 for midpoint)."),
                io.Combo.Input("method", options=["midpoint", "euler"], default="midpoint", tooltip="ODE integration numerical solver."),
                io.Float.Input("cfg", default=1.0, min=0.0, max=10.0, step=0.1, tooltip="Classifier-free guidance strength (1.0 = standard unguided, >1.0 activates negative prompt subtraction)."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, tooltip="Random seed for ODE sampling noise."),
            ],
            outputs=[
                io.Audio.Output(display_name="target_audio"),
                io.Audio.Output(display_name="residual_audio"),
            ],
        )

    @classmethod
    def execute(cls, model, audio, prompt: str, negative_prompt: str = "", clip=None, steps: int = 16, method: str = "midpoint", cfg: float = 1.0, seed: int = 0) -> io.NodeOutput:
        comfy.model_management.load_models_gpu([model])
        inner: SAMAudioModel = model.model
        device = comfy.model_management.get_torch_device()
        dtype = inner.proj.weight.dtype

        waveform = _resample_audio(audio, target_sr=inner.sample_rate)
        text_features, text_mask = _encode_text_prompt(prompt, clip=clip, device=device, dtype=dtype)

        uncond_features, uncond_mask = None, None
        if cfg > 1.0:
            neg_text = negative_prompt if (negative_prompt and negative_prompt.strip()) else ""
            uncond_features, uncond_mask = _encode_text_prompt(neg_text, clip=clip, device=device, dtype=dtype)
        target_wav, residual_wav = inner.separate(
            waveform=waveform,
            text_features=text_features,
            text_mask=text_mask,
            steps=steps,
            method=method,
            seed=seed,
            cfg=cfg,
            uncond_text_features=uncond_features,
            uncond_text_mask=uncond_mask,
            device=device,
        )
        comfy.model_management.soft_empty_cache()

        out_target = {"waveform": target_wav.float().cpu(), "sample_rate": inner.sample_rate}
        out_residual = {"waveform": residual_wav.float().cpu(), "sample_rate": inner.sample_rate}
        return io.NodeOutput(out_target, out_residual)


class SAMAudioSpanSeparate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SAMAudioSpanSeparate",
            display_name="SAM-Audio Span Separate",
            category="audio/separation",
            description="Isolate sounds by specifying time ranges where the target sound occurs (+) or does not occur (-).",
            inputs=[
                io.Custom("SAM_AUDIO").Input("model", tooltip="The loaded SAM-Audio model."),
                io.Audio.Input("audio", tooltip="Input audio mixture."),
                io.Float.Input("start_time", default=0.0, min=0.0, max=3600.0, step=0.1, tooltip="Anchor span start time in seconds."),
                io.Float.Input("end_time", default=1.0, min=0.0, max=3600.0, step=0.1, tooltip="Anchor span end time in seconds."),
                io.Combo.Input("mode", options=["present (+)", "absent (-)"], default="present (+)", tooltip="Whether sound IS present or NOT present in this span."),
                io.String.Input("prompt", default="", multiline=True, optional=True, tooltip="Optional text description to assist separation."),
                io.Clip.Input("clip", optional=True, tooltip="Optional T5-base CLIP text encoder."),
                io.Int.Input("steps", default=16, min=1, max=128, step=1, tooltip="ODE integration steps."),
                io.Combo.Input("method", options=["midpoint", "euler"], default="midpoint"),
                io.Float.Input("cfg", default=1.0, min=0.0, max=10.0, step=0.1, tooltip="Classifier-free guidance strength (1.0 = standard unguided)."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
            ],
            outputs=[
                io.Audio.Output(display_name="target_audio"),
                io.Audio.Output(display_name="residual_audio"),
            ],
        )

    @classmethod
    def execute(cls, model, audio, start_time: float, end_time: float, mode: str = "present (+)", prompt: str = "", clip=None, steps: int = 16, method: str = "midpoint", cfg: float = 1.0, seed: int = 0) -> io.NodeOutput:
        comfy.model_management.load_models_gpu([model])
        inner: SAMAudioModel = model.model
        device = comfy.model_management.get_torch_device()
        dtype = inner.proj.weight.dtype

        waveform = _resample_audio(audio, target_sr=inner.sample_rate)
        text_features, text_mask = None, None
        if prompt and prompt.strip() and clip is not None:
            text_features, text_mask = _encode_text_prompt(prompt, clip=clip, device=device, dtype=dtype)
        sign = "+" if "present" in mode else "-"
        anchors = [[(sign, float(start_time), float(end_time))]]

        target_wav, residual_wav = inner.separate(
            waveform=waveform,
            text_features=text_features,
            text_mask=text_mask,
            anchors=anchors,
            steps=steps,
            method=method,
            cfg=cfg,
            seed=seed,
            device=device,
        )
        comfy.model_management.soft_empty_cache()

        out_target = {"waveform": target_wav.float().cpu(), "sample_rate": inner.sample_rate}
        out_residual = {"waveform": residual_wav.float().cpu(), "sample_rate": inner.sample_rate}
        return io.NodeOutput(out_target, out_residual)


class SAMAudioVisualSeparate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SAMAudioVisualSeparate",
            display_name="SAM-Audio Visual Separate",
            category="audio/separation",
            description="Isolate sounds associated with a visual object in video frames using an optional object mask (e.g. from SAM3 video tracker).",
            inputs=[
                io.Custom("SAM_AUDIO").Input("model", tooltip="The loaded SAM-Audio model."),
                io.Audio.Input("audio", tooltip="Input audio mixture."),
                io.Image.Input("images", tooltip="Video frames [N, H, W, 3]."),
                io.Mask.Input("mask", optional=True, tooltip="Object mask for video frames [N, H, W] (e.g. from SAM3 tracker)."),
                io.String.Input("prompt", default="", multiline=True, optional=True, tooltip="Optional text description."),
                io.Clip.Input("clip", optional=True, tooltip="Optional T5-base CLIP text encoder."),
                io.Int.Input("steps", default=16, min=1, max=128, step=1, tooltip="ODE integration steps."),
                io.Combo.Input("method", options=["midpoint", "euler"], default="midpoint"),
                io.Float.Input("cfg", default=1.0, min=0.0, max=10.0, step=0.1, tooltip="Classifier-free guidance strength (1.0 = standard unguided)."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
            ],
            outputs=[
                io.Audio.Output(display_name="target_audio"),
                io.Audio.Output(display_name="residual_audio"),
            ],
        )

    @classmethod
    def execute(cls, model, audio, images, mask=None, prompt: str = "", clip=None, steps: int = 16, method: str = "midpoint", cfg: float = 1.0, seed: int = 0) -> io.NodeOutput:
        comfy.model_management.load_models_gpu([model])
        inner: SAMAudioModel = model.model
        device = comfy.model_management.get_torch_device()
        dtype = inner.proj.weight.dtype

        waveform = _resample_audio(audio, target_sr=inner.sample_rate)
        text_features, text_mask = None, None
        if prompt and prompt.strip() and clip is not None:
            text_features, text_mask = _encode_text_prompt(prompt, clip=clip, device=device, dtype=dtype)
        video_frames = images.to(device=device, dtype=dtype)
        video_mask = mask.to(device=device, dtype=dtype).unsqueeze(1) if mask is not None else None

        target_wav, residual_wav = inner.separate(
            waveform=waveform,
            text_features=text_features,
            text_mask=text_mask,
            video_frames=video_frames,
            video_mask=video_mask,
            steps=steps,
            method=method,
            cfg=cfg,
            seed=seed,
            device=device,
        )
        comfy.model_management.soft_empty_cache()

        out_target = {"waveform": target_wav.float().cpu(), "sample_rate": inner.sample_rate}
        out_residual = {"waveform": residual_wav.float().cpu(), "sample_rate": inner.sample_rate}
        return io.NodeOutput(out_target, out_residual)


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


NODE_CLASS_MAPPINGS = {
    "SAMAudioModelLoader": SAMAudioModelLoader,
    "SAMAudioTextSeparate": SAMAudioTextSeparate,
    "SAMAudioSpanSeparate": SAMAudioSpanSeparate,
    "SAMAudioVisualSeparate": SAMAudioVisualSeparate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAMAudioModelLoader": "Load SAM-Audio Model",
    "SAMAudioTextSeparate": "SAM-Audio Text Separate",
    "SAMAudioSpanSeparate": "SAM-Audio Span Separate",
    "SAMAudioVisualSeparate": "SAM-Audio Visual Separate",
}
