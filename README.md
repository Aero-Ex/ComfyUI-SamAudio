# ComfyUI-SAM-Audio

dependency-free ComfyUI implementation of Meta's **SAM-Audio** (Segment Anything Model for Audio).

Isolates any sound in an audio mixture using text descriptions, temporal span anchors, or video tracking cues.

## Features

- **Dependency-Free:** Runs directly on PyTorch and native ComfyUI operations (no external `sam_audio`, `dacvae`, `torchcodec`, or `imagebind` packages required).
- **All 3 Prompting Modalities:**
  - **Text Separation (`SAMAudioTextSeparate`):** Isolate vocals, speech, instruments, sound effects using natural language.
  - **Temporal Span Separation (`SAMAudioSpanSeparate`):** Pinpoint exact timestamps where the target sound occurs (`+` present) or does not occur (`-` absent).
  - **Visual-Guided Separation (`SAMAudioVisualSeparate`):** Isolate sounds tied to visual objects using video frames and tracking masks (e.g. from native SAM 3 tracking).

## Model Downloads & Setup

### 1. SAM-Audio Models
Download converted BF16 `.safetensors` from Hugging Face:  
**[Aero-Ex/ComfyUI-SamAudio](https://huggingface.co/Aero-Ex/ComfyUI-SamAudio)**

| Checkpoint | Size | Description |
| :--- | :---: | :--- |
| **[`sam_audio_base.safetensors`](https://huggingface.co/Aero-Ex/ComfyUI-SamAudio/resolve/main/sam_audio_base.safetensors)** | 1.73 GB | 12-layer DiT, fast daily driver |
| **[`sam_audio_large.safetensors`](https://huggingface.co/Aero-Ex/ComfyUI-SamAudio/resolve/main/sam_audio_large.safetensors)** | 6.38 GB | 22-layer DiT, maximum separation precision |

Place downloaded model(s) into:
```
ComfyUI/models/sam-audio/
```

CLI Quick Download:
```bash
hf download Aero-Ex/ComfyUI-SamAudio --local-dir models/sam-audio
```

### 2. T5-Base Text Encoder
All separation nodes need it. Load it with **Load CLIP**, type `stable_audio`.

Download `t5-base.safetensors` from:  
**[ComfyUI-Wiki/t5-base](https://huggingface.co/ComfyUI-Wiki/t5-base/resolve/main/t5-base.safetensors)**

Place into:
```
ComfyUI/models/text_encoders/
```

CLI Quick Download:
```bash
hf download Comfy-Org/t5-base t5-base.safetensors --local-dir models/text_encoders
```

### 3. Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Aero-Ex/ComfyUI-SamAudio.git
```
## Node Overview

| Node | Category | Inputs | Outputs |
| :--- | :--- | :--- | :--- |
| **Load SAM-Audio Model** | `audio/loaders` | `model_name` (`.safetensors`) | `model` |
| **SAM-Audio Text Separate** | `audio/separation` | `model`, `audio`, `prompt`, `negative_prompt`, `clip`, `steps`, `method`, `cfg`, `seed` | `target_audio`, `residual_audio` |
| **SAM-Audio Span Separate** | `audio/separation` | `model`, `audio`, `start_time`, `end_time`, `mode`, `prompt`, `clip`, `steps`, `method`, `cfg`, `seed` | `target_audio`, `residual_audio` |
| **SAM-Audio Visual Separate** | `audio/separation` | `model`, `audio`, `images`, `mask`, `prompt`, `clip`, `steps`, `method`, `cfg`, `seed` | `target_audio`, `residual_audio` |

## Example Workflow

An example workflow is included in `workflows/audio_sam_audio_separation.json`.

## Differences from upstream

- `cfg` / `negative_prompt` are an addition; upstream has no guidance. `cfg = 1.0` matches upstream.
- The DACVAE decoder's audio watermark is not applied.
- Audio is downmixed to mono and resampled to 48 kHz; outputs are trimmed to the input length.
- Codec encode/decode falls back to tiling when it runs out of VRAM, so long audio can differ slightly from a single pass.
- `chunk_long_audio` (on by default) separates audio longer than 15 s in crossfaded 15 s chunks. The model loses the target on long clips: on a 101 s clip, one pass returned a silent target, upstream too. Turn it off to match upstream exactly.
