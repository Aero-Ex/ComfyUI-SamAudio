import os
import folder_paths

# Register model directories for SAM-Audio
for _p in [
    os.path.join(folder_paths.models_dir, "sam-audio"),
    os.path.join(folder_paths.models_dir, "sam_audio"),
]:
    folder_paths.add_model_folder_path("sam_audio", _p)

from .nodes_sam_audio import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    SAMAudioExtension,
    comfy_entrypoint,
)

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "SAMAudioExtension",
    "comfy_entrypoint",
]
