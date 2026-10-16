import os
import folder_paths

folder_paths.add_model_folder_path("sam_audio", os.path.join(folder_paths.models_dir, "sam-audio"))
folder_paths.folder_names_and_paths["sam_audio"] = (folder_paths.folder_names_and_paths["sam_audio"][0], {".safetensors"})

from .nodes_sam_audio import comfy_entrypoint

__all__ = ["comfy_entrypoint"]
