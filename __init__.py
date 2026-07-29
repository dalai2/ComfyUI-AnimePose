"""ComfyUI-AnimePose - pose estimation for anime / illustrated characters.

ComfyUI loads this file directly from custom_nodes/. The implementation lives
in the `anime_pose` package next to it, which keeps it importable by normal
means from scripts and tests (the repo directory name contains a hyphen and so
can't itself be a module name).
"""

from .anime_pose import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
