"""ComfyUI-AnimePose - node registration.

Pose estimation for anime / illustrated characters, built on
ShuhongChen/bizarre-pose-estimator (Chen & Zwicker, WACV 2022). Output is an
OpenPose-18 skeleton on black, interchangeable with what DWPreprocessor emits,
so it drops straight into an existing ControlNetApply (OpenPose) graph.

No Docker and no Detectron2: that keypoint branch is reproduced with
torchvision's KeypointRCNN, and weights download themselves into
ComfyUI/models/anime_pose/ on first use.

Optional environment overrides:
    ANIMEPOSE_POSE_CKPT=<path>   ANIMEPOSE_SEG_CKPT=<path>
"""

from __future__ import annotations

import torch

from .inference import (
    DEFAULT_BBOX_PCT, DEFAULT_BORDER, DEFAULT_CONF, DEFAULT_MASK_TOL,
)

_pipeline = None


def _get_pipeline():
    """Load the models once, on first execution rather than at import time, so
    a missing checkpoint can't stop ComfyUI from starting."""
    global _pipeline
    if _pipeline is None:
        from .inference import PosePipeline
        _pipeline = PosePipeline(
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    return _pipeline


class AnimePoseEstimator:
    """Pose estimation for illustrated characters.

    Useful where a photo-trained person detector refuses to fire, which on
    stylized art is often. The underlying model emits all 17 keypoints with no
    notion of visibility, which is why using it naively grows legs on a
    bust-shot; this node recovers a per-keypoint confidence from the model's
    own heatmaps and drops joints it has no evidence for, so the skeleton stops
    at what is actually in the picture.
    """

    CATEGORY = "ControlNet Preprocessors/Pose"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("pose_image",)
    FUNCTION = "estimate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "confidence": ("FLOAT", {
                    "default": DEFAULT_CONF, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Minimum heatmap confidence to draw a joint. "
                               "Raise it if the skeleton invents limbs that "
                               "aren't in the image; lower it (0 = upstream "
                               "behaviour, always draw all 17) if joints you "
                               "can see go missing.",
                }),
                "mask_gate": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Also drop joints that land off the character "
                               "silhouette, using the segmentation this node "
                               "already computes.",
                }),
                "mask_tolerance": ("FLOAT", {
                    "default": DEFAULT_MASK_TOL, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "How far off the silhouette a joint may sit, as "
                               "a fraction of the character's size.",
                }),
                "border_margin": ("FLOAT", {
                    "default": DEFAULT_BORDER, "min": 0.0, "max": 0.25, "step": 0.01,
                    "tooltip": "Drop joints pinned against the edge of the "
                               "model's crop. Useful for tightly framed input; "
                               "0 disables it.",
                }),
                "bbox_trim": ("FLOAT", {
                    "default": DEFAULT_BBOX_PCT, "min": 0.0, "max": 0.4, "step": 0.01,
                    "tooltip": "Ignore near-empty rows/columns when framing the "
                               "character, so ribbons, scarves, smoke or loose "
                               "hair don't inflate the crop and shrink the body. "
                               "Raise it for very wispy designs; 0 uses the raw "
                               "silhouette extents.",
                }),
            },
        }

    def estimate(self, image: torch.Tensor,
                 confidence: float = DEFAULT_CONF,
                 mask_gate: bool = True,
                 mask_tolerance: float = DEFAULT_MASK_TOL,
                 border_margin: float = DEFAULT_BORDER,
                 bbox_trim: float = DEFAULT_BBOX_PCT):
        """image: (B, H, W, 3) float [0,1] -> same shape, skeleton on black."""
        result = _get_pipeline().run_tensor(
            image,
            conf_threshold=confidence,
            use_mask_gate=mask_gate,
            mask_tolerance=mask_tolerance,
            border_margin=border_margin,
            bbox_percentile=bbox_trim,
        )
        return (result,)


NODE_CLASS_MAPPINGS = {
    "AnimePoseEstimator": AnimePoseEstimator,
    # Pre-rename id, so workflows saved before this existed still load.
    "BizarrePosePreprocessor": AnimePoseEstimator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimePoseEstimator": "Anime Pose Estimator",
    "BizarrePosePreprocessor": "Anime Pose Estimator (old id)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
