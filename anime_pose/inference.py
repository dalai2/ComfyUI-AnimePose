"""Standalone inference pipeline for ComfyUI-AnimePose.

Reproduces `_scripts/pose_estimator.py::infer_pose` from the original repo
without Docker, Detectron2 or PyTorch Lightning:

    segmenter -> character bbox -> exact upstream crop -> pose head
    -> per-keypoint heatmap confidence -> COCO-17 -> OpenPose-18

Two things it does that upstream does not:

  * It keeps the per-keypoint heatmap confidence instead of discarding it, so
    joints the model has no evidence for can be dropped rather than drawn at
    whatever pixel happened to win the argmax. That is the fix for skeletons
    sprouting legs on a bust-shot.
  * It gates keypoints against the character segmentation mask, which the
    pipeline already computes for free.

Coordinate conventions are a running hazard here. Upstream works in
(row, col); PIL, cv2 and this module's public surface work in (x, y).
Everything below the `_pose_on_crop` boundary is (row, col); everything above
it is (x, y).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TFF
from PIL import Image

from . import geometry as geo
from .models import load_pose_checkpoint, load_seg_checkpoint
from .pose_convert import COCO17_NAMES, render_pose_png_array
from .weights import resolve_pose_ckpt, resolve_seg_ckpt

# From the checkpoints' own hyper_parameters -- identical across every
# released pose checkpoint, so they are constants rather than inputs.
SEG_INPUT_SIZE = 256      # largs.bg_seg.size
POSE_CROP_SIZE = 256      # largs.danbooru_coco.size
POSE_CROP_PAD  = 0.1      # largs.danbooru_coco.padding
SEG_BBOX_THRESH = 0.5

NUM_COCO = 17

# Defaults calibrated on this repo's test set (8 illustrations) by comparing
# each keypoint's heatmap score against the Detectron2 reference position, and
# by re-running every image cropped to its top 45% so the leg joints are known
# to be absent:
#
#   score >= 0.50  -> median localisation error   0 px
#   score 0.25-0.50 -> median  13 px
#   score <  0.15  -> median  37 px, mean 178 px   <- the hallucinations
#
# 0.15 sits at that cliff: on full-body images it keeps ~84% of joints, on the
# bust crops it removes ~71% of the phantom legs. Raise it if skeletons still
# grow limbs that aren't there; lower it if joints you can see go missing.
DEFAULT_CONF = 0.15
DEFAULT_MASK_TOL = 0.04
# Off by default: the model's input crop already includes 10% padding around
# the character, so joints essentially never land in the outer band and this
# gate rejected nothing on the test set. Kept because it is the right tool for
# tightly-cropped inputs.
DEFAULT_BORDER = 0.0
# Trim rows/columns holding under this fraction of the busiest one before
# taking the character bbox. Guards against thin things attached to the
# character (ribbons, scarves, smoke, loose hair, a drawn weapon) inflating the
# box and shrinking the character inside the model's 256px crop. 0 = upstream
# behaviour (absolute extremes).
DEFAULT_BBOX_PCT = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_pil(image) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(image)
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    return image


def _alpha_bg_white(img: Image.Image) -> Image.Image:
    """Composite over white -> RGB. Matches upstream's `.alpha_bg(1)`."""
    if img.mode in ("RGBA", "LA") or "transparency" in img.info:
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        return bg
    return img.convert("RGB")


def _resize_min(img: Image.Image, s: int) -> Image.Image:
    """Resize so the shorter side is exactly `s` (up or down), keeping aspect.

    Upstream's `resize_min` always resizes; the previous version of this file
    skipped upscaling, which silently changed the segmenter's input scale.
    """
    w, h = img.size
    if w < h:
        new = (s, int(h * s / w))
    else:
        new = (int(w * s / h), s)
    return img.resize(new, Image.BILINEAR)


# ─────────────────────────────────────────────────────────────────────────────
# Stages
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _segment(model, pil_rgb: Image.Image, device: torch.device) -> np.ndarray:
    """Character probability map at the original resolution, (H, W) float."""
    small = _resize_min(pil_rgb, SEG_INPUT_SIZE)
    t = TFF.to_tensor(small).unsqueeze(0).to(device)
    out = model(t)
    prob = out["softmax"][0, 1].float().cpu()
    prob_pil = TFF.to_pil_image(prob).resize(pil_rgb.size, Image.BICUBIC)
    return np.asarray(prob_pil, dtype=np.float32) / 255.0


def _crop_for_pose(pil: Image.Image, cropbox) -> Image.Image:
    """Apply an upstream cropbox to an image (out-of-bounds pads with black)."""
    fc = geo.pixel_ij(cropbox[0], rounding=True)
    fs = geo.pixel_ij(cropbox[1], rounding=True)
    ts = geo.pixel_ij(cropbox[2], rounding=True)
    return TFF.resized_crop(
        pil.convert("RGB"),
        top=fc[0], left=fc[1], height=fs[0], width=fs[1],
        size=list(ts),
        interpolation=TFF.InterpolationMode.BILINEAR,
    )


@torch.no_grad()
def _pose_on_crop(model, crop: Image.Image, device: torch.device,
                  smoothing: float = 0.1):
    """Run the pose head. Returns (kps_rowcol, scores) for the COCO-17 subset,
    in crop pixel coordinates."""
    t = TFF.to_tensor(crop).unsqueeze(0).to(device)
    out = model(t, smoothing=smoothing)
    kps = out["keypoints"][0, :NUM_COCO].float().cpu().numpy()          # (17,2) row,col
    scores = out["keypoint_scores"][0, :NUM_COCO].float().cpu().numpy()  # (17,)
    return kps, scores


# ─────────────────────────────────────────────────────────────────────────────
# Presence gating
# ─────────────────────────────────────────────────────────────────────────────

def _mask_gate(kps_xy: np.ndarray, mask: np.ndarray, tolerance: float) -> np.ndarray:
    """True where the keypoint sits on (or near) the character silhouette.

    `tolerance` is a fraction of the character's bbox diagonal; the mask is
    dilated by that much before testing, so joints just outside the outline
    (a hand at the very edge of a sleeve, a foot cut off by antialiasing)
    still pass, while a leg invented over empty background does not.
    """
    H, W = mask.shape
    if tolerance > 0:
        rows = np.any(mask, axis=1).nonzero()[0]
        cols = np.any(mask, axis=0).nonzero()[0]
        if len(rows) == 0:
            return np.ones(len(kps_xy), dtype=bool)
        diag = float(np.hypot(rows[-1] - rows[0], cols[-1] - cols[0]))
        r = max(1, int(round(diag * tolerance)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        test = cv2.dilate(mask.astype(np.uint8), k).astype(bool)
    else:
        test = mask

    ok = np.zeros(len(kps_xy), dtype=bool)
    for i, (x, y) in enumerate(kps_xy):
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H:
            ok[i] = bool(test[yi, xi])
    return ok


def _border_gate(kps_rowcol: np.ndarray, crop_size: int, margin: float) -> np.ndarray:
    """False for keypoints pinned against the edge of the model's input crop.

    The crop is built to contain the whole character plus padding, so a real
    joint has no reason to land in the outermost band. Joints the model cannot
    place tend to get pushed there.
    """
    if margin <= 0:
        return np.ones(len(kps_rowcol), dtype=bool)
    m = crop_size * margin
    r, c = kps_rowcol[:, 0], kps_rowcol[:, 1]
    return ((r > m) & (r < crop_size - m) & (c > m) & (c < crop_size - m))


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class PosePipeline:
    """Loads both models once; call run() / run_tensor() per image."""

    def __init__(
        self,
        pose_ckpt: "str | Path | None" = None,
        seg_ckpt: "str | Path | None" = None,
        device: str = "cuda",
        allow_download: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        pose_ckpt = pose_ckpt or resolve_pose_ckpt(allow_download)
        seg_ckpt  = seg_ckpt  or resolve_seg_ckpt(allow_download)
        print(f"[anime-pose] device={self.device}")
        print(f"[anime-pose]   seg  ckpt: {seg_ckpt}")
        print(f"[anime-pose]   pose ckpt: {pose_ckpt}")
        self.seg_model  = load_seg_checkpoint(seg_ckpt, self.device)
        self.pose_model = load_pose_checkpoint(pose_ckpt, self.device)

    @torch.no_grad()
    def estimate(
        self,
        image,
        conf_threshold: float = DEFAULT_CONF,
        mask_tolerance: float = DEFAULT_MASK_TOL,
        use_mask_gate: bool = True,
        border_margin: float = DEFAULT_BORDER,
        bbox_percentile: float = DEFAULT_BBOX_PCT,
    ) -> dict:
        """Estimate one image. Returns a dict with (x, y) keypoints in original
        image coordinates, per-keypoint scores, and the rejection reasons.

        Keypoints rejected by any gate come back with score 0, which
        `pose_convert` renders as absent.
        """
        pil = _to_pil(image)
        W, H = pil.size

        # 1. segmentation -> character bbox, in (row, col)
        prob = _segment(self.seg_model, _alpha_bg_white(pil), self.device)
        mask = prob > SEG_BBOX_THRESH
        if bbox_percentile > 0:
            bbox = geo.percentile_bbox(mask, keep=bbox_percentile)
        else:
            bbox = geo.alpha_bbox(prob, thresh=SEG_BBOX_THRESH, allow_empty=True)

        # 2. upstream's exact framing, and the transform back out of it
        cb = geo.pose_cropbox(bbox, POSE_CROP_SIZE, POSE_CROP_PAD)
        icb = geo.cropbox_inverse((H, W), *cb)
        crop = _crop_for_pose(pil, cb)

        # 3. pose head (crop space, row/col)
        kps_crop, scores = _pose_on_crop(self.pose_model, crop, self.device)

        # 4. back to original image coordinates, then to (x, y)
        kps_full_rc = geo.cropbox_points(kps_crop, *icb)
        kps_xy = kps_full_rc[:, ::-1].copy()

        # 5. presence gating
        reasons = [None] * NUM_COCO
        keep = scores >= conf_threshold
        for i in np.nonzero(~keep)[0]:
            reasons[i] = "low_confidence"

        if border_margin > 0:
            ok = _border_gate(kps_crop, POSE_CROP_SIZE, border_margin)
            for i in np.nonzero(keep & ~ok)[0]:
                reasons[i] = "crop_border"
            keep &= ok

        if use_mask_gate:
            ok = _mask_gate(kps_xy, mask, mask_tolerance)
            for i in np.nonzero(keep & ~ok)[0]:
                reasons[i] = "off_character"
            keep &= ok

        final_scores = np.where(keep, scores, 0.0)

        return {
            "keypoints_xy": kps_xy,
            "scores": final_scores,
            "raw_scores": scores,
            "keep": keep,
            "reasons": reasons,
            "image_size": (H, W),
            "bbox": bbox,
            "mask": mask,
        }

    def run(
        self,
        image,
        conf_threshold: float = DEFAULT_CONF,
        mask_tolerance: float = DEFAULT_MASK_TOL,
        use_mask_gate: bool = True,
        border_margin: float = DEFAULT_BORDER,
        bbox_percentile: float = DEFAULT_BBOX_PCT,
        out_json: "Optional[str | Path]" = None,
    ) -> np.ndarray:
        """Full pipeline -> OpenPose-18 skeleton as (H, W, 3) uint8 RGB."""
        res = self.estimate(
            image,
            conf_threshold=conf_threshold,
            mask_tolerance=mask_tolerance,
            use_mask_gate=use_mask_gate,
            border_margin=border_margin,
            bbox_percentile=bbox_percentile,
        )

        if out_json is not None:
            payload = {
                "image_size": list(res["image_size"]),
                "keypoints": {
                    name: {
                        "x": round(float(res["keypoints_xy"][i, 0]), 2),
                        "y": round(float(res["keypoints_xy"][i, 1]), 2),
                        "score": round(float(res["raw_scores"][i]), 4),
                        "kept": bool(res["keep"][i]),
                        "reason": res["reasons"][i],
                    }
                    for i, name in enumerate(COCO17_NAMES)
                },
                "source": "torchvision",
            }
            Path(out_json).write_text(json.dumps(payload, indent=2))

        # A gated-out keypoint has score 0, which falls below any positive
        # threshold, so pose_convert drops it and every limb touching it.
        return render_pose_png_array(
            res["image_size"],
            res["keypoints_xy"],
            coco_scores=res["scores"],
            conf_threshold=1e-6,
        )

    def run_tensor(self, comfyui_tensor: torch.Tensor, **kw) -> torch.Tensor:
        """ComfyUI wrapper: (B, H, W, 3) float [0,1] -> same shape."""
        results = []
        for i in range(comfyui_tensor.shape[0]):
            frame = (comfyui_tensor[i].cpu().numpy() * 255).astype(np.uint8)
            results.append(torch.from_numpy(self.run(frame, **kw)).float() / 255.0)
        return torch.stack(results)
