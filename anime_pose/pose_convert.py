"""
COCO-17 -> OpenPose-18 keypoint conversion and skeleton rendering.

Pure numpy + opencv, no torch/docker dependency. This is meant to run in
ComfyUI's own venv (or anywhere) and produce a PNG that is a drop-in
replacement for what comfyui_controlnet_aux's DWPreprocessor node outputs,
so it can feed a ControlNetApply(OpenPose) node directly.

Source of truth for the two schemas:
  - COCO-17 order: ShuhongChen/bizarre-pose-estimator, _util/keypoints_v0.py
    (`ukey.coco_keypoints`), also the order `_scripts/pose_estimator.py`
    returns in `ans['keypoints']`.
  - OpenPose-18 order + rendering style: this machine's own
    custom_nodes/comfyui_controlnet_aux/src/custom_controlnet_aux/dwpose/
    {body.py, util.py} (draw_bodypose, limbSeq, colors, stick width).
"""

from __future__ import annotations

import numpy as np
import cv2
import math

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

COCO17_NAMES = [
    "nose", "eye_left", "eye_right", "ear_left", "ear_right",
    "shoulder_left", "shoulder_right", "elbow_left", "elbow_right",
    "wrist_left", "wrist_right", "hip_left", "hip_right",
    "knee_left", "knee_right", "ankle_left", "ankle_right",
]

OPENPOSE18_NAMES = [
    "nose", "neck", "shoulder_right", "elbow_right", "wrist_right",
    "shoulder_left", "elbow_left", "wrist_left", "hip_right", "knee_right",
    "ankle_right", "hip_left", "knee_left", "ankle_left",
    "eye_left", "eye_right", "ear_left", "ear_right",
]

# OpenPose-18 index -> COCO-17 index (None = synthesized, not a direct copy)
_OP18_FROM_COCO17 = [
    0,     # nose
    None,  # neck (midpoint of shoulders)
    6,     # shoulder_right
    8,     # elbow_right
    10,    # wrist_right
    5,     # shoulder_left
    7,     # elbow_left
    9,     # wrist_left
    12,    # hip_right
    14,    # knee_right
    16,    # ankle_right
    11,    # hip_left
    13,    # knee_left
    15,    # ankle_left
    1,     # eye_left
    2,     # eye_right
    3,     # ear_left
    4,     # ear_right
]

assert len(_OP18_FROM_COCO17) == len(OPENPOSE18_NAMES) == 18
assert len(COCO17_NAMES) == 17


def coco17_to_openpose18(coco_kps: np.ndarray, coco_scores: np.ndarray | None = None,
                          conf_threshold: float = 0.05) -> np.ndarray:
    """Convert COCO-17 keypoints to OpenPose-18 order.

    Args:
        coco_kps: (17, 2) array of (x, y) in COCO-17 order (see COCO17_NAMES).
        coco_scores: optional (17,) confidence array. bizarre-pose-estimator's
            own CLI emits all 17 points unconditionally with no score, which is
            what makes its skeletons sprout limbs that aren't in the picture;
            `inference.py` recovers a real per-point score from the model's
            heatmaps and passes it here. Defaults to "all present" so callers
            that genuinely have no scores behave like upstream.
        conf_threshold: points with score below this become NaN (missing).

    Returns:
        (18, 3) array of (x, y, score) in OpenPose-18 order. Missing points
        are NaN in every column.
    """
    coco_kps = np.asarray(coco_kps, dtype=np.float64)
    assert coco_kps.shape == (17, 2), f"expected (17,2), got {coco_kps.shape}"

    if coco_scores is None:
        coco_scores = np.ones(17, dtype=np.float64)
    else:
        coco_scores = np.asarray(coco_scores, dtype=np.float64)
        assert coco_scores.shape == (17,)

    out = np.full((18, 3), np.nan, dtype=np.float64)

    for op_idx, coco_idx in enumerate(_OP18_FROM_COCO17):
        if coco_idx is None:
            continue
        if coco_scores[coco_idx] < conf_threshold:
            continue
        out[op_idx, 0] = coco_kps[coco_idx, 0]
        out[op_idx, 1] = coco_kps[coco_idx, 1]
        out[op_idx, 2] = coco_scores[coco_idx]

    # neck = midpoint of shoulders, only if both present
    l_idx, r_idx = COCO17_NAMES.index("shoulder_left"), COCO17_NAMES.index("shoulder_right")
    if coco_scores[l_idx] >= conf_threshold and coco_scores[r_idx] >= conf_threshold:
        neck_xy = (coco_kps[l_idx] + coco_kps[r_idx]) / 2.0
        neck_score = min(coco_scores[l_idx], coco_scores[r_idx])
        out[OPENPOSE18_NAMES.index("neck")] = [neck_xy[0], neck_xy[1], neck_score]

    return out


# ---------------------------------------------------------------------------
# Rendering (mirrors comfyui_controlnet_aux dwpose/util.py: draw_bodypose)
# ---------------------------------------------------------------------------

# 1-indexed pairs into the 18-point OpenPose body, identical to the ones used
# by draw_bodypose() in custom_controlnet_aux so the visual style matches.
_LIMB_SEQ = [
    [2, 3], [2, 6], [3, 4], [4, 5],
    [6, 7], [7, 8], [2, 9], [9, 10],
    [10, 11], [2, 12], [12, 13], [13, 14],
    [2, 1], [1, 15], [15, 17], [1, 16],
    [16, 18],
]

_COLORS = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85],
]

_STICKWIDTH = 4


def draw_openpose18(canvas: np.ndarray, keypoints18: np.ndarray) -> np.ndarray:
    """Draw an 18-point OpenPose body skeleton onto `canvas` in place.

    keypoints18: (18, 3) array of (x, y, score) in pixel coordinates
        (same convention as coco17_to_openpose18's output). NaN rows are
        treated as missing.
    """
    keypoints18 = np.asarray(keypoints18, dtype=np.float64)
    assert keypoints18.shape == (18, 3)

    def present(i):
        return not np.isnan(keypoints18[i, 0]) and not np.isnan(keypoints18[i, 1])

    for (k1, k2), color in zip(_LIMB_SEQ, _COLORS):
        i1, i2 = k1 - 1, k2 - 1
        if not present(i1) or not present(i2):
            continue
        x1, y1 = keypoints18[i1, 0], keypoints18[i1, 1]
        x2, y2 = keypoints18[i2, 0], keypoints18[i2, 1]
        mX, mY = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        length = math.hypot(x2 - x1, y2 - y1)
        angle = math.degrees(math.atan2(y1 - y2, x1 - x2))
        polygon = cv2.ellipse2Poly(
            (int(mX), int(mY)), (int(length / 2), _STICKWIDTH), int(angle), 0, 360, 1
        )
        cv2.fillConvexPoly(canvas, polygon, [int(c * 0.6) for c in color])

    for i, color in enumerate(_COLORS):
        if not present(i):
            continue
        x, y = int(keypoints18[i, 0]), int(keypoints18[i, 1])
        cv2.circle(canvas, (x, y), 4, color, thickness=-1)

    return canvas


def render_pose_png(image_hw: tuple[int, int], coco_kps: np.ndarray, output_path: str,
                     coco_scores: np.ndarray | None = None, conf_threshold: float = 0.05,
                     background: np.ndarray | None = None) -> str:
    """High-level entry point: COCO-17 keypoints -> OpenPose-18 skeleton PNG.

    Args:
        image_hw: (height, width) of the source image the keypoints were
            estimated on (bizarre-pose-estimator's `infer_pose` already
            transforms keypoints back into this coordinate space).
        coco_kps: (17, 2) array, COCO-17 order.
        output_path: where to write the PNG.
        background: optional (H, W, 3) BGR image to draw the skeleton on
            top of (useful for debugging alignment). Defaults to solid black,
            matching what ControlNetApply(OpenPose) expects.

    Returns:
        output_path
    """
    h, w = image_hw
    if background is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        canvas = background.copy()
        assert canvas.shape[:2] == (h, w)

    kp18 = coco17_to_openpose18(coco_kps, coco_scores, conf_threshold)
    draw_openpose18(canvas, kp18)

    cv2.imwrite(output_path, canvas)
    return output_path


def render_pose_png_array(
    image_hw: tuple[int, int],
    coco_kps: np.ndarray,
    coco_scores: np.ndarray | None = None,
    conf_threshold: float = 0.05,
    background: np.ndarray | None = None,
) -> np.ndarray:
    """Same as render_pose_png but returns a (H, W, 3) uint8 RGB numpy array
    instead of writing to disk.  Used by the ComfyUI node.
    """
    h, w = image_hw
    if background is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        canvas = background.copy()

    kp18 = coco17_to_openpose18(coco_kps, coco_scores, conf_threshold)
    draw_openpose18(canvas, kp18)

    # cv2 uses BGR internally; convert to RGB for ComfyUI
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
