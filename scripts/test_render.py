"""
Self-test for pose_convert.py that needs no Docker and no real model:
builds synthetic COCO-17 keypoints for a standing pose and a dynamic
(one-arm-raised, one-leg-bent) pose, converts + renders both, and prints
a sanity check of the OpenPose-18 output.

Run with ComfyUI's own venv (has numpy + opencv already):
    python scripts/test_render.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anime_pose.pose_convert import (  # noqa: E402
    COCO17_NAMES, OPENPOSE18_NAMES, coco17_to_openpose18, render_pose_png,
)

W, H = 512, 768


def standing_pose() -> np.ndarray:
    kp = {
        "nose": (256, 80),
        "eye_left": (246, 70), "eye_right": (266, 70),
        "ear_left": (236, 75), "ear_right": (276, 75),
        "shoulder_left": (210, 150), "shoulder_right": (302, 150),
        "elbow_left": (190, 260), "elbow_right": (322, 260),
        "wrist_left": (180, 360), "wrist_right": (332, 360),
        "hip_left": (225, 400), "hip_right": (287, 400),
        "knee_left": (220, 540), "knee_right": (292, 540),
        "ankle_left": (215, 680), "ankle_right": (297, 680),
    }
    return np.array([kp[name] for name in COCO17_NAMES], dtype=np.float64)


def dynamic_pose() -> np.ndarray:
    # right arm raised overhead, left leg kicked out to the side
    kp = {
        "nose": (256, 90),
        "eye_left": (246, 80), "eye_right": (266, 80),
        "ear_left": (236, 85), "ear_right": (276, 85),
        "shoulder_left": (210, 160), "shoulder_right": (300, 150),
        "elbow_left": (185, 270), "elbow_right": (340, 90),
        "wrist_left": (175, 370), "wrist_right": (350, 20),
        "hip_left": (225, 410), "hip_right": (287, 410),
        "knee_left": (120, 470), "knee_right": (292, 550),
        "ankle_left": (60, 430), "ankle_right": (297, 690),
    }
    return np.array([kp[name] for name in COCO17_NAMES], dtype=np.float64)


def run(name, coco_kps):
    kp18 = coco17_to_openpose18(coco_kps)
    print(f"--- {name} ---")
    for op_name, row in zip(OPENPOSE18_NAMES, kp18):
        print(f"  {op_name:16s} {row}")
    out_path = f"./_out_{name}.png"
    render_pose_png((H, W), coco_kps, out_path)
    print(f"  saved -> {out_path}\n")


if __name__ == "__main__":
    run("standing", standing_pose())
    run("dynamic", dynamic_pose())
