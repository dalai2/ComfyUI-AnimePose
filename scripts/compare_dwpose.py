"""Compare DWPose against this node on the same images.

There is no hand-annotated ground truth here, so this measures the part that is
objective -- coverage, i.e. who produces a skeleton at all and with how many
joints -- and writes a side-by-side sheet for judging quality by eye.

Usage:
    python scripts/compare_dwpose.py --out compare.png test/*.png

Requires comfyui_controlnet_aux installed with its DWPose checkpoints; point
--aux at it if it isn't in the default ComfyUI location.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_AUX = Path.home() / "Documents/ComfyUI/custom_nodes/comfyui_controlnet_aux"
CELL = 300


def run_dwpose(images: list[Path], aux: Path) -> tuple[dict, dict]:
    # the package expects its own src/ on sys.path, not the repo root
    sys.path.insert(0, str(aux / "src"))
    from custom_controlnet_aux.dwpose import DwposeDetector

    det = DwposeDetector.from_pretrained(
        "hr16/DWPose-TorchScript-BatchSize5", "yzd-v/DWPose",
        det_filename="yolox_l.onnx",
        pose_filename="dw-ll_ucoco_384_bs5.torchscript.pt",
        torchscript_device="cuda" if torch.cuda.is_available() else "cpu",
    )
    renders, stats = {}, {}
    for p in images:
        src = np.array(Image.open(p).convert("RGB"))
        img, js = det(src, include_body=True, include_hand=False,
                      include_face=False, image_and_json=True,
                      detect_resolution=512, output_type="np")
        people = js.get("people", []) if isinstance(js, dict) else []
        n_joints = 0
        if people:
            kp = np.array(people[0].get("pose_keypoints_2d", [])).reshape(-1, 3)
            n_joints = int((kp[:, 2] > 0).sum())
        renders[p.name] = cv2.resize(img, (src.shape[1], src.shape[0]))
        stats[p.name] = (len(people), n_joints)
        print(f"  DWPose  {p.name:<28} people={len(people)} joints={n_joints}")
    del det
    torch.cuda.empty_cache()
    return renders, stats


def run_anime_pose(images: list[Path]) -> tuple[dict, dict]:
    from anime_pose.inference import PosePipeline
    from anime_pose.pose_convert import render_pose_png_array

    pipe = PosePipeline()
    renders, stats = {}, {}
    for p in images:
        r = pipe.estimate(Image.open(p).convert("RGB"))
        renders[p.name] = render_pose_png_array(
            r["image_size"], r["keypoints_xy"],
            coco_scores=r["scores"], conf_threshold=1e-6)
        stats[p.name] = (int(r["keep"].sum()), float(r["raw_scores"].mean()))
        print(f"  anime   {p.name:<28} joints={stats[p.name][0]}/17 "
              f"mean_score={stats[p.name][1]:.3f}")
    return renders, stats


def fit(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = CELL / max(h, w)
    r = cv2.resize(img, (int(w * s), int(h * s)))
    pad = np.zeros((CELL, CELL, 3), np.uint8)
    pad[:r.shape[0], :r.shape[1]] = r
    return pad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("compare_dwpose.png"))
    ap.add_argument("--aux", type=Path, default=DEFAULT_AUX,
                    help="path to comfyui_controlnet_aux")
    args = ap.parse_args()

    images = [p for p in args.images if p.exists()]
    if not images:
        print("no input images found")
        return 1
    if not args.aux.exists():
        print(f"comfyui_controlnet_aux not found at {args.aux}; pass --aux")
        return 1

    dw_r, dw_s = run_dwpose(images, args.aux)
    bp_r, bp_s = run_anime_pose(images)

    print(f"\n{'image':<30}{'DWPose':>18}{'anime-pose':>18}")
    dw_fail = bp_fail = 0
    for p in images:
        n_people, n_joints = dw_s[p.name]
        kept, _ = bp_s[p.name]
        dw_fail += n_people == 0
        bp_fail += kept == 0
        left = f"{n_joints} joints" if n_people else "NO DETECTION"
        print(f"{p.name:<30}{left:>18}{str(kept) + ' joints':>18}")
    print(f"\nno skeleton at all:  DWPose {dw_fail}/{len(images)}   "
          f"anime-pose {bp_fail}/{len(images)}")

    rows = []
    for p in images:
        src = np.array(Image.open(p).convert("RGB"))
        rows.append(np.hstack([
            fit(src),
            fit(cv2.addWeighted(src, 0.4, dw_r[p.name], 1.0, 0)),
            fit(cv2.addWeighted(src, 0.4, bp_r[p.name], 1.0, 0)),
        ]))
    sheet = np.vstack(rows)
    hdr = np.zeros((32, sheet.shape[1], 3), np.uint8)
    for x, t in [(90, "source"), (350, "DWPose"), (640, "anime-pose")]:
        cv2.putText(hdr, t, (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imwrite(str(args.out), cv2.cvtColor(np.vstack([hdr, sheet]), cv2.COLOR_RGB2BGR))
    print("->", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
