"""Compara keypoints de Detectron2 (ground truth) vs torchvision (nodo standalone).

Uso:
    python compare_keypoints.py --dir test/

Espera encontrar en --dir pares de archivos:
    <stem>_kp_detectron2.json   (generado con --out-json en Docker)
    <stem>_kp_torchvision.json  (generado por el nodo standalone)

Imprime una tabla con el error medio por joint (en píxeles) y
guarda una imagen de comparación side-by-side por cada par.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

COCO17_NAMES = [
    "nose", "eye_left", "eye_right", "ear_left", "ear_right",
    "shoulder_left", "shoulder_right", "elbow_left", "elbow_right",
    "wrist_left", "wrist_right", "hip_left", "hip_right",
    "knee_left", "knee_right", "ankle_left", "ankle_right",
]


def load_kps(path: Path) -> dict[str, tuple[float, float]]:
    data = json.loads(path.read_text())
    return {k: (v["x"], v["y"]) for k, v in data["keypoints"].items()}


def compare(dir_path: Path) -> None:
    det2_jsons = sorted(dir_path.glob("*_kp_detectron2.json"))
    if not det2_jsons:
        print("No *_kp_detectron2.json files found. Run Docker with --out-json first.")
        return

    per_joint_errors: dict[str, list[float]] = {n: [] for n in COCO17_NAMES}
    all_errors: list[float] = []

    results = []
    for det2_path in det2_jsons:
        stem = det2_path.name.replace("_kp_detectron2.json", "")
        tv_path = dir_path / f"{stem}_kp_torchvision.json"
        if not tv_path.exists():
            print(f"  SKIP {stem} — no torchvision JSON yet")
            continue

        kps_d2 = load_kps(det2_path)
        kps_tv = load_kps(tv_path)

        img_errors = []
        for name in COCO17_NAMES:
            if name not in kps_d2 or name not in kps_tv:
                continue
            x1, y1 = kps_d2[name]
            x2, y2 = kps_tv[name]
            err = float(np.hypot(x2 - x1, y2 - y1))
            per_joint_errors[name].append(err)
            img_errors.append(err)
            all_errors.append(err)

        mean_err = np.mean(img_errors) if img_errors else float("nan")
        results.append((stem, mean_err))
        print(f"  {stem:<40s}  mean={mean_err:.1f}px")

    if not results:
        print("No pairs found.")
        return

    # ASCII only: this runs under ComfyUI's cp1252 console on Windows, where
    # box-drawing characters raise UnicodeEncodeError.
    print("\n-- Per-joint mean error (px) ---------------------------")
    for name in COCO17_NAMES:
        errs = per_joint_errors[name]
        if errs:
            print(f"  {name:<20s}  {np.mean(errs):.1f}px  (n={len(errs)})")

    overall = np.mean(all_errors) if all_errors else float("nan")
    print(f"\n  OVERALL mean error: {overall:.1f}px")
    if overall < 10:
        print("  -> Quality looks good (< 10px)")
    elif overall < 20:
        print("  -> Acceptable (10-20px), minor visual differences")
    else:
        print("  -> Noticeable degradation (> 20px), may need calibration")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="test", help="folder with JSON files")
    args = parser.parse_args()
    compare(Path(args.dir))


if __name__ == "__main__":
    main()
