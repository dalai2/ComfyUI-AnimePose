"""Build the trimmed weight files to publish.

The released pose checkpoint is 351 MB, but 313 MB of that is the Detectron2
R101 keypoint branch, which this node never loads -- it reproduces that branch
with torchvision instead. Stripping it leaves 37.5 MB of weights that are
actually used.

    python scripts/make_weights.py --out dist_weights

Then upload the contents of that folder to the HuggingFace repo named in
node/weights.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anime_pose import weights as W  # noqa: E402
from anime_pose.weights import POSE_PUBLISH_NAME, SEG_PUBLISH_NAME  # noqa: E402

# Everything the standalone model actually consumes. `rcnn.*` is deliberately
# absent: TorchvisionKeypointDetector brings its own pretrained weights.
POSE_KEEP_PREFIXES = ("resnet.", "keypoint_head.")


def mb(sd: dict) -> float:
    return sum(t.numel() * t.element_size() for t in sd.values()) / 1e6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", type=Path, default=None,
                    help="source pose checkpoint (.ckpt or .safetensors)")
    ap.add_argument("--seg", type=Path, default=None,
                    help="source segmenter checkpoint")
    ap.add_argument("--out", type=Path, default=Path("dist_weights"))
    args = ap.parse_args()

    pose_src = args.pose or W.resolve_pose_ckpt(allow_download=False)
    seg_src = args.seg or W.resolve_seg_ckpt(allow_download=False)
    args.out.mkdir(parents=True, exist_ok=True)

    pose = W.load_state_dict(pose_src)
    trimmed = {k: v.contiguous() for k, v in pose.items()
               if k.startswith(POSE_KEEP_PREFIXES)}
    dropped = len(pose) - len(trimmed)
    print(f"pose: {pose_src}")
    print(f"  {len(pose)} tensors, {mb(pose):.1f} MB")
    print(f"  keeping {len(trimmed)} ({mb(trimmed):.1f} MB), "
          f"dropping {dropped} rcnn.* ({mb(pose) - mb(trimmed):.1f} MB)")
    save_file(trimmed, str(args.out / POSE_PUBLISH_NAME),
              metadata={"source": "ShuhongChen/bizarre-pose-estimator "
                                  "feat_concat+data.ckpt",
                        "license": "AGPL-3.0",
                        "note": "rcnn.* (Detectron2 R101) stripped; unused"})

    seg = {k: v.contiguous() for k, v in W.load_state_dict(seg_src).items()}
    print(f"seg:  {seg_src}")
    print(f"  {len(seg)} tensors, {mb(seg):.1f} MB (all used, nothing stripped)")
    save_file(seg, str(args.out / SEG_PUBLISH_NAME),
              metadata={"source": "ShuhongChen/bizarre-pose-estimator "
                                  "character_bg_seg epoch=0096",
                        "license": "AGPL-3.0"})

    total = sum(f.stat().st_size for f in args.out.iterdir()) / 1e6
    print(f"\nwrote {args.out}/  ({total:.1f} MB total)")
    for f in sorted(args.out.iterdir()):
        print(f"  {f.name:<40} {f.stat().st_size/1e6:7.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
