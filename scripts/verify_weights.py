"""Verify the published weights against the paper's official checkpoints.

This node ships repackaged weights, so "trust me" isn't good enough. This
checks tensor-by-tensor that every weight actually shipped is bit-identical to
the corresponding tensor in the official release, and reports exactly what was
left out (the Detectron2 `rcnn.*` branch, which the node never loads).

Usage:
    python scripts/verify_weights.py --estimator ../bizarre-pose-estimator

`--estimator` is a checkout with the official checkpoints extracted into
_train/*/runs/, from the paper's Google Drive folder. Use --published to point
at the files instead of wherever the node keeps them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anime_pose import weights as W  # noqa: E402
from anime_pose.weights import POSE_PUBLISH_NAME, SEG_PUBLISH_NAME  # noqa: E402

OFFICIAL = {
    POSE_PUBLISH_NAME: (
        "_train/character_pose_estim/runs/feat_concat+data.ckpt",
        ("rcnn.",),   # intentionally not shipped
    ),
    SEG_PUBLISH_NAME: (
        "_train/character_bg_seg/runs/eyeless_alaska_vulcan0000/checkpoints/"
        "epoch=0096-val_f1=0.9508-val_loss=0.0483.ckpt",
        (),
    ),
}


def compare(official: Path, published: Path, expected_dropped: tuple) -> bool:
    a = W.load_state_dict(official)
    b = W.load_state_dict(published)

    print(f"\n=== {published.name}")
    print(f"  official: {official}")
    print(f"  tensors: official={len(a)}  published={len(b)}")

    missing = sorted(set(a) - set(b))
    extra = sorted(set(b) - set(a))

    intentional = [k for k in missing if k.startswith(expected_dropped)] \
        if expected_dropped else []
    unexplained = [k for k in missing if k not in set(intentional)]

    if intentional:
        print(f"  intentionally dropped: {len(intentional)} tensors "
              f"({', '.join(expected_dropped)})")
    if unexplained:
        print(f"  UNEXPECTEDLY MISSING ({len(unexplained)}): {unexplained[:5]}")
    if extra:
        print(f"  NOT IN OFFICIAL ({len(extra)}): {extra[:5]}")

    same, differing, max_delta = 0, [], 0.0
    for k in sorted(set(a) & set(b)):
        ta, tb = a[k].cpu(), b[k].cpu()
        if ta.shape == tb.shape and torch.equal(ta, tb):
            same += 1
        else:
            differing.append(k)
            if ta.shape == tb.shape:
                max_delta = max(max_delta, (ta.float() - tb.float()).abs().max().item())

    total = same + len(differing)
    print(f"  bit-identical: {same}/{total}   differing: {len(differing)}"
          f"   max delta: {max_delta}")
    if differing:
        print(f"  first differing: {differing[:5]}")

    ok = not (unexplained or extra or differing)
    print("  ->", "OK" if ok else "MISMATCH")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", required=True,
                    help="bizarre-pose-estimator checkout with the official "
                         ".ckpt files extracted")
    ap.add_argument("--published", type=Path, default=None,
                    help="folder holding the published safetensors "
                         "(default: wherever the node keeps them)")
    args = ap.parse_args()

    est = Path(args.estimator)
    pub_dir = args.published or W.models_dir()
    all_ok = True

    for filename, (rel, dropped) in OFFICIAL.items():
        official, published = est / rel, pub_dir / filename
        if not official.exists():
            print(f"\nSKIP {filename}: no official checkpoint at {official}")
            all_ok = False
            continue
        if not published.exists():
            print(f"\nSKIP {filename}: not present at {published}")
            all_ok = False
            continue
        all_ok &= compare(official, published, dropped)

    print("\nALL VERIFIED" if all_ok else "\nINCOMPLETE OR MISMATCHED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
