"""
Drop-in replacement for `python3 -m _scripts.pose_estimator <img> <ckpt>` that
runs INSIDE the bizarre-pose-estimator docker container.

Copy this file (and pose_convert.py) into the repo root before running, e.g.
from the repo root on the host:

    cp /path/to/ComfyUI-AnimePose/scripts/run_pose_estimator.py .
    cp /path/to/ComfyUI-AnimePose/anime_pose/pose_convert.py .

Then, inside the container shell (make/shell_docker):

    python3 run_pose_estimator.py ./_samples/megumin.png \
        ./_train/character_pose_estim/runs/feat_concat+data.ckpt \
        --out ./_samples/megumin_openpose.png

Mirrors the model-loading logic of `_scripts/pose_estimator.py` (segmenter
checkpoint path is fixed by the repo itself; the pose-estimator model class
is chosen from the checkpoint filename: 'feat_concat' -> passup.Model,
'feat_match' -> fermat.Model), but returns/saves an OpenPose-18 skeleton PNG
instead of the repo's own COCO-style debug visualization.
"""

import argparse
import sys

from _util.util_v1 import *  # noqa: F401,F403  (repo convention: brings in np, torch, etc.)
import _util.util_v1 as uutil  # noqa: F401
import _util.pytorch_v1 as utorch  # noqa: F401
from _util.twodee_v0 import *  # noqa: F401,F403  (brings in I, the image wrapper class)
import _util.twodee_v0 as u2d  # noqa: F401
import _util.keypoints_v0 as ukey  # noqa: F401

from pose_convert import render_pose_png


SEGMENTER_CKPT = (
    "./_train/character_bg_seg/runs/eyeless_alaska_vulcan0000/checkpoints/"
    "epoch=0096-val_f1=0.9508-val_loss=0.0483.ckpt"
)


def load_models(fn_model: str):
    from _train.character_bg_seg.models.alaska import Model as CharacterBGSegmenter
    model_segmenter = CharacterBGSegmenter.load_from_checkpoint(SEGMENTER_CKPT)

    if "feat_concat" in fn_model:
        from _train.character_pose_estim.models.passup import Model as CharacterPoseEstimator
    elif "feat_match" in fn_model:
        from _train.character_pose_estim.models.fermat import Model as CharacterPoseEstimator
    else:
        raise ValueError("checkpoint filename must contain 'feat_concat' or 'feat_match'")
    model_pose = CharacterPoseEstimator.load_from_checkpoint(fn_model, strict=False)

    return model_pose, model_segmenter


def abbox(img, thresh=0.5, allow_empty=False):
    # copied verbatim from the repo's own _scripts/pose_estimator.py
    img = I(img).np()  # noqa: F821
    assert len(img) in [1, 4], "image must be mode L or RGBA"
    a = img[-1] > thresh
    xlim = np.any(a, axis=1).nonzero()[0]  # noqa: F821
    ylim = np.any(a, axis=0).nonzero()[0]  # noqa: F821
    if len(xlim) == 0 and allow_empty:
        xlim = np.asarray([0, a.shape[0]])  # noqa: F821
    if len(ylim) == 0 and allow_empty:
        ylim = np.asarray([0, a.shape[1]])  # noqa: F821
    axmin, axmax = max(int(xlim.min()) - 1, 0), min(int(xlim.max()) + 1, a.shape[0])
    aymin, aymax = max(int(ylim.min()) - 1, 0), min(int(ylim.max()) + 1, a.shape[1])
    return [(axmin, aymin), (axmax - axmin, aymax - aymin)]


def infer_segmentation(self, images, bbox_thresh=0.5, return_more=True):
    # copied verbatim from the repo's own _scripts/pose_estimator.py
    # (it's a free function taking the model as `self`, not a bound method)
    anss = []
    _size = self.hparams.largs.bg_seg.size
    self.eval()
    for img in images:
        oimg = img
        img = I(img).resize_min(_size).convert("RGBA").alpha_bg(1).convert("RGB").pil()  # noqa: F821
        timg = TF.to_tensor(img)[None].to(self.device)  # noqa: F821
        with torch.no_grad():  # noqa: F821
            out = self(timg)
        ans = TF.to_pil_image(out["softmax"][0, 1].float().cpu()).resize(oimg.size[::-1])  # noqa: F821
        ans = {"segmentation": I(ans)}  # noqa: F821
        ans["bbox"] = abbox(ans["segmentation"], thresh=bbox_thresh, allow_empty=True)
        anss.append(ans)
    return anss


def infer_pose(self, segmenter, images, smoothing=0.1, pad_factor=1):
    # copied verbatim from the repo's own _scripts/pose_estimator.py
    # (it's a free function taking the model as `self`, not a bound method)
    self.eval()
    try:
        largs = self.hparams.largs.adds_keypoints
    except Exception:
        largs = self.hparams.largs.danbooru_coco
    _s = largs.size
    _p = _s * largs.padding
    anss = []
    segs = infer_segmentation(segmenter, images)
    for img, seg in zip(images, segs):
        oimg = img
        ans = {"segmentation_output": seg}
        bbox = seg["bbox"]
        cb = u2d.cropbox_sequence(
            [
                [bbox[0], bbox[1], bbox[1]],
                resize_square_dry(bbox[1], _s),  # noqa: F821
                [-_p * pad_factor / 2, _s + _p * pad_factor, _s],
            ]
        )
        icb = u2d.cropbox_inverse(oimg.size, *cb)
        img = u2d.cropbox(img, *cb)
        img = img.convert("RGBA").alpha(0).convert("RGB")
        ans["bbox"] = bbox
        ans["cropbox"] = cb
        ans["cropbox_inverse"] = icb
        ans["input_image"] = img

        timg = img.tensor()[None].to(self.device)
        with torch.no_grad():  # noqa: F821
            out = self(timg, smoothing=smoothing, return_more=True)
        ans["out"] = out

        kps = out["keypoints"][0].cpu().numpy()
        kps = u2d.cropbox_points(kps, *icb)
        ans["keypoints"] = kps

        anss.append(ans)
    return anss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fn_img")
    parser.add_argument("fn_model")
    parser.add_argument("--out", default=None, help="output PNG path (default: <img>_openpose.png)")
    parser.add_argument("--conf", type=float, default=0.05, help="min confidence to keep a keypoint")
    args = parser.parse_args()

    out_path = args.out or (args.fn_img.rsplit(".", 1)[0] + "_openpose.png")

    img = I(args.fn_img)  # noqa: F821  (I comes from _util.twodee_v0 via star-import)
    model_pose, model_segmenter = load_models(args.fn_model)

    # infer_pose/infer_segmentation are free functions in the repo's own script,
    # not bound methods on the Model class — called as infer_pose(model, segmenter, images)
    ans = infer_pose(model_pose, model_segmenter, [img])

    bbox = ans[0]["bbox"]
    # this checkpoint (trained with the extended ADDS keypoint set) returns
    # more than 17 points, but the first 17 rows are the base COCO-17 set in
    # ukey.coco_keypoints order (coco_keypoints_ext just appends extras after them)
    coco_kps = ans[0]["keypoints"][: len(ukey.coco_keypoints)]  # (17, 2), (row, col) order

    # this repo's own convention (confirmed in _train/character_pose_estim/models/passup.py's
    # forward(), where the heatmap argmax is unpacked as `kps // W, kps % W`) is (row, col) i.e.
    # (y, x) -- NOT the (x, y) pixel convention pose_convert.py/cv2 expect. Same for I.size,
    # which this repo's own _util/twodee_v0.py returns as (h, w), not PIL's (w, h). Swap both.
    coco_kps_xy = coco_kps[:, ::-1]

    print(f"bounding box\n\ttop-left: {bbox[0]}\n\tsize: {bbox[1]}")
    for name, (y, x) in zip(ukey.coco_keypoints, coco_kps):
        print(f"\t({x:.2f}, {y:.2f}) {name}")

    h, w = img.size  # this repo's I.size is (height, width), not PIL's (width, height)
    render_pose_png((h, w), coco_kps_xy, out_path, conf_threshold=args.conf)
    print(f"OpenPose skeleton saved to {out_path}")


if __name__ == "__main__":
    sys.exit(main())
