# ComfyUI-AnimePose

Pose estimation for **anime and illustrated characters**, as a ControlNet
OpenPose preprocessor for ComfyUI.

Most pose preprocessors are built for photographs. They start with a person
detector trained on photos, and on stylized illustration that detector often
just doesn't fire — you get an empty image and nothing to work with. This node
uses a model trained on illustrations instead
([bizarre-pose-estimator](https://github.com/ShuhongChen/bizarre-pose-estimator),
Chen & Zwicker, WACV 2022), and outputs an OpenPose-18 skeleton that drops
straight into an existing `ControlNetApply (OpenPose)` graph.

No Docker, no Detectron2, weights download themselves on first use.

> **This is packaging around someone else's research.** The model, the weights
> and the science are Shuhong Chen and Matthias Zwicker's. See [NOTICE](NOTICE).

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/dalai2/ComfyUI-AnimePose
```

Restart ComfyUI. The node is **Anime Pose Estimator**, under
`ControlNet Preprocessors/Pose`.

On first execution it downloads ~271 MB into `ComfyUI/models/anime_pose/`.
Nothing else to install — `torch`, `torchvision`, `kornia`, `opencv-python`
and `safetensors` all ship with ComfyUI.

## Inputs

| input | default | what it does |
|---|---|---|
| `confidence` | 0.15 | Minimum heatmap confidence to draw a joint. Raise it if the skeleton invents limbs; lower it if visible joints go missing. 0 disables the gate. |
| `mask_gate` | on | Also drop joints that land off the character silhouette. |
| `mask_tolerance` | 0.04 | How far off the silhouette a joint may still sit, as a fraction of character size. |
| `border_margin` | 0.0 | Drop joints pinned to the edge of the model's crop. For tightly framed input. |
| `bbox_trim` | 0.05 | Ignore near-empty rows/columns when framing the character. |

If something looks wrong, `confidence` is almost always the knob you want.

## What this adds over running the model directly

### Keypoints can be absent

The underlying model reads out keypoints with a per-channel `argmax`. `argmax`
always returns a pixel, so all 17 joints come back with a position no matter
how weak the evidence — feed it a bust-shot and you get a full skeleton with
invented hips, knees and ankles.

The evidence is available, it was just being discarded. This node keeps the
**unblurred sigmoid heatmap peak** per channel as a confidence. (Localisation
still uses the blurred map, as upstream does; that blur is huge — sigma =
0.1 × 128px — and flattens exactly the peak values that carry the presence
signal, so confidence has to be read before it.)

Measured on 8 test illustrations, scoring each keypoint against the reference
Detectron2 pipeline:

| heatmap score | median localisation error |
|---|---|
| >= 0.50 | 0 px |
| 0.25 – 0.50 | 13 px |
| < 0.15 | 37 px (mean 178 px) |

`corr(score, log1p(error)) = -0.64`. Re-running every image cropped to its top
45%, so the legs are known to be absent, the leg joints' mean score falls from
0.305 to 0.140 while upper-body joints stay at 0.384. The default threshold of
0.15 sits at that cliff: on full-body images it keeps ~84% of joints, on the
bust crops it removes ~71% of the phantom legs.

**Is the score meaningful?** More than it looks. The head is trained with
`BCEWithLogitsLoss` against gaussian-blob targets, so a channel is pushed
toward zero everywhere it has no keypoint — suppression is trained behaviour,
not an accident:

| input | mean score |
|---|---|
| flat grey / random noise (no character) | 0.002 |
| real background crop, no character | 0.040 |
| full-body illustration, leg joints | 0.305 |
| same images cropped to top 45%, leg joints | 0.140 |

Given nothing to find, the model outputs essentially zero. That is what makes
thresholding work.

The caveat is where suppression is only *partial*: when a character **is**
present but part of the body is out of frame, the learned body prior still
fires — a visible torso implies hips somewhere. That is why bust-crop legs land
at 0.140 rather than 0.002, and why ~27% of them survive the default
threshold. A real improvement, not a guarantee.

### Robust character framing

The pose model is fed a crop built from the character's segmentation bounding
box. Taking the absolute extremes of that mask means anything thin attached to
the character — ribbons, scarves, smoke, loose hair, a drawn weapon — drags the
box out to the frame edge, shrinking the character inside the model's 256px
crop. It also makes the box hypersensitive to single stray mask pixels.

`bbox_trim` discards rows and columns holding under 5% of the busiest one.
Measured against the reference pipeline this cut mean error from **43.3 px to
30.4 px**, with the worst outliers improving most (`wrist_right` 297 → 129 px,
`ankle_left` 137 → 61 px).

## Where it fits alongside other preprocessors

Honest version, measured on 10 images (8 anime illustrations, 2 painted splash
arts), body keypoints only, against DWPose:

|  | DWPose | this node |
|---|---|---|
| produced no skeleton at all | 7 / 10 | 0 / 10 |

That gap is the reason this node exists. DWPose is gated by a YOLOX person
detector that frequently refuses to fire on stylized art, and then emits an
empty image — no partial result, nothing to fall back on.

But coverage is not quality, and on quality it does **not** win:

- **Where DWPose fires, its skeleton is at least as good as this one**, usually
  cleaner and more complete. It is the better tool when it works.
- **Where this one fires, quality varies by body region.** Head, neck,
  shoulders and raised arms are reliably right. Hips, knees and ankles fail
  often, especially in dark, occluded or heavily-costumed regions.

So treat them as complementary: photographic-style estimator first, this one
when it returns nothing. The confidence gate is what makes that fallback safe —
a partial skeleton of joints the model is sure about conditions ControlNet
fine, whereas a complete but wrong one fights the generation.

Reproduce with `scripts/compare_dwpose.py` (needs `comfyui_controlnet_aux`).

## Known limitations

- **Single full-body character.** The upstream authors are explicit that the
  model targets single full-body characters; the segmenter produces one
  bounding box, so multiple characters get one skeleton spanning all of them.
- **Out-of-domain art degrades badly.** The model is trained on danbooru-style
  illustration. On painted, semi-realistic art (game splash art, for instance)
  the upper body usually survives but the legs are frequently wrong.
- **The confidence gate has no anatomical model.** It is per-joint, so it can
  drop a mid-limb joint and leave the ones beyond it floating.
- `feat_match` checkpoints are a different architecture (`fermat.Model`) and
  are not supported; only `feat_concat` is.

## Weights

Published weights are a repackaging of the paper's release: same tensors, in
safetensors, with the Detectron2 `rcnn.*` branch stripped from the pose
checkpoint because this node never loads it — torchvision's KeypointRCNN
stands in for that branch. That takes the pose file from 351 MB to 37.5 MB.

| file | size |
|---|---|
| `anime_pose_head.safetensors` | 37.5 MB |
| `character_bg_seg.safetensors` | 233 MB |

Don't take that on faith — check it:

```bash
python scripts/verify_weights.py --estimator ../bizarre-pose-estimator
```

That compares every shipped tensor against the official Google Drive
checkpoints and reports what was intentionally left out. Expected result:
336/336 and 690/690 bit-identical, 573 `rcnn.*` tensors intentionally dropped.

You can also bring your own: put the original `.ckpt` files in
`ComfyUI/models/anime_pose/`, or set `ANIMEPOSE_POSE_CKPT` /
`ANIMEPOSE_SEG_CKPT`. Both formats load, and an existing local file always
wins over downloading.

## How it runs without Detectron2

The original pipeline needs Detectron2, which is painful to build on Windows.
`anime_pose/models.py` reimplements the checkpoints' `nn.Module`s directly and
replaces the Detectron2 branch with torchvision. Getting that faithful took
four fixes, each worth knowing if you touch this code:

1. **The RCNN branch produces a dense tensor, not point detections.** Upstream's
   `PretrainedKeypointDetector` doesn't run detection at all: it forces one ROI
   over the whole image and reads `pred_keypoint_heatmaps`, documented by
   detectron2 as the *raw keypoint logits* — a dense `(N, 17, 56, 56)` map.
   Reproduced via torchvision's
   `keypoint_roi_pool → keypoint_head → keypoint_predictor` with a full-image
   box. The backbone differs (R50-FPN vs R101-FPN), but the tensor kind and
   logit scale match.
2. **`margs.size` is 128, not 256.** The head runs at half the crop resolution.
3. **The ResNet input size is 256, from the danbooru tagger checkpoint**, and is
   independent of `margs.size`. Conflating the two halves feature resolution.
4. **Crop geometry is bbox → centred square → 256px → 10% pad**, composed
   through the repo's cropbox algebra (ported to numpy in
   `anime_pose/geometry.py`). A rectangular crop changes both the aspect the
   model sees and the inverse transform used to map keypoints back.

Before these: 219 px mean error against the reference. After: 30.4 px, with
head keypoints essentially exact. Check with
`python scripts/compare_keypoints.py --dir <dir>`.

## Files

| path | what |
|---|---|
| `anime_pose/inference.py` | pipeline: segment → frame → pose → gate → render |
| `anime_pose/models.py` | standalone `nn.Module`s mirroring the checkpoints |
| `anime_pose/geometry.py` | numpy port of upstream's cropbox algebra |
| `anime_pose/weights.py` | checkpoint discovery + download |
| `anime_pose/pose_convert.py` | COCO-17 → OpenPose-18 + DWPose-style renderer |
| `scripts/` | weight building/verification, accuracy and DWPose comparisons |

## License

AGPL-3.0, inherited from the upstream project. See [LICENSE](LICENSE) and
[NOTICE](NOTICE). If you deploy this as a network service, AGPL section 13
applies.
