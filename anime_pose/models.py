"""Standalone PyTorch models for ComfyUI-AnimePose.

Replaces two deps that don't install cleanly on Windows/CUDA13:
  - Detectron2  → torchvision keypointrcnn_resnet50_fpn
  - PyTorch Lightning → plain torch.load + strict=False

All nn.Module subclasses here mirror the original bizarre-pose-estimator
structures **exactly** (same layer names, same forward logic) so that weights
saved from the original .ckpt files load correctly with strict=False.
Only the Detectron2 RCNN weights are excluded — they live under the 'rcnn.*'
prefix in the checkpoint and are replaced by fresh torchvision weights.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as TT


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Helpers shared by both models
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Exact replica of _util/helper_models_v0.py ResBlock.
    Layer names and forward logic MUST match for checkpoint weights to load.
    """

    def __init__(
        self,
        depth: int,
        channels: int,
        kernel: int,
        channels_in: Optional[int] = None,
        activation=nn.ReLU,
        normalization=nn.BatchNorm2d,
    ):
        super().__init__()
        self.channels_in = channels_in

        od: OrderedDict = OrderedDict()
        for i in range(depth):
            chin = channels_in if (channels_in is not None and i == 0) else channels
            od[f"conv{i}"] = nn.Conv2d(
                chin, channels,
                kernel_size=kernel, padding=kernel // 2,
                bias=True, padding_mode="replicate",
            )
            if activation is not None:
                od[f"act{i}"] = activation()
            if normalization is not None:
                od[f"norm{i}"] = normalization(channels)
        self.net = nn.Sequential(od)

        od_tail: OrderedDict = OrderedDict()
        if activation is not None:
            od_tail[f"act{depth}"] = activation()
        if normalization is not None:
            od_tail[f"norm{depth}"] = normalization(channels)
        self.net_tail = nn.Sequential(od_tail)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channels_in is None:
            return self.net_tail(x + self.net(x))
        # When channels_in != channels the skip goes from conv0 output onwards
        head = self.net[0](x)          # first Conv2d only (channels_in → channels)
        t = head
        for body in self.net[1:]:      # act0, norm0, conv1, act1, norm1, conv2, …
            t = body(t)
        return self.net_tail(head + t)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Torchvision-based Keypoint detector (replaces Detectron2)
# ─────────────────────────────────────────────────────────────────────────────

class TorchvisionKeypointDetector(nn.Module):
    """Drop-in replacement for PretrainedKeypointDetector (Detectron2).

    What the original actually produces matters a lot here, and it is easy to
    get wrong. `PretrainedKeypointDetector` does **not** run detection: it
    forces a single ROI covering the whole image and reads back
    `pred_keypoint_heatmaps`, which detectron2 documents as "the raw keypoint
    logits as passed to [keypoint_rcnn_inference]" -- i.e. a dense
    ``(N, 17, 56, 56)`` tensor of unnormalised spatial logits over the ROI
    grid, never resampled to image resolution.

    So the tensor `converter_rcnn` (Conv2d 17->32) was trained on is a dense
    56x56 logit map, not a set of point detections. torchvision exposes the
    same intermediate, so this reproduces it by calling the keypoint branch
    directly with a full-image box:

        backbone -> keypoint_roi_pool -> keypoint_head -> keypoint_predictor

    Both frameworks train that head with cross-entropy over the flattened
    56x56 grid, so the logit scale is comparable. The remaining difference is
    the backbone (torchvision R50-FPN vs detectron2 R101-FPN) -- a domain
    shift the trained head tolerates, unlike a change of tensor *kind*.
    """

    ROI_SIDE = 56   # keypoint_predictor output side, both frameworks

    def __init__(self):
        super().__init__()
        weights = torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        # min/max size match detectron2's INPUT.{MIN,MAX}_SIZE_TEST defaults
        # (800 / 1333) used by the model-zoo config the original loads.
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
            weights=weights, min_size=800, max_size=1333,
        )
        self.model.eval()
        # frozen - this is a fixed feature extractor, never fine-tuned
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(
        self,
        img: torch.Tensor,          # (bs, 3, H, W)  values in [0, 1]
        return_more: bool = False,
    ) -> dict:
        m = self.model
        m.eval()

        # GeneralizedRCNNTransform: ImageNet-normalise + resize shortest edge
        # to 800. Same resize rule detectron2 applies, so the forced ROI ends
        # up covering the same content.
        images, _ = m.transform(list(img.unbind(0)))
        features = m.backbone(images.tensors)

        # One ROI per image, covering the whole (post-transform) image. This is
        # the "forces them to use my bboxes" branch of the original.
        boxes = [
            torch.tensor([[0.0, 0.0, float(w), float(h)]],
                         dtype=img.dtype, device=img.device)
            for (h, w) in images.image_sizes
        ]

        x = m.roi_heads.keypoint_roi_pool(features, boxes, images.image_sizes)
        x = m.roi_heads.keypoint_head(x)
        logits = m.roi_heads.keypoint_predictor(x)   # (bs, 17, 56, 56)

        ans = {"keypoint_heatmaps": logits}
        if return_more:
            ans["features"] = features
            ans["image_sizes"] = images.image_sizes
        return ans


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Standalone ResNet feature extractor (mirrors ResnetFeatureExtractor)
# ─────────────────────────────────────────────────────────────────────────────

class StandaloneResNetExtractor(nn.Module):
    """Mirrors passup.ResnetFeatureExtractor.

    The released checkpoints set ``margs.resnet_inferserve_query`` to the
    danbooru tagger ("kate") checkpoint, not 'torchvision' -- so this is the
    third branch of the original, whose preprocessing comes from that tagger:
    ``TT.Resize(largs.danbooru_sfw.size)`` = 256, and ImageNet normalisation
    (kate/models.py uses the standard ImageNet mean/std, same as the
    torchvision branch). The *weights* differ from torchvision's ResNet50, but
    those are stored in the pose .ckpt under 'resnet.*' and load from there.

    Note this 256 is the tagger's input size and is independent of the pose
    head's ``margs.size``; conflating the two silently halves the resolution
    every feature is extracted at.
    """

    TAGGER_SIZE = 256   # largs.danbooru_sfw.size of the kate checkpoint

    def __init__(self, size: int = TAGGER_SIZE):
        super().__init__()
        self.resize = TT.Resize(size, antialias=False)
        self.resnet_preprocess = TT.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        _resnet = torchvision.models.resnet50(weights=None)
        self.conv1  = _resnet.conv1
        self.bn1    = _resnet.bn1
        self.relu   = _resnet.relu
        self.maxpool = _resnet.maxpool
        self.layer1 = _resnet.layer1
        self.layer2 = _resnet.layer2
        self.layer3 = _resnet.layer3

    def forward(self, x: torch.Tensor) -> dict:
        x = self.resize(x)
        x = self.resnet_preprocess(x)
        x = self.conv1(x);  ans = {}
        x = self.bn1(x)
        x = self.relu(x);   ans["conv1"] = x
        x = self.maxpool(x)
        x = self.layer1(x); ans["layer1"] = x
        x = self.layer2(x); ans["layer2"] = x
        x = self.layer3(x); ans["layer3"] = x
        return ans


# ─────────────────────────────────────────────────────────────────────────────
# 4.  ResnetFeatureConverter (mirrors passup.ResnetFeatureConverter)
# ─────────────────────────────────────────────────────────────────────────────

class StandaloneResNetConverter(nn.Module):
    def __init__(self, size: int = 128):
        super().__init__()
        self.size = size
        self.resizer = TT.Resize(size, antialias=False)
        self.conv1  = nn.Conv2d(64,   64,  kernel_size=1)
        self.layer1 = nn.Conv2d(256,  64,  kernel_size=1)
        self.layer2 = nn.Conv2d(512,  64,  kernel_size=1)
        self.layer3 = nn.Conv2d(1024, 64,  kernel_size=1)
        self.head   = nn.Conv2d(64*4, 128, kernel_size=3, padding=1, padding_mode="replicate")
        self.relu       = nn.LeakyReLU()
        self.batchnorm  = nn.BatchNorm2d(64*4)

    def forward(self, feats: dict) -> torch.Tensor:
        combined = torch.cat([
            self.resizer(self.conv1 (feats["conv1" ])),
            self.resizer(self.layer1(feats["layer1"])),
            self.resizer(self.layer2(feats["layer2"])),
            self.resizer(self.layer3(feats["layer3"])),
        ], dim=1)
        return self.head(self.relu(self.batchnorm(combined)))


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Full standalone pose model (passup.Model without PL / Detectron2)
# ─────────────────────────────────────────────────────────────────────────────

class StandalonePoseModel(nn.Module):
    """Mirrors passup.Model.  Load weights with load_pose_checkpoint()."""

    NUM_KPS_OUT = 17 + 8   # matches original head output (25 channels)
    # margs.size in every released pose checkpoint. The head runs at 128px,
    # NOT at the 256px crop resolution.
    DEFAULT_SIZE = 128

    def __init__(self, size: int = DEFAULT_SIZE):
        super().__init__()
        self.size = size
        self.resizer = TT.Resize(size, antialias=False)

        # sub-modules — names must match checkpoint keys
        self.resnet = StandaloneResNetExtractor()
        self.rcnn   = TorchvisionKeypointDetector()   # replaces Detectron2

        self.keypoint_head = nn.ModuleDict({
            "converter_resnet": StandaloneResNetConverter(size),
            "converter_rcnn":   nn.Conv2d(17, 32, kernel_size=1, padding=0),
            "head": nn.Sequential(
                nn.Conv2d(128 + 32 + 3, 128, kernel_size=3, padding=1, padding_mode="replicate"),
                nn.LeakyReLU(),
                nn.BatchNorm2d(128),
                ResBlock(3, 64, 3, channels_in=128),
                ResBlock(3, 32, 3, channels_in=64),
                nn.Conv2d(32, self.NUM_KPS_OUT, kernel_size=3, padding=1, padding_mode="replicate"),
            ),
        })

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,        # (bs, 3, H, W)  values in [0, 1]
        smoothing: float = 0.1,
    ) -> dict:
        import kornia

        # extract features
        feats_resnet = self.resnet(x)
        feats_rcnn   = self.rcnn(x)

        feats = torch.cat([
            self.resizer(x),
            self.resizer(self.keypoint_head["converter_resnet"](feats_resnet)),
            self.resizer(self.keypoint_head["converter_rcnn"](feats_rcnn["keypoint_heatmaps"])),
        ], dim=1)

        hms = self.keypoint_head["head"](feats)

        # Unblurred per-pixel probability. This is what carries the "is there
        # any evidence for this joint at all" signal — the original repo throws
        # it away and only argmaxes the blurred map, which is why every joint
        # always comes back with a position even when it isn't in the picture.
        hmp_raw = torch.sigmoid(hms)

        hmp = hmp_raw
        if smoothing and smoothing > 0:
            ksig = max(hmp.shape[-2:]) * smoothing
            kern = max(3, int(ksig) * 2 + 1)
            hmp = kornia.filters.gaussian_blur2d(
                hmp,
                kernel_size=(kern, kern),
                sigma=(ksig, ksig),
                border_type="reflect",
            )
        # Localisation comes from the blurred map (same as upstream): the blur
        # makes argmax pick the centre of the response blob instead of a noisy
        # single-pixel spike.
        bs, C, Hh, Wh = hmp.shape
        kps_flat = hmp.view(bs, C, -1).argmax(-1)
        rows = torch.div(kps_flat, Wh, rounding_mode="floor")
        cols = kps_flat % Wh
        kps = torch.stack([
            rows * (x.shape[2] / Hh),
            cols * (x.shape[3] / Wh),
        ], dim=2)   # (bs, 25, 2)  in (row, col) = (y, x)

        # Confidence comes from the *unblurred* map. Two flavours, both useful:
        #   peak        — strongest evidence anywhere in the channel. Robust to
        #                 the blur dragging the argmax off a sharp peak.
        #   at_argmax   — evidence exactly where we ended up placing the joint.
        # A joint that isn't in frame has no strong response anywhere, so its
        # peak stays low even though argmax still returns *some* pixel.
        scores_peak = hmp_raw.view(bs, C, -1).max(-1).values           # (bs, C)
        scores_at   = hmp_raw.view(bs, C, -1).gather(
            -1, kps_flat.unsqueeze(-1)
        ).squeeze(-1)                                                  # (bs, C)

        return {
            "keypoints": kps,
            "keypoint_heatmaps_prob": hmp,
            "keypoint_scores": scores_peak,
            "keypoint_scores_at_argmax": scores_at,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Standalone background segmenter (alaska.Model without PL)
# ─────────────────────────────────────────────────────────────────────────────

class StandaloneSegmenter(nn.Module):
    """Mirrors alaska.Model. Load weights with load_seg_checkpoint()."""

    def __init__(self):
        super().__init__()
        import torchvision.models.segmentation as seg_models
        import torchvision.models.segmentation.deeplabv3 as dlv3

        self.deeplab = seg_models.deeplabv3_resnet101(weights=None)
        self.deeplab.aux_classifier = None
        self.deeplab.classifier = nn.Sequential(
            dlv3.ASPP(2048, [12, 24, 36]),
            nn.Conv2d(256, 64,  kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64,  16,  kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
        )
        self.final_head = nn.Sequential(
            nn.Conv2d(16 + 3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 8,   kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(8,  2,   kernel_size=1, stride=1),
        )
        self.deeplab_preprocess = TT.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> dict:
        """rgb: (bs, 3, H, W) in [0, 1].  Returns {'softmax': (bs, 2, H, W)}."""
        normed = self.deeplab_preprocess(rgb)
        out_dl  = self.deeplab(normed)["out"]
        out_fin = self.final_head(torch.cat([out_dl, normed], dim=1))
        softmax = torch.softmax(out_fin, dim=1)
        return {"softmax": softmax, "raw": out_fin}


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Weight loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip_prefix(state_dict: dict, prefix: str) -> dict:
    """Remove a common prefix from all keys (PyTorch Lightning adds none, but
    just in case the ckpt was saved with DataParallel wrapping)."""
    return {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    }


def load_pose_checkpoint(
    ckpt_path: str | Path,
    device: str | torch.device = "cuda",
    size: int = StandalonePoseModel.DEFAULT_SIZE,
) -> StandalonePoseModel:
    """Load the pose estimator from a .ckpt file.

    Weights that transfer (resnet.*, keypoint_head.*) are loaded.
    Weights that don't transfer (rcnn.* — Detectron2) are skipped.
    """
    from .weights import load_state_dict

    model = StandalonePoseModel(size=size).to(device)
    sd = load_state_dict(ckpt_path, device=device)

    # Strip rcnn.* from the checkpoint BEFORE load_state_dict.
    # strict=False skips missing/unexpected keys but still raises on shape
    # mismatches — and the Detectron2 rcnn.* keys in the ckpt share names
    # with our torchvision rcnn but have incompatible shapes (e.g. bbox_pred
    # [4,1024] vs [8,1024]). Dropping them here lets torchvision keep its
    # own pretrained weights while everything else transfers cleanly.
    rcnn_keys = [k for k in sd if k.startswith("rcnn.")]
    sd_filtered = {k: v for k, v in sd.items() if not k.startswith("rcnn.")}
    missing, unexpected = model.load_state_dict(sd_filtered, strict=False)
    print(f"[pose] Loaded {len(sd_filtered) - len(unexpected)} / {len(sd_filtered)} params")
    # NOTE: keep runtime prints pure ASCII. ComfyUI on Windows often runs with a
    # cp1252 console, where printing a non-ASCII char raises UnicodeEncodeError
    # and takes the whole node down during model load.
    print(f"[pose] Skipped {len(rcnn_keys)} rcnn.* keys (Detectron2 -> replaced by torchvision)")
    if unexpected:
        print(f"[pose] Other unexpected keys: {unexpected[:5]}")
    non_rcnn_missing = [k for k in missing if not k.startswith("rcnn.")]
    if non_rcnn_missing:
        print(f"[pose] Missing keys: {non_rcnn_missing[:5]}")

    model.eval()
    return model


def load_seg_checkpoint(
    ckpt_path: str | Path,
    device: str | torch.device = "cuda",
) -> StandaloneSegmenter:
    """Load the background segmenter from a .ckpt file."""
    from .weights import load_state_dict

    model = StandaloneSegmenter().to(device)
    sd    = load_state_dict(ckpt_path, device=device)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[seg]  Loaded {len(sd) - len(unexpected)} / {len(sd)} params")
    if unexpected:
        print(f"[seg]  Unexpected: {unexpected[:5]}")
    if missing:
        print(f"[seg]  Missing: {missing[:5]}")

    model.eval()
    return model
