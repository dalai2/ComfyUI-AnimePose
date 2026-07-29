"""Checkpoint discovery and first-run download.

Goal: install the node and it works. No Docker, no Google Drive, no manual
unzipping, no environment variables.

Search order for each checkpoint:
  1. explicit env var (ANIMEPOSE_POSE_CKPT / ANIMEPOSE_SEG_CKPT)
  2. ComfyUI's models/anime_pose/ (via folder_paths when running inside
     ComfyUI, else a directory next to this package)
  3. a sibling bizarre-pose-estimator/ checkout, if you already have the
     original .ckpt files from the paper's Google Drive
  4. download from HuggingFace into (2)

What gets downloaded is a repackaging of the paper's released weights: the
same tensors, in safetensors (no pickle on load), with the Detectron2 R101
keypoint branch stripped out of the pose checkpoint because this node never
loads it -- torchvision's KeypointRCNN stands in for that branch. Rebuild the
files with `scripts/make_weights.py`; verify any mirror against the official
checkpoints with `scripts/verify_weights.py`.
"""

from __future__ import annotations

import os
import shutil
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Publishing target. Set HF_REPO to the HuggingFace repo holding the files
# produced by scripts/make_weights.py.
# ---------------------------------------------------------------------------
HF_REPO = os.environ.get("ANIMEPOSE_HF_REPO", "CHANGEME/ComfyUI-AnimePose-weights")
HF_REVISION = "main"

POSE_PUBLISH_NAME = "anime_pose_head.safetensors"      # ~37.5 MB
SEG_PUBLISH_NAME = "character_bg_seg.safetensors"      # ~233 MB

# Names the original Google-Drive release uses, if the user already has them.
_LEGACY_POSE = "_train/character_pose_estim/runs/feat_concat+data.ckpt"
_LEGACY_SEG = (
    "_train/character_bg_seg/runs/eyeless_alaska_vulcan0000/checkpoints/"
    "epoch=0096-val_f1=0.9508-val_loss=0.0483.ckpt"
)

SUBDIR = "anime_pose"


def _hf_url(filename: str) -> str:
    return f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}/{filename}"


def models_dir() -> Path:
    """Where to keep (and download) the checkpoints."""
    try:
        import folder_paths  # provided by ComfyUI at runtime
        base = Path(folder_paths.models_dir)
    except Exception:
        base = Path(__file__).resolve().parent.parent / "models"
    d = base / SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _legacy_estimator_dirs() -> list[Path]:
    """Plausible locations of an existing bizarre-pose-estimator checkout."""
    here = Path(__file__).resolve()
    return [
        here.parent.parent.parent / "bizarre-pose-estimator",
        here.parent.parent / "bizarre-pose-estimator",
    ]


def _download(url: str, dest: Path) -> Path:
    """Stream `url` to `dest`, atomically, with a coarse progress line."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[anime-pose] downloading {dest.name} from {HF_REPO}")
    try:
        with urllib.request.urlopen(url) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            done, step = 0, max(int(total) // 20, 1)
            next_mark = step
            with open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if total and done >= next_mark:
                        print(f"[anime-pose]   {100*done/total:5.1f}%  "
                              f"({done/1e6:.0f}/{total/1e6:.0f} MB)")
                        sys.stdout.flush()
                        next_mark += step
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    print(f"[anime-pose] saved {dest}")
    return dest


def _resolve(env_var: str, filename: str, legacy_rel: str,
             allow_download: bool) -> Path:
    override = os.environ.get(env_var)
    if override:
        p = Path(override)
        if not p.exists():
            raise FileNotFoundError(f"{env_var} points at a missing file: {p}")
        return p

    target = models_dir() / filename
    if target.exists():
        return target

    for est in _legacy_estimator_dirs():
        legacy = est / legacy_rel
        if legacy.exists():
            print(f"[anime-pose] using existing checkpoint {legacy}")
            return legacy

    if not allow_download:
        raise FileNotFoundError(
            f"Checkpoint not found: {target}\n"
            f"Download it from https://huggingface.co/{HF_REPO} and put it "
            f"there, or set {env_var}."
        )

    if HF_REPO.startswith("CHANGEME"):
        raise RuntimeError(
            "No weights repository configured.\n"
            "Set HF_REPO in node/weights.py (or the ANIMEPOSE_HF_REPO env var) "
            "to the HuggingFace repo holding the files built by "
            "scripts/make_weights.py, or place the checkpoints manually in "
            f"{target.parent}."
        )

    free = shutil.disk_usage(target.parent).free
    if free < 1_000_000_000:
        raise RuntimeError(
            f"Not enough free space in {target.parent} "
            f"({free/1e9:.1f} GB free, need ~1 GB)."
        )
    return _download(_hf_url(filename), target)


def resolve_pose_ckpt(allow_download: bool = True) -> Path:
    return _resolve("ANIMEPOSE_POSE_CKPT", POSE_PUBLISH_NAME,
                    _LEGACY_POSE, allow_download)


def resolve_seg_ckpt(allow_download: bool = True) -> Path:
    return _resolve("ANIMEPOSE_SEG_CKPT", SEG_PUBLISH_NAME,
                    _LEGACY_SEG, allow_download)


def load_state_dict(path: str | Path, device="cpu") -> dict:
    """Load either a safetensors file or a Lightning .ckpt into a state_dict."""
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path), device=str(device))

    import torch
    # Lightning .ckpt is a pickle; only reachable for files the user placed
    # themselves (every auto-downloaded file is safetensors).
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt.get("state_dict", ckpt)
