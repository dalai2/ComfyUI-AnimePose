"""Verbatim numpy port of the cropbox algebra in bizarre-pose-estimator's
`_util/twodee_v0.py`.

The pose model is extremely sensitive to how the character is framed: it was
trained on a very specific crop (bbox -> centred square -> 256px -> 10% pad),
and feeding it anything else silently degrades the keypoints by hundreds of
pixels. So rather than approximate that framing, this module reproduces the
original's cropbox composition exactly.

Everything here is in the upstream repo's **(row, col) = (y, x)** convention,
not the (x, y) convention cv2/PIL use. Convert at the boundary, not in here.

A "cropbox" is the triple ``(from_corner, from_size, to_size)``: take the
rectangle at ``from_corner`` of size ``from_size`` out of the source image
(padding with zeros where it falls outside), then resize it to ``to_size``.
"""

from __future__ import annotations

import math

import numpy as np

CropBox = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


def _pixel_rounder(n: float, rounding):
    if rounding is True or rounding == "round":
        return round(n)
    if rounding == "ceil":
        return math.ceil(n)
    if rounding == "floor":
        return math.floor(n)
    return n


def pixel_ij(x, rounding=True) -> tuple:
    """Normalise a scalar or pair into a 2-tuple, optionally rounding.

    A bare scalar broadcasts to both axes -- upstream relies on this (e.g. the
    padding cropbox is written as three scalars).
    """
    if isinstance(x, np.ndarray):
        x = x.tolist()
    seq = x if isinstance(x, (tuple, list)) else (x, x)
    return tuple(_pixel_rounder(i, rounding) for i in seq)


def resize_square_dry(size: tuple[float, float], s: float) -> CropBox:
    """Cropbox that centres a (h, w) rectangle inside a square of side `s`."""
    h, w = size
    from_corner = (0, -(h - w) // 2) if h >= w else (-(w - h) // 2, 0)
    return (from_corner, (max(h, w),) * 2, (s, s))


def cropbox_compose(cba: CropBox, cbb: CropBox) -> CropBox:
    """Compose two cropboxes: apply `cba` first, then `cbb`."""
    fca, fsa, tsa = [pixel_ij(q, rounding=False) for q in cba]
    fcb, fsb, tsb = [pixel_ij(q, rounding=False) for q in cbb]
    sfx = fsa[0] / tsa[0]
    sfy = fsa[1] / tsa[1]
    return (
        (fca[0] + fcb[0] * sfx, fca[1] + fcb[1] * sfy),
        (fsb[0] * sfx, fsb[1] * sfy),
        tsb,
    )


def cropbox_sequence(cropboxes: list[CropBox]) -> CropBox:
    """Compose a list of cropboxes applied in order."""
    ans = cropboxes[-1]
    for c in range(len(cropboxes) - 2, -1, -1):
        ans = cropbox_compose(cropboxes[c], ans)
    return ans


def cropbox_points(pts: np.ndarray, from_corner, from_size, to_size) -> np.ndarray:
    """Map (N, 2) points in (row, col) through a cropbox."""
    pts = np.asarray(pts, dtype=np.float64)
    assert pts.ndim == 2 and pts.shape[1] == 2, f"expected (N,2), got {pts.shape}"
    fc = np.asarray(pixel_ij(from_corner, rounding=False), dtype=np.float64)[None]
    fs = pixel_ij(from_size, rounding=False)
    ts = pixel_ij(to_size, rounding=False)
    sf = np.asarray([ts[0] / fs[0], ts[1] / fs[1]], dtype=np.float64)[None]
    return (pts - fc) * sf


def cropbox_inverse(origin_size, from_corner, from_size, to_size) -> CropBox:
    """Cropbox that undoes the given one, back into `origin_size`."""
    origin_size = pixel_ij(origin_size, rounding=False)
    from_corner = pixel_ij(from_corner, rounding=False)
    from_size = pixel_ij(from_size, rounding=False)
    to_size = pixel_ij(to_size, rounding=False)
    sx, sy = to_size[0] / from_size[0], to_size[1] / from_size[1]
    return (
        (-from_corner[0] * sx, -from_corner[1] * sy),
        (origin_size[0] * sx, origin_size[1] * sy),
        origin_size,
    )


def pose_cropbox(bbox, size: int, padding: float, pad_factor: float = 1.0) -> CropBox:
    """The exact framing `_scripts/pose_estimator.py::infer_pose` feeds the model.

    bbox: ((row0, col0), (h, w)) of the character, from the segmenter.
    size / padding: `largs.danbooru_coco.{size,padding}` out of the checkpoint
        (256 and 0.1 for every released pose checkpoint).
    """
    _s = size
    _p = _s * padding
    return cropbox_sequence([
        (bbox[0], bbox[1], bbox[1]),
        resize_square_dry(bbox[1], _s),
        (-_p * pad_factor / 2, _s + _p * pad_factor, _s),
    ])


def percentile_bbox(mask: np.ndarray, keep: float = 0.05):
    """Bounding box that ignores rows/columns holding almost no mask.

    `alpha_bbox` takes the absolute extremes, so anything thin attached to the
    character -- ribbons, scarves, smoke, loose hair, a drawn weapon -- drags
    the box out to the edge of the frame. The character then shrinks inside the
    256px crop the pose model sees, and accuracy drops with it.

    This keeps only the rows and columns whose mask pixel count exceeds `keep`
    times the busiest row/column, which is where the body actually is. Thin
    appendages contribute a handful of pixels per row and fall away; limbs do
    not. `keep=0` reproduces `alpha_bbox`.

    Returns ((row0, col0), (h, w)); falls back to the full extent if the
    threshold would empty the box.
    """
    m = np.asarray(mask, dtype=bool)
    if keep <= 0 or not m.any():
        return alpha_bbox(m.astype(np.float32), thresh=0.5)

    row_counts, col_counts = m.sum(1), m.sum(0)
    rows = np.nonzero(row_counts > row_counts.max() * keep)[0]
    cols = np.nonzero(col_counts > col_counts.max() * keep)[0]
    if len(rows) == 0 or len(cols) == 0:
        return alpha_bbox(m.astype(np.float32), thresh=0.5)

    r0, r1 = int(rows[0]), int(rows[-1])
    c0, c1 = int(cols[0]), int(cols[-1])
    # grow by a pixel on each side, matching alpha_bbox
    r0, c0 = max(r0 - 1, 0), max(c0 - 1, 0)
    r1, c1 = min(r1 + 1, m.shape[0]), min(c1 + 1, m.shape[1])
    return ((r0, c0), (r1 - r0, c1 - c0))


def alpha_bbox(mask: np.ndarray, thresh: float = 0.5, allow_empty: bool = True):
    """Port of `abbox`: bounding box of `mask > thresh`, grown by 1px.

    Returns ((row0, col0), (h, w)). Note upstream's axis naming looks swapped
    (it calls the row axis "x"); the values are (row, col).
    """
    a = mask > thresh
    rows = np.any(a, axis=1).nonzero()[0]
    cols = np.any(a, axis=0).nonzero()[0]
    if len(rows) == 0:
        if not allow_empty:
            raise ValueError("empty segmentation mask")
        rows = np.asarray([0, a.shape[0]])
    if len(cols) == 0:
        if not allow_empty:
            raise ValueError("empty segmentation mask")
        cols = np.asarray([0, a.shape[1]])
    r0, r1 = max(int(rows.min()) - 1, 0), min(int(rows.max()) + 1, a.shape[0])
    c0, c1 = max(int(cols.min()) - 1, 0), min(int(cols.max()) + 1, a.shape[1])
    return ((r0, c0), (r1 - r0, c1 - c0))
