"""Geometry and descriptive analysis for new H&E-only slides."""
from __future__ import annotations
import cv2
import numpy as np


def analysis_geometry(header, mpp=.325):
    if not np.isfinite(mpp) or mpp <= 0:
        raise ValueError("Analysis MPP must be finite and positive")
    if header.get("format") == "svs":
        from henets.wsi import calibrated_mpp
        native_x, native_y = calibrated_mpp({"openslide.mpp-x": header.get("mpp_x"),
                                            "openslide.mpp-y": header.get("mpp_y")})
        width, height = int(header["width"]), int(header["height"])
        if min(width, height) <= 0:
            raise ValueError("Invalid SVS level-0 dimensions")
        scale_x, scale_y = native_x/mpp, native_y/mpp
        shape = (int(np.floor(height*scale_y)), int(np.floor(width*scale_x)))
        return shape, np.diag([scale_x, scale_y, 1.])
    native_mpp = float(header["cap_res"])
    if int(header["scan_scale"]) != 40 or not .1 < native_mpp < 1:
        raise ValueError("Expected calibrated 40x KFB supported by the current SDK bridge")
    scale = native_mpp / mpp
    shape = (int(np.floor(int(header["height"]) * scale)),
             int(np.floor(int(header["width"]) * scale)))
    return shape, np.diag([scale, scale, 1.])


def he_tissue_mask(bgr, policy="legacy"):
    """Select candidate tissue from the 10.4 um/pixel analysis preview.

    ``stain_components`` is an optical foreground policy for scanned SVS glass
    backgrounds. It uses no model predictions or clinical outcomes. Small pale
    spaces within stained components are filled; large glass gaps stay open.
    This is a sampling mask, not a pathologic tissue or artifact annotation.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    if policy == "legacy":
        # Preserve the original KFB selection exactly.
        raw = ((hsv[..., 1] >= 20) | (bgr.mean(2) <= 220)).astype(np.uint8)
        kernel_size, minimum_component = 3, 32
    elif policy == "stain_components":
        raw = ((hsv[..., 1] >= 20) & (bgr.min(2) < 240)).astype(np.uint8)
        kernel_size, minimum_component = 5, 64
    else:
        raise ValueError(f"Unknown preview tissue policy: {policy}")
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    keep = np.zeros(raw.shape, np.uint8)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= minimum_component:
            keep[labels == i] = 255
    if policy == "stain_components":
        # Fill only enclosed gaps <=256 preview pixels (~0.028 mm^2), never
        # the exterior glass background or large lumina between fragments.
        n, holes, areas, _ = cv2.connectedComponentsWithStats((keep == 0).astype(np.uint8), 8)
        boundary = set(np.concatenate((holes[0], holes[-1], holes[:, 0], holes[:, -1])))
        for i in range(1, n):
            if i not in boundary and areas[i, cv2.CC_STAT_AREA] <= 256:
                keep[holes == i] = 255
    return keep


def eligible_compartment(row, min_tissue=.5, purity=.6):
    if int(row["center224_tissue_pixels"]) / 224**2 < min_tissue:
        return "low_tissue"
    if float(row["center224_purity"]) < purity:
        return "mixed"
    return row["center224_dominant_compartment"]


def describe(values):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return dict(n=0, mean=None, median=None, q25=None, q75=None)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite scores")
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)),
                q25=float(np.percentile(x, 25)), q75=float(np.percentile(x, 75)))


def black_line_qc(rgb, threshold=30, line_length=32):
    """Flag long near-black scan lines, not ordinary small dark nuclei."""
    dark = (rgb.max(2) < threshold).astype(np.uint8)
    horizontal = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((1,line_length),np.uint8))
    vertical = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((line_length,1),np.uint8))
    pixels = int(((horizontal>0)|(vertical>0)).sum())
    return dict(center_black_pixel_fraction=float(dark.mean()), center_black_line_pixels=pixels,
                artifact_review=bool(pixels >= line_length))
