from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def full_resolution_homography(
    preview_homography: np.ndarray,
    he_downsample: float = 32.0,
    if_downsample: float = 32.0,
) -> np.ndarray:
    """Convert an H&E-preview→IF-preview homography to native pixel coordinates."""
    he_to_preview = np.diag([1 / he_downsample, 1 / he_downsample, 1.0])
    preview_to_if = np.diag([if_downsample, if_downsample, 1.0])
    result = preview_to_if @ preview_homography @ he_to_preview
    return result / result[2, 2]


def load_full_resolution_homography(registration_json: Path) -> np.ndarray:
    record = json.loads(registration_json.read_text(encoding="utf-8"))
    matrix = np.asarray(record["homography_he_preview_to_exported_dapi"], dtype=np.float64)
    return full_resolution_homography(matrix)


def tissue_mask(dapi_preview: np.ndarray) -> np.ndarray:
    """Create a conservative tissue mask from a display-normalized DAPI overview."""
    if dapi_preview.ndim != 2:
        raise ValueError("DAPI preview must be grayscale")
    blurred = cv2.GaussianBlur(dapi_preview, (5, 5), 0)
    _threshold, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    minimum_area = max(64, int(mask.size * 0.00005))
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum_area:
            cleaned[labels == label] = 255
    return cleaned


def patch_grid(
    mask: np.ndarray,
    full_width: int,
    full_height: int,
    patch_size: int = 512,
    stride: int = 512,
    mask_downsample: int = 32,
    minimum_tissue_fraction: float = 0.35,
) -> list[dict[str, int | float]]:
    integral = cv2.integral((mask > 0).astype(np.uint8))
    records: list[dict[str, int | float]] = []
    for y in range(0, full_height - patch_size + 1, stride):
        for x in range(0, full_width - patch_size + 1, stride):
            x0, y0 = x // mask_downsample, y // mask_downsample
            x1 = min(mask.shape[1], (x + patch_size + mask_downsample - 1) // mask_downsample)
            y1 = min(mask.shape[0], (y + patch_size + mask_downsample - 1) // mask_downsample)
            area = max(1, (x1 - x0) * (y1 - y0))
            tissue = (integral[y1, x1] - integral[y0, x1]
                      - integral[y1, x0] + integral[y0, x0])
            fraction = float(tissue / area)
            if fraction >= minimum_tissue_fraction:
                records.append({"if_x": x, "if_y": y, "patch_size": patch_size,
                                "tissue_fraction": fraction})
    return records


def patch_inside_source(
    if_x: int,
    if_y: int,
    patch_size: int,
    he_to_if: np.ndarray,
    he_width: int,
    he_height: int,
    margin: int = 8,
) -> bool:
    corners = np.float32([[[if_x, if_y], [if_x + patch_size, if_y],
                           [if_x + patch_size, if_y + patch_size],
                           [if_x, if_y + patch_size]]])
    source = cv2.perspectiveTransform(corners, np.linalg.inv(he_to_if))[0]
    return bool(
        source[:, 0].min() >= margin
        and source[:, 1].min() >= margin
        and source[:, 0].max() < he_width - margin
        and source[:, 1].max() < he_height - margin
    )
