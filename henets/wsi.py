"""Bounded-region WSI readers with a shared native-coordinate, BGR interface.

SVS calibration is taken only from OpenSlide's explicit MPP properties. The
reader never infers microns per pixel from objective power or slide filenames.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def calibrated_mpp(properties):
    values = []
    for key in ("openslide.mpp-x", "openslide.mpp-y"):
        try:
            value = float(properties[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"SVS lacks a numeric {key}; calibrated MPP is required") from exc
        if not math.isfinite(value) or not .1 < value < 1:
            raise ValueError(f"Unsupported {key}={value!r}; expected calibrated 20x/40x H&E")
        values.append(value)
    return tuple(values)


def rgba_to_bgr(image):
    """Composite unpremultiplied OpenSlide RGBA onto white, then return BGR."""
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    rgb = np.asarray(Image.alpha_composite(background, rgba).convert("RGB"))
    return rgb[..., ::-1].copy()


class OpenSlideReader:
    """Keep at most one OpenSlide handle; no whole-slide level-0 decoding."""

    def __init__(self, opener=None):
        if opener is None:
            import openslide
            opener = openslide.OpenSlide
        self._opener = opener
        self._path = None
        self._slide = None

    def _get(self, path):
        resolved = str(Path(path).resolve())
        if self._path != resolved:
            self.close()
            self._slide = self._opener(resolved)
            self._path = resolved
        return self._slide

    def header(self, path):
        slide = self._get(path)
        mpp_x, mpp_y = calibrated_mpp(slide.properties)
        width, height = map(int, slide.dimensions)
        if width <= 0 or height <= 0:
            raise ValueError("Invalid SVS level-0 dimensions")
        power = slide.properties.get("openslide.objective-power")
        # Objective power is descriptive; it never supplies missing calibration.
        return dict(format="svs", reader="OpenSlide", width=width, height=height,
                    mpp_x=mpp_x, mpp_y=mpp_y, objective_power=power,
                    vendor=slide.properties.get("openslide.vendor"),
                    level_downsamples=[float(x) for x in slide.level_downsamples],
                    color_handling="native_RGB_no_ICC_transform_alpha_on_white")

    def read_region(self, path, x, y, width, height):
        if width <= 0 or height <= 0:
            raise ValueError("Region dimensions must be positive")
        slide = self._get(path)
        return rgba_to_bgr(slide.read_region((int(x), int(y)), 0, (int(width), int(height))))

    def read_preview(self, path, downsample=32, chunk_size=4096):
        if downsample < 1 or int(downsample) != downsample or chunk_size < downsample:
            raise ValueError("Preview requires an integer downsample and bounded positive chunks")
        header = self.header(path)
        slide = self._get(path)
        downsample = int(downsample)
        out_width = math.ceil(header["width"] / downsample)
        out_height = math.ceil(header["height"] / downsample)
        canvas = np.full((out_height, out_width, 3), 255, np.uint8)
        level = slide.get_best_level_for_downsample(downsample)
        level_downsample = float(slide.level_downsamples[level])
        if not math.isfinite(level_downsample) or level_downsample <= 0:
            raise ValueError("Invalid OpenSlide pyramid downsample")
        step = max(1, int(chunk_size) // downsample)
        scale = level_downsample / downsample
        transform = np.array([[scale, 0, 0], [0, scale, 0]], dtype=float)
        for y in range(0, out_height, step):
            for x in range(0, out_width, step):
                width, height = min(step, out_width-x), min(step, out_height-y)
                # read_region locations are always in level-0 coordinates.
                size = (math.ceil((width-1)/scale)+2, math.ceil((height-1)/scale)+2)
                tile = rgba_to_bgr(slide.read_region((x*downsample, y*downsample), level, size))
                canvas[y:y+height, x:x+width] = cv2.warpAffine(
                    tile, transform, (width, height), flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        return canvas

    def close(self):
        if self._slide is not None:
            self._slide.close()
        self._slide = None
        self._path = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class WSIReader:
    """Route KFB to its unchanged vendor bridge and SVS to OpenSlide."""

    def __init__(self, kfb_factory=None, svs_factory=OpenSlideReader):
        self._factories = {".kfb": kfb_factory, ".svs": svs_factory}
        self._readers = {}

    def _reader(self, path):
        suffix = Path(path).suffix.lower()
        if suffix not in self._factories or self._factories[suffix] is None:
            raise ValueError(f"Unsupported WSI format or unconfigured reader: {suffix}")
        if suffix not in self._readers:
            self._readers[suffix] = self._factories[suffix]()
        return self._readers[suffix]

    def header(self, path):
        return self._reader(path).header(path)

    def read_region(self, path, x, y, width, height):
        return self._reader(path).read_region(path, x, y, width, height)

    def read_preview(self, path, downsample=32, chunk_size=4096):
        return self._reader(path).read_preview(path, downsample, chunk_size)

    def close(self):
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
