from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


class TiledTiffReader:
    """Random-access reader for a single-channel tiled TIFF."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.tif = tifffile.TiffFile(path)
        self.page = self.tif.pages[0]
        if not self.page.is_tiled or self.page.samplesperpixel != 1:
            raise ValueError(f"Expected single-channel tiled TIFF: {path}")
        self.height, self.width = self.page.shape[:2]
        self.tile_height = self.page.tilelength
        self.tile_width = self.page.tilewidth
        self.tiles_across = (self.width + self.tile_width - 1) // self.tile_width

    def read_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        if x < 0 or y < 0 or x + width > self.width or y + height > self.height:
            raise ValueError(f"Region outside {self.path.name}: {(x, y, width, height)}")
        output = np.empty((height, width), dtype=self.page.dtype)
        first_tx, last_tx = x // self.tile_width, (x + width - 1) // self.tile_width
        first_ty, last_ty = y // self.tile_height, (y + height - 1) // self.tile_height
        for ty in range(first_ty, last_ty + 1):
            for tx in range(first_tx, last_tx + 1):
                index = ty * self.tiles_across + tx
                self.tif.filehandle.seek(self.page.dataoffsets[index])
                encoded = self.tif.filehandle.read(self.page.databytecounts[index])
                decoded, _position, _shape = self.page.decode(encoded, index)
                tile = decoded[0, :, :, 0]
                tile_x, tile_y = tx * self.tile_width, ty * self.tile_height
                src_x0, src_y0 = max(x, tile_x) - tile_x, max(y, tile_y) - tile_y
                src_x1 = min(x + width, tile_x + tile.shape[1]) - tile_x
                src_y1 = min(y + height, tile_y + tile.shape[0]) - tile_y
                dst_x0, dst_y0 = max(tile_x, x) - x, max(tile_y, y) - y
                output[dst_y0:dst_y0 + src_y1 - src_y0,
                       dst_x0:dst_x0 + src_x1 - src_x0] = tile[src_y0:src_y1, src_x0:src_x1]
        return output

    def close(self) -> None:
        self.tif.close()

    def __enter__(self) -> "TiledTiffReader":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
