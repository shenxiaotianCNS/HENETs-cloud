"""Write a calibrated synthetic tiled slide for an installation smoke test."""
import argparse
from pathlib import Path

import numpy as np
import tifffile


def make_demo(path):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 1024
    y, x = np.indices((n, n))
    spots = ((x % 33 - 16)**2 + (y % 29 - 14)**2) < 45
    rgb = np.empty((n, n, 3), dtype=np.uint8)
    rgb[:] = (222, 155, 186)
    rgb[spots] = (125, 68, 141)
    description = ("Aperio Image Library v11.2.1\n1024x1024 [0,0 1024x1024] "
                   "(256x256) JPEG/RGB Q=90|AppMag = 40|MPP = 0.325")
    tifffile.imwrite(path, rgb, tile=(256, 256), photometric="rgb", compression="jpeg",
                     description=description, metadata=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    make_demo(parser.parse_args().output)
