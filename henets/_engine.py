"""Numerical inference kernels from HENETs. No training, study data, or paths are bundled.

Adaptations: one OpenSlide reader, per-case progress, and explicit mounted checkpoints.
"""
from __future__ import annotations
import copy
import csv
import gc
import json
import time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
import tifffile
import torch
from torch import nn
from PIL import Image
from .external_he import analysis_geometry, he_tissue_mask
from .wsi import OpenSlideReader
from .patches import patch_grid
from .tissue import BCSSTissueModel
from .tissue_dense import predict_full_block, dense_labels, compartment_stats
from .infer import sha256, encoder_class
MARKERS = ('DAPI','MPO','CitH3')


class ProteinHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, len(MARKERS))
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class LastStageProteinModel(nn.Module):
    def __init__(self, stage: nn.Module, norm: nn.Module) -> None:
        super().__init__()
        self.stage = stage
        self.norm = norm
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = ProteinHead()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.stage(tokens)
        tokens = self.norm(tokens)
        features = self.pool(tokens.transpose(1, 2)).flatten(1)
        return self.head(features)


def he_nets_score(mpo: np.ndarray, cith3: np.ndarray) -> np.ndarray:
    """Fixed-scale 0-100 regional score requiring both predicted proteins."""
    return 100.0 * np.sqrt(np.clip(mpo, 0, 1) * np.clip(cith3, 0, 1))


def read_aligned_he(
    bridge: OpenSlideReader,
    he_path: Path,
    he_shape: tuple[int, int],
    he_to_if: np.ndarray,
    if_x: int,
    if_y: int,
    patch_size: int,
) -> np.ndarray:
    inverse = np.linalg.inv(he_to_if)
    corners = np.float32([[[if_x, if_y], [if_x + patch_size, if_y],
                           [if_x + patch_size, if_y + patch_size],
                           [if_x, if_y + patch_size]]])
    he_corners = cv2.perspectiveTransform(corners, inverse)[0]
    margin = 32
    x0 = max(0, int(np.floor(he_corners[:, 0].min())) - margin)
    y0 = max(0, int(np.floor(he_corners[:, 1].min())) - margin)
    x1 = min(he_shape[1], int(np.ceil(he_corners[:, 0].max())) + margin)
    y1 = min(he_shape[0], int(np.ceil(he_corners[:, 1].max())) + margin)
    width = min(he_shape[1] - x0, ((x1 - x0 + 255) // 256) * 256)
    height = min(he_shape[0] - y0, ((y1 - y0 + 255) // 256) * 256)
    roi = bridge.read_region(he_path, x0, y0, width, height)
    local_to_patch = np.array([[1, 0, -if_x], [0, 1, -if_y], [0, 0, 1]], float)
    local_to_native = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], float)
    transform = local_to_patch @ he_to_if @ local_to_native
    return cv2.warpPerspective(roi, transform, (patch_size, patch_size),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(255, 255, 255))


def write_json(path, value):
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
    temp.replace(path)


def write_csv(path, rows):
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    temp.replace(path)


def load_csv(path):
    with path.open(encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def prepare(config, out):
    with OpenSlideReader() as bridge:
        for sample in config["samples"]:
            sid = sample["sample_id"]
            folder = out / sid
            folder.mkdir(exist_ok=True)
            source = Path(sample["he_path"])
            header = bridge.header(source)
            shape, transform = analysis_geometry(header, config["analysis_mpp"])
            header["format"] = "OpenSlide"
            header["source_suffix"] = source.suffix.lower()
            record = dict(**sample, header=header, analysis_shape=list(shape),
                          he_to_analysis=transform.tolist(), source_bytes=source.stat().st_size,
                          source_sha256=sha256(source))
            info = folder / "slide.json"
            if info.exists():
                if json.loads(info.read_text()) != record:
                    raise ValueError(f"Source slide changed: {sid}")
                if (folder / "grid.csv").exists():
                    continue
            print(f"PREVIEW {sid} native={header['width']}x{header['height']}", flush=True)
            native = bridge.read_preview(source, downsample=32)
            h, w = shape
            # Warp rather than resize to preserve pixel coordinates at canvas edges.
            preview = cv2.warpPerspective(native, transform, ((w+31)//32, (h+31)//32),
                                          borderValue=(255, 255, 255))
            cv2.imwrite(str(folder / "HE_preview.jpg"), preview, [cv2.IMWRITE_JPEG_QUALITY, 95])
            mask = he_tissue_mask(preview, policy=config.get("preview_tissue_policy", "legacy"))
            cv2.imwrite(str(folder / "tissue_selection.png"), mask)
            rows = patch_grid(mask, w, h, minimum_tissue_fraction=config["minimum_preview_tissue_fraction"])
            for r in rows:
                r["sample_id"] = sid
                r["analysis_x"] = r.pop("if_x")
                r["analysis_y"] = r.pop("if_y")
                r["patch_id"] = f"{sid}_x{r['analysis_x']}_y{r['analysis_y']}"
            if not rows:
                raise ValueError(f"No tissue windows: {sid}")
            write_json(info, record)
            write_csv(folder / "grid.csv", rows)
            print(f"PREPARED {sid} windows={len(rows)}", flush=True)


def grouped_rows(rows, core):
    groups = defaultdict(list)
    for row in rows:
        x, y = int(row["analysis_x"]), int(row["analysis_y"])
        groups[x//core*core, y//core*core].append(row)
    return groups


def read_context(bridge, info, x, y, core):
    shape = (info["header"]["height"], info["header"]["width"])
    return read_aligned_he(bridge, Path(info["he_path"]), shape,
                           np.array(info["he_to_analysis"]), x-256, y-256, core+512)[..., ::-1].copy()


def process(model, rgb, rows, x, y, core, folder, *, compressed_crops=False, save_confidence=True):
    size = core+512
    scaled = cv2.resize(rgb, (round(size*.325/.25),)*2, interpolation=cv2.INTER_LINEAR)
    probabilities = predict_full_block(model, scaled, collapse=True)
    probabilities = cv2.resize(probabilities, (size, size), interpolation=cv2.INTER_LINEAR)[256:-256, 256:-256]
    center_rgb = rgb[256:-256, 256:-256]
    labels, low, confidence = dense_labels(probabilities, center_rgb)
    covered = np.zeros((core, core), bool)
    results, crops = [], []
    for r in rows:
        dx, dy = int(r["analysis_x"])-x, int(r["analysis_y"])-y
        patch = labels[dy:dy+512, dx:dx+512]
        center = patch[144:368, 144:368]
        covered[dy:dy+512, dx:dx+512] = True
        results.append(dict(patch_id=r["patch_id"], sample_id=r["sample_id"],
                            analysis_x=int(r["analysis_x"]), analysis_y=int(r["analysis_y"]),
                            **compartment_stats(patch), **compartment_stats(center, "center224_"),
                            center224_low_confidence_fraction=float((low[dy+144:dy+368, dx+144:dx+368] & (center>0)).sum()/max((center>0).sum(), 1))))
        crops.append(center_rgb[dy+144:dy+368, dx+144:dx+368].copy())
    labels[~covered] = 255
    confidence[~covered] = 0
    stem = f"x{x}_y{y}"
    Image.fromarray(labels).save(folder / f"{stem}_labels.png", compress_level=3)
    if save_confidence:
        Image.fromarray(confidence).save(folder / f"{stem}_confidence.png", compress_level=3)
    save_crops = np.savez_compressed if compressed_crops else np.savez
    save_crops(folder / f"{stem}_crops.npz", rgb=np.stack(crops), ids=np.array([r["patch_id"] for r in results]))
    write_json(folder / f"{stem}.json", dict(rows=results))
    return results


def package(folder, info, rows, groups, core):
    h, w = info["analysis_shape"]
    @lru_cache(maxsize=24)
    def block(x, y):
        return np.asarray(Image.open(folder / "blocks" / f"x{x}_y{y}_labels.png"))
    def tiles():
        blank = np.full((512, 512), 255, np.uint8)
        for y in range(0, h, 512):
            for x in range(0, w, 512):
                bx, by = x//core*core, y//core*core
                yield block(bx, by)[y-by:y-by+512, x-bx:x-bx+512] if (bx, by) in groups else blank
    temporary = folder / "compartments.tif.tmp"
    tifffile.imwrite(temporary, data=tiles(), shape=(h, w), dtype=np.uint8, bigtiff=True,
                    tile=(512, 512), compression="deflate", compressionargs={"level": 3}, maxworkers=4,
                    resolution=(10000/.325,)*2, resolutionunit="CENTIMETER",
                    metadata=dict(axes="YX", mpp=.325, coordinate_system="HE_resampled_0.325um",
                                  he_to_analysis=info["he_to_analysis"],
                                  classes={0:"background", 1:"tumor", 2:"stroma", 3:"other", 255:"unprocessed"},
                                  scope="HE_tissue_selected_512_grid"))
    temporary.replace(folder / "compartments.tif")
    preview = np.full(((h+31)//32, (w+31)//32), 255, np.uint8)
    for x, y in groups:
        small = block(x, y)[::32, ::32]
        hh, ww = min(small.shape[0], preview.shape[0]-y//32), min(small.shape[1], preview.shape[1]-x//32)
        preview[y//32:y//32+hh, x//32:x//32+ww] = small[:hh, :ww]
    Image.fromarray(preview).save(folder / "compartments_preview.png")
    block.cache_clear()
    write_csv(folder / "regional_compartments.csv", rows)


def segment(config, out, max_blocks=None):
    model = BCSSTissueModel()
    model.load_state_dict(torch.load(config["bcss_weights"], map_location="cpu", weights_only=True))
    model.cuda().eval()
    core = config["core_size"]
    start = time.monotonic()
    count = 0
    with OpenSlideReader() as bridge, ThreadPoolExecutor(max_workers=1) as executor:
        for sample in config["samples"]:
            folder = out / sample["sample_id"]
            info = json.loads((folder / "slide.json").read_text())
            rows = load_csv(folder / "grid.csv")
            groups = grouped_rows(rows, core)
            blocks = folder / "blocks"
            blocks.mkdir(exist_ok=True)
            stats, remaining = [], []
            for (x, y), rr in sorted(groups.items(), key=lambda z:(z[0][1], z[0][0])):
                marker = blocks / f"x{x}_y{y}.json"
                if marker.exists():
                    suffixes = ["_labels.png", "_crops.npz"]
                    if config.get("save_confidence", True):
                        suffixes.append("_confidence.png")
                    if not all((blocks / f"x{x}_y{y}{suffix}").exists() for suffix in suffixes):
                        raise ValueError(f"Incomplete committed block {marker}")
                    stats.extend(json.loads(marker.read_text())["rows"])
                else:
                    remaining.append((x, y))
            print(f"SEGMENT {sample['sample_id']} total={len(groups)} remaining={len(remaining)}", flush=True)
            future = None
            for i, (x, y) in enumerate(remaining):
                if future is None:
                    future = executor.submit(read_context, bridge, info, x, y, core)
                rgb = future.result()
                future = None
                if i+1 < len(remaining):
                    nx, ny = remaining[i+1]
                    future = executor.submit(read_context, bridge, info, nx, ny, core)
                stats.extend(process(model, rgb, groups[x,y], x, y, core, blocks,
                                     compressed_crops=config.get("compressed_crops", False),
                                     save_confidence=config.get("save_confidence", True)))
                count += 1
                progress = dict(stage="segment", sample=sample["sample_id"], sample_blocks_done=len(groups)-len(remaining)+i+1,
                                sample_blocks_total=len(groups), computed=count, elapsed_seconds=time.monotonic()-start)
                write_json(folder / "inference_progress.json", progress)
                if i%20 == 0:
                    print(json.dumps(progress), flush=True)
                if max_blocks and count >= max_blocks:
                    return
            stats.sort(key=lambda r:(r["analysis_y"], r["analysis_x"]))
            package(folder, info, stats, groups, core)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def load_ensemble(config):
    CTransPath = encoder_class(config["ctranspath_root"])
    encoder = CTransPath(num_classes=0).cuda().eval()
    weights = torch.load(config["ctranspath_weights"], map_location="cpu", weights_only=True)
    encoder.load_state_dict(weights.get("model", weights))
    heads, stages = [], []
    for fold in range(9):
        fs = torch.load(Path(config["weights_root"]) / f"path2space_protein/fold{fold}_heads.pt", map_location="cpu", weights_only=True)["heads"]
        ts = torch.load(Path(config["weights_root"]) / f"path2space_last_stage/fold{fold}_models.pt", map_location="cpu", weights_only=True)["models"]
        if len(fs) != 5 or len(ts) != 5:
            raise ValueError("Expected five seeds per fold")
        hh, tt = [], []
        for state in fs:
            head = ProteinHead()
            head.load_state_dict(state)
            hh.append(head.cuda().eval())
        for state in ts:
            model = LastStageProteinModel(copy.deepcopy(encoder.layers[3]), copy.deepcopy(encoder.norm))
            model.load_state_dict(state)
            tt.append(model.cuda().eval())
        heads.append(hh)
        stages.append(tt)
    return encoder, heads, stages


@torch.inference_mode()
def score_batch(rgb, ensemble):
    encoder, heads, stages = ensemble
    x = torch.from_numpy(rgb.transpose(0,3,1,2).copy()).cuda().float()/255
    x = (x-torch.tensor([.485,.456,.406], device="cuda")[None,:,None,None])/torch.tensor([.229,.224,.225], device="cuda")[None,:,None,None]
    with torch.autocast("cuda", dtype=torch.float16):
        tokens = encoder.pos_drop(encoder.patch_embed(x))
        for layer in encoder.layers[:3]:
            tokens = layer(tokens)
        final = encoder.norm(encoder.layers[3](tokens))
        pooled = final.mean(1).float()
        mp = torch.stack([torch.stack([m(pooled).float()[:,1] for m in fold]).mean(0) for fold in heads])
        ci = torch.stack([torch.stack([m(tokens.float()).float()[:,2] for m in fold]).mean(0) for fold in stages])
    mpo, cit = mp.mean(0).cpu().numpy(), ci.mean(0).cpu().numpy()
    return mpo, cit, he_nets_score(mpo, cit), he_nets_score(mp.cpu().numpy(), ci.cpu().numpy()).std(0)


def protein(config, out):
    ensemble = load_ensemble(config)
    for sample in config["samples"]:
        folder = out / sample["sample_id"]
        stats = load_csv(folder / "regional_compartments.csv")
        index = {r["patch_id"]:r for r in stats}
        scores = folder / "protein_blocks"
        scores.mkdir(exist_ok=True)
        results = []
        files = sorted((folder / "blocks").glob("*_crops.npz"))
        start = time.monotonic()
        for i, path in enumerate(files):
            target = scores / (path.stem.replace("_crops", "")+".json")
            if target.exists():
                results.extend(json.loads(target.read_text())["rows"])
                continue
            with np.load(path) as data:
                ids, rgb = data["ids"], data["rgb"]
            mpo, cit, score, spread = score_batch(rgb, ensemble)
            batch = []
            for j, pid in enumerate(ids):
                batch.append(dict(**index[str(pid)], predicted_MPO=float(mpo[j]), predicted_CitH3=float(cit[j]),
                                  HE_NETs_score=float(score[j]), score_fold_sd=float(spread[j])))
            write_json(target, dict(rows=batch))
            results.extend(batch)
            if i%20 == 0:
                progress = dict(stage="protein", sample=sample["sample_id"], blocks_done=i+1,
                                blocks_total=len(files), elapsed_seconds=time.monotonic()-start)
                write_json(folder / "inference_progress.json", progress)
                print(json.dumps(progress), flush=True)
        if len(results) != len(stats) or len({r['patch_id'] for r in results}) != len(stats):
            raise ValueError("Incomplete or duplicate protein rows")
        write_csv(folder / "regional_scores.csv", sorted(results, key=lambda r:(int(r["analysis_y"]), int(r["analysis_x"]))))
    del ensemble
    gc.collect()
    torch.cuda.empty_cache()
