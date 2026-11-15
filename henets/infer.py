"""Portable per-slide entry point for frozen HENETs inference on calibrated H&E WSIs."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import threading


HERE = Path(__file__).resolve().parent
MODEL_LOCK = HERE / "model_lock.json"
SINGLE_FILE_SUFFIXES = {".svs", ".tif", ".tiff", ".ndpi", ".scn"}
PROTOCOL = {
    "analysis_mpp": 0.325, "patch_size": 512, "center_size": 224, "core_size": 3072,
    "minimum_preview_tissue_fraction": 0.35, "minimum_center_tissue_fraction": 0.5,
    "compartment_purity": 0.6, "preview_tissue_policy": "stain_components",
    "compressed_crops": True, "save_confidence": False,
}
_IMPORT_LOCK = threading.Lock()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(config_path, sample_id):
    path = Path(config_path).resolve(strict=True)
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    for key, expected in PROTOCOL.items():
        if key in cfg and cfg[key] != expected:
            raise ValueError(f"{key} must be {expected!r} for this model version")
    if cfg.get("device", "cuda") != "cuda":
        raise ValueError("Reference inference uses CUDA; use preview for CPU slide preparation")
    samples = cfg["samples"]
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", s["sample_id"]) for s in samples):
        raise ValueError("Sample IDs must be unique, safe directory names (letters, digits, dot, dash, underscore)")
    ids = [s["sample_id"].casefold() for s in samples]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample IDs")
    selected = [s for s in samples if s["sample_id"] == sample_id]
    if len(selected) != 1:
        raise ValueError("Sample ID not present in current configuration")
    source = (path.parent / selected[0]["he_path"]).resolve(strict=True)
    if not source.is_file():
        raise ValueError("Expected an OpenSlide-readable slide file")
    if source.suffix.lower() not in SINGLE_FILE_SUFFIXES:
        raise ValueError("This release accepts single-file SVS, calibrated tiled TIFF, NDPI or SCN. "
                         "Multi-file MRXS/VMS/VMU and SDK-dependent KFB are not supported.")
    # Resolve relative paths with respect to the configuration file, never the
    # process CWD. Do not serialize optional clinical metadata from user manifests.
    cfg = dict(PROTOCOL, samples=[{"sample_id": sample_id, "he_path": str(source)}],
               **{key: str((path.parent/cfg[key]).resolve()) for key in ("output_directory","weights_root","ctranspath_root")})
    out = Path(cfg["output_directory"])
    if (out/sample_id).is_symlink():
        raise ValueError("Case directory cannot be a symlink")
    if source.is_relative_to(out/sample_id):
        raise ValueError("Keep source WSI outside the case output directory")
    cfg["bcss_weights"] = str(Path(cfg["weights_root"])/"best_adapted.pth")
    cfg["ctranspath_weights"] = str(Path(cfg["weights_root"])/"ctranspath.pth")
    return cfg


def _encoder_tree(root):
    return Path(root) / "path2space/frozen/ctrans"


def _validate_encoder(root, expected):
    tree = _encoder_tree(root)
    for name, digest in expected.items():
        data = (tree/name).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"Path2Space encoder source mismatch: {name}")
    return tree


def encoder_class(root):
    """Load exactly the upstream encoder subpackage, bypassing RNA pipeline imports."""
    expected = json.loads(MODEL_LOCK.read_text(encoding="utf-8"))["encoder_source_lf_sha256"]
    tree = _validate_encoder(root, expected)
    key = "_henets_encoder_" + hashlib.sha256(str(tree.resolve()).encode()).hexdigest()
    with _IMPORT_LOCK:
        if key not in sys.modules:
            spec = importlib.util.spec_from_file_location(key, tree/"__init__.py", submodule_search_locations=[str(tree)])
            module = importlib.util.module_from_spec(spec)
            sys.modules[key] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                for name in tuple(sys.modules):
                    if name == key or name.startswith(key+"."):
                        sys.modules.pop(name)
                raise
        return sys.modules[key].CTransPath


def verify_models(cfg):
    """Hash externally mounted, unmodified artifacts before any deserialization."""
    locked = json.loads(MODEL_LOCK.read_text(encoding="utf-8"))
    root = Path(cfg["weights_root"])
    expected_paths = {"ctranspath.pth","best_adapted.pth"} | {
        f"{group}/fold{fold}_{suffix}.pt" for fold in range(9)
        for group,suffix in (("path2space_protein","heads"),("path2space_last_stage","models"))
    }
    if set(locked["weights"]) != expected_paths:
        raise ValueError("Incomplete model fingerprint inventory")
    _validate_encoder(cfg["ctranspath_root"], locked["encoder_source_lf_sha256"])
    actual = {name: sha256(root/name) for name in locked["weights"]}
    if actual != locked["weights"] and actual != locked.get("archived_weights"):
        raise ValueError("Weights do not match a complete HENETs release")
    return {**locked, "weights": actual}


@contextmanager
def _case_lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/".inference.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            if handle.seek(0, 2) == 0:
                handle.write(b"0"); handle.flush()
            handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        # OS closes/releases this lock even following an interrupted case.
        yield


def preview(config_path, sample_id):
    """Prepare only calibrated geometry, tissue preview and grid, without a GPU or weights."""
    from . import _engine
    cfg = load_config(config_path, sample_id)
    out = Path(cfg["output_directory"]); folder = out/sample_id
    with _case_lock(folder):
        _engine.prepare(cfg, out)
    return folder


def run(config_path, sample_id):
    """Infer one case; raw predictions/compartment mask/caches remain for downstream QC."""
    import cv2
    import openslide
    import torch
    from . import _engine
    cfg = load_config(config_path, sample_id)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the reference inference path. For CPU input inspection use --preview.")
    locked = verify_models(cfg)
    out = Path(cfg["output_directory"]); folder = out/sample_id
    torch.set_num_threads(4); cv2.setNumThreads(2)
    record = dict(schema_version=1, config=cfg, model_version=locked["model_version"],
                  weights=locked["weights"], encoder_source=locked["encoder_source_lf_sha256"],
                  method="RGB ImageNet scaling, no stain normalization; mean raw MPO and CitH3 across 9 folds x 5 seeds, shared 0-100 score.",
                  runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,opencv=cv2.__version__,
                               openslide=openslide.__version__, openslide_native=openslide.__library_version__,
                               device=torch.cuda.get_device_name()))
    with _case_lock(folder):
        info = folder/"inference_metadata.json"
        if info.exists():
            if json.loads(info.read_text(encoding="utf-8")) != record:
                raise ValueError("Model, settings or runtime changed; use a new output directory")
        elif any((folder/k).exists() for k in ("blocks","protein_blocks","regional_scores.csv")):
            raise ValueError("Unbound case output already exists")
        else:
            _engine.write_json(info, record)
        if (folder/"cleanup_receipt.json").exists():
            raise ValueError("This case was already finalized or cleaned; choose a new output directory for re-inference")
        _engine.prepare(cfg,out)
        _engine.segment(cfg,out)
        _engine.protein(cfg,out)
    return folder


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True,type=Path)
    parser.add_argument("--sample",required=True)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--check",action="store_true",help="Check mounted model/code fingerprints; no inference.")
    mode.add_argument("--preview",action="store_true",help="Build slide preview and grid only; no CUDA required.")
    args=parser.parse_args()
    if args.check:
        bundle=verify_models(load_config(args.config,args.sample))
        print(f"Verified {len(bundle['weights'])} model files and encoder source")
    else:
        print((preview if args.preview else run)(args.config,args.sample))
