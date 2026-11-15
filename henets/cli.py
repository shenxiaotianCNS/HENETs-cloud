"""Run one H&E slide or a manifest of slides, optionally distributed by sample."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from .infer import PROTOCOL as DEFAULTS


CACHE_FILES = {"blocks": re.compile(r"x\d+_y\d+(?:\.json|_labels\.png|_confidence\.png|_crops\.npz)"),
               "protein_blocks": re.compile(r"x\d+_y\d+\.json")}


def write_json(path, value):
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent,
                                     prefix=".henets-", delete=False) as tmp:
        name = Path(tmp.name)
        try:
            json.dump(value, tmp, indent=2, ensure_ascii=False, allow_nan=False)
            tmp.write("\n")
        except BaseException:
            tmp.close()
            name.unlink(missing_ok=True)
            raise
    try:
        os.replace(name, path)
    finally:
        name.unlink(missing_ok=True)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def load_samples(manifest=None, slide=None, sample_id=None):
    if manifest:
        with manifest.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"sample_id", "he_path"} <= set(reader.fieldnames or ()):
                raise ValueError("CSV requires sample_id and he_path")
            rows = list(reader)
        root = manifest.resolve().parent
    else:
        if not sample_id:
            raise ValueError("Single-slide mode requires --sample-id")
        rows = [dict(sample_id=sample_id, he_path=str(slide))]
        root = Path.cwd()
    samples, ids, slides = [], set(), set()
    for row in rows:
        sid = row["sample_id"]
        if (not sid or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", sid)
                or sid.casefold() in ids or sid.casefold() in {"report","runs","weights"}):
            raise ValueError(f"Unsafe, reserved or duplicate sample ID: {sid!r}")
        if not row.get("he_path"):
            raise ValueError(f"Missing slide path for {sid}")
        path = (root / row["he_path"]).resolve()
        if path in slides:
            raise ValueError(f"Slide is listed twice: {path}")
        item = dict(sample_id=sid, he_path=str(path))
        if row.get("cancer_type"):
            item["cancer_type"] = row["cancer_type"]
        ids.add(sid.casefold())
        slides.add(path)
        samples.append(item)
    if not samples:
        raise ValueError("Empty slide manifest")
    return samples


def select(samples, sample, shard):
    if sample:
        rows = [row for row in samples if row["sample_id"] == sample]
        if not rows:
            raise ValueError(f"Unknown sample: {sample}")
        return rows
    if shard:
        match = re.fullmatch(r"(\d+)/(\d+)", shard)
        if not match:
            raise ValueError("--shard expects a zero-based INDEX/COUNT")
        index, count = map(int, match.groups())
        if not 0 <= index < count:
            raise ValueError("Require 0 <= INDEX < COUNT")
        return samples[index::count]
    return samples


@contextmanager
def exclusive(path):
    # One OS-held lock per sample. Concurrent jobs for distinct samples can proceed.
    with path.open("a+b") as fp:
        if os.name == "nt":
            import msvcrt
            if fp.seek(0, 2) == 0:
                fp.write(b"0")
                fp.flush()
            fp.seek(0)
            msvcrt.locking(fp.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def clear_verified_cache(folder):
    files, directories = [], []
    for name, pattern in CACHE_FILES.items():
        directory = folder / name
        if directory.is_symlink():
            raise ValueError(f"Unexpected linked cache: {directory}")
        if not directory.exists():
            continue
        if not directory.is_dir() or directory.resolve().parent != folder.resolve():
            raise ValueError(f"Invalid cache directory: {directory}")
        directories.append(directory)
        for path in directory.iterdir():
            if path.is_symlink() or not path.is_file() or not pattern.fullmatch(path.name):
                raise ValueError(f"Unexpected cache entry: {path}")
            files.append(path)
    for path in files:
        path.unlink()
    for directory in directories:
        directory.rmdir()


def process(row, args, code_id):
    from . import export, infer
    slide = Path(row["he_path"])
    if not slide.is_file():
        raise FileNotFoundError(slide)
    out = args.out.resolve()
    folder = out / row["sample_id"]
    if folder.is_symlink():
        raise ValueError(f"Linked case directory: {folder}")
    folder.mkdir(exist_ok=True)
    config = dict(DEFAULTS, samples=[row], output_directory=str(out),
                  weights_root=str(args.weights.resolve()), ctranspath_root=str(args.encoder.resolve()))
    with exclusive(folder / ".case.lock"):
        job, bound = folder / "job.json", folder / "input_binding.json"
        binding = dict(slide_sha256=digest(slide), implementation_sha256=code_id, config=config)
        if bound.is_file():
            if json.loads(bound.read_text(encoding="utf-8")) != binding:
                raise ValueError(f"Inputs or code changed for {row['sample_id']}. Use a separate output directory.")
        else:
            if any(p.name != ".case.lock" and not p.name.startswith(".henets-") for p in folder.iterdir()):
                raise ValueError(f"Unbound output already exists in {folder}")
            write_json(bound, binding)
        write_json(job, config)
        if not (folder / "export/export_manifest.json").is_file():
            infer.run(job, row["sample_id"])
            export.export_case(config, row["sample_id"])
        export.verify_case(config, row["sample_id"])
        if not args.keep_cache:
            clear_verified_cache(folder)
        print(f"Completed {row['sample_id']}: {folder / 'export/index.html'}", flush=True)


def code_binding():
    package = Path(__file__).parent
    h = hashlib.sha256()
    for path in sorted([*package.glob("*.py"), package / "model_lock.json"]):
        h.update(path.name.encode() + b"\0" + path.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for command in ("run", "report"):
        p = actions.add_parser(command)
        inputs = p.add_mutually_exclusive_group(required=True)
        inputs.add_argument("--manifest", type=Path)
        inputs.add_argument("--slide", type=Path)
        p.add_argument("--sample-id", help="Anonymous identifier in single-slide mode")
        p.add_argument("--out", type=Path, required=True)
        if command == "run":
            p.add_argument("--weights", type=Path, required=True)
            p.add_argument("--encoder", type=Path, required=True)
            subset = p.add_mutually_exclusive_group()
            subset.add_argument("--sample")
            subset.add_argument("--shard", metavar="INDEX/COUNT")
            p.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args()
    if args.manifest and args.sample_id:
        parser.error("--sample-id is for --slide")
    samples = load_samples(args.manifest, args.slide, args.sample_id)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.action == "run":
        signature = code_binding()
        for row in select(samples, args.sample, args.shard):
            process(row, args, signature)
        if args.sample or args.shard:
            return
    from .export import export_index
    with exclusive(args.out / ".report.lock"):
        export_index(dict(DEFAULTS, samples=samples, output_directory=str(args.out.resolve())))
    print("Cohort report:", args.out.resolve() / "report/index.html")


if __name__ == "__main__":
    main()
