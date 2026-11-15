# GPU runtime and model mounts

The inference engine processes a calibrated H&E slide without cancer-specific inputs. Acceptance by the file reader does not establish predictive accuracy in the submitted tissue. This release retains the trained ICC protein and compartment models; it has no separate model for each cancer.

Build the Dockerfile **in the exported repository**. It installs a stable PyTorch 2.7.1 / torchvision 0.22.1 CUDA 12.8 pair through the PyTorch base image and a pinned inference/export stack. Use an NVIDIA host driver compatible with this CUDA runtime and make the GPU visible to Docker (for example, `--gpus all`). CPU-only preview is supported. Full scoring uses CUDA and mixed precision; no automatic CPU fallback or reproducibility claim across GPU hardware or PyTorch versions is made. The originating validation used a different PyTorch build. Use a small known slide to compare scores and masks before a new deployment.

For a source installation, install the matching PyTorch/torchvision wheels for your GPU from the [PyTorch installation guidance](https://pytorch.org/get-started/previous-versions/), then `pip install -r requirements-inference.txt`. Linux x86_64 and a local POSIX-style filesystem are intended for Docker; an object-store URL is not a filesystem path. Each worker may serve different slides under the same output directory. Budget for one working context plus weights per GPU worker: an equipment-specific VRAM threshold is not guaranteed. The frozen BCSS context is intentionally not split into smaller contexts on out-of-memory errors.

Mount the model bundle from the [public assets release](https://github.com/shenxiaotianCNS/HENETs-models/releases/tag/assets-v0.1.1) as a **read-only** directory:

```text
weights/
  ctranspath.pth
  best_adapted.pth
  path2space_protein/fold0_heads.pt ... fold8_heads.pt
  path2space_last_stage/fold0_models.pt ... fold8_models.pt
encoder/
  path2space/frozen/ctrans/__init__.py
  path2space/frozen/ctrans/ctranspath.py
  path2space/frozen/ctrans/swin_transformer.py
```

The public bundle omits training split identifiers. The loader also accepts the complete original v0.1.0 bundle and records the fingerprints actually used. The model and encoder source fingerprints are in `henets/model_lock.json`. The encoder directory is the `ge_model` directory from [Path2Space companion, pinned revision](https://github.com/eldadshulman/path2space-companion/tree/dd69f503e0f290ff4c5628ce2f723cbd7c638808/ge_model). Only the indicated encoder subpackage is needed; the Path2Space gene-expression model, its weights, and its normalization pipeline are not used. The external code is trusted executable Python; mount it from the expected source, not a user-uploadable directory. Source fingerprints ignore CRLF vs LF; model fingerprints use exact file bytes. A generic ImageNet model or unadapted public BCSS weights will fail the checks. No weight or encoder download is attempted.

Given a JSON config with `samples` (each with `sample_id` and `he_path`), `output_directory`, `weights_root`, and `ctranspath_root`:

```bash
python -m henets.infer --config config.json --sample example --check
python -m henets.infer --config config.json --sample example --preview
```

`--check` verifies the model mounts. `--preview` uses no weights to inspect the OpenSlide header, record geometry, generate the tissue-selection grid and export previews. Use a **separate scratch output directory** for this low-level preview, or use the normal cohort CLI from start to finish. This release accepts only **single-file** SVS, calibrated tiled TIFF (.tif/.tiff), NDPI, or SCN that the installed OpenSlide vendor reader actually decodes. It requires explicit numeric X/Y microns per pixel strictly between 0.1 and 1 and tissue windows on the frozen grid. A matching extension alone is insufficient. MRXS/VMS/VMU multi-file slides are rejected because their sidecars are not bound by the single-file input fingerprint. Ordinary raster images and proprietary KFB (whose Windows SDK is not part of the Linux port) are not accepted. Newly supported vendor formats need geometry/color and model-quality checks in your own cohort.

`henets.infer.run(config_path, sample_id)` returns the case directory. Inputs stay intact; only per-case inference outputs are written. `regional_scores.csv` contains all proposed windows **before QC**. Use the end-to-end cohort CLI for QC-filtered patient features and spatial reports. Reusing a working directory requires matching settings, source, encoder, weights, runtime and GPU model. Do not share one case across concurrent workers.
