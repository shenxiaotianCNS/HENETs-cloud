# Prepare paired inputs from the Genomic Data Commons

Run these commands from the package root on a cloud machine with Python 3.10+ and HTTPS access to `api.gdc.cancer.gov`. The GDC helper uses the Python standard library and does not require a token, GPU or model weights. It queries only public, released files. It does not upload anything to GDC.

```bash
# Metadata only. The destination must be new. Change the TCGA project ID as needed.
python -m henets.gdc prepare --project TCGA-LIHC --out datasets/lihc

# Read manifest.csv and snapshot.json before committing storage. The latter lists the selected bytes per file type.
# Download only the H&E for the first matched participant, preserving raw/<kind>/<uuid>/<file>.
sample=$(python -c "import csv; print(next(csv.DictReader(open('datasets/lihc/manifest.csv',encoding='utf-8')))['sample_id'])")
python -m henets.gdc download --manifest datasets/lihc/manifest.csv --sample "$sample" --kind he

# Download the matching molecular and clinical files for the same participant.
python -m henets.gdc download --manifest datasets/lihc/manifest.csv --sample "$sample" --kind rna --kind clinical

# Optional: omit --sample to fetch all matched patients. This can require substantial disk space.
# python -m henets.gdc download --manifest datasets/lihc/manifest.csv
```

`manifest.csv` supplies `sample_id` and `he_path` for the image runner. Paths are relative to the manifest directory. To process one downloaded slide, use the image runner's `--manifest datasets/lihc/manifest.csv --sample "$sample"` options. Omit `--sample` after downloading the full image cohort.

Each row represents one patient with one primary-tumor diagnostic SVS, one primary-tumor STAR–Counts TSV and one BCR clinical XML. Case UUID, patient barcode and TCGA project must agree. For slide and RNA we require associated sample metadata of type `Primary Tumor`; records with mixed sample types are excluded. This rule may yield few or no matches in cohorts dominated by metastatic or blood-derived samples. The tool does not filter by histology or anatomy; a project ID alone does **not** establish that a patient meets a particular histologic subtype definition.

If several files qualify, the tool chooses one deterministically by specimen barcode(s), filename, then file UUID. Each chosen file retains its source URI, size, MD5, specimen barcode(s) and eligible file count. Different slide and RNA barcodes identify different specimens from the same patient. Even identical barcodes do not establish spatial correspondence. `excluded.csv` reports which kinds were missing for each excluded patient; cases with no eligible triple are never silently turned into H&E-only cases. `gdc_cases.json` and `gdc_files.json` record the queried metadata. `snapshot.json` records query filters, selection rules, counts, time and SHA-256 hashes; preparation refuses to overwrite an existing snapshot.

Downloads resume interrupted `.partial` files only when the GDC server returns a matching `Content-Range`. An OS file lock prevents concurrent writers for the same UUID within one snapshot. Each completed or previously downloaded file is checked against the manifest size and MD5. A corrupt existing original is left untouched; a failed complete partial is retained for inspection. The helper does not bypass the GDC API or use controlled data. Configure a trusted CA bundle with `SSL_CERT_FILE` if the system trust store is unavailable; do not disable TLS verification. Use storage that supports advisory locks and same-volume atomic file publication; on POSIX this also requires hard-link support. For object stores, stage on a local or block filesystem first.

The downloader provides public files and a patient-matched manifest for inference. Measured RNA represents bulk tissue. Define histology eligibility, time origin and censoring for the project before building the endpoint table from curated follow-up data. Clinical XML alone does not supply a harmonized OS/PFI table. For RNA and immune scoring and patient-level correlations or survival analyses, see [ASSOCIATIONS.md](ASSOCIATIONS.md).

Official sources: [GDC search, filters and pagination](https://docs.gdc.cancer.gov/API/Users_Guide/Search_and_Retrieval/), [GDC download endpoint and UUIDs](https://docs.gdc.cancer.gov/API/Users_Guide/Downloading_Files/).
