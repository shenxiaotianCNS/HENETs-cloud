"""Optional patient-level, within-cancer RNA and survival analyses."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from contextlib import contextmanager
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from .tcga_analysis import bh_adjust, correlation, fit_cox, mcp_counter, read_star_tpm

KEYS = ["sample_id", "patient_id", "cancer_type"]
FEATURES = ["overall_median", "tumor_median", "stroma_median", "other_median", "tumor_minus_stroma"]
ENDPOINTS = ("OS", "PFI")
TERM = "KEGG_NET_FORMATION"
GENE_SHA = "9eae4b586d28a4b5e257ee82c8acd537314cb9b861611e9f5c211ee056e3352e"
MCP_SHA = "408f6c5d02c8f9bd2f1599c598367881853520934f8ce398a4886a8b296922bb"
GSEA = dict(sample_norm_method="rank", weight=.25, correl_norm_type="rank",
            min_size=10, max_size=2000, permutation_num=0, threads=1, seed=721)
MISSING = {"", "na", "nan", "n/a"}


def csv_table(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        names = next(reader, [])
        if len(set(names)) != len(names) or not set(required).issubset(names):
            raise ValueError(f"CSV requires unique headers including {required}")
        rows = list(reader)
    if not rows or any(len(r) != len(names) for r in rows):
        raise ValueError("Empty CSV or inconsistent field count")
    return pd.DataFrame(rows, columns=names)


def identity(frame):
    for col in (name for name in KEYS if name in frame):
        values = frame[col]
        if not values.str.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}").all():
            raise ValueError(f"{col} requires a nonempty pseudonymous code (ASCII letters/digits/_/-)")
        if col != "cancer_type" and values.str.casefold().duplicated().any():
            raise ValueError(f"Duplicate {col}. Supply one slide and one row per independent patient.")
        if col == "cancer_type" and values.nunique() != values.str.casefold().nunique():
            raise ValueError("Inconsistent capitalization for cancer_type")


def number(series):
    clean = series.str.strip()
    result = pd.to_numeric(clean.mask(clean.str.lower().isin(MISSING)), errors="raise").astype(float)
    if np.isinf(result).any():
        raise ValueError(f"Non-finite value in {series.name}")
    return result


@contextmanager
def publication(out):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(f"Choose a new output directory: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".henets-analysis-", dir=out.parent) as temp:
        stage = Path(temp) / "result"
        stage.mkdir()
        yield stage
        stage.rename(out)


def write_meta(stage, meta):
    files = sorted(stage.glob("*.csv"))
    meta["output_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (stage / "analysis_metadata.json").write_text(json.dumps(meta, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def planned_scores(frame):
    return [c for c in frame if c == "rna_nets_score" or c.startswith("mcp_")]


def run(features, patients, out, rna_scores=None, scores=None, bootstraps=2000):
    if not 100 <= bootstraps <= 2000:
        raise ValueError("--bootstraps must be in [100,2000]")
    image = csv_table(features, ["sample_id", *FEATURES])
    people = csv_table(patients, KEYS)
    identity(image[["sample_id"]])
    identity(people)
    if set(image.sample_id) != set(people.sample_id):
        raise ValueError("Patients and HE features must cover exactly the same sample_id set")
    if set(FEATURES) & set(people):
        raise ValueError("Patient input may not override image features")
    if rna_scores:
        rna = csv_table(rna_scores, KEYS)
        identity(rna)
        keys = people[KEYS].merge(rna[KEYS], on=KEYS, how="right", indicator=True)
        if not keys._merge.eq("both").all():
            raise ValueError("RNA sample/patient/cancer identities do not match the patient table")
        molecular = planned_scores(rna) if scores is None else [s for s in scores if s in rna]
        if set(molecular) & set(people):
            raise ValueError("Ambiguous molecular column present in both RNA and patient inputs")
        people = people.merge(rna[KEYS + molecular], on=KEYS, how="left", validate="one_to_one").fillna("")
    scores = list(scores) if scores is not None else planned_scores(people)
    reserved = set(KEYS + FEATURES + [f"{e}_{f}" for e in ENDPOINTS for f in ("days","event")])
    if (len(set(scores)) != len(scores) or set(scores) & reserved or not set(scores).issubset(people)
            or any(not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]{0,79}", s) for s in scores)):
        raise ValueError("Choose distinct numeric molecular score columns present in the input")
    frame = people[KEYS].merge(image[["sample_id", *FEATURES]], on="sample_id", validate="one_to_one", how="left")
    for feature in FEATURES:
        frame[feature] = number(frame[feature])
    for score in scores:
        frame[score] = number(people[score])
    for e in ENDPOINTS:
        d, v = e+"_days", e+"_event"
        if (d in people) != (v in people):
            raise ValueError(f"Supply {d} and {v} as a pair of columns")
        frame[d] = number(people[d]) if d in people else np.nan
        frame[v] = number(people[v]) if v in people else np.nan
        if not frame[v].dropna().isin([0, 1]).all():
            raise ValueError(f"{v} must be 0 or 1 or blank")
        reason = np.select([frame[[d,v]].isna().all(axis=1), frame[[d,v]].isna().any(axis=1), frame[d] <= 0],
                           ["missing_pair", "incomplete_pair", "nonpositive_days"], default="eligible")
        frame[e+"_status"] = reason
        frame.loc[reason != "eligible", [d,v]] = np.nan
    corr_rows, surv_rows, missing = [], [], []
    for cancer, group in frame.groupby("cancer_type", sort=True):
        local = []
        for feature in FEATURES:
            for score in scores:
                local.append(dict(cancer_type=cancer, image_feature=feature, molecular_score=score,
                                  **correlation(group[feature], group[score], n_boot=bootstraps)))
        for row, q in zip(local, bh_adjust([r["p_value"] for r in local], family_size=len(local))):
            row["bh_q_within_cancer"] = q
        corr_rows.extend(local)
        for col in FEATURES + scores:
            missing.append(dict(cancer_type=cancer, variable=col, patients=len(group), eligible=int(group[col].notna().sum())))
        for e in ENDPOINTS:
            obj, _ = fit_cox(group, "overall_median", e)
            estimates = ["hr_per_sd","ci_low","ci_high","p_value"]
            if obj["status"] == "ok" and (not np.isfinite([obj[k] for k in estimates]).all() or min(obj[k] for k in estimates[:3]) <= 0):
                obj.update(status="failed_nonfinite_estimate", **dict.fromkeys(estimates, np.nan))
            surv_rows.append(dict(cancer_type=cancer, **obj))
            missing.append(dict(cancer_type=cancer, variable=e, patients=len(group), eligible=int(group[e+"_status"].eq("eligible").sum())))
    corr_cols = ["cancer_type","image_feature","molecular_score","n","rho","p_value","ci_low","ci_high","bootstrap_valid","status","bh_q_within_cancer"]
    with publication(out) as dest:
        frame.to_csv(dest/"patients.csv", index=False)
        pd.DataFrame(corr_rows, columns=corr_cols).to_csv(dest/"correlations.csv", index=False)
        pd.DataFrame(surv_rows).to_csv(dest/"survival_cox.csv", index=False)
        pd.DataFrame(missing).to_csv(dest/"missingness.csv", index=False)
        inputs = dict(features=features, patients=patients)
        if rna_scores:
            inputs["rna_scores"] = rna_scores
        write_meta(dest, dict(method="Exploratory within-cancer patient-level associations", bootstrap_replicates=bootstraps,
                              seed=721, image_features=FEATURES, molecular_scores=scores,
                              patients_per_cancer={k:int(v) for k,v in frame.cancer_type.value_counts().items()},
                              input_sha256={k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in inputs.items()}))


def score_rna(manifest, gene_sets, markers, out):
    import gseapy
    if gseapy.__version__ != "1.3.2":
        raise ValueError("This scoring implementation requires gseapy==1.3.2")
    manifest = Path(manifest)
    cohort = csv_table(manifest, ["sample_id","patient_id","rna_path"])
    if "cancer_type" not in cohort:
        if "project_id" not in cohort:
            raise ValueError("RNA manifest requires cancer_type or a GDC project_id")
        cohort["cancer_type"] = cohort.project_id
    identity(cohort)
    refs = [Path(p).read_bytes() for p in (gene_sets, markers)]
    if [hashlib.sha256(x).hexdigest() for x in refs] != [GENE_SHA, MCP_SHA]:
        raise ValueError("RNA reference SHA-256 mismatch (see ASSOCIATIONS.md)")
    sets = json.loads(refs[0])["gene_sets"]
    marker_table = pd.read_csv(io.BytesIO(refs[1]), sep="\t")
    rows, audits, seen, coverage_rows = [], [], set(), []
    for cancer, group in cohort.groupby("cancer_type", sort=True):
        universe = None
        for start in range(0,len(group),8):
            batch = {}
            for row in group.iloc[start:start+8].itertuples(index=False):
                if not row.rna_path.strip():
                    raise ValueError(f"Missing RNA path for {row.sample_id}")
                raw = (manifest.parent / row.rna_path).read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                if "rna_md5" in cohort and hashlib.md5(raw).hexdigest() != row.rna_md5.lower():
                    raise ValueError(f"GDC RNA MD5 mismatch for {row.sample_id}")
                if digest in seen or ("rna_sha256" in cohort and digest != row.rna_sha256.lower()):
                    raise ValueError(f"Duplicate RNA file or input checksum mismatch for {row.sample_id}")
                seen.add(digest)
                values, _ = read_star_tpm(io.BytesIO(raw))
                if universe is None:
                    universe = values.index
                    if not set(sets[TERM]) <= set(universe):
                        raise ValueError(f"{cancer}: missing targets from locked NET gene-set mapping")
                elif not values.index.equals(universe):
                    raise ValueError(f"{cancer}: inconsistent STAR Ensembl universe/order")
                batch[row.sample_id] = values
                audits.append(dict(cancer_type=cancer,sample_id=row.sample_id,patient_id=row.patient_id,
                                   rna_sha256=digest,ensembl_gene_count=len(values)))
            expr = pd.DataFrame(batch)
            result = gseapy.ssgsea(data=expr, gene_sets=sets, outdir=None, no_plot=True, **GSEA).res2d
            es = result.loc[result.Term == TERM].set_index("Name").ES.astype(float)
            if es.index.has_duplicates or set(es.index) != set(batch) or not np.isfinite(es).all():
                raise ValueError("Expected one finite raw ssGSEA ES per sample")
            immune, coverage = mcp_counter(np.log2(expr+1), marker_table)
            if len(coverage) != 10 or (coverage.coverage < .5).any() or not np.isfinite(immune).all().all():
                raise ValueError(f"{cancer}: insufficient MCP-counter marker coverage")
            if start == 0:
                coverage_rows.append(coverage.assign(cancer_type=cancer))
            rows.append(immune.join(es.rename("rna_nets_score"), validate="one_to_one").rename_axis("sample_id").reset_index())
    merged = cohort[KEYS].merge(pd.concat(rows,ignore_index=True), on="sample_id",validate="one_to_one")
    with publication(out) as dest:
        merged.to_csv(dest/"rna_scores.csv",index=False)
        pd.DataFrame(audits).to_csv(dest/"rna_input_audit.csv",index=False)
        pd.concat(coverage_rows,ignore_index=True).to_csv(dest/"immune_marker_coverage.csv",index=False)
        write_meta(dest, dict(method="STAR unstranded TPM, KEGG NET formation raw ssGSEA ES and MCP-counter",
                              gseapy_version=gseapy.__version__, ssGSEA=GSEA, batch_size=8, net_ensembl_targets=len(sets[TERM]),
                              input_sha256=dict(manifest=hashlib.sha256(manifest.read_bytes()).hexdigest(),gene_sets=GENE_SHA,markers=MCP_SHA)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    analysis = sub.add_parser("run",help="Within-cancer Spearman and unadjusted OS/PFI Cox models")
    for key in ("features","patients","out"):
        analysis.add_argument("--"+key,type=Path,required=True)
    analysis.add_argument("--rna-scores",type=Path)
    analysis.add_argument("--scores", nargs="+",help="Molecular columns; defaults to rna_nets_score and mcp_*")
    analysis.add_argument("--bootstraps",type=int,default=2000)
    rna = sub.add_parser("rna",help="Score GDC-style STAR gene-count TSVs")
    for key in ("manifest","gene-sets","markers","out"):
        rna.add_argument("--"+key,type=Path,required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    (run if command == "run" else score_rna)(**args)


if __name__ == "__main__":
    main()
