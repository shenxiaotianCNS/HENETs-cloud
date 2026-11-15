"""Export a spatial protein-score grid and compartment summaries from HENETs inference.

Input: one inference folder per pseudonymous sample. No source slide paths,
scanner headers, raw label/macro images, or expression/clinical data are exported.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from henets.external_he import black_line_qc, describe, eligible_compartment
from henets.tiled_tiff import TiledTiffReader


CLASSES = ("tumor", "stroma", "other")
ELIGIBLE = (*CLASSES, "mixed")
CATEGORIES = ("unprocessed", *ELIGIBLE, "low_tissue", "artifact_review")
SPACING = 512
CENTER = 224
MPP = .325
THRESHOLD = 20
LABELS = {0:"background", 1:"tumor", 2:"stroma", 3:"other", 255:"unprocessed"}
PALETTE = {0:(255,255,255), 1:(225,64,75), 2:(40,170,125), 3:(105,125,185), 255:(220,220,220)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_id(value):
    require(isinstance(value,str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}",value)
            and not value.endswith("."), "Use a path-safe pseudonymous sample_id")
    return value


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=tuple(rows[0]),lineterminator="\n")
        w.writeheader();w.writerows(rows)


def json_file(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n",encoding="utf-8")


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def check_config(config):
    require(config.get("analysis_mpp")==MPP and config.get("patch_size")==SPACING and config.get("center_size")==CENTER
            and config.get("minimum_center_tissue_fraction")==.5 and config.get("compartment_purity")==.6,
            "Unsupported geometry or HE eligibility thresholds for this export version")
    ids=[safe_id(s["sample_id"]) for s in config["samples"]]
    require(bool(ids) and len(set(s.casefold() for s in ids))==len(ids),"Empty or ambiguous sample IDs")
    return ids


def center_qc(folder):
    result={}
    for path in sorted((folder/"blocks").glob("*_crops.npz")):
        with np.load(path,allow_pickle=False) as data:
            ids,rgb=data["ids"],data["rgb"]
            require(ids.ndim==1 and rgb.dtype==np.uint8 and rgb.shape==(len(ids),CENTER,CENTER,3),"Invalid center RGB cache")
            for pid,center in zip(ids,rgb):
                key=str(pid)
                require(key not in result,"Duplicate cached center")
                result[key]=black_line_qc(center)
    require(bool(result),"No RGB center caches for initial QC export")
    return result


def prepared_rows(folder, sid, cfg):
    info=json.loads((folder/"slide.json").read_text(encoding="utf-8"))
    require(info.get("sample_id")==sid,"Inference sample differs from selected pseudonym")
    h,w=(int(v) for v in info["analysis_shape"])
    transform=np.asarray(info["he_to_analysis"],dtype=float)
    require(transform.shape==(3,3) and np.isfinite(transform).all(),"Invalid slide coordinate transform")
    inverse=np.linalg.inv(transform)
    scores=read_csv(folder/"regional_scores.csv"); grid=read_csv(folder/"grid.csv");qc=center_qc(folder)
    key=lambda r:(r["patch_id"],int(r["analysis_x"]),int(r["analysis_y"]))
    require(len(scores)==len(qc)==len(grid)>0 and {key(r) for r in scores}=={key(r) for r in grid}
            and set(qc)=={r["patch_id"] for r in scores} and all(r["sample_id"]==sid for r in grid),"Incomplete/misaligned inference grid")
    rows=[];locations=set()
    for r in scores:
        pid,x,y=key(r)
        require(r["sample_id"]==sid and pid==f"{sid}_x{x}_y{y}" and x>=0 and y>=0 and
                x%SPACING==0 and y%SPACING==0 and x+SPACING<=w and y+SPACING<=h and (x,y) not in locations,"Invalid analysis coordinate")
        locations.add((x,y))
        for prefix,n in (("",SPACING),("center224_",CENTER)):
            counts=[int(r[prefix+k+"_pixels"]) for k in CLASSES]
            tissue=int(r[prefix+"tissue_pixels"]); bg=int(r[prefix+"background_pixels"])
            require(min(counts+[bg])>=0 and sum(counts)==tissue and tissue+bg==n*n and int(r[prefix+"unprocessed_pixels"])==0,
                    "Invalid processed mask pixel counts")
            require(abs(float(r[prefix+"purity"])-max(counts)/max(tissue,1))<1e-7 and
                    r[prefix+"dominant_compartment"]==(CLASSES[int(np.argmax(counts))] if tissue else "background"),"Invalid mask class or purity")
        mpo,cit,score,sd=[float(r[k]) for k in ("predicted_MPO","predicted_CitH3","HE_NETs_score","score_fold_sd")]
        expected=100*math.sqrt(min(max(mpo,0),1)*min(max(cit,0),1))
        require(all(math.isfinite(v) for v in (mpo,cit,score,sd)) and sd>=0 and 0<=score<=100 and abs(expected-score)<1e-4,
                "Invalid protein prediction or HENETs score")
        qc_entry=qc[pid]
        cat="artifact_review" if qc_entry["artifact_review"] else eligible_compartment(r,cfg["minimum_center_tissue_fraction"],cfg["compartment_purity"])
        require(cat in CATEGORIES[1:],"Invalid comparison compartment")
        native=inverse@np.array([x+256,y+256,1.])
        require(np.isfinite(native).all() and native[2]!=0,"Invalid native HE coordinate")
        row=dict(sample_id=sid,patch_id=pid,analysis_x=x,analysis_y=y,center_left_x=x+144,center_top_y=y+144,
                 native_center_x=float(native[0]/native[2]),native_center_y=float(native[1]/native[2]),
                 comparison_compartment=cat,eligible=cat in ELIGIBLE,predicted_MPO=mpo,predicted_CitH3=cit,
                 HE_NETs_score=score,score_fold_sd=sd,center_tissue_fraction=tissue/CENTER**2,
                 center_purity=float(r["center224_purity"]),center_dominant_compartment=r["center224_dominant_compartment"],
                 center_tumor_pixels=counts[0],center_stroma_pixels=counts[1],center_other_pixels=counts[2],
                 center_background_pixels=bg,**qc_entry)
        rows.append(row)
    rows.sort(key=lambda row:(row["analysis_y"],row["analysis_x"]))
    # Independent final-TIFF/cached-label comparison at up to 12 fixed spatial positions.
    with TiledTiffReader(folder/"compartments.tif") as reader:
        require((reader.height,reader.width)==(h,w),"Compartment mask dimensions differ")
        md=json.loads(reader.page.description)
        require(md.get("coordinate_system")=="HE_resampled_0.325um" and md.get("mpp")==MPP and
                np.allclose(md["he_to_analysis"],transform) and md.get("classes")=={str(k):v for k,v in LABELS.items()},"Invalid TIFF label or coordinate metadata")
        checks=np.linspace(0,len(rows)-1,min(12,len(rows)),dtype=int)
        core=int(cfg["core_size"])
        for i in checks:
            r=rows[int(i)];x,y=r["analysis_x"],r["analysis_y"];bx=x//core*core;by=y//core*core
            with Image.open(folder/"blocks"/f"x{bx}_y{by}_labels.png") as im:
                tile=np.asarray(im.crop((x-bx,y-by,x-bx+512,y-by+512)))
            roi=reader.read_region(x,y,512,512)
            require(np.array_equal(tile,roi),"TIFF ROI does not match inference labels")
            center=roi[144:368,144:368]
            require(all(np.count_nonzero(center==k)==r["center_"+LABELS[k]+"_pixels"] for k in range(4)),
                    "Final TIFF does not match center class counts")
    return rows,(h,w),transform,len(checks)


def summarize(sid, rows):
    eligible=[r for r in rows if r["eligible"]]
    overall=describe([r["HE_NETs_score"] for r in eligible])
    patient=dict(sample_id=sid,total_windows=len(rows),eligible_windows=len(eligible),overall_median=overall["median"],
                 overall_mean=overall["mean"],low_tissue_windows=sum(r["comparison_compartment"]=="low_tissue" for r in rows),
                 artifact_review_windows=sum(r["comparison_compartment"]=="artifact_review" for r in rows))
    compartments=[]
    for c in ELIGIBLE:
        vals=[r["HE_NETs_score"] for r in eligible if r["comparison_compartment"]==c]
        stats=describe(vals);usable=len(vals)>=THRESHOLD
        patient[c+"_windows"]=len(vals);patient[c+"_median"]=stats["median"] if usable else None
        compartments.append(dict(sample_id=sid,compartment=c,**stats,usable_patient_median=usable,
                                 patient_level_median=patient[c+"_median"]))
    tm,sm=patient["tumor_median"],patient["stroma_median"]
    patient["tumor_minus_stroma"]=tm-sm if tm is not None and sm is not None else None
    return patient,compartments


def emit_grid(directory, folder, rows, shape, transform):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    h,w=shape
    gridshape=(h//SPACING,w//SPACING)
    raw=np.full(gridshape,np.nan,np.float32);category=np.zeros(gridshape,np.uint8)
    for row in rows:
        i,j=row["analysis_y"]//SPACING,row["analysis_x"]//SPACING
        raw[i,j]=row["HE_NETs_score"];category[i,j]=CATEGORIES.index(row["comparison_compartment"])
    valid=(category>=1)&(category<=4)
    eligible=np.where(valid,raw,np.nan)
    np.savez_compressed(directory/"score_grid.npz",raw_score=raw,eligible_score=eligible,category=category,
                        stride_px=SPACING,center_size_px=CENTER,center_offset_px=144,analysis_mpp=MPP,
                        analysis_shape_yx=np.asarray(shape),native_to_analysis=transform)
    # Preview labels have native-to-analysis coordinates and a fixed 32x downsample.
    with Image.open(folder/"compartments_preview.png") as im:
        labels=np.asarray(im)
    require(labels.shape==((h+31)//32,(w+31)//32) and set(np.unique(labels)).issubset(LABELS),"Invalid compartment preview")
    rgb=np.empty((*labels.shape,3),np.uint8)
    for label,color in PALETTE.items():
        rgb[labels==label]=color
    # Render only tissue-pixel locations sampled in an eligible window; labels do
    # not enable a pixel-level NET measurement. Keep full native canvas extents.
    heat=np.full(labels.shape,np.nan,np.float32)
    for row in rows:
        if row["eligible"]:
            i,j=row["analysis_y"]//32,row["analysis_x"]//32
            heat[i:i+16,j:j+16]=row["HE_NETs_score"]
    heat[~np.isin(labels,[1,2,3])]=np.nan
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout="constrained")
    try:
        axes[0].imshow(rgb,interpolation="nearest")
        axes[0].legend(handles=[Patch(color=np.array(PALETTE[i])/255,label=LABELS[i]) for i in (1,2,3,0,255)],ncol=3,fontsize=7,loc="lower center")
        axes[0].set_title("Predicted tissue compartments")
        axes[1].set_facecolor("white")
        plot=axes[1].imshow(np.ma.masked_invalid(heat),interpolation="nearest",cmap="viridis",vmin=0,vmax=100)
        axes[1].set_title("Eligible centers: HENETs score")
        for ax in axes:ax.set_axis_off()
        fig.colorbar(plot,ax=axes[1],shrink=.75,label="Predicted score (0–100)")
        fig.savefig(directory/"spatial_fixed_0_100.png",dpi=150,facecolor="white")
    finally:
        plt.close(fig)


def number(value):
    return "—" if value in (None,"") else f"{float(value):.2f}"


def case_html(sid,patient,comps):
    trs="".join(f'<tr><td>{r["compartment"]}</td><td>{r["n"]}</td><td>{number(r["median"])}</td><td>{number(r["patient_level_median"])}</td></tr>' for r in comps)
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>{html.escape(sid)} | HENETs</title>
<style>body{{max-width:1000px;margin:2rem auto;padding:1rem;font:16px/1.5 system-ui}}img{{max-width:100%}}td,th{{text-align:left;padding:.4rem;border-bottom:1px solid #ccc}}</style>
<h1>{html.escape(sid)} · Spatial HENETs</h1><p>Whole-slide median: {number(patient['overall_median'])}; {patient['eligible_windows']:,} eligible of {patient['total_windows']:,} sampled windows. The whole-slide median includes mixed centers. Low tissue: {patient['low_tissue_windows']:,}; flagged scan lines: {patient['artifact_review_windows']:,}.</p>
<img src="spatial_fixed_0_100.png" alt="Compartment labels and fixed 0–100 HENETs score map">
<table><tr><th>Compartment</th><th>Eligible centers</th><th>Descriptive median</th><th>Case-level median (≥20 centers)</th></tr>{trs}</table>
<p>Each score estimates MPO/CitH3-associated signal from one 72.8 µm center at a 166.4 µm sampling interval. The map uses the same 0–100 range across slides. Colored display cells are not pixel-wise NET measurements. Missing compartment values do not imply absence of tissue. The near-black line screen does not rule out folds, pigment, stripes, or blur. Neither the protein score nor these compartment labels establish accuracy in a new cancer type.</p>
<p><a href="regional_scores_qc.csv">Coordinates and scores</a> · <a href="score_grid.npz">Numeric score grid</a> · <a href="../compartments.tif">Full-resolution compartment mask (analysis coordinates)</a> · <a href="compartment_summary.csv">Compartment statistics</a> · <a href="patient_image_features.csv">Case summary</a> · <a href="qc.json">QC and grid legend</a></p></html>"""


def verify_case(config, sample_id):
    ids=check_config(config);require(sample_id in ids,"Unknown sample")
    folder=Path(config["output_directory"])/sample_id;d=folder/"export"
    record=json.loads((d/"export_manifest.json").read_text(encoding="utf-8"))
    required={"regional_scores_qc.csv","patient_image_features.csv","compartment_summary.csv","qc.json","index.html",
              "score_grid.npz","spatial_fixed_0_100.png"}
    require(record["sample_id"]==sample_id and record["schema_version"]==1 and set(record["outputs"])==required,"Invalid export manifest")
    for name,digest in record["outputs"].items():require(sha(d/name)==digest,"Changed retained export: "+name)
    require(sha(folder/"compartments.tif")==record["compartments_tif_sha256"],"Changed compartment mask")
    return record


def export_case(config, sample_id):
    """Render self-contained tables/maps in SAMPLE/export; leave inference files alone."""
    ids=check_config(config);require(sample_id in ids,"Unknown sample")
    folder=Path(config["output_directory"])/sample_id;target=folder/"export"
    if target.exists():
        return verify_case(config,sample_id)
    rows,shape,transform,checks=prepared_rows(folder,sample_id,config)
    patient,comps=summarize(sample_id,rows)
    with tempfile.TemporaryDirectory(dir=folder,prefix=".export-") as scratch:
        d=Path(scratch)/"publish";d.mkdir()
        write_csv(d/"regional_scores_qc.csv",rows);write_csv(d/"patient_image_features.csv",[patient]);write_csv(d/"compartment_summary.csv",comps)
        emit_grid(d,folder,rows,shape,transform)
        json_file(d/"qc.json",dict(sample_id=sample_id,tiff_rois_verified=checks,summary=patient,
            analysis_shape_yx=shape,analysis_mpp=MPP,native_to_analysis=transform.tolist(),center_size_px=224,stride_px=512,
            coordinate_origin="Zero-based; analysis_x/y are 512-window top left; score center is x+256,y+256.",
            score_grid="Array indices [window_y//512,window_x//512]; eligible_score is NaN when excluded/unprocessed; raw_score is retained for sampled windows.",
            grid_categories=dict(enumerate(CATEGORIES)),tiff_labels=LABELS,fixed_color_limits=[0,100],
            score="100 * sqrt(clip(mean_MPO,0,1) * clip(mean_CitH3,0,1)); no per-slide rescaling",
            eligibility="Artifact has priority. Center >=50% tissue; dominant class >=60% of tissue pixels, else mixed. Background and unprocessed pixels are not compartments.",
            minimum_compartment_centers=20,low_count_medians="Descriptive median for any nonempty class; case-level estimate blank for n<20.",
            qc_limits="Paired NET measurements and expert compartment labels are unavailable. Scan-line QC only; not a general artifact detector."))
        (d/"index.html").write_text(case_html(sample_id,patient,comps),encoding="utf-8")
        record=dict(schema_version=1,sample_id=sample_id,total_windows=len(rows),eligible_windows=patient["eligible_windows"],
                    outputs={p.name:sha(p) for p in sorted(d.iterdir())},compartments_tif_sha256=sha(folder/"compartments.tif"))
        json_file(d/"export_manifest.json",record);d.rename(target)
    return verify_case(config,sample_id)


def export_index(config):
    """Build the cohort report only when all configured sample exports verify."""
    ids=check_config(config);root=Path(config["output_directory"]);target=root/"report"
    patients=[];comps=[];digests={}
    for sid in ids:
        verify_case(config,sid)
        d=root/sid/"export";digests[sid]=sha(d/"export_manifest.json")
        patients.extend(read_csv(d/"patient_image_features.csv"));comps.extend(read_csv(d/"compartment_summary.csv"))
    if target.exists():
        rec=json.loads((target/"completion.json").read_text(encoding="utf-8"))
        require(rec.get("complete") is True and rec["case_manifests"]==digests and set(rec["outputs"])=={"index.html","patient_image_features.csv","compartment_summary.csv"},"Changed cohort membership")
        for n,h in rec["outputs"].items():require(sha(target/n)==h,"Changed cohort report")
        return rec
    with tempfile.TemporaryDirectory(dir=root,prefix=".report-") as scratch:
        d=Path(scratch)/"publish";d.mkdir()
        write_csv(d/"patient_image_features.csv",patients);write_csv(d/"compartment_summary.csv",comps)
        rows="".join(f'<tr><td><a href="../{r["sample_id"]}/export/index.html">{r["sample_id"]}</a></td><td>{r["eligible_windows"]}</td><td>{number(r["overall_median"])}</td><td>{number(r["tumor_median"])}</td><td>{number(r["stroma_median"])}</td><td>{number(r["other_median"])}</td></tr>' for r in patients)
        (d/"index.html").write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><title>HENETs cohort</title><h1>HENETs spatial protein scores</h1><p>{len(ids)} slides. One row per configured slide; independent patients are not assumed. Compartment medians require ≥20 eligible centers. These are H&amp;E predictions, not measured NET counts.</p><table><tr><th>Slide</th><th>Eligible windows</th><th>Whole slide</th><th>Tumor</th><th>Stroma</th><th>Other</th></tr>{rows}</table><p><a href="patient_image_features.csv">Slide summaries</a> · <a href="compartment_summary.csv">All compartments</a></p></html>',encoding="utf-8")
        rec=dict(complete=True,case_manifests=digests,n_samples=len(ids),outputs={p.name:sha(p) for p in d.iterdir()})
        json_file(d/"completion.json",rec);d.rename(target)
    return rec


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",required=True,type=Path)
    select=p.add_mutually_exclusive_group(required=True);select.add_argument("--sample");select.add_argument("--index",action="store_true")
    args=p.parse_args();cfg=json.loads(args.config.read_text(encoding="utf-8-sig"))
    print(json.dumps(export_index(cfg) if args.index else export_case(cfg,args.sample)))
