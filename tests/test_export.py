"""Small calibrated HE fixture: sparse/mixed/background/flagged windows and TIFF export."""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import tifffile

from henets import export as exp


def write_csv(path,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=tuple(rows[0]));w.writeheader();w.writerows(rows)


def fixture(tmp_path,only_excluded=False):
    sid="case-001";d=tmp_path/sid;blocks=d/"blocks";blocks.mkdir(parents=True)
    cats=(["empty","line"] if only_excluded else ["tumor"]*20+["stroma","other","mixed","empty","line"])
    width=512*5;height=512*((len(cats)+4)//5)+32
    full=np.full((height,width),255,np.uint8); rows=[]; grid=[]
    for index,cat in enumerate(cats):
        x=index%5*512;y=index//5*512;pid=f"{sid}_x{x}_y{y}"
        tile=np.zeros((512,512),np.uint8)
        tile[:]=dict(tumor=1,stroma=2,other=3,mixed=1,empty=0,line=1)[cat]
        if cat=="mixed":tile[:,256:]=2
        rgb=np.full((224,224,3),145,np.uint8)
        if cat=="line":rgb[150:153,:,0:3]=0
        v=float((index+1)/100); row=dict(sample_id=sid,patch_id=pid,analysis_x=x,analysis_y=y,
                predicted_MPO=v,predicted_CitH3=v,HE_NETs_score=100*v,score_fold_sd=.2)
        for prefix,arr in (("",tile),("center224_",tile[144:368,144:368])):
            co=np.bincount(arr.ravel(),minlength=4);t=int(sum(co[1:4]))
            for i,k in enumerate(("background","tumor","stroma","other")):row[prefix+k+"_pixels"]=int(co[i])
            row[prefix+"tissue_pixels"]=t;row[prefix+"unprocessed_pixels"]=0
            row[prefix+"dominant_compartment"]=("tumor","stroma","other")[int(np.argmax(co[1:4]))] if t else "background"
            row[prefix+"purity"]=float(max(co[1:4])/max(t,1))
        rows.append(row);grid.append({k:row[k] for k in ("sample_id","patch_id","analysis_x","analysis_y")})
        full[y:y+512,x:x+512]=tile
        Image.fromarray(tile).save(blocks/f"x{x}_y{y}_labels.png")
        np.savez_compressed(blocks/f"x{x}_y{y}_crops.npz",ids=np.array([pid]),rgb=rgb[None])
    write_csv(d/"grid.csv",grid);write_csv(d/"regional_scores.csv",rows)
    trans=np.diag([.8,.8,1.]).tolist()
    (d/"slide.json").write_text(json.dumps(dict(sample_id=sid,analysis_shape=[height,width],
        he_to_analysis=trans,he_path="PRIVATE_SOURCE_AND_PATIENT_NAME",header={"scanner_id":"PRIVATE_HEADER"})))
    Image.fromarray(full[::32,::32]).save(d/"compartments_preview.png")
    tifffile.imwrite(d/"compartments.tif",full,tile=(512,512),photometric="minisblack",compression="deflate",
                     metadata=dict(axes="YX",mpp=.325,classes=exp.LABELS,coordinate_system="HE_resampled_0.325um",he_to_analysis=trans))
    return d,dict(output_directory=str(tmp_path),samples=[dict(sample_id=sid,he_path="PRIVATE_SOURCE_AND_PATIENT_NAME")],
                  analysis_mpp=.325,patch_size=512,center_size=224,minimum_center_tissue_fraction=.5,compartment_purity=.6,
                  core_size=512)


def test_export_and_verify_after_cache_removed(tmp_path):
    d,cfg=fixture(tmp_path);record=exp.export_case(cfg,d.name)
    assert record["total_windows"]==25 and record["eligible_windows"]==23
    table=exp.read_csv(d/"export/regional_scores_qc.csv")
    assert table[0]["native_center_x"]=="320.0"
    assert [r["comparison_compartment"] for r in table[-5:]]==["stroma","other","mixed","low_tissue","artifact_review"]
    p=exp.read_csv(d/"export/patient_image_features.csv")[0]
    assert (float(p["overall_median"]),int(p["tumor_windows"]),float(p["tumor_median"]))==(12.,20,10.5)
    assert p["stroma_median"]==p["other_median"]==p["tumor_minus_stroma"]==""
    summary=exp.read_csv(d/"export/compartment_summary.csv")
    assert float(summary[1]["median"])==21. and summary[1]["patient_level_median"]==""
    with np.load(d/"export/score_grid.npz") as grid:
        assert grid["eligible_score"].shape==(5,5) and np.isnan(grid["eligible_score"][-1,-1])
        assert grid["raw_score"][-1,-1]==25.
    for f in (d/"export").iterdir():
        if f.suffix in (".html",".json",".csv"):
            assert "PRIVATE" not in f.read_text(encoding="utf-8")
    # Exclude caches from the verification/read path after export.
    (d/"blocks").rename(d/"fixture_cache_moved")
    assert exp.verify_case(cfg,d.name)==record and exp.export_index(cfg)["complete"]
    (d/"export/regional_scores_qc.csv").write_text("changed",encoding="utf-8")
    with pytest.raises(ValueError,match="Changed retained export"):
        exp.verify_case(cfg,d.name)


def test_all_excluded_stays_missing_not_zero(tmp_path):
    d,cfg=fixture(tmp_path,only_excluded=True)
    assert exp.export_case(cfg,d.name)["eligible_windows"]==0
    assert exp.read_csv(d/"export/patient_image_features.csv")[0]["overall_median"]==""
    with np.load(d/"export/score_grid.npz") as grid:
        assert np.isnan(grid["eligible_score"]).all() and np.isnan(grid["raw_score"][0,2])
    with pytest.raises(ValueError,match="Unknown"):
        exp.export_case(cfg,"../private")
