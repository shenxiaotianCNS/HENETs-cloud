from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from henets import associate


def tables(folder):
    samples = [f"S{i}" for i in range(24)]
    patients = pd.DataFrame(dict(sample_id=samples,patient_id=[f"P{i}" for i in range(24)],
                                 cancer_type=["LIVER"]*12+["BREAST"]*12,
                                 OS_event=[0,1]*12,OS_days=[11,3,7,4,9,6,1,12,2,10,5,8]*2))
    patients["PFI_days"] = ["0","3","5"]+[""]*21
    patients["PFI_event"] = ["1","1",""]+[""]*21
    images = pd.DataFrame({"sample_id": samples})
    for name in associate.FEATURES:
        images[name] = np.tile(np.arange(12),2)
    images["other_median"] = ""
    rna = patients[associate.KEYS].copy()
    rna["rna_nets_score"] = list(range(12))+list(range(11,-1,-1))
    for df,name in [(patients,"patients"),(images.sample(frac=1,random_state=721),"features"),(rna.iloc[:-1],"rna")]:
        df.to_csv(folder/(name+".csv"),index=False)
    return [folder/(x+".csv") for x in ("features","patients","rna")]


def test_associations_separate_cancers_handle_missing_endpoints(tmp_path):
    features,patients,rna = tables(tmp_path)
    out = tmp_path/"out"
    associate.run(features,patients,out,rna_scores=rna,bootstraps=100)
    corr = pd.read_csv(out/"correlations.csv")
    rho = corr.query("image_feature == 'overall_median'").set_index("cancer_type")
    assert rho.loc["LIVER","rho"] == pytest.approx(1)
    assert rho.loc["BREAST","rho"] == pytest.approx(-1)
    assert rho.loc["BREAST","n"] == 11
    assert len(corr) == 10 and (corr.query("image_feature=='other_median'").n == 0).all()
    surv = pd.read_csv(out/"survival_cox.csv")
    assert len(surv) == 4 and (surv.query("endpoint=='OS'").status == "ok").all()
    assert (surv.query("endpoint=='PFI'").status == "skipped_fewer_than_5_events").all()
    checked = pd.read_csv(out/"patients.csv").set_index("sample_id")
    assert checked.loc["S0","PFI_status"] == "nonpositive_days"
    assert checked.loc["S2","PFI_status"] == "incomplete_pair"
    assert np.isnan(checked.loc["S23","rna_nets_score"])
    with pytest.raises(FileExistsError):
        associate.run(features,patients,out,bootstraps=100)


def test_reject_repeated_patient_and_mismatched_rna_identity(tmp_path):
    features,patients,rna = tables(tmp_path)
    out = tmp_path/"out"
    mol = pd.read_csv(rna)
    mol.loc[0,"cancer_type"] = "OTHER"
    mol.to_csv(rna,index=False)
    with pytest.raises(ValueError,match="identities"):
        associate.run(features,patients,out,rna_scores=rna,bootstraps=100)
    p = pd.read_csv(patients)
    p.loc[23,"patient_id"] = p.loc[0,"patient_id"]
    p.to_csv(patients,index=False)
    with pytest.raises(ValueError,match="Duplicate patient_id"):
        associate.run(features,patients,out,bootstraps=100)
    assert not out.exists()


def test_reject_invalid_event_and_safely_omit_unavailable_sources(tmp_path):
    features,patients,_ = tables(tmp_path)
    p = pd.read_csv(patients)
    p.loc[0,"OS_event"] = 2
    p.to_csv(patients,index=False)
    with pytest.raises(ValueError,match="OS_event"):
        associate.run(features,patients,tmp_path/"bad",bootstraps=100)
    p = p[associate.KEYS]
    p.to_csv(patients,index=False)
    associate.run(features,patients,tmp_path/"missing",bootstraps=100)
    assert pd.read_csv(tmp_path/"missing/correlations.csv").empty
    assert pd.read_csv(tmp_path/"missing/survival_cox.csv").n.eq(0).all()


def test_rna_raw_es_matches_direct_and_per_sample_controls(tmp_path,monkeypatch):
    import hashlib
    import json
    gseapy = pytest.importorskip("gseapy")
    n = 90
    ensembl = [f"ENSG{i:011d}" for i in range(n)]
    sets = {associate.TERM:ensembl[:20],"SECOND_SET":ensembl[1:15]}
    genes = tmp_path/"gene_sets.json"
    genes.write_text(json.dumps(dict(gene_sets=sets)),encoding="utf-8")
    markers = tmp_path/"markers.tsv"
    pd.DataFrame({"Cell population":[f"pop{i}" for i in range(10)],"ENSEMBL ID":ensembl[30:40]}).to_csv(markers,sep="\t",index=False)
    monkeypatch.setattr(associate,"GENE_SHA",hashlib.sha256(genes.read_bytes()).hexdigest())
    monkeypatch.setattr(associate,"MCP_SHA",hashlib.sha256(markers.read_bytes()).hexdigest())
    sample_rows = []
    expressions = {}
    for i in range(10):
        sample = f"S{i}"
        tpm = np.random.default_rng(i).uniform(0,20,size=n)
        expressions[sample] = tpm
        f = tmp_path/(sample+".tsv")
        pd.DataFrame({"gene_id":[g+".2" for g in ensembl],"gene_name":ensembl,"tpm_unstranded":tpm}).to_csv(f,sep="\t",index=False)
        sample_rows.append(dict(sample_id=sample,patient_id=f"P{i}",project_id="C",rna_path=f.name))
    manifest = tmp_path/"rna_manifest.csv"
    pd.DataFrame(sample_rows).to_csv(manifest,index=False)
    associate.score_rna(manifest,genes,markers,tmp_path/"scores")
    res = pd.read_csv(tmp_path/"scores/rna_scores.csv").set_index("sample_id")
    direct = gseapy.ssgsea(data=pd.DataFrame(expressions,index=ensembl),gene_sets=sets,
                            outdir=None,no_plot=True,**associate.GSEA).res2d
    ref = direct.loc[direct.Term==associate.TERM].set_index("Name").ES.astype(float)
    np.testing.assert_allclose(res.rna_nets_score,ref.reindex(res.index),atol=1e-10)
    assert res.loc["S9","mcp_pop0"] == pytest.approx(np.log2(1+expressions["S9"][30]))
    assert len(pd.read_csv(tmp_path/"scores/immune_marker_coverage.csv")) == 10
    sample_rows[-1]["rna_path"] = sample_rows[0]["rna_path"]
    pd.DataFrame(sample_rows).to_csv(manifest,index=False)
    with pytest.raises(ValueError,match="Duplicate RNA"):
        associate.score_rna(manifest,genes,markers,tmp_path/"duplicate")
    assert not (tmp_path/"duplicate").exists()
