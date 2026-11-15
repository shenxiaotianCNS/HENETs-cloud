"""Patient-level TCGA research analysis; no spatial RNA truth is implied."""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


HE_FEATURES = ["he_all_median", "he_tumor_median", "he_stroma_median", "he_tumor_minus_stroma"]


def parse_kegg_genes(text: str) -> list[dict]:
    """Read only the GENE section of the official KEGG flat-file entry."""
    found, rows = False, []
    for line in text.splitlines():
        field = line[:12].strip()
        if field:
            found = field == "GENE"
        if found:
            match = re.match(r"\s*(\d+)\s+([^;\s]+);", line[12:])
            if match:
                rows.append({"entrez_id": match[1], "symbol": match[2]})
    if not rows:
        raise ValueError("KEGG entry has no parsed GENE section")
    return rows


def read_star_tpm(path: str | Path) -> tuple[pd.Series, pd.DataFrame]:
    frame = pd.read_csv(path, sep="\t", comment="#", dtype={"gene_id": str})
    required = {"gene_id", "gene_name", "tpm_unstranded"}
    if not required.issubset(frame):
        raise ValueError(f"Missing STAR TPM columns: {required - set(frame)}")
    frame = frame[frame.gene_id.str.startswith("ENSG")].copy()
    # GENCODE pseudoautosomal Y copies end in .<version>_PAR_Y. Preserve that
    # suffix so a distinct Y feature is not silently collapsed into its X locus.
    frame["ensembl_id"] = frame.gene_id.str.replace(r"\.\d+(?=(_PAR_Y)?$)", "", regex=True)
    frame["tpm_unstranded"] = pd.to_numeric(frame.tpm_unstranded, errors="raise")
    if frame.ensembl_id.duplicated().any():
        raise ValueError("Duplicate version-stripped Ensembl IDs; resolve explicitly")
    if not np.isfinite(frame.tpm_unstranded).all() or (frame.tpm_unstranded < 0).any():
        raise ValueError("Invalid TPM values")
    return frame.set_index("ensembl_id").tpm_unstranded, frame[["ensembl_id", "gene_name"]]


def mcp_counter(log_expression: pd.DataFrame, markers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Faithful ENSEMBL_ID port of official MCPcounter appendSignatures().

    The official R implementation averages each population's available distinct
    marker features. Input here is explicitly log2(TPM+1), not absolute fractions.
    """
    scores, coverage = {}, []
    for population, group in markers.groupby("Cell population", sort=False):
        ids = set(group["ENSEMBL ID"].dropna().astype(str)) - {"NA", ""}
        overlap = sorted(ids.intersection(log_expression.index))
        column = "mcp_" + re.sub(r"[^a-z0-9]+", "_", population.lower()).strip("_")
        scores[column] = log_expression.loc[overlap].mean(axis=0) if overlap else np.nan
        coverage.append(dict(score=column, population=population, total_markers=len(ids),
                             matched_markers=len(overlap), coverage=len(overlap)/max(len(ids), 1),
                             missing_markers=";".join(sorted(ids-set(overlap)))))
    return pd.DataFrame(scores, index=log_expression.columns), pd.DataFrame(coverage)


def bh_adjust(values, *, family_size: int | None = None) -> np.ndarray:
    """BH adjustment, optionally retaining a prespecified family with missing tests.

    Non-testable entries stay NaN. A fixed family counts them in the denominator
    (equivalent to padding with P=1), without reporting fabricated P-values.
    """
    values = np.asarray(values, dtype=float)
    output = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    size = len(valid) if family_size is None else family_size
    if int(size) != size or size < len(valid):
        raise ValueError("BH family size must be an integer at least the number of valid tests")
    if ((values[valid] < 0) | (values[valid] > 1)).any():
        raise ValueError("BH P-values must be in [0, 1]")
    order = valid[np.argsort(values[valid])]
    if len(order):
        adjusted = values[order] * size / np.arange(1, len(order)+1)
        output[order] = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
    return output


def correlation(x, y, *, seed=721, n_boot=2000) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    result = dict(n=len(x), rho=np.nan, p_value=np.nan, ci_low=np.nan, ci_high=np.nan,
                  bootstrap_valid=0, status="insufficient_or_constant")
    if len(x) < 5 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return result
    result["rho"], result["p_value"] = map(float, stats.spearmanr(x, y))
    draw = np.random.default_rng(seed).integers(0, len(x), (n_boot, len(x)))
    rx, ry = stats.rankdata(x[draw], axis=1), stats.rankdata(y[draw], axis=1)
    rx -= rx.mean(axis=1, keepdims=True)
    ry -= ry.mean(axis=1, keepdims=True)
    denom = np.sqrt((rx*rx).sum(axis=1) * (ry*ry).sum(axis=1))
    boot = np.divide((rx*ry).sum(axis=1), denom, out=np.full(n_boot, np.nan), where=denom>0)
    boot = boot[np.isfinite(boot)]
    result.update(bootstrap_valid=len(boot), status="ok")
    if len(boot) >= n_boot*.9:
        result["ci_low"], result["ci_high"] = map(float, np.quantile(boot, [.025, .975]))
    return result


def partial_spearman(x, y, covariate) -> dict:
    frame = pd.DataFrame(dict(x=x, y=y, z=covariate)).replace([np.inf, -np.inf], np.nan).dropna()
    result = dict(n=len(frame), partial_rho=np.nan, p_value=np.nan, status="insufficient_or_constant")
    if len(frame) < 6 or (frame.nunique() < 2).any():
        return result
    ranked = frame.rank(method="average").to_numpy()
    design = np.column_stack([np.ones(len(frame)), ranked[:, 2]])
    residual = ranked[:, :2] - design @ np.linalg.lstsq(design, ranked[:, :2], rcond=None)[0]
    residual -= residual.mean(axis=0)
    rank_scale = np.linalg.norm(ranked[:, :2]-ranked[:, :2].mean(axis=0), axis=0)
    tolerance = 10*np.finfo(float).eps*max(design.shape)*rank_scale
    if (np.linalg.norm(residual, axis=0) <= tolerance).any():
        result["status"] = "undefined_rank_collinearity"
        return result
    rho = float(np.corrcoef(residual.T)[0, 1])
    rho = float(np.clip(rho, -1, 1))
    df = len(frame)-3
    t = rho * np.sqrt(df/max(1-rho*rho, np.finfo(float).eps))
    result.update(partial_rho=rho, p_value=float(2*stats.t.sf(abs(t), df)), status="ok")
    return result


def aggregate_he(frame: pd.DataFrame, min_compartment_centers=20) -> pd.DataFrame:
    required = {"sample_id", "HE_NETs_score", "comparison_compartment"}
    if not required.issubset(frame):
        raise ValueError(f"Missing HE columns: {required - set(frame)}")
    if "patch_id" in frame and frame.duplicated(["sample_id", "patch_id"]).any():
        raise ValueError("Duplicate patient/patch coordinates")
    eligible = {"tumor", "stroma", "other", "mixed"}
    rows = []
    for patient, group in frame.groupby("sample_id", sort=True):
        row = dict(patient_id=patient, n_all_windows=len(group))
        valid = group[group.comparison_compartment.isin(eligible)]
        row.update(n_eligible_centers=len(valid), he_all_median=valid.HE_NETs_score.median())
        for compartment in ["tumor", "stroma", "other"]:
            scores = valid.loc[valid.comparison_compartment == compartment, "HE_NETs_score"]
            row[f"n_{compartment}_centers"] = len(scores)
            row[f"he_{compartment}_median"] = scores.median() if len(scores) >= min_compartment_centers else np.nan
        row["he_tumor_minus_stroma"] = row["he_tumor_median"]-row["he_stroma_median"]
        row["n_artifact_review"] = int((group.comparison_compartment == "artifact_review").sum())
        row["n_low_tissue"] = int((group.comparison_compartment == "low_tissue").sum())
        rows.append(row)
    return pd.DataFrame(rows)


def stage_binary(stage) -> float:
    text = str(stage).strip().upper().replace("STAGE", "").strip()
    if re.fullmatch(r"(III|IV)[ABC]?", text):
        return 1.
    if re.fullmatch(r"(I|II)[ABC]?", text):
        return 0.
    return np.nan


def clinical_table(patient_ids, cdr: pd.DataFrame, cases: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use frozen CDR endpoints as coherent records; never silently mix times/status.

    Current GDC records independently determine histologic cohort eligibility and
    baseline covariates. CDR/current-GDC discrepancies are audit rows, not overwrites.
    """
    cdr = cdr.set_index("bcr_patient_barcode", verify_integrity=True)
    lookup = {case["submitter_id"]: case for case in cases}
    rows, conflicts = [], []
    for patient in patient_ids:
        case = lookup[patient]
        primary = [d for d in case.get("diagnoses", []) if str(d.get("classification_of_tumor", "")).lower() == "primary"]
        if len(primary) != 1:
            raise ValueError(f"Expected one primary diagnosis for {patient}")
        diagnosis, demographic = primary[0], case.get("demographic", {})
        age = diagnosis.get("age_at_diagnosis")
        row = dict(patient_id=patient, age_years=float(age)/365.25 if age is not None else np.nan,
                   stage=diagnosis.get("ajcc_pathologic_stage"),
                   stage_advanced=stage_binary(diagnosis.get("ajcc_pathologic_stage")),
                   gdc_vital_status=demographic.get("vital_status"),
                   gdc_days_to_death=demographic.get("days_to_death"),
                   gdc_days_to_last_follow_up=diagnosis.get("days_to_last_follow_up"),
                   clinical_endpoint_source="TCGA-CDR 2018 frozen", cdr_available=patient in cdr.index)
        for endpoint in ["OS", "PFI", "DFI", "DSS"]:
            row[endpoint+"_event"] = np.nan
            row[endpoint+"_days"] = np.nan
        if patient in cdr.index:
            source = cdr.loc[patient]
            row["cdr_type"] = source.get("type")
            for endpoint in ["OS", "PFI", "DFI", "DSS"]:
                event = pd.to_numeric(source.get(endpoint), errors="coerce")
                days = pd.to_numeric(source.get(endpoint+".time"), errors="coerce")
                if event in [0, 1] and np.isfinite(days) and days > 0:
                    row[endpoint+"_event"], row[endpoint+"_days"] = int(event), float(days)
                elif pd.notna(event) or pd.notna(days):
                    conflicts.append(dict(patient_id=patient, field=endpoint, issue="invalid_or_zero_time_endpoint_excluded", cdr_value=f"{event};{days}", gdc_value=""))
            current_event = {"Alive": 0, "Dead": 1}.get(demographic.get("vital_status"))
            if current_event is not None and pd.notna(source.get("OS")) and current_event != int(source["OS"]):
                conflicts.append(dict(patient_id=patient, field="OS_event", issue="CDR_GDC_discordance_keep_frozen_CDR", cdr_value=source["OS"], gdc_value=current_event))
            current_time = demographic.get("days_to_death") if current_event == 1 else diagnosis.get("days_to_last_follow_up")
            if current_time is not None and pd.notna(source.get("OS.time")) and abs(float(current_time)-float(source["OS.time"])) > 1:
                conflicts.append(dict(patient_id=patient, field="OS_days", issue="CDR_GDC_discordance_keep_frozen_CDR", cdr_value=source["OS.time"], gdc_value=current_time))
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(conflicts, columns=["patient_id", "field", "issue", "cdr_value", "gdc_value"])


def fit_cox(frame: pd.DataFrame, feature: str, endpoint: str, adjusted=False) -> tuple[dict, pd.DataFrame]:
    from statsmodels.duration.hazard_regression import PHReg
    covariates = [feature] + (["age_years", "stage_advanced"] if adjusted else [])
    cols = [endpoint+"_days", endpoint+"_event"] + covariates
    data = frame[cols].replace([np.inf, -np.inf], np.nan).dropna()
    data = data[data[endpoint+"_days"] > 0]
    events = int(data[endpoint+"_event"].sum())
    result = dict(feature=feature, endpoint=endpoint, adjusted=adjusted, n=len(data), events=events,
                  covariates=";".join(covariates), status="not_fitted", hr_per_sd=np.nan,
                  ci_low=np.nan, ci_high=np.nan, p_value=np.nan, ph_residual_time_rho=np.nan,
                  ph_residual_time_p=np.nan, warning="")
    if events < (20 if adjusted else 5):
        result["status"] = "skipped_fewer_than_20_events" if adjusted else "skipped_fewer_than_5_events"
        return result, pd.DataFrame()
    if len(data) < 10 or (data[covariates].nunique() < 2).any():
        result["status"] = "skipped_insufficient_or_constant_covariate"
        return result, pd.DataFrame()
    design = data[covariates].copy()
    mean, sd = design[feature].mean(), design[feature].std(ddof=1)
    design[feature] = (design[feature]-mean)/sd
    result.update(feature_mean=float(mean), feature_sd=float(sd))
    if adjusted:
        design["age_years"] = (design.age_years-design.age_years.mean())/10
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = PHReg(data[endpoint+"_days"], design, status=data[endpoint+"_event"], ties="efron").fit(disp=False, maxiter=200)
        result["warning"] = "; ".join(str(w.message) for w in caught)
        if not np.isfinite(fit.params).all() or not np.isfinite(fit.bse).all():
            raise ValueError("Non-finite coefficient or standard error")
        if any("converg" in str(w.message).lower() for w in caught):
            raise ValueError("Optimizer did not converge")
        bounds = fit.conf_int()[0]
        result.update(status="ok", hr_per_sd=float(np.exp(fit.params[0])),
                      ci_low=float(np.exp(bounds[0])), ci_high=float(np.exp(bounds[1])), p_value=float(fit.pvalues[0]))
        residuals = np.asarray(fit.schoenfeld_residuals)
        event_mask = data[endpoint+"_event"].to_numpy() == 1
        residual_frame = pd.DataFrame(dict(time=data.loc[event_mask, endpoint+"_days"].to_numpy(),
                                          residual=residuals[event_mask, 0]))
        residual_frame = residual_frame.replace([np.inf, -np.inf], np.nan).dropna()
        if len(residual_frame) >= 5 and residual_frame.residual.nunique() > 1:
            rho, p = stats.spearmanr(np.log(residual_frame.time), residual_frame.residual)
            result.update(ph_residual_time_rho=float(rho), ph_residual_time_p=float(p))
        return result, residual_frame
    except (ValueError, np.linalg.LinAlgError) as exc:
        result.update(status="failed_fit", warning=(result["warning"]+"; "+str(exc)).strip("; "))
        return result, pd.DataFrame()


def kaplan_meier(times, events) -> pd.DataFrame:
    """Product-limit steps with log-log Greenwood confidence limits."""
    times, events = np.asarray(times, float), np.asarray(events, int)
    survival, greenwood, rows = 1., 0., [dict(time=0., survival=1., ci_low=1., ci_high=1., at_risk=len(times), events=0)]
    for time in np.unique(times):
        risk = int((times >= time).sum())
        deaths = int(((times == time) & (events == 1)).sum())
        survival *= 1-deaths/risk
        if risk > deaths and deaths:
            greenwood += deaths/(risk*(risk-deaths))
        low, high = survival, survival
        if 0 < survival < 1:
            se = np.sqrt(greenwood)/abs(np.log(survival))
            center = np.log(-np.log(survival))
            low, high = np.exp(-np.exp(center+1.96*se)), np.exp(-np.exp(center-1.96*se))
        rows.append(dict(time=float(time), survival=float(survival), ci_low=float(low), ci_high=float(high), at_risk=risk, events=deaths))
    return pd.DataFrame(rows)
