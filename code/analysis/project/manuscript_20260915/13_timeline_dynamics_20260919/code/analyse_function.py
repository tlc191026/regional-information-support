"""Common-reference regional probability curves with joint match-level uncertainty."""

from pathlib import Path
import sys, os, json, hashlib, gc, shutil, time

OUT = Path(__file__).resolve().parents[1]
BASE = OUT.parent / "03_reproducibility"
OLD = OUT.parent / "02_timeline"
os.environ["MPLCONFIGDIR"] = str(OUT / "qa/mplconfig")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[k] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, str(BASE / "code"))
sys.path.insert(0, str(BASE / "runtime_deps"))
import statistics_core as core

core.LOGS = OUT / "logs"
import numpy as np
import pandas as pd
from scipy import sparse, special, stats

REGIONS = ["NA1", "EUW1", "KR"]
LABELS = {"NA1": "NA", "EUW1": "EUW", "KR": "KR"}


def dump(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding="utf-8")


def save(df, rel):
    df.to_csv(OUT / rel, index=False, encoding="utf-8-sig")


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def scip(logp):
    if not np.isfinite(logp):
        return "<1e-300"
    z = logp / np.log(10)
    i = int(np.floor(z))
    return f"{10**(z-i):.10f}e{i:+d}" if z < -3 else f"{np.exp(logp):.12g}"


def normal_p(z):
    return scip(np.log(2) + stats.norm.logsf(abs(z)))


def rcs(z, knots):
    a, b = knots[-2:]
    result = []
    for k in knots[:-2]:
        result.append(
            (
                np.maximum(z - k, 0) ** 3
                - (b - k) / (b - a) * np.maximum(z - a, 0) ** 3
                + (a - k) / (b - a) * np.maximum(z - b, 0) ** 3
            )
            / (knots[-1] - knots[0]) ** 2
        )
    return np.column_stack(result)


def fit_joint(df, spline=False):
    design = core.Design(["region", "tier", "patch"], core.STATE).fit(df)
    x0 = design.transform(df)
    z = df.maintenance_diff_z.to_numpy(float)
    basis = [z]
    knots = np.quantile(z, [0.05, 0.35, 0.65, 0.95])
    if spline:
        basis.extend(rcs(z, knots).T)
    extras = []
    extra_names = []
    for r in REGIONS:
        for k, v in enumerate(basis):
            extras.append(v * df.region.eq(r).to_numpy())
            extra_names.append(f"m_{r}_{k}")
    x = sparse.hstack([x0, sparse.csr_matrix(np.column_stack(extras))], format="csr")
    names = design.names + extra_names
    y = df.first_post_neutral_blue.to_numpy(float)
    model = core.fit_logistic(
        x, y, label="regional probability " + ("rcs" if spline else "linear")
    )
    beta = model.coef_[0]
    p = model.predict_proba(x)[:, 1]
    res = y - p
    h = (x.T @ x.multiply((p * (1 - p))[:, None])).toarray()
    bread = np.linalg.pinv(h, rcond=1e-12)
    meat = np.zeros_like(h)
    strata = (df.region.astype(str) + "|" + df.tier.astype(str)).to_numpy()
    for s in np.unique(strata):
        ix = np.flatnonzero(strata == s)
        xs = x[ix]
        rs = res[ix]
        sums = np.asarray(xs.T @ rs).ravel()
        meat += (len(ix) / (len(ix) - 1)) * (
            (xs.T @ xs.multiply(rs[:, None] ** 2)).toarray()
            - np.outer(sums, sums) / len(ix)
        )
    cov = bread @ meat @ bread
    cov = (cov + cov.T) / 2
    return dict(
        x=x,
        beta=beta,
        cov=cov,
        names=names,
        design=design,
        knots=knots,
        spline=spline,
        diagnostics=model.fit_diagnostics_,
    )


def curves(fit, df, reference_weights, grid, offset):
    x = fit["x"]
    beta = fit["beta"]
    names = fit["names"]
    cov = fit["cov"]
    wref = reference_weights
    idx_change = [j for j, n in enumerate(names) if n.startswith(("region_", "m_"))]
    eta = x @ beta - x[:, idx_change] @ beta[idx_change]
    rows = []
    gradients = {}
    for r in REGIONS:
        for u in grid:
            z = u - offset
            values = np.zeros(len(names))
            for j in idx_change:
                n = names[j]
                if n.startswith("region_"):
                    values[j] = float(n == "region_" + r)
                elif n.startswith("m_" + r + "_"):
                    k = int(n.rsplit("_", 1)[1])
                    values[j] = (
                        z if k == 0 else rcs(np.array([z]), fit["knots"])[0, k - 1]
                    )
            prob = special.expit(eta + values @ beta)
            weighted = wref * prob * (1 - prob)
            gradient = np.asarray(x.T @ weighted).ravel()
            gradient[idx_change] = values[idx_change] * weighted.sum()
            mean = float(wref @ prob)
            se = float(np.sqrt(max(0, gradient @ cov @ gradient)))
            gradients[r, float(u)] = gradient
            rows.append(
                dict(
                    region=r,
                    display_region=LABELS[r],
                    maintenance_difference_sd=u,
                    model_predictor_z=z,
                    probability=mean,
                    ci_low=mean - 1.96 * se,
                    ci_high=mean + 1.96 * se,
                    se=se,
                    n_model_matches=len(df),
                    n_reference_matches=int((wref > 0).sum()),
                    method="Common-reference standardization; fixed reference distribution; stratified match-score delta-method 95% CI",
                    specification=(
                        "restricted_cubic_spline" if fit["spline"] else "linear_logit"
                    ),
                )
            )
        log(
            "Calculated probability curve "
            + LABELS[r]
            + (" spline" if fit["spline"] else " linear")
        )
    table = pd.DataFrame(rows)
    effects = []
    for r in REGIONS:
        a = table[(table.region == r) & (table.maintenance_difference_sd == -1)].iloc[0]
        b = table[(table.region == r) & (table.maintenance_difference_sd == 1)].iloc[0]
        g = gradients[r, 1.0] - gradients[r, -1.0]
        delta = b.probability - a.probability
        se = np.sqrt(max(0, g @ cov @ g))
        effects.append(
            dict(
                region=r,
                display_region=LABELS[r],
                p_at_minus1=a.probability,
                p_at_plus1=b.probability,
                probability_difference=delta,
                ci_low=delta - 1.96 * se,
                ci_high=delta + 1.96 * se,
                se=se,
                p_value=normal_p(delta / se),
                contrast="+1 versus -1 common SD of uncentred blue-minus-red maintenance",
                n_matches=int(df.region.eq(r).sum()),
                specification=table.specification.iloc[0],
            )
        )
    contrasts = []
    for a, b in [("KR", "NA1"), ("KR", "EUW1"), ("EUW1", "NA1")]:
        ea = next(v for v in effects if v["region"] == a)
        eb = next(v for v in effects if v["region"] == b)
        g = (gradients[a, 1.0] - gradients[a, -1.0]) - (
            gradients[b, 1.0] - gradients[b, -1.0]
        )
        delta = ea["probability_difference"] - eb["probability_difference"]
        se = np.sqrt(max(0, g @ cov @ g))
        contrasts.append(
            dict(
                contrast=LABELS[a] + "-" + LABELS[b],
                difference_of_probability_changes=delta,
                ci_low=delta - 1.96 * se,
                ci_high=delta + 1.96 * se,
                p_value=normal_p(delta / se),
            )
        )
    return table, pd.DataFrame(effects), pd.DataFrame(contrasts)


def main():
    for p in ["source_data", "tables", "data", "qa", "logs"]:
        (OUT / p).mkdir(exist_ok=True, parents=True)
    copy = [
        "early_adjusted_region_means.csv",
        "early_region_contrasts.csv",
        "primary_prospective_region_slopes.csv",
        "absolute_probabilities.csv",
        "absolute_probability_change.json",
        "sensitivity_models.csv",
        "timeline_sample_flow.csv",
        "timeline_valid_coverage_0_10.csv",
        "participant_disjoint_selection_composition.csv",
        "region_functional_heterogeneity.json",
    ]
    for name in copy:
        shutil.copyfile(OLD / "tables" / name, OUT / "tables" / ("frozen_" + name))
    paths = [
        BASE / "data/analysis_scored_0_10.parquet",
        BASE / "data/maintenance_scale_primary.json",
        OUT.parent / "manuscript_20260805.docx",
    ]
    manifest = {str(p): sha(p) for p in paths}
    dump(OUT / "qa/function_input_manifest.json", manifest)
    df = pd.read_parquet(paths[0])
    assert len(df) == 2223546 and df.match_id.is_unique
    df = df.dropna(
        subset=["first_post_neutral_blue", "maintenance_diff_z", *core.STATE]
    ).reset_index(drop=True)
    assert len(df) == 2210921
    scale = json.loads(paths[1].read_text(encoding="utf-8"))
    offset = scale["diff_mean"] / scale["diff_sd"]
    rawsd = df.maintenance_diff_raw.to_numpy(float) / scale["diff_sd"]
    assert np.max(np.abs(rawsd - offset - df.maintenance_diff_z)) < 1e-12
    counts = (
        df.groupby(["tier", "patch", "region"]).size().unstack("region", fill_value=0)
    )
    common = counts[counts.min(axis=1) >= 30]
    joint = common.sum(axis=1)
    weights = joint / joint.groupby(level=0).transform("sum") / 7
    ix = pd.MultiIndex.from_frame(df[["tier", "patch"]])
    per_record = (weights / joint).reindex(ix).fillna(0).to_numpy(copy=True)
    per_record /= per_record.sum()
    ref = pd.DataFrame(
        {
            "tier": [k[0] for k in joint.index],
            "patch": [k[1] for k in joint.index],
            "n_matches": joint.values,
            "reference_weight": weights.values,
        }
    )
    save(ref, "source_data/function_reference_weights.csv")
    support = []
    for r in REGIONS:
        a = rawsd[df.region.eq(r)]
        lo, hi = np.quantile(a, [0.01, 0.99])
        support.append(
            dict(
                region=r,
                q01=lo,
                q99=hi,
                n_matches=len(a),
                fraction_between_minus2_plus2=float((abs(a) <= 2).mean()),
            )
        )
    assert max(v["q01"] for v in support) < -2 and min(v["q99"] for v in support) > 2
    grid = np.linspace(-2, 2, 81)
    save(pd.DataFrame(support), "source_data/Fig3d_support.csv")
    hist = []
    bins = np.linspace(-2, 2, 81)
    for r in REGIONS:
        a = rawsd[df.region.eq(r)]
        ct, _ = np.histogram(a, bins)
        for lo, hi, n in zip(bins[:-1], bins[1:], ct):
            hist.append(
                dict(
                    region=r,
                    left=lo,
                    right=hi,
                    mid=(lo + hi) / 2,
                    count=int(n),
                    fraction_of_region=n / len(a),
                    n_matches=len(a),
                )
            )
    save(pd.DataFrame(hist), "source_data/Fig3d_observed_difference_distribution.csv")
    # Reproduce the existing pooled coefficient before extending its display.
    log("Reproduce frozen pooled model")
    pooled = core.binary_fit(df)
    estimate = pooled["effects"][0]["beta"]
    old = pd.read_csv(
        OLD / "tables/primary_prospective_region_slopes.csv", keep_default_na=False
    )
    oldbeta = float(old.loc[old.region.eq("pooled"), "beta"].iloc[0])
    assert abs(estimate - oldbeta) < 2e-7
    del pooled
    gc.collect()
    log("Fit common-control region interaction")
    fit = fit_joint(df)
    table, effects, contrasts = curves(fit, df, per_record, grid, offset)
    save(table, "source_data/Fig3d_adjusted_probability_curves.csv")
    save(effects, "tables/regional_probability_changes.csv")
    save(contrasts, "tables/regional_probability_change_contrasts.csv")
    coeff = []
    for r in REGIONS:
        j = fit["names"].index("m_" + r + "_0")
        b = fit["beta"][j]
        se = np.sqrt(fit["cov"][j, j])
        coeff.append(
            dict(
                region=r,
                display_region=LABELS[r],
                beta=b,
                se=se,
                odds_ratio=np.exp(b),
                or_low=np.exp(b - 1.96 * se),
                or_high=np.exp(b + 1.96 * se),
                p_value=normal_p(b / se),
                n_matches=int(df.region.eq(r).sum()),
            )
        )
    C = np.zeros((2, len(fit["names"])))
    for i, r in enumerate(["EUW1", "KR"]):
        C[i, fit["names"].index("m_" + r + "_0")] = 1
        C[i, fit["names"].index("m_NA1_0")] = -1
    d = C @ fit["beta"]
    cv = C @ fit["cov"] @ C.T
    wald = float(d @ np.linalg.pinv(cv) @ d)
    hetero = dict(
        wald_chi2=wald,
        df=2,
        p_value=scip(stats.chi2.logsf(wald, 2)),
        n_matches=len(df),
        scale="common global SD",
    )
    save(pd.DataFrame(coeff), "tables/regional_common_control_slopes.csv")
    dump(OUT / "tables/regional_interaction_test.json", hetero)
    np.savez_compressed(
        OUT / "data/linear_probability_fit.npz",
        beta=fit["beta"],
        cov=fit["cov"],
        names=np.array(fit["names"]),
        offset=offset,
    )
    diag = {
        "pooled_beta_reproduction_abs_error": abs(estimate - oldbeta),
        "n_primary": len(df),
        "n_common_reference": int((per_record > 0).sum()),
        "display_difference_zero_is_equal_maintenance": True,
        "predictor_centering_offset_in_SD": offset,
        "linear": fit["diagnostics"],
        "interaction": hetero,
    }
    del fit
    gc.collect()
    log("Fit flexible shape sensitivity")
    fit = fit_joint(df, True)
    cur, eff, con = curves(fit, df, per_record, grid, offset)
    save(cur, "source_data/Supplementary_probability_spline_curves.csv")
    save(eff, "tables/spline_probability_changes.csv")
    idx = [
        j
        for j, n in enumerate(fit["names"])
        if n.startswith("m_") and not n.endswith("_0")
    ]
    b = fit["beta"][idx]
    v = fit["cov"][np.ix_(idx, idx)]
    w = float(b @ np.linalg.pinv(v) @ b)
    diag["spline"] = {
        **fit["diagnostics"],
        "knots_in_centered_SD": fit["knots"].tolist(),
        "nonlinear_wald_chi2": w,
        "df": len(idx),
        "p_value": scip(stats.chi2.logsf(w, len(idx))),
    }
    delta = cur.merge(
        table,
        on=["region", "maintenance_difference_sd"],
        suffixes=("_spline", "_linear"),
    )
    diag["spline_linear_max_probability_difference"] = float(
        (delta.probability_spline - delta.probability_linear).abs().max()
    )
    diag["spline_change_by_region"] = eff[
        ["region", "probability_difference", "ci_low", "ci_high"]
    ].to_dict("records")
    diag["spline_monotonic_on_display_grid"] = {
        r: bool((np.diff(cur[cur.region == r].probability) >= 0).all()) for r in REGIONS
    }
    np.savez_compressed(
        OUT / "data/spline_probability_fit.npz",
        beta=fit["beta"],
        cov=fit["cov"],
        names=np.array(fit["names"]),
        knots=fit["knots"],
        offset=offset,
    )
    dump(OUT / "qa/function_analysis_audit.json", diag)
    print(json.dumps(diag, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
