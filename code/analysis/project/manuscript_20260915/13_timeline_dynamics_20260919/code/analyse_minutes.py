"""Natural-unit minute profiles using common tier-by-patch poststratification."""

from pathlib import Path
import os, sys, json, time

OUT = Path(__file__).resolve().parents[1]
BASE = OUT.parent / "03_reproducibility/data"
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[k] = "1"
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy import stats

REGIONS = ["NA1", "EUW1", "KR"]
LABELS = {"NA1": "NA", "EUW1": "EUW", "KR": "KR"}
COMPONENTS = [
    "valid_wards_placed",
    "valid_ward_kills",
    "control_ward_purchases",
    "control_ward_undo_net",
]


def save(df, rel):
    df.to_csv(OUT / rel, index=False, encoding="utf-8-sig")


def dump(rel, v):
    (OUT / rel).write_text(
        json.dumps(v, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


def sp(log10p):
    if log10p >= -3:
        return f"{10**log10p:.12g}"
    power = int(np.floor(log10p))
    return f"{10**(log10p-power):.10f}e{power:+d}"


def summarize(meta, y, weights, eligible_cells, specification):
    rows = []
    means = {}
    covs = {}
    composition = []
    for r in REGIONS:
        mean = np.zeros(40)
        cov = np.zeros((40, 40))
        n_total = 0
        group = (
            meta[meta.region.eq(r)].groupby(["tier", "patch"], observed=True).indices
        )
        region_indices = np.flatnonzero(meta.region.eq(r).to_numpy())
        for cell, loc in group.items():
            if cell not in eligible_cells:
                continue
            ix = region_indices[loc]
            a = y[ix]
            mu = a.mean(0)
            n = len(ix)
            weight = float(weights[cell])
            n_total += n
            dev = a - mu
            mean += weight * mu
            cov += weight**2 * (dev.T @ dev) / (n * (n - 1))
            composition.append(
                dict(
                    specification=specification,
                    region=r,
                    tier=cell[0],
                    patch=cell[1],
                    n_matches=n,
                    weight=weight,
                )
            )
        means[r] = mean
        covs[r] = cov
        for j, c in enumerate(COMPONENTS):
            for k in range(10):
                i = j * 10 + k
                se = np.sqrt(max(0, cov[i, i]))
                lo = mean[i] - 1.96 * se
                hi = mean[i] + 1.96 * se
                rows.append(
                    dict(
                        region=r,
                        display_region=LABELS[r],
                        component=c,
                        minute=k + 1,
                        time_midpoint=k + 0.5,
                        mean_rate=mean[i],
                        se=se,
                        ci_low=lo,
                        ci_high=hi,
                        n_matches=n_total,
                        n_teams=2 * n_total,
                        unit="events per team per minute",
                        specification=specification,
                        interval_method="Fixed-reference poststratification; within-cell match-cluster covariance; normal pointwise 95% CI",
                    )
                )
    return pd.DataFrame(rows), means, covs, pd.DataFrame(composition)


def simple_profiles(meta, y, by_tier):
    rows = []
    for r in REGIONS:
        use = meta.region.eq(r).to_numpy()
        idx = np.flatnonzero(use)
        groups = (
            [
                np.flatnonzero((meta.region.eq(r) & meta.tier.eq(t)).to_numpy())
                for t in sorted(meta.tier.unique())
            ]
            if by_tier
            else [idx]
        )
        mu = np.zeros(40)
        cov = np.zeros((40, 40))
        for ix in groups:
            a = y[ix]
            m = a.mean(0)
            dev = a - m
            mu += m / len(groups)
            cov += (dev.T @ dev) / (len(ix) * (len(ix) - 1) * len(groups) ** 2)
        for j, c in enumerate(COMPONENTS):
            for k in range(10):
                se = np.sqrt(cov[j * 10 + k, j * 10 + k])
                m = mu[j * 10 + k]
                rows.append(
                    dict(
                        region=r,
                        component=c,
                        minute=k + 1,
                        mean_rate=m,
                        ci_low=m - 1.96 * se,
                        ci_high=m + 1.96 * se,
                        n_matches=len(idx),
                        specification=(
                            "full_cohort_tier_equal"
                            if by_tier
                            else "full_cohort_unadjusted"
                        ),
                    )
                )
    return pd.DataFrame(rows)


def main():
    summary_path = OUT / "qa/minute_parse_summary.json"
    if not summary_path.exists():
        raise RuntimeError("Full parsing has not completed")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(summary) == 465 and sum(x["n_matches"] for x in summary) == 2223546
    assert all(
        x["historical_payload_identity"] and x["all_frozen_component_totals_equal"]
        for x in summary
    )
    log("Load minute counts and immutable cohort metadata")
    columns = [
        f"{s}_{c}_m{k:02d}"
        for s in ["blue", "red"]
        for c in COMPONENTS
        for k in range(1, 11)
    ]
    records = pd.concat(
        [
            pd.read_parquet(p)
            for p in sorted((OUT / "data/minute_shards").glob("*.parquet"))
        ],
        ignore_index=True,
    )
    assert len(records) == 2223546 and records.match_id.is_unique
    meta = pd.read_parquet(
        BASE / "timeline_features_0_10.parquet",
        columns=["match_id", "region", "tier", "patch", "analysis_cutoff_ms"],
    )
    records = meta.merge(records, on="match_id", how="left", validate="one_to_one")
    assert not records[columns].isna().any().any()
    y = (
        records[columns[:40]].to_numpy(dtype=float)
        + records[columns[40:]].to_numpy(dtype=float)
    ) / 2
    last_duration = (records.analysis_cutoff_ms.to_numpy(float) - 540000) / 60000
    assert (last_duration > 0).all() and (abs(last_duration - 1) < 0.017).all()
    for j in range(4):
        y[:, j * 10 + 9] /= last_duration
    meta = records[["match_id", "region", "tier", "patch", "analysis_cutoff_ms"]].copy()
    del records
    counts = (
        meta.groupby(["tier", "patch", "region"]).size().unstack("region", fill_value=0)
    )
    common = counts[counts.min(axis=1) >= 30]
    joint = common.sum(1)
    weights = joint / joint.groupby(level=0).transform("sum") / 7
    assert abs(weights.sum() - 1) < 1e-12
    log("Estimate shared-reference profiles and joint covariance")
    profiles, means, covs, composition = summarize(
        meta, y, weights, set(common.index), "common_tier_patch_reference"
    )
    save(profiles, "source_data/Fig3abc_minute_profiles.csv")
    save(composition, "source_data/minute_reference_composition.csv")
    np.savez_compressed(
        OUT / "data/minute_profile_covariance.npz",
        **{f"cov_{r}": covs[r] for r in REGIONS},
        **{f"mean_{r}": means[r] for r in REGIONS},
    )
    rows = []
    critical = float(stats.norm.isf(0.05 / (2 * 90)))
    for a, b in [("KR", "NA1"), ("KR", "EUW1"), ("EUW1", "NA1")]:
        delta = means[a] - means[b]
        cov = covs[a] + covs[b]
        for j, c in enumerate(COMPONENTS[:3]):
            for k in range(10):
                i = j * 10 + k
                se = float(np.sqrt(max(0, cov[i, i])))
                z = delta[i] / se if se > 0 else 0.0
                lp = (np.log(2) + stats.norm.logsf(abs(z))) / np.log(10)
                rows.append(
                    dict(
                        contrast=LABELS[a] + "-" + LABELS[b],
                        component=c,
                        minute=k + 1,
                        mean_difference=delta[i],
                        se=se,
                        ci_low=delta[i] - 1.96 * se,
                        ci_high=delta[i] + 1.96 * se,
                        simultaneous_low=delta[i] - critical * se,
                        simultaneous_high=delta[i] + critical * se,
                        z=z,
                        log10_p=lp,
                        p_value=sp(lp),
                        family="90 exploratory regional-by-component-by-minute contrasts",
                        simultaneous_method="Bonferroni 95% across 90 contrasts",
                    )
                )
    contrasts = pd.DataFrame(rows)
    order = np.argsort(contrasts.log10_p.to_numpy())
    holm = np.empty(90)
    holm[order] = np.minimum(
        0,
        np.maximum.accumulate(
            contrasts.log10_p.to_numpy()[order] + np.log10(np.arange(90, 0, -1))
        ),
    )
    contrasts["log10_p_holm"] = holm
    contrasts["p_holm"] = [sp(p) for p in holm]
    save(contrasts, "tables/minute_regional_contrasts.csv")
    # Five-minute blocks use the covariance of the component minute means.
    blocks = []
    for r in REGIONS:
        for j, c in enumerate(COMPONENTS[:3]):
            for start, end in [(0, 5), (5, 10)]:
                ix = np.arange(j * 10 + start, j * 10 + end)
                v = means[r][ix].mean()
                se = np.sqrt(covs[r][np.ix_(ix, ix)].sum()) / len(ix)
                blocks.append(
                    dict(
                        region=r,
                        component=c,
                        window=f"{start}-{end}",
                        mean_rate=v,
                        ci_low=v - 1.96 * se,
                        ci_high=v + 1.96 * se,
                    )
                )
    save(pd.DataFrame(blocks), "tables/five_minute_block_rates.csv")
    log("Calculate full-cohort descriptive and disjoint-player diagnostics")
    raw = simple_profiles(meta, y, False)
    tier = simple_profiles(meta, y, True)
    save(pd.concat([raw, tier]), "source_data/minute_full_cohort_sensitivity.csv")
    disjoint = pd.read_parquet(
        BASE / "all_participant_disjoint_matches.parquet"
    ).match_id
    use = meta.match_id.isin(disjoint).to_numpy()
    md = meta.loc[use].reset_index(drop=True)
    yd = y[use]
    dc = md.groupby(["tier", "patch", "region"]).size().unstack("region", fill_value=0)
    valid = dc.index[(dc.min(1) >= 5) & dc.index.isin(common.index)]
    dw = weights.reindex(valid)
    dw = dw / dw.groupby(level=0).transform("sum") / 7
    prof, dm, dv, dcmp = summarize(
        md, yd, dw, set(valid), "all_participant_disjoint_common_reference"
    )
    save(prof, "source_data/minute_disjoint_player_profiles.csv")
    save(dcmp, "tables/minute_disjoint_composition.csv")
    selection = []
    for r in REGIONS:
        nr = int(meta.region.eq(r).sum())
        nc = int(composition.loc[composition.region.eq(r), "n_matches"].sum())
        nd = int(dcmp.loc[dcmp.region.eq(r), "n_matches"].sum())
        selection.append(
            dict(
                region=r,
                full_matches=nr,
                common_reference_matches=nc,
                excluded_from_common=nr - nc,
                disjoint_matches_before_cell_support=int(md.region.eq(r).sum()),
                disjoint_common_matches=nd,
            )
        )
    save(pd.DataFrame(selection), "tables/minute_sample_flow.csv")
    diagnostics = []
    for a, b in [("KR", "NA1"), ("KR", "EUW1"), ("EUW1", "NA1")]:
        for j, c in enumerate(COMPONENTS[:3]):
            for k in range(10):
                i = j * 10 + k
                di = dm[a][i] - dm[b][i]
                se = np.sqrt(dv[a][i, i] + dv[b][i, i])
                main = means[a][i] - means[b][i]
                diagnostics.append(
                    dict(
                        contrast=LABELS[a] + "-" + LABELS[b],
                        component=c,
                        minute=k + 1,
                        main_difference=main,
                        disjoint_difference=di,
                        disjoint_ci_low=di - 1.96 * se,
                        disjoint_ci_high=di + 1.96 * se,
                        same_direction=bool(np.sign(di) == np.sign(main)),
                    )
                )
    save(pd.DataFrame(diagnostics), "tables/minute_disjoint_direction_check.csv")
    # Net purchase sensitivity is a paired estimand within the same match.
    undo = []
    for r in REGIONS:
        for k in range(10):
            c = np.zeros(40)
            c[30 + k] = 1
            c[20 + k] = -1
            delta = c @ means[r]
            se = np.sqrt(max(0, c @ covs[r] @ c))
            undo.append(
                dict(
                    region=r,
                    minute=k + 1,
                    net_minus_gross=delta,
                    ci_low=delta - 1.96 * se,
                    ci_high=delta + 1.96 * se,
                )
            )
    save(pd.DataFrame(undo), "tables/minute_purchase_undo_sensitivity.csv")
    audit = dict(
        n_full_matches=len(meta),
        n_common_matches=int(composition.n_matches.sum()),
        n_common_cells=len(common),
        retained_fraction=float(composition.n_matches.sum() / len(meta)),
        n_disjoint_matches=len(md),
        n_disjoint_common_matches=int(dcmp.n_matches.sum()),
        n_minute_contrasts=90,
        n_Holm_p_below_05=int((contrasts.log10_p_holm < np.log10(0.05)).sum()),
        bonferroni_critical_z=critical,
        all_raw_payload_hashes_match=True,
        all_minute_sums_match_frozen=True,
        disjoint_same_direction_count=int(
            sum(v["same_direction"] for v in diagnostics)
        ),
        minimum_main_rate=float(
            profiles[profiles.component.isin(COMPONENTS[:3])].mean_rate.min()
        ),
        last_bin_duration_minutes_range=[
            float(last_duration.min()),
            float(last_duration.max()),
        ],
        largest_absolute_net_purchase_change=float(
            pd.DataFrame(undo).net_minus_gross.abs().max()
        ),
    )
    dump("qa/minute_analysis_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    print(pd.DataFrame(blocks).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
