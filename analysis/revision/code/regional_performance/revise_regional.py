"""Recompute regional signal composition and supported percentile gradients."""
from __future__ import annotations

import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import sys
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "baseline"
WORK = ROOT / "revision"
OUT = WORK / "tables/regional_performance"
QA = WORK / "qa/regional_performance"
OUT.mkdir(parents=True, exist_ok=True)
QA.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(OLD / "code"))
from config import PING_FIELDS, PING_GROUPS, REGIONS, POSITIONS, TIERS, child_seed
from estimation import player_order, cell_sums, stratified_weights

FAMILIES = ["caution", "initiation", "intent", "request", "enemy_information", "tactical"]
METRICS = ["control_wards_bought", "wards_killed", "wards_placed", "get_back_pings", "push_pings", "all_in_pings", "caution", "initiation", "total_pings"]
CONTRASTS = [("KR", "NA1"), ("KR", "EUW1"), ("EUW1", "NA1")]
GRID = np.linspace(.25, .97, 73)

def read_rows(filename, tiers):
    columns = ["puuid", "region", "tier", "team_position", "duration_min", "wards_placed", "wards_killed", "control_wards_bought", *PING_FIELDS]
    rows = pd.read_parquet(OLD / "data" / filename, columns=columns)
    rows = rows[rows.team_position.isin(POSITIONS) & rows.tier.isin(tiers)].copy()
    for key in ["region", "tier", "team_position"]:
        rows[key] = rows[key].astype("category")
    for family, fields in PING_GROUPS.items():
        rows[family] = rows[fields].sum(axis=1)
    rows["total_pings"] = rows[PING_FIELDS].sum(axis=1)
    return rows

def composition(rows, codes, players, weights, tiers, sample):
    position = pd.Categorical(rows.team_position, categories=POSITIONS).codes
    totals = cell_sums(codes, len(players), position, len(POSITIONS), rows[FAMILIES].to_numpy(float))
    point = np.zeros((3, len(FAMILIES)))
    oldpoint = point.copy()
    boot = np.zeros((len(weights), 3, len(FAMILIES)))
    oldboot = boot.copy()
    for r, region in enumerate(REGIONS):
        for tier in tiers:
            members = np.flatnonzero((players.region == region).to_numpy() & (players.tier == tier).to_numpy())
            s = totals[members].sum(axis=0)
            sb = (weights[:, members] @ totals[members].reshape(len(members), -1)).reshape(len(weights), len(POSITIONS), len(FAMILIES))
            if (s.sum(axis=-1) <= 0).any() or (sb.sum(axis=-1) <= 0).any():
                raise ValueError(f"Empty signal-count cell: {sample} {region} {tier}")
            point[r] += (s / s.sum(axis=-1, keepdims=True)).mean(axis=0) / len(tiers)
            boot[:, r] += (sb / sb.sum(axis=-1, keepdims=True)).mean(axis=1) / len(tiers)
            ss, ssb = s.sum(axis=0), sb.sum(axis=1)
            oldpoint[r] += ss / ss.sum() / len(tiers)
            oldboot[:, r] += ssb / ssb.sum(axis=-1, keepdims=True) / len(tiers)
    result, comparisons, contrast_rows = [], [], []
    for r, region in enumerate(REGIONS):
        for k, family in enumerate(FAMILIES):
            lo, hi = np.quantile(boot[:, r, k], [.025, .975])
            result.append(dict(sample=sample, region=region, family=family, estimate=point[r, k], ci_low=lo, ci_high=hi, weighting="equal_tier_and_position"))
            olo, ohi = np.quantile(oldboot[:, r, k], [.025, .975])
            comparisons.append(dict(sample=sample, region=region, family=family, old_tier_only=oldpoint[r, k], old_ci_low=olo, old_ci_high=ohi, revised_tier_position=point[r, k], difference=point[r, k]-oldpoint[r, k]))
    for a, b in CONTRASTS:
        i, j = REGIONS.index(a), REGIONS.index(b)
        for k, family in enumerate(FAMILIES):
            diff = boot[:, i, k] - boot[:, j, k]
            ratio = boot[:, i, k] / boot[:, j, k]
            contrast_rows.append(dict(sample=sample, contrast=f"{a}-{b}", family=family, difference=point[i,k]-point[j,k], difference_ci_low=np.quantile(diff,.025), difference_ci_high=np.quantile(diff,.975), ratio=point[i,k]/point[j,k], ratio_ci_low=np.quantile(ratio,.025), ratio_ci_high=np.quantile(ratio,.975)))
    assert np.allclose(point.sum(axis=1), 1)
    return result, comparisons, contrast_rows

def spline_curves(rows, codes, players, weights, sample, percentiles, knots):
    p = players.merge(percentiles, on="puuid", how="left", validate="one_to_one").percentile.to_numpy()
    rates = rows[METRICS].to_numpy(float) / rows.duration_min.to_numpy(float)[:, None] * 10
    rates = pd.DataFrame(rates).groupby(codes, sort=True).mean().to_numpy()
    def basis(x):
        return np.column_stack([np.ones_like(x), x, x*x, x*x*x, *[np.maximum(x-k,0)**3 for k in knots]])
    curves, gradients, diagnostics = [], [], []
    store = {}
    for region in REGIONS:
        use = (players.region == region).to_numpy() & np.isfinite(p)
        pr = p[use]
        supported = pr.min() <= GRID[0] and pr.max() >= GRID[-1]
        diagnostics.append(dict(sample=sample, region=region, n_players=int(use.sum()), min_percentile=float(pr.min()), max_percentile=float(pr.max()), interval_supported=bool(supported), n_near_lower=int(np.abs(pr-.25).lt(.025).sum()) if isinstance(pr,pd.Series) else int((np.abs(pr-.25)<.025).sum()), n_near_upper=int((np.abs(pr-.97)<.025).sum()), missing_percentiles=int(((players.region==region).to_numpy() & ~np.isfinite(p)).sum())))
        if not supported:
            raise ValueError(f"Unsupported percentile interval: {sample} {region}")
        x, xg = basis(pr), basis(GRID)
        w = weights[:, use].astype(float)
        xprod = np.einsum("ni,nj->nij", x, x).reshape(len(x), -1)
        xtx = (w @ xprod).reshape(len(w), x.shape[1], x.shape[1])
        base_xtx = x.T @ x
        for k, metric in enumerate(METRICS):
            y = rates[use, k]
            beta = np.linalg.solve(base_xtx, x.T @ y)
            betab = np.linalg.solve(xtx, (w @ (x*y[:,None]))[...,None])[...,0]
            point, boots = xg @ beta, betab @ xg.T
            low, high = np.quantile(boots, [.025,.975], axis=0)
            for g, est, lo, hi in zip(GRID, point, low, high):
                curves.append(dict(sample=sample, region=region, metric=metric, percentile=g, estimate=est, ci_low=lo, ci_high=hi))
            delta, deltab = point[-1]-point[0], boots[:,-1]-boots[:,0]
            store[(region, metric)] = (delta, deltab)
            gradients.append(dict(sample=sample, region=region, metric=metric, estimate=delta, ci_low=np.quantile(deltab,.025), ci_high=np.quantile(deltab,.975), lower_percentile=.25, upper_percentile=.97))
            diagnostics.append(dict(sample=sample, region=region, metric=metric, min_prediction=float(point.min()), min_ci=float(low.min()), negative_point=bool((point<0).any()), condition_number=float(np.linalg.cond(base_xtx)), basis="ordinary cubic truncated-power spline", knots=";".join(map(str,knots))))
    for a,b in CONTRASTS:
        for metric in METRICS:
            da, ba = store[(a,metric)]; db, bb = store[(b,metric)]
            delta = ba-bb
            gradients.append(dict(sample=sample, region=f"{a}-{b}", metric=metric, estimate=da-db, ci_low=np.quantile(delta,.025), ci_high=np.quantile(delta,.975), lower_percentile=.25, upper_percentile=.97))
    return curves, gradients, diagnostics

def main():
    percentiles = pd.read_parquet(OLD / "data/player_percentiles.parquet", columns=["puuid","percentile"])
    configs = [("discovery", "study1_focal_rows.parquet", TIERS), ("discovery_six", "study1_focal_rows.parquet", TIERS[1:]), ("holdout", "study1_holdout_rows_primary.parquet", TIERS[1:]), ("holdout_strict", "study1_holdout_rows_strict.parquet", TIERS[1:])]
    allcurves=[]; allgrad=[]; allqa=[]; allcomp=[]; allcomparison=[]; allcontrast=[]; counts=[]
    frozen_knots=None
    for sample, filename, tiers in configs:
        print(f"Loading {sample}",flush=True)
        rows=read_rows(filename,tiers)
        codes,players=player_order(rows)
        seed=child_seed(f"regional_revision_{sample}")
        weights=stratified_weights(players,seed)
        p=players.merge(percentiles,on="puuid",how="left").percentile.to_numpy()
        if sample=="discovery":
            frozen_knots=np.quantile(p[(players.tier!="IRON").to_numpy() & np.isfinite(p)],[.25,.5,.75])
        c,co,ct=composition(rows,codes,players,weights,tiers,sample)
        allcomp+=c;allcomparison+=co;allcontrast+=ct
        cur,gr,qa=spline_curves(rows,codes,players,weights,sample,percentiles,frozen_knots)
        allcurves+=cur;allgrad+=gr;allqa+=qa
        counts.append(dict(sample=sample,n_player_matches=len(rows),n_players=len(players),seed=int(seed),bootstrap_replicates=1000,tiers=tiers,knots=frozen_knots.tolist()))
        for stem,records in [("percentile_curves",allcurves),("percentile_gradients",allgrad),("ping_composition_adjusted",allcomp),("ping_composition_old_new",allcomparison),("ping_composition_contrasts",allcontrast)]:
            export=pd.DataFrame(records)
            export["region_definition"]="server of the sampled player listing"
            export.to_csv(OUT/f"{stem}.csv",index=False)
        pd.DataFrame(allqa).to_csv(QA/"percentile_diagnostics.csv",index=False)
        (QA/"regional_run_info.json").write_text(json.dumps(counts,indent=2),encoding="utf-8")
        print(f"Completed {sample}: {len(players):,} players; {len(rows):,} records",flush=True)
        del rows,codes,players,weights,p
        gc.collect()
    natural=pd.read_csv(OLD/"tables/study1/regional_means_natural.csv",keep_default_na=False)
    natural=natural[natural.metric.isin(["wards_placed","wards_killed","control_wards_bought","get_back_pings","on_my_way_pings","push_pings","all_in_pings"]) & natural.scale.eq("per_10_min")].copy()
    natural["region_definition"]="server of the sampled player listing"
    natural.to_csv(OUT/"regional_natural_units_existing.csv",index=False)
    print("Regional analysis finished",flush=True)

if __name__=="__main__":
    main()
