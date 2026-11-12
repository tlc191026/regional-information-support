"""Fit matched performance specifications and export comparable probability curves."""
from __future__ import annotations
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import gc
import hashlib
import json
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import special, stats
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/"baseline"
WORK=ROOT/"revision"
OUT=WORK/"tables/regional_performance"
DATA=WORK/"data/regional_performance"
QA=WORK/"qa/regional_performance"
for directory in (OUT,DATA,QA):
    directory.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(OLD/"code"))
from config import REGIONS, TIMELINE_FEATURES_10, child_seed
from performance_link import STATE, MAP, COMPONENTS, OUTCOME, add_differences, map_control, SCALE

BASE_FIELDS=["gold_at_window","xp_at_window","level_at_window","kills_pre","neutral_objectives_pre","turret_plates_pre",*COMPONENTS]
TEAM_COLS=["match_id","window","window_valid","blue_win","first_post_neutral_blue","first_monster_type",*[f"{side}_{field}" for side in ("blue","red") for field in BASE_FIELDS]]
KEEP=["match_id","region","tier","patch",OUTCOME,"first_monster_type","maintenance_diff_z",*STATE,*MAP,*[f"{c}_diff" for c in COMPONENTS]]
CURVE_GRID=np.linspace(-2,2,81)
SEED=child_seed("common_probability_reference_20261023")

def save_json(path,payload):
    path.write_text(json.dumps(payload,indent=2,default=str),encoding="utf-8")

def prepare():
    flow=[]
    discovery_path=DATA/"performance_discovery.parquet"
    rebuild_discovery=(not discovery_path.exists()) or pq.ParquetFile(discovery_path).metadata.num_rows!=2210921
    if rebuild_discovery:
        print("Preparing discovery features",flush=True)
        columns=[*TEAM_COLS,"analysis_eligible","region","tier","patch"]
        df=pd.read_parquet(TIMELINE_FEATURES_10,columns=columns)
        total=len(df)
        df=df[df.analysis_eligible.astype(bool)&df.region.isin(REGIONS)].copy()
        eligible=len(df)
        df=add_differences(df).merge(map_control("discovery"),on="match_id",how="left",validate="one_to_one")
        df=df.dropna(subset=[OUTCOME,"maintenance_diff_z",*STATE,*MAP])
        df[KEEP].to_parquet(discovery_path,index=False)
        flow.append(dict(sample="discovery",timeline_records=total,analysis_eligible=eligible,n_complete=len(df)))
        del df;gc.collect()
    if not all((DATA/f"performance_{name}.parquet").exists() for name in ("holdout","holdout_strict")):
        print("Preparing holdout team and metadata records",flush=True)
        frames=[]
        for path in sorted((OLD/"data/timeline_team_holdout").glob("*.parquet")):
            frames.append(pd.read_parquet(path,columns=TEAM_COLS,filters=[("window","==","0_10")]))
        team=pd.concat(frames,ignore_index=True);del frames
        assert not team.match_id.duplicated().any()
        metadata=[]
        for path in sorted((OLD/"data/participants/holdout").glob("*.parquet")):
            f=pd.read_parquet(path,columns=["match_id","team_id","win","patch"])
            f=f[f.team_id.eq(100)].drop_duplicates("match_id")[["match_id","win","patch"]]
            metadata.append(f)
        meta=pd.concat(metadata,ignore_index=True).drop_duplicates("match_id").rename(columns={"win":"blue_win_raw"})
        team=team.merge(meta,on="match_id",how="inner",validate="one_to_one")
        maps=map_control("holdout")
        team=add_differences(team).merge(maps,on="match_id",how="left",validate="one_to_one")
        del meta,metadata,maps;gc.collect()
        for variant,label in [("primary","holdout"),("strict","holdout_strict")]:
            hp=pd.read_parquet(OLD/f"data/holdout/{variant}/holdout_player_matches.parquet",columns=["puuid","match_id","region","tier","contains_focal"])
            player_count=hp.puuid.nunique();match_count=hp.match_id.nunique()
            if variant=="strict":
                assert player_count==23193 and match_count==548256 and not hp.contains_focal.any()
            hp=hp.sort_values(["match_id","puuid"]).drop_duplicates("match_id")
            df=team.merge(hp[["match_id","region","tier"]],on="match_id",how="inner",validate="one_to_one")
            df["region"]=df.match_id.str.split("_").str[0]
            timeline_records=len(df)
            eligible=df.window_valid.astype(bool)&df.blue_win.astype(float).eq(df.blue_win_raw.astype(float))&df.region.isin(REGIONS)
            df=df[eligible].dropna(subset=[OUTCOME,"maintenance_diff_z",*STATE,*MAP]).copy()
            assert not df.match_id.duplicated().any()
            df[KEEP].to_parquet(DATA/f"performance_{label}.parquet",index=False)
            flow.append(dict(sample=label,source_players=player_count,source_unique_matches=match_count,timeline_records=timeline_records,analysis_eligible=int(eligible.sum()),n_complete=len(df)))
            del hp,df;gc.collect()
        del team;gc.collect()
    if flow:
        flow_path=QA/"performance_sample_flow.csv"
        prior=pd.read_csv(flow_path) if flow_path.exists() else pd.DataFrame()
        pd.concat([prior,pd.DataFrame(flow)],ignore_index=True).drop_duplicates("sample",keep="last").to_csv(flow_path,index=False)
    cache_checks=[]
    for label in ["discovery","holdout","holdout_strict"]:
        path=DATA/f"performance_{label}.parquet"
        frame=pd.read_parquet(path)
        match_region=frame.match_id.str.split("_").str[0]
        wrong=int((frame.region!=match_region).sum())
        outside=int((~match_region.isin(REGIONS)).sum())
        if label!="discovery" and (wrong or outside):
            frame["region"]=match_region
            frame=frame[frame.region.isin(REGIONS)].copy()
            frame.to_parquet(path,index=False)
        cache_checks.append(dict(sample=label,region_prefix_disagreements=wrong,outside_region_records=outside,region_definition="frozen discovery metadata" if label=="discovery" else "match identifier prefix",n_complete=len(frame)))
        del frame;gc.collect()
    pd.DataFrame(cache_checks).to_csv(QA/"performance_region_validation.csv",index=False)
    flow_path=QA/"performance_sample_flow.csv"
    if flow_path.exists():
        flow_table=pd.read_csv(flow_path)
        if "n_complete_before_region_check" not in flow_table:
            flow_table["n_complete_before_region_check"]=flow_table.n_complete
        final_counts={row["sample"]:row["n_complete"] for row in cache_checks}
        flow_table["n_complete"]=flow_table["sample"].map(final_counts)
        flow_table.to_csv(flow_path,index=False)

def reference():
    supports={};common_tiers=None;common_patches=None
    for sample in ["discovery","holdout","holdout_strict"]:
        df=pd.read_parquet(DATA/f"performance_{sample}.parquet",columns=["region","tier","patch"])
        for region in REGIONS:
            x=df[df.region.eq(region)]
            tiers=set(x.tier.astype(str));patches=set(x.patch.astype(str))
            common_tiers=tiers if common_tiers is None else common_tiers&tiers
            common_patches=patches if common_patches is None else common_patches&patches
            supports[f"{sample}/{region}"]=dict(n=len(x),tiers=sorted(tiers),patches=sorted(patches))
        del df
    df=pd.read_parquet(DATA/"performance_discovery.parquet")
    ref=df[df.tier.isin(common_tiers)&df.patch.isin(common_patches)].copy()
    available=len(ref)
    ref=ref.sample(n=min(100000,len(ref)),random_state=SEED).reset_index(drop=True)
    ref.to_parquet(DATA/"common_probability_reference.parquet",index=False)
    save_json(QA/"probability_reference.json",dict(seed=SEED,n_reference=len(ref),n_available=available,common_tiers=sorted(common_tiers),common_patches=sorted(common_patches),supports=supports,interpretation="Fixed random sample of observed discovery covariates on categorical support common to all nine region-by-sample models. The same covariate rows are used for every curve; uncertainty is conditional on this empirical reference distribution."))
    del df;gc.collect()
    return ref

def design(df,predictors,cats,conts,meta=None):
    if meta is None:
        meta={"levels":{c:sorted(df[c].astype(str).unique()) for c in cats},"means":df[conts].mean().to_dict(),"sds":df[conts].std(ddof=0).replace(0,1).to_dict()}
    blocks=[np.ones((len(df),1))];names=["intercept"]
    for c in cats:
        values=df[c].astype(str).to_numpy()
        unknown=set(values)-set(meta["levels"][c])
        if unknown:
            raise ValueError(f"Unsupported reference category {c}: {unknown}")
        for level in meta["levels"][c][1:]:
            blocks.append((values==level).astype(float)[:,None]);names.append(f"{c}_{level}")
    for c in conts:
        blocks.append(((df[c].to_numpy(float)-meta["means"][c])/meta["sds"][c])[:,None]);names.append(c)
    for p in predictors:
        blocks.append(df[p].to_numpy(float)[:,None]);names.append(p)
    return np.column_stack(blocks),names,meta

def fit(df,predictors,cats,conts,sample,model_name,region,ref=None):
    x,names,meta=design(df,predictors,cats,conts)
    y=df[OUTCOME].to_numpy(float)
    model=LogisticRegression(C=1e8,solver="newton-cholesky",fit_intercept=False,max_iter=200,tol=1e-10)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always");model.fit(x,y)
    beta=model.coef_[0];eta=x@beta;p=special.expit(eta);resid=y-p
    bread=np.linalg.pinv((x*(p*(1-p))[:,None]).T@x)
    meat=(x*(resid*resid)[:,None]).T@x
    covariance=bread@meat@bread
    rows=[]
    for name in predictors:
        j=names.index(name);b=beta[j];se=np.sqrt(covariance[j,j]);p1=special.expit(eta+b)
        w0=p*(1-p);w1=p1*(1-p1)
        grad=x.T@(w1-w0)/len(x);grad[j]+=w1.mean()
        diff=(p1-p).mean();sed=np.sqrt(grad@covariance@grad)
        rows.append(dict(sample=sample,model=model_name,region=region,predictor=name,beta=b,se=se,odds_ratio=np.exp(b),or_low=np.exp(b-1.96*se),or_high=np.exp(b+1.96*se),p=2*stats.norm.sf(abs(b/se)),n_matches=len(df),outcome_mean=y.mean(),ame_step=1.,prob_change=diff,prob_change_low=diff-1.96*sed,prob_change_high=diff+1.96*sed,prob_change_se=sed,region_definition="frozen discovery metadata" if sample=="discovery" else "match identifier prefix"))
    curve=[]
    if ref is not None:
        xr,rnames,_=design(ref,predictors,cats,conts,meta)
        assert names==rnames
        j=names.index("maintenance_diff_z")
        base=xr@beta-xr[:,j]*beta[j]
        xr[:,j]=0.
        for g in CURVE_GRID:
            pr=special.expit(base+beta[j]*g);w=pr*(1-pr)
            gradient=xr.T@w/len(xr);gradient[j]=g*w.mean()
            point=pr.mean();se_p=np.sqrt(gradient@covariance@gradient)
            se_logit=se_p/(point*(1-point))
            lo,hi=special.expit(special.logit(point)+np.array([-1.96,1.96])*se_logit)
            curve.append(dict(sample=sample,region=region,maintenance_difference_sd=g,probability=point,ci_low=lo,ci_high=hi,n_reference=len(ref)))
        del xr
    tag=f"{sample}_{model_name}_{region}"
    np.savez(DATA/f"model_{tag}.npz",beta=beta,covariance=covariance)
    save_json(DATA/f"model_{tag}.json",dict(names=names,design=meta,cats=cats,conts=conts,predictors=predictors,n_matches=len(df),n_iterations=model.n_iter_.tolist(),warnings=[str(c.message) for c in caught],covariance="HC0 across unique matches"))
    del x,eta,p,model;gc.collect()
    return rows,curve

def main():
    prepare();ref=reference();results=[];curves=[];support=[];samples=[]
    for sample in ["discovery","holdout","holdout_strict"]:
        df=pd.read_parquet(DATA/f"performance_{sample}.parquet")
        print(f"Fitting {sample}: {len(df):,} matches",flush=True)
        rows,_=fit(df,["maintenance_diff_z"],["region","tier","patch"],STATE+MAP,sample,"M1_map_control","pooled")
        results+=rows
        natural=[f"{c}_diff" for c in COMPONENTS]
        rows,_=fit(df,natural,["region","tier","patch"],STATE+MAP,sample,"M3_natural_units","pooled")
        results+=rows
        for region in REGIONS:
            sub=df[df.region.eq(region)].copy()
            rows,curve=fit(sub,["maintenance_diff_z"],["tier","patch"],STATE+MAP,sample,"M1_map_control",region,ref)
            results+=rows;curves+=curve
            rows,_=fit(sub,natural,["tier","patch"],STATE+MAP,sample,"M3_natural_units",region)
            results+=rows
            bins=np.linspace(-4,4,161);count,edges=np.histogram(sub.maintenance_diff_z,bins=bins)
            for lo,hi,n in zip(edges[:-1],edges[1:],count):
                support.append(dict(sample=sample,region=region,bin_low=lo,bin_high=hi,mid=(lo+hi)/2,count=n))
            samples.append(dict(sample=sample,region=region,n_matches=len(sub),n_below_support=int((sub.maintenance_diff_z<bins[0]).sum()),n_above_support=int((sub.maintenance_diff_z>bins[-1]).sum()),min_difference=sub.maintenance_diff_z.min(),max_difference=sub.maintenance_diff_z.max()))
            print(f"Completed {sample} {region}",flush=True)
            del sub;gc.collect()
        pd.DataFrame(results).to_csv(OUT/"performance_models.csv",index=False)
        pd.DataFrame(curves).to_csv(OUT/"performance_probability_curves.csv",index=False)
        pd.DataFrame(support).to_csv(OUT/"performance_support.csv",index=False)
        pd.DataFrame(samples).to_csv(QA/"performance_support_coverage.csv",index=False)
        if sample=="discovery":
            prefix=df.match_id.str.split("_").str[0]
            excluded=df[df.region.ne(prefix) | ~prefix.isin(REGIONS)].copy()
            excluded["match_region"]=excluded.match_id.str.split("_").str[0]
            excluded.groupby(["region","match_region","tier"],observed=True).size().rename("n_matches").reset_index().to_csv(QA/"discovery_region_prefix_disagreements.csv",index=False)
            consistent=df[df.region.eq(prefix)&prefix.isin(REGIONS)].copy()
            sensitivity=[]
            for region in ["pooled",*REGIONS]:
                sub=consistent if region=="pooled" else consistent[consistent.region.eq(region)]
                cats=["region","tier","patch"] if region=="pooled" else ["tier","patch"]
                rows,_=fit(sub,["maintenance_diff_z"],cats,STATE+MAP,"discovery_prefix_consistent","M1_map_control",region)
                for row in rows:
                    row["excluded_outside_target_servers"]=len(excluded)
                sensitivity+=rows
            pd.DataFrame(sensitivity).to_csv(OUT/"discovery_region_prefix_sensitivity.csv",index=False)
            del consistent,excluded;gc.collect()
        del df;gc.collect()
    old=[]
    for sample in ["discovery","holdout","holdout_strict"]:
        tab=pd.read_csv(OLD/f"tables/performance/{sample}/performance_models.csv")
        tab=tab[tab.model.isin(["M1_map_control","M3_natural_units"])].copy();tab["sample"]=sample;old.append(tab)
    old=pd.concat(old,ignore_index=True)
    compare=pd.DataFrame(results).merge(old,on=["sample","model","region","predictor"],suffixes=("_revised","_old"),validate="one_to_one")
    for col in ["beta","odds_ratio","prob_change","n_matches"]:
        compare[f"change_{col}"]=compare[f"{col}_revised"]-compare[f"{col}_old"]
    compare.to_csv(QA/"performance_old_new.csv",index=False)
    save_json(QA/"performance_revision_notes.json",dict(maintenance_scale=SCALE,strict_population="23,193 eligible listed players; 548,256 unique matches in the strict source list; one record per eligible match",probability_curves="Marginal predictions over one shared discovery reference distribution; 95% delta-method intervals on the logit scale",probability_change="Average change from increasing the observed maintenance difference by one fixed discovery SD, evaluated over each estimation sample's actual covariate distribution",curve_sd_grid=CURVE_GRID.tolist()))
    print("Performance analysis finished",flush=True)

if __name__=="__main__":
    main()
