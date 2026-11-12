"""Check sample separation, estimator parity and interval calculations."""
from __future__ import annotations
import os
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:os.environ[key]="1"
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy import stats
from config import DATA, REGIONS, write_json
from cluster_sensitivity import MultiwayCodes, REGION_OFFSETS
from twoway import ClusterCodes
from study2_analysis import full_match_lookup

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"tables/peer"


def ids(frame):
    return set(frame.region.astype(str)+"_"+frame.mnum.astype(str))


def main():
    checks=[]
    def check(name,value,detail=None):
        assert bool(value),f"{name}: {detail}"
        checks.append({"check":name,"passed":True,"detail":detail})
    manifest=json.loads((ROOT/"qa/peer_source_fingerprints.json").read_text())
    check("source_code_unchanged",all(hashlib.sha256((DATA.parent/"code"/name).read_bytes()).hexdigest()==sha for name,sha in manifest.items()))
    check("region_qualified_key",REGION_OFFSETS["NA1"]+1 != REGION_OFFSETS["KR"]+1)
    rng=np.random.default_rng(191026);n=120;psi=rng.normal(size=(n,2))/n
    unique=np.arange(n)
    c=MultiwayCodes(unique,unique,unique).covariance(psi)
    expected=n/(n-1)*(psi.T@psi)
    check("independent_cluster_reduction",np.allclose(c["three_way"][0],expected))
    focal=np.repeat(np.arange(12),10);match=np.tile(np.arange(20),6)
    c=MultiwayCodes(focal,focal,match).covariance(psi)
    expected=ClusterCodes(focal,match).covariance(psi)
    check("duplicate_cluster_dimension_reduction",np.allclose(c["three_way"][0],expected))
    c=MultiwayCodes(focal,unique,match).covariance(psi)
    check("unique_recipient_reduction",np.allclose(c["three_way"][0],c["two_way"][0]))
    strict=pd.read_parquet(DATA/"holdout/strict/holdout_player_matches.parquet")
    primary=pd.read_parquet(DATA/"holdout/primary/holdout_player_matches.parquet")
    strict_habits=pd.read_parquet(ROOT/"data/peer/strict/type_rows.parquet",columns=["pid","match_id","region","mnum"])
    discovery_habits=pd.read_parquet(DATA/"c1/type_rows.parquet",columns=["region","mnum"])
    lookup=full_match_lookup("discovery")
    discovery_habits["match_id"]=pd.concat([g.mnum.map(lookup[region]) for region,g in discovery_habits.groupby("region")]).sort_index()
    assert discovery_habits.match_id.notna().all()
    dh=set(discovery_habits.match_id)
    strict_ids=set(strict.match_id)
    check("strict_no_discovery_focal_matches",not strict.contains_focal.any())
    check("strict_habit_source_containment",set(strict_habits.match_id)<=strict_ids)
    check("strict_no_discovery_habit_overlap",len(strict_ids&dh)==0)
    check("strict_source_size",strict.puuid.nunique()==23193 and len(strict_ids)==548256)
    count_rows=[]
    for sample,links in [("holdout",primary),("strict",strict)]:
        for stage,subset in [("all",links),("s1",links[links.in_s1]),("outcome",links[links.s2_set.eq("outcome")]),("habit",links[links.s2_set.eq("type")])]:
            mids=set(subset.match_id)
            count_rows.append({"sample":sample,"stage":stage,"players":subset.puuid.nunique(),"player_match_rows":len(subset),"unique_matches":len(mids),"overlap_discovery_habit_matches":len(mids&dh)})
        a=set(zip(links.loc[links.s2_set.eq("outcome"),"puuid"],links.loc[links.s2_set.eq("outcome"),"match_id"]))
        b=set(zip(links.loc[links.s2_set.eq("type"),"puuid"],links.loc[links.s2_set.eq("type"),"match_id"]))
        check(f"{sample}_within_player_habit_outcome_disjoint",not(a&b))
    overlap_path=OUT/"sample_overlap.csv"
    overlap_csv=pd.DataFrame(count_rows).to_csv(index=False).encode("utf-8")
    if not overlap_path.exists() or overlap_path.read_bytes()!=overlap_csv:
        overlap_path.write_bytes(overlap_csv)
    numeric=primary.drop_duplicates("match_id").copy();numeric["mnum"]=numeric.match_id.str.split("_").str[-1].astype("int64")
    excluded=set(numeric.loc[numeric.contains_focal,"mnum"])
    false_ids=sorted(set(numeric.loc[numeric.mnum.isin(excluded),"match_id"])-set(numeric.loc[numeric.contains_focal,"match_id"]))
    write_json(ROOT/"qa/peer_original_numeric_key_collisions.json",{"previous_extra_excluded_matches":len(false_ids),"match_ids":false_ids,"new_rule":"Strict outcome and habit membership is selected using full original match identifiers."})
    for sample in ["discovery","holdout","strict"]:
        main=pd.read_csv(OUT/sample/"main_effects.csv")
        sensitivity=pd.read_csv(OUT/sample/"cluster_sensitivity.csv")
        keys=["region","relation","exposure","outcome","method"]
        old=main[main.exposure.eq(main.outcome)]
        new=sensitivity[sensitivity.spec.eq("two_way")]
        merged=old.merge(new,on=keys,suffixes=("_main","_check"))
        delta=float(abs(merged.b_main-merged.b_check).max())
        check(f"{sample}_point_estimate_parity",delta<1e-9,delta)
        variance_delta=float(abs(merged.se_main-merged.se_check).max())
        check(f"{sample}_two_way_se_parity",variance_delta<1e-9,variance_delta)
        check(f"{sample}_tost_matches_90pct_interval",np.array_equal(main.tost_equivalent.to_numpy(),((main.ci90_low_std>-.02)&(main.ci90_high_std<.02)).to_numpy()))
        f=pd.read_csv(OUT/sample/"dynamics_self.csv")
        for key,g in f.groupby(["exposure","outcome","method"]):
            regional=g[g.region.isin(REGIONS)];pooled=g[g.region.eq("pooled")].iloc[0]
            assert np.isclose(pooled.b,regional.b.mean()) and np.isclose(pooled.se**2,(regional.se**2).sum()/9)
        check(f"{sample}_minute_pooling_with_variance",True)
    synthetic=[(-.005,.001,True,True,False),(0.,.03,False,False,False),(.06,.005,True,False,True)]
    for b,se,sig,equiv,meaningful in synthetic:
        p=2*stats.norm.sf(abs(b/se));lo90=b-1.645*se;hi90=b+1.645*se;lo95=b-1.96*se;hi95=b+1.96*se
        assert (p<.05)==sig and (lo90>-.02 and hi90<.02)==equiv and (lo95>.02 or hi95<-.02)==meaningful
    check("nonexclusive_inference_cases",True)
    own=pd.read_csv(OUT/"own_tendency_figure_source.csv",keep_default_na=False)
    outcomes={"control_wards","wards_killed","wards_placed","initiation","caution","total_pings"}
    own_checks={
        "all_six_outcomes_in_each_sample":all(
            set(own.loc[own["sample"].eq(sample)&own.region.eq(region)&own.relation.eq(relation)&own.term.eq(term),"outcome"])==outcomes
            for sample in ["discovery","holdout","strict"]
            for region in [*REGIONS,"pooled"]
            for relation in ["teammate","opponent"]
            for term in ["recipient_own_habit","focal_partner_habit"]),
        "own_coefficients_within_intervals":bool(
            np.isfinite(own[["b_std","ci_low_std","ci_high_std"]].to_numpy()).all()
            and (own.ci_low_std<=own.b_std).all() and (own.b_std<=own.ci_high_std).all()),
        "correct_model_only":bool(own.model.eq("own_and_partner").all()
            and own.term.isin(["recipient_own_habit","focal_partner_habit"]).all())}
    for name,passed in own_checks.items():check(name,passed)
    write_json(ROOT/"qa/peer_own_model_checks.json",own_checks)
    conclusions=[]
    for sample in ["discovery","holdout","strict"]:
        frame=pd.read_csv(OUT/sample/"cluster_sensitivity.csv")
        pooled=frame[frame.method.eq("oriv")&frame.spec.eq("three_way")&frame.region.eq("pooled")
            &frame.relation.eq("teammate")&frame.exposure.eq(frame.outcome)]
        check(f"{sample}_six_three_way_teammate_estimates",len(pooled)==6 and set(pooled.outcome)==outcomes)
        n_equivalent=int(pooled.tost_equivalent.sum())
        conclusions.append({"sample":sample,"six_pooled_teammate_equivalent":n_equivalent==6,"n_equivalent":n_equivalent})
    write_json(ROOT/"qa/peer_three_way_conclusions.json",conclusions)
    write_json(ROOT/"qa/peer_validation.json",{"checks":checks,"n_checks":len(checks),"all_passed":True})


if __name__=="__main__":main()
