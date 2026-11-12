"""Harmonize scales, inference labels and figure source tables across peer samples."""
from __future__ import annotations
import os
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:os.environ[key]="1"
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from scipy import stats
import study2_analysis as analysis
from config import DATA, REGIONS, write_json

ROOT=Path(__file__).resolve().parents[2]
OLD=DATA.parent/"tables/study2"
OUT=ROOT/"tables/peer"
COMMON_PATH=OLD/"discovery/standardization.json"
COMMON=json.loads(COMMON_PATH.read_text())
EXTRA=COMMON["extra_outcome_sd"]


def infer(frame):
    frame=frame.copy()
    frame["significant_two_sided_05"]=frame.p.lt(.05)
    frame["tost_p"]=np.maximum(stats.norm.sf((frame.b_std+.02)/frame.se_std),stats.norm.cdf((frame.b_std-.02)/frame.se_std))
    frame["meaningful_beyond_02"]=frame.beyond_sesoi
    frame["region_weight"]=np.where(frame.region.eq("pooled"),1/3,np.nan)
    return frame


def scale(frame,extra=None):
    frame=analysis.add_standardized(frame,COMMON,extra)
    if "p" not in frame:frame["p"]=2*stats.norm.sf(abs(frame.b/frame.se))
    frame["ci_low"],frame["ci_high"]=frame.b-1.96*frame.se,frame.b+1.96*frame.se
    return infer(frame)


def pool_self(frame):
    records=[]
    for keys,g in frame.groupby(["exposure","outcome","method"]):
        r=g[g.region.isin(REGIONS)]
        assert len(r)==3
        records.append({"exposure":keys[0],"outcome":keys[1],"method":keys[2],"region":"pooled","b":r.b.mean(),"se":np.sqrt(np.square(r.se).sum()/9),"n_self":r.n_self.sum()})
    return pd.concat([frame,pd.DataFrame(records)],ignore_index=True)


def own_table(path,sample):
    f=pd.read_csv(path,keep_default_na=False)
    f=f[f.model.isin(["with_other_habit","own_and_partner"])].copy()
    f["term"]=f.term.replace({"other_habit":"recipient_own_habit","focal_habit":"focal_partner_habit"})
    f["model"]="own_and_partner"
    pooled=[]
    for keys,g in f.groupby(["relation","outcome","term"]):
        assert set(g.region)==set(REGIONS)
        pooled.append({"relation":keys[0],"outcome":keys[1],"term":keys[2],"model":"own_and_partner","region":"pooled","b":g.b.mean(),"v":g.v.sum()/9,"n_rows":g.n_rows.sum()})
    f=pd.concat([f,pd.DataFrame(pooled)],ignore_index=True)
    f["sample"]=sample;f["se"]=np.sqrt(f.v)
    f["ci_low"],f["ci_high"]=f.b-1.96*f.se,f.b+1.96*f.se
    f["p"]=2*stats.norm.sf(abs(f.b/f.se))
    f["std_factor"]=[COMMON["pooled"]["pooled"][x]["sd_t"]/COMMON["pooled"]["pooled"][x]["sd_y"] for x in f.outcome]
    for name in ["b","se","ci_low","ci_high"]:f[f"{name}_std"]=f[name]*f.std_factor
    return f


def classifications(discovery,holdout,label):
    a=discovery[(discovery.method=="oriv")&(discovery.exposure==discovery.outcome)&discovery.region.eq("pooled")]
    b=holdout[(holdout.method=="oriv")&(holdout.exposure==holdout.outcome)&holdout.region.eq("pooled")]
    cols=["exposure","outcome","relation","b_std","ci_low_std","ci_high_std","p","tost_p","tost_equivalent","beyond_sesoi"]
    out=a[cols].merge(b[cols],on=["exposure","outcome","relation"],suffixes=("_discovery","_validation"))
    out["sample"]=label
    out["same_direction"]=np.sign(out.b_std_discovery)==np.sign(out.b_std_validation)
    out["significance_supported"]=out.same_direction&out.p_validation.lt(.05)
    out["equivalence_supported"]=out.tost_equivalent_discovery&out.tost_equivalent_validation
    out["meaningful_effect_supported"]=out.same_direction&out.beyond_sesoi_discovery&out.beyond_sesoi_validation
    out["equivalence_assessment"]=np.where(out.tost_equivalent_discovery,np.where(out.tost_equivalent_validation,"supported","not supported"),"not applicable to discovery result")
    return out


def main():
    all_own=[];main_tables={};sample_rows=[]
    for sample in ["discovery","holdout","strict"]:
        src=OLD/sample if sample!="strict" else OUT/"study2/holdout_strict"
        dst=OUT/sample;dst.mkdir(parents=True,exist_ok=True)
        for filename in ["main_effects.csv","dynamics.csv","moderators.csv","robustness.csv"]:
            if not (src/filename).exists():continue
            f=pd.read_csv(src/filename,keep_default_na=False)
            f=scale(f,EXTRA if filename in ["dynamics.csv","robustness.csv"] else None)
            f["sample"]=sample
            f.to_csv(dst/filename,index=False)
            if filename=="main_effects.csv":main_tables[sample]=f
        analysis.primary_family(main_tables[sample]).to_csv(dst/"primary_family.csv",index=False)
        self_frame=pd.read_csv(src/"dynamics_self.csv",keep_default_na=False)
        self_frame=scale(pool_self(self_frame),EXTRA)
        self_frame["sample"],self_frame["relation"]=sample,"self"
        self_frame.to_csv(dst/"dynamics_self.csv",index=False)
        own_source = dst/"own_model_raw.csv" if (dst/"own_model_raw.csv").exists() else src/"twoway_design.csv"
        own=own_table(own_source,sample)
        own.to_csv(dst/"own_tendency_associations.csv",index=False);all_own.append(own)
        s=main_tables[sample]
        counts=s[(s.exposure=="control_wards")&(s.outcome=="control_wards")&(s.method=="oriv")&(s.relation=="teammate")&s.region.isin(REGIONS)]
        sample_rows.extend(counts[["sample","region","n_teammate","n_opponent","n_focal","n_match"]].to_dict("records"))
    pd.concat(all_own).to_csv(OUT/"own_tendency_figure_source.csv",index=False)
    pd.concat([classifications(main_tables["discovery"],main_tables[s],s) for s in ["holdout","strict"]]).to_csv(OUT/"replication_classification.csv",index=False)
    pd.DataFrame(sample_rows).to_csv(OUT/"sample_counts.csv",index=False)
    heterogeneity=[]
    for sample, frame in main_tables.items():
        sub=frame[frame.region.eq("pooled") & frame.relation.eq("difference") & frame.method.eq("oriv") & frame.exposure.eq(frame.outcome)]
        heterogeneity.extend(sub[["sample","outcome","heterogeneity_wald","heterogeneity_p"]].to_dict("records"))
    pd.DataFrame(heterogeneity).to_csv(OUT/"regional_heterogeneity.csv",index=False)
    region_rows=[]
    for sample,variant in [("holdout","primary"),("strict","strict")]:
        links=pd.read_parquet(DATA/"holdout"/variant/"holdout_player_matches.parquet")
        links["match_server"]=links.match_id.str.split("_").str[0]
        for (region,stage), group in links.groupby(["region","s2_set"]):
            region_rows.append({"sample":sample,"listing_region":region,"stage":stage,"n_links":len(group),"n_distinct_matches":group.match_id.nunique(),"n_match_server_differs":int(group.match_server.ne(region).sum())})
    pd.DataFrame(region_rows).to_csv(OUT/"region_definition_audit.csv",index=False)

    write_json(ROOT/"qa/peer_scale_provenance.json",{"source":str(COMMON_PATH),"sha256":hashlib.sha256(COMMON_PATH.read_bytes()).hexdigest(),"regional_pooling_weights":{r:1/3 for r in REGIONS},"SESOI_SD":.02,"scales":COMMON,"self_tendency_model":"Recipient's cross-match tendency jointly entered with focal partner tendency; own_only in original output is not used.","minute_pooling":"Equal-region mean and variance sum divided by nine, not a raw mean of standardized estimates."})


if __name__=="__main__":main()
