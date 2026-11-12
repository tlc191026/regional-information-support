"""Compare two-way and three-way clustered inference for matched behavioural outcomes."""
from __future__ import annotations
import os
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[key]="1"
import argparse
import gc
import json
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import study2_analysis as analysis
from twoway import residualize
from config import DATA, REGIONS, write_json, log

ROOT=Path(__file__).resolve().parents[2]
COMMON=json.loads((DATA.parent/"tables/study2/discovery/standardization.json").read_text())
REGION_OFFSETS={r:(i+1)*10**12 for i,r in enumerate(REGIONS)}


class MultiwayCodes:
    """Precompute all intersections for focal, recipient and match clusters."""
    def __init__(self, focal, recipient, match):
        arrays=[pd.factorize(v)[0] for v in [focal,recipient,match]]
        self.parts=[]
        self.counts={}
        for k in range(1,4):
            for dims in itertools.combinations(range(3),k):
                codes=arrays[dims[0]]
                for dim in dims[1:]:
                    codes=pd.factorize(codes.astype(np.int64)*(int(arrays[dim].max())+1)+arrays[dim])[0]
                n=int(codes.max())+1
                self.parts.append((dims,codes,n,1.0 if k%2 else -1.0))
                self.counts["_".join(["focal","recipient","match"][d] for d in dims)]=n

    def covariance(self,psi):
        two=np.zeros((psi.shape[1],psi.shape[1]));three=two.copy()
        for dims,codes,n,sign in self.parts:
            sums=np.column_stack([np.bincount(codes,weights=psi[:,j],minlength=n) for j in range(psi.shape[1])])
            cov=sign*n/(n-1)*(sums.T@sums)
            three+=cov
            if 1 not in dims:
                two+=cov
        result={}
        for name,cov in [("two_way",two),("three_way",three)]:
            eig,vec=np.linalg.eigh((cov+cov.T)/2)
            result[name]=((vec*np.clip(eig,0,None))@vec.T,float(eig.min()))
        return result


def estimate(rows,types,region):
    blocks={}
    for relation in ["teammate","opponent"]:
        r=rows[rows.relation.eq(relation)]
        tcols=[f"{p}_{e}" for e in analysis.NAMES for p in ["T","TA","TB"]]
        cats,conts,tv=analysis.controls(r,types,exposure_cols=tcols)
        values=np.column_stack([r[[f"Y_{n}" for n in analysis.NAMES]].to_numpy(float),tv[tcols].to_numpy(float)])
        res=residualize(values,cats,conts,chunk=120_000)
        blocks[relation]={"res":dict(zip(analysis.NAMES+tcols,res.T)),"focal":r.focal_pid.to_numpy(),"recipient":r.pid.to_numpy(),"match":r.mnum.to_numpy(),"n":len(r)}
        del r,values,res,cats,conts,tv
    cs=MultiwayCodes(np.concatenate([b["focal"] for b in blocks.values()]),np.concatenate([b["recipient"] for b in blocks.values()]),np.concatenate([b["match"] for b in blocks.values()]))
    out=[];eigen=[]
    for outcome in analysis.NAMES:
        for method in ["ols","oriv"]:
            bs=[];psi=np.zeros((sum(b["n"] for b in blocks.values()),2));start=0
            for j,relation in enumerate(["teammate","opponent"]):
                block=blocks[relation];res=block["res"]
                if method=="oriv":b,p=analysis.oriv_slope(res[outcome],res[f"TA_{outcome}"],res[f"TB_{outcome}"])
                else:b,p=analysis.slope(res[outcome],res[f"T_{outcome}"])
                bs.append(b);psi[start:start+block["n"],j]=p;start+=block["n"]
            covs=cs.covariance(psi)
            for inference,(cov,min_eig) in covs.items():
                out.append({"exposure":outcome,"outcome":outcome,"method":method,"region":region,"spec":inference,
                    "b_teammate":bs[0],"b_opponent":bs[1],"v_teammate":cov[0,0],"v_opponent":cov[1,1],"c_teammate_opponent":cov[0,1],
                    "n_teammate":blocks["teammate"]["n"],"n_opponent":blocks["opponent"]["n"],"n_focal":cs.counts["focal"],"n_match":cs.counts["match"]})
                eigen.append({"region":region,"outcome":outcome,"method":method,"inference":inference,"minimum_covariance_eigenvalue":min_eig})
    return out,cs.counts,eigen


def add_flags(frame):
    from scipy import stats
    frame=frame.copy()
    frame["significant_two_sided_05"]=frame.p.lt(.05)
    frame["tost_p"]=np.maximum(stats.norm.sf((frame.b_std+.02)/frame.se_std),stats.norm.cdf((frame.b_std-.02)/frame.se_std))
    frame["meaningful_beyond_02"]=frame.beyond_sesoi
    return frame


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--sample",choices=["discovery","holdout","strict"],required=True);args=parser.parse_args()
    sample=args.sample
    if sample=="strict":analysis.c1_folder=lambda name:ROOT/"data/peer/strict"
    types=analysis.load_types("discovery" if sample=="discovery" else "holdout",10)
    records=[];qa={};eigen=[]
    for region in REGIONS:
        log(f"{sample} {region}: three-way sensitivity loading")
        rows=analysis.load_rows("discovery" if sample=="discovery" else "holdout",region,types,{},None,None)
        rows=rows[rows.relation.ne("self")]
        rec,count,ev=estimate(rows,types,region)
        records+=rec;qa[region]=count;eigen+=ev
        log(f"{sample} {region}: three-way sensitivity complete")
        del rows;gc.collect()
    out=ROOT/"tables/peer"/sample;out.mkdir(parents=True,exist_ok=True)
    result=analysis.add_standardized(analysis.combine(records,["exposure","outcome","method","spec"]),COMMON)
    add_flags(result).to_csv(out/"cluster_sensitivity.csv",index=False)
    pd.DataFrame(records).to_csv(out/"cluster_covariance_source.csv",index=False)
    pd.DataFrame(eigen).to_csv(out/"cluster_covariance_diagnostics.csv",index=False)
    write_json(ROOT/"qa"/f"peer_clusters_{sample}.json",qa)
    log(f"{sample}: sensitivity saved")


if __name__=="__main__":main()
