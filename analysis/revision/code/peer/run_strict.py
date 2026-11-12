"""Estimate peer associations using the common strict holdout and discovery scales."""
from __future__ import annotations
import os
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[key] = "1"
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import study2_analysis as analysis
from config import DATA, write_json
from twoway import residualize, ClusterCodes

ROOT = Path(__file__).resolve().parents[2]
COMMON = json.loads((DATA.parent / "tables" / "study2" / "discovery" / "standardization.json").read_text())


def recipient_tendencies(rows, types, outcomes):
    """Estimate each recipient's own tendency and the focal partner's tendency jointly."""
    keys = pd.MultiIndex.from_frame(rows.loc[rows.relation.eq("self"), ["focal_pid", "mnum"]])
    out = []
    for relation in ["teammate", "opponent"]:
        r = rows[rows.relation.eq(relation) & rows.pid.isin(types.index)]
        r = r[pd.MultiIndex.from_arrays([r.pid.to_numpy(), r.mnum.to_numpy()]).isin(keys)]
        if len(r) < 1000:
            continue
        cats, conts, tv = analysis.controls(r, types, exposure_cols=tuple(f"T_{n}" for n in outcomes))
        own = types.loc[r.pid.to_numpy(), [f"T_{n}" for n in outcomes]]
        for outcome in outcomes:
            partner_raw = tv[f"T_{outcome}"].to_numpy(float)
            own_raw = own[f"T_{outcome}"].to_numpy(float)
            interaction = (partner_raw-partner_raw.mean())*(own_raw-own_raw.mean())
            res = residualize(np.column_stack([r[f"Y_{outcome}"].to_numpy(float), own_raw, partner_raw, interaction]), cats, conts)
            y, x = res[:,0], res[:,1:]
            clusters = ClusterCodes(r.focal_pid.to_numpy(), r.mnum.to_numpy())
            for model, use in [("own_only_corrected", [0]), ("own_and_partner", [0,1]), ("own_partner_interaction", [0,1,2])]:
                xx=x[:,use]
                inv=np.linalg.inv(xx.T@xx)
                b=inv@(xx.T@y)
                psi=(xx*(y-xx@b)[:,None])@inv
                cov=clusters.covariance(psi)
                for k, term in enumerate(["recipient_own_habit", "focal_partner_habit", "interaction"][:len(use)]):
                    out.append({"relation":relation,"outcome":outcome,"model":model,"term":term,"b":float(b[k]),"v":float(cov[k,k]),"n_rows":len(r),"n_focal":clusters.nf,"n_match":clusters.nm})
    return out


def main():
    analysis.c1_folder = lambda sample: ROOT / "data" / "peer" / "strict"
    analysis.strict_excluded_matches = lambda: set()
    analysis.sd_table = lambda all_rows, types: {k:COMMON[k] for k in ["pooled","region"]}
    analysis.acc_sd = lambda acc: COMMON["extra_outcome_sd"]
    analysis.fit_twoway_design = lambda rows,types,outcomes: recipient_tendencies(rows,types,analysis.NAMES)
    sys.argv=[__file__,"--sample","holdout","--variant","strict","--skip-robustness"]
    analysis.main()
    write_json(ROOT/"qa"/"peer_common_scale.json", {
        "source":str(DATA.parent/"tables/study2/discovery/standardization.json"),
        "region_weights":{"NA1":1/3,"EUW1":1/3,"KR":1/3},
        "habit_scale":"Discovery pooled reliable split-half SD for ORIV; observed habit SD for OLS.",
        "outcome_scale":"Discovery pooled teammate outcome SD; minute-specific discovery SDs for dynamics.",
        "parameters":COMMON})


if __name__=="__main__":
    main()
