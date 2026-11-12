"""Estimate recipient and partner tendencies for all six behavioural outcomes."""
from __future__ import annotations
import os
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:os.environ[key]="1"
import argparse
import gc
from pathlib import Path
import pandas as pd
import study2_analysis as analysis
from run_strict import recipient_tendencies
from config import REGIONS, log


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--sample",choices=["discovery","holdout"],required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    types=analysis.load_types(args.sample,10);records=[]
    for region in REGIONS:
        log(f"{args.sample} {region}: recipient tendency models")
        rows=analysis.load_rows(args.sample,region,types,{},None,None)
        records.extend([{**rec,"region":region} for rec in recipient_tendencies(rows,types,analysis.NAMES)])
        del rows;gc.collect()
    out=root/"tables/peer"/args.sample;out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(out/"own_model_raw.csv",index=False)
    log(f"{args.sample}: recipient tendency models saved")


if __name__=="__main__":main()
