"""Reproduce the corrected peer analyses using existing local data."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--skip-build",action="store_true");args=parser.parse_args()
    code=Path(__file__).resolve().parent;work=code.parents[1]
    (work/"logs").mkdir(parents=True,exist_ok=True)
    env=dict(os.environ)
    env.update({name:"1" for name in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","PYTHONDONTWRITEBYTECODE"]})
    steps=[] if args.skip_build else [("build_strict.py",[],"peer_strict_build")]
    steps += [("run_strict.py",[],"peer_strict_analysis")]
    steps += [("cluster_sensitivity.py",["--sample",s],f"peer_clusters_{s}") for s in ["discovery","holdout","strict"]]
    steps += [("own_tendencies.py",["--sample",s],f"peer_own_{s}") for s in ["discovery","holdout"]]
    steps += [("assemble_peer_results.py",[],"peer_assembly"),("validate_peer.py",[],"peer_validation"),("export_summary_tables.py",[],"peer_export")]
    for script,extra,name in steps:
        print(f"Running {name}",flush=True)
        with (work/"logs"/f"{name}.log").open("w",encoding="utf-8") as log:
            subprocess.run([sys.executable,"-X","utf8","-B",str(code/script),*extra],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print("Peer revision completed",flush=True)


if __name__=="__main__":main()
