"""Shared paths, fixed analysis definitions, and reproducibility utilities."""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys, json, hashlib, time
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy import stats

WORK = Path(__file__).resolve().parents[1]
REV = WORK.parent
ROOT = REV.parent
OLD = REV / 'theory'
SI = REV / 'supplementary'
DATA = ROOT / 'baseline/data'
REGIONS = ['NA1', 'EUW1', 'KR']
SAMPLES = ['discovery', 'holdout', 'strict']
POSITIONS = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']
BEHAVIOURS = ['wards_placed','control_wards','wards_killed']
PAIRS = [(a,b) for a in POSITIONS for b in POSITIONS if a != b]
SEED = 191026
for sub in ['code','data','tables','logs','qa','manuscript','figures/main','figures/supplementary','source_data']:
    (WORK/sub).mkdir(parents=True, exist_ok=True)

def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)

def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=lambda x:x.item() if isinstance(x,np.generic) else str(x)), encoding='utf-8')

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2), b''): h.update(chunk)
    return h.hexdigest()

def holm(values):
    p=np.asarray(values,float); order=np.argsort(p); out=np.empty_like(p)
    out[order]=np.minimum(1,np.maximum.accumulate(p[order]*np.arange(len(p),0,-1)))
    return out

def estimate_record(estimate, variance):
    se=float(np.sqrt(max(float(variance),0)))
    p=float(2*stats.norm.sf(abs(estimate/se))) if se>0 else float('nan')
    return dict(estimate=float(estimate),se=se,ci_low=float(estimate-1.95996398454*se),
                ci_high=float(estimate+1.95996398454*se),p=p)

def role_contrasts():
    masks={
        'to_support':[b=='UTILITY' for a,b in PAIRS],
        'from_support':[a=='UTILITY' for a,b in PAIRS],
        'non_support':[a!='UTILITY' and b!='UTILITY' for a,b in PAIRS],
        'support_jungle':[set((a,b))=={'UTILITY','JUNGLE'} for a,b in PAIRS],
        'other_support':['UTILITY' in (a,b) and 'JUNGLE' not in (a,b) for a,b in PAIRS],
        'non_jungle_to_support':[b=='UTILITY' and a!='JUNGLE' for a,b in PAIRS],
    }
    out={k:np.asarray(v,float)/sum(v) for k,v in masks.items()}
    out['support_jungle_minus_other_support']=out['support_jungle']-out['other_support']
    out['to_support_minus_non_support']=out['to_support']-out['non_support']
    out.update({'|'.join(pair):np.eye(20)[j] for j,pair in enumerate(PAIRS)})
    return out

def region_check(frame, column='mnum'):
    regions=frame[column].astype(str).str.split('_').str[0]
    if regions.str.startswith('TW').any():
        raise ValueError('Taiwan matches are excluded from all analyses')
    return regions.value_counts().to_dict()
