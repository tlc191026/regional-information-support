"""Shared paths, fixed definitions, and output utilities."""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys
sys.dont_write_bytecode=True
import hashlib, json, time
import numpy as np
from scipy import stats

WORK=Path(__file__).resolve().parents[1]
REV=WORK.parent
ROOT=REV.parent
DATA=ROOT/'baseline/data'
REGIONS=['NA1','EUW1','KR']
TIERS=['IRON','BRONZE','SILVER','GOLD','PLATINUM','EMERALD','DIAMOND']
POSITIONS=['TOP','JUNGLE','MIDDLE','BOTTOM','UTILITY']
PROCESSES=['maintenance','signalling']
GROUPS={
 'maintenance':['wards_placed','wards_killed','control_wards_bought'],
 'state':['enemy_missing_pings','enemy_vision_pings','command_pings'],
 'request':['need_vision_pings','assist_me_pings'],
 'action':['on_my_way_pings','push_pings','all_in_pings','get_back_pings']}
FAMILIES={'caution':['get_back_pings','danger_pings'],'initiation':['push_pings','all_in_pings'],
 'intent':['on_my_way_pings'],'request':['assist_me_pings','need_vision_pings'],
 'enemy_information':['enemy_missing_pings','enemy_vision_pings','vision_cleared_pings'],
 'tactical':['command_pings','hold_pings','bait_pings']}
SEED=191026
REPS=1000
for name in ['code','tables','data','logs','qa','manuscript','source_data/main','source_data/supplementary','figures/main','figures/supplementary']:
    (WORK/name).mkdir(parents=True,exist_ok=True)

def log(message):
    print(time.strftime('%H:%M:%S'),message,flush=True)

def dump(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8388608),b''):h.update(chunk)
    return h.hexdigest()

def holm(p):
    p=np.asarray(p,float);order=np.argsort(p)
    out=np.empty_like(p);out[order]=np.minimum(1,np.maximum.accumulate(p[order]*np.arange(len(p),0,-1)))
    return out

def summary(estimate,draws):
    draws=np.asarray(draws);se=float(draws.std(ddof=1))
    lo,hi=np.quantile(draws,[.025,.975])
    return dict(estimate=float(estimate),ci_low=float(lo),ci_high=float(hi),se=se,p=float(2*stats.norm.sf(abs(estimate/se))) if se>0 else float('nan'))
