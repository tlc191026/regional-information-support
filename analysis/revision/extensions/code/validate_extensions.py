"""Check replicated estimates, sample boundaries, and calculation identities."""
from common_revision import *
sys.path.insert(0,str(REV/'code/peer'))
from cluster_sensitivity import COMMON
checks=[]
def check(name,ok,detail=None):
    checks.append(dict(check=name,pass_=bool(ok),detail=detail))
for sample in SAMPLES:
    for region in REGIONS:
        old=np.load(OLD/f'data/roles_{sample}_{region}.npz')
        for j,b in enumerate(BEHAVIOURS):
            z=np.load(WORK/f'data/peer_{sample}_{region}_base_{b}.npz')
            factor=COMMON['pooled']['pooled'][b]['sd_t_reliable']/COMMON['pooled']['pooled'][b]['sd_y']
            error=float(abs(z['beta'][j]*factor-old['beta'][j]).max())
            check(f'{sample}/{region}/{b}: reproduce frozen same-activity slopes',error<1e-7,error)
            for inf in ['two_way','three_way']:
                error=float(abs(z[inf][j*20:(j+1)*20,j*20:(j+1)*20]*factor**2-old[inf][j]).max())
                check(f'{sample}/{region}/{b}/{inf}: covariance replication',error<1e-8,error)
        t=np.load(WORK/f'data/team_{sample}_{region}.npz')
        check(f'{sample}/{region}: whole team equals own plus four teammates',np.allclose(t['beta'][0]+t['beta'][1],t['beta'][2],atol=1e-8))
        a=json.loads((WORK/f'qa/peer_server_audit_{sample}_{region}.json').read_text())
        check(f'{sample}/{region}: no Taiwan matches',not any(k.startswith('TW') for k in a['match_server_counts']))
    a=json.loads((WORK/f'qa/timeline_input_{sample}.json').read_text())
    check(f'{sample}: event windows sum to original ten-minute counts',all(v['n_different']==0 for v in a['count_checks']),a['count_checks'])
    check(f'{sample}: state and event cutoffs agree',a['boundary_max_difference_ms']<=1)
    check(f'{sample}: actual timeline servers',set(a['servers'])==set(REGIONS))
    f=pd.read_parquet(WORK/f'data/timeline_windows_{sample}.parquet',columns=['match_id','delay_ms','delay_group'])
    check(f'{sample}: one row per match and subsequent objective',not f.match_id.duplicated().any() and f.delay_ms.gt(0).all())
for p in (WORK/'qa').glob('timeline_model_*.json'):
    a=json.loads(p.read_text());check(p.stem+': convergence',not a['warnings'] and a['score_max']<1e-7,a)
for p in (WORK/'data').glob('peer_*.json'):
    a=json.loads(p.read_text());check(p.stem+': residualization',a['residual_crossproduct_max_per_row']<1e-6)
dump(WORK/'qa/analysis_validation.json',dict(passed=sum(c['pass_'] for c in checks),failed=sum(not c['pass_'] for c in checks),checks=checks))
print('PASS',sum(c['pass_'] for c in checks),'FAIL',sum(not c['pass_'] for c in checks))
for c in checks:
    if not c['pass_']:print(c)
assert all(c['pass_'] for c in checks)
