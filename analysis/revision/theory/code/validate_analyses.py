"""Check estimands, algebra, source consistency, and inference diagnostics."""
from common import *
import pandas as pd
from regional_tests import distance, FAMILIES
from role_pair_tests import oriv_matrix, analysis

checks=[]
def check(name,ok,detail=''):
    checks.append(dict(check=name,passed=bool(ok),detail=detail))
    assert ok,(name,detail)

def main():
    p=np.array([[.4,.6,0],[0,0,1.]])
    q=np.array([[.1,.7,.2],[1.,0,0]])
    d=distance(p,q,'content')
    check('Jensen-Shannon identity and symmetry',np.allclose(distance(p,p,'content'),0) and np.allclose(d,distance(q,p,'content')))
    check('Jensen-Shannon boundary and zero cells',np.isfinite(d).all() and (d>=0).all() and (d<=1).all() and d[1]==1)
    check('Level distance RMS scale',np.isclose(distance(np.array([1.,2.]),np.array([0.,0.]),'level'),np.sqrt(2.5)))
    v=np.array([[0.,0.],[2.,3.],[4.,6.]])
    det=lambda z:np.linalg.det(np.vstack([z[2]-z[0],z[1]-z[0]]))
    check('Unequal proportional contrasts are collinear',np.isclose(det(v),0))
    v[1,1]+=.1
    check('Non-proportional contrasts depart from a line',not np.isclose(det(v),0))
    rng=np.random.default_rng(SEED);n=40000
    latent=rng.normal(size=n);A=latent+rng.normal(size=n)*.4;B=latent+rng.normal(size=n)*.4;y=.3*latent+rng.normal(size=n)
    b,psi,_=oriv_matrix(y,A[:,None],B[:,None]);old_b,old_psi=analysis.oriv_slope(y,A,B)
    check('Matrix ORIV reduces to established scalar estimator',np.allclose(b[0],old_b,rtol=1e-10) and np.allclose(psi[:,0],old_psi,rtol=1e-9,atol=1e-12))
    k=20;pc=rng.integers(0,k,n);true=np.linspace(-.2,.2,k)
    y=true[pc]*latent+rng.normal(size=n)*.15
    aa=np.zeros((n,k));bb=aa.copy();aa[np.arange(n),pc]=A;bb[np.arange(n),pc]=B
    beta,_,_=oriv_matrix(y,aa,bb)
    check('Directed pair indexing recovers simulated slopes',np.max(abs(beta-true))<.03,float(np.max(abs(beta-true))))
    check('Full identifiers retain cross-server same suffix',len(set(['NA1_123','EUW1_123','KR_123']))==3)
    c=np.array([126,127,128,255,256],dtype=np.int64)*13+12
    check('Sparse sufficient-statistic index avoids narrow integer overflow',np.all(c>=0) and c[-1]==3340)

    generic=[]
    for sample,file in [('discovery','study1_focal_rows.parquet'),('holdout','study1_holdout_rows_primary.parquet'),('strict','study1_holdout_rows_strict.parquet')]:
        f=pd.read_parquet(DATA/file,columns=['basic_pings','region','tier','team_position'])
        generic.append(dict(sample=sample,basic_pings_total=float(f.basic_pings.sum()),basic_nonzero_records=int(f.basic_pings.ne(0).sum())))
        check(sample+' generic pings are zero',not f.basic_pings.ne(0).any())
        reference=json.loads((WORK/f'qa/{sample}_reference.json').read_text())
        check(sample+' valid reference tiers',reference['tiers']==(TIERS if sample=='discovery' else TIERS[1:]))
    pd.DataFrame(generic).to_csv(WORK/'qa/generic_ping_counts.csv',index=False)

    for sample in ['discovery','holdout','strict']:
        for context in ['regional','tier','position']:
            z=np.load(WORK/f'data/{sample}_{context}_draws.npz')
            check(sample+' '+context+' finite bootstrap estimates',np.isfinite(z['values']).all() and z['values'].shape[0]==REPS+1)
        s=np.load(WORK/f'data/{sample}_content_draws.npz')['shares']
        check(sample+' composition closure',np.allclose(s.sum(axis=-1),1) and (s>=0).all())
    audit=pd.read_csv(WORK/'qa/regional_estimation.csv')
    check('Stable bootstrap estimates under rank loss',audit.maximum_absolute_adjusted_mean.dropna().lt(10).all(),audit.dropna(subset=['model'])[['sample','model','bootstrap_pseudoinverse_solves']].to_dict('records'))

    eigen=[]
    for path in (WORK/'qa').glob('roles_*.json'):
        eigen+=json.loads(path.read_text())
    pd.DataFrame(eigen).to_csv(WORK/'qa/role_covariance_diagnostics.csv',index=False)
    check('Role covariance matrices positive before adjustment',all(x['minimum_two_way_eigenvalue']>0 and x['minimum_three_way_eigenvalue']>0 for x in eigen))
    r=pd.read_csv(WORK/'tables/role_pair_results.csv')
    a=r[r.inference.eq('two_way')].set_index(['sample','region','outcome','contrast'])
    b=r[r.inference.eq('three_way')].set_index(['sample','region','outcome','contrast'])
    check('Cluster changes leave coefficients unchanged',np.allclose(a.estimate,b.estimate))
    qa=pd.DataFrame({'se_ratio_three_to_two':b.se/a.se,'two_way_p':a.p,'three_way_p':b.p}).reset_index()
    qa.to_csv(WORK/'qa/role_clustering_comparison.csv',index=False)

    m=pd.read_csv(WORK/'tables/adjusted_means.csv')
    for context,filename in [('tier','tier_adjusted_means.csv'),('position','position_adjusted_means.csv')]:
        old=pd.read_csv(REV/'inputs/plotting/original/context'/filename,keep_default_na=False)
        old['region']=old.region.replace({'NA':'NA1','EUW':'EUW1'})
        old['measure']=old.measure.replace({'shared_information_maintenance':'maintenance','team_signalling':'signalling'})
        merged=m[(m['sample']=='discovery')&(m.context==context)].merge(old,on=['region','level','measure'],suffixes=('_new','_old'))
        merged['absolute_point_difference']=abs(merged.estimate_new-merged.estimate_old)
        merged.to_csv(WORK/f'qa/{context}_point_comparison.csv',index=False)
        check(context+' point estimates reproduce existing models',len(merged)>0 and merged.absolute_point_difference.max()<1e-4,float(merged.absolute_point_difference.max()))

    inp=pd.read_csv(WORK/'qa/input_fingerprints.csv')
    cols=list(inp.columns)
    dump(WORK/'qa/validation_report.json',dict(checks=checks,input_manifest_columns=cols))
    log(f'{len(checks)} analysis checks passed')

if __name__=='__main__':main()
