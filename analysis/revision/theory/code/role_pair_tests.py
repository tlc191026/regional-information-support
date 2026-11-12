"""Estimate directed role-specific ORIV slopes and joint role contrasts."""
from common import *
import pandas as pd
import itertools, gc, argparse
sys.path.insert(0,str(REV/'code/peer'))
import study2_analysis as analysis
from twoway import residualize
from cluster_sensitivity import MultiwayCodes, COMMON

PAIRS=[(a,b) for a in POSITIONS for b in POSITIONS if a!=b]
OUTCOMES=['wards_placed','control_wards','wards_killed']
original_folder=analysis.c1_folder

def oriv_matrix(y,A,B):
    aa=A.T@A;bb=B.T@B;ab=A.T@B
    p1=np.linalg.solve(bb,ab.T);p2=np.linalg.solve(aa,ab)
    F1=B@p1;F2=A@p2
    gram=A.T@F1+B.T@F2
    inv=np.linalg.inv(gram)
    beta=inv@(F1.T@y+F2.T@y)
    psi=((F1*(y-A@beta)[:,None])+(F2*(y-B@beta)[:,None]))@inv.T
    return beta,psi,dict(condition_number=float(np.linalg.cond(gram)),minimum_first_stage_eigenvalue=float(min(np.linalg.eigvalsh(aa).min(),np.linalg.eigvalsh(bb).min())))

def compute_region(sample,region):
    base='discovery' if sample=='discovery' else 'holdout'
    analysis.c1_folder=(lambda name:REV/'data/peer/strict') if sample=='strict' else original_folder
    types=analysis.load_types(base,10)
    rows=analysis.load_rows(base,region,types,{},None,None)
    rows=rows[rows.relation.eq('teammate')&rows.focal_position.ne(rows.team_position)].reset_index(drop=True)
    assert not rows[['focal_pid','pid','mnum']].duplicated().any()
    log(f'{sample} {region}: {len(rows):,} directed teammate records')
    labels=rows.focal_position.astype(str)+'|'+rows.team_position.astype(str)
    pc=pd.Categorical(labels,categories=['|'.join(x) for x in PAIRS]).codes
    assert (pc>=0).all() and set(np.unique(pc))==set(range(20))
    cs=MultiwayCodes(rows.focal_pid.to_numpy(),rows.pid.to_numpy(),rows.mnum.to_numpy())
    outcomes=[];betas=[];cov2=[];cov3=[];diagnostics=[]
    for outcome in OUTCOMES:
        log(f'{sample} {region}: fit {outcome}')
        cats,conts,tv=analysis.controls(rows,types,exposure_cols=(f'TA_{outcome}',f'TB_{outcome}'))
        n=len(rows);values=np.zeros((n,41));values[:,0]=rows[f'Y_{outcome}']
        values[np.arange(n),1+pc]=tv[f'TA_{outcome}'].to_numpy()
        values[np.arange(n),21+pc]=tv[f'TB_{outcome}'].to_numpy()
        res=residualize(values,cats,conts,chunk=100000)
        del values,tv,cats,conts;gc.collect()
        beta,psi,diag=oriv_matrix(res[:,0],res[:,1:21],res[:,21:41])
        cov=cs.covariance(psi)
        scale=COMMON['pooled']['pooled'][outcome]['sd_t_reliable']/COMMON['pooled']['pooled'][outcome]['sd_y']
        betas.append(beta*scale);cov2.append(cov['two_way'][0]*scale**2);cov3.append(cov['three_way'][0]*scale**2)
        diagnostics.append(dict(sample=sample,region=region,outcome=outcome,scale=scale,
            minimum_two_way_eigenvalue=cov['two_way'][1],minimum_three_way_eigenvalue=cov['three_way'][1],**diag,**cs.counts))
        del res,beta,psi,cov;gc.collect()
    counts=pd.Series(pc).value_counts().sort_index().to_numpy()
    np.savez_compressed(WORK/f'data/roles_{sample}_{region}.npz',beta=np.array(betas),two_way=np.array(cov2),three_way=np.array(cov3),counts=counts,outcomes=OUTCOMES,pairs=np.array(PAIRS))
    dump(WORK/f'qa/roles_{sample}_{region}.json',diagnostics)

def collect():
    rows=[];counts=[]
    cj=PAIRS.index(('UTILITY','JUNGLE'));jc=PAIRS.index(('JUNGLE','UTILITY'))
    pair_mean=np.zeros(20);pair_mean[[cj,jc]]=.5
    other=np.full(20,1/18);other[[cj,jc]]=0
    contrasts={**{'|'.join(p):np.eye(20)[i] for i,p in enumerate(PAIRS)},
        'support_jungle_mean':pair_mean,'other_18_mean':other,'support_jungle_minus_others':pair_mean-other}
    for sample in ['discovery','holdout','strict']:
        objects={r:np.load(WORK/f'data/roles_{sample}_{r}.npz') for r in REGIONS}
        for region in REGIONS+['pooled']:
            B=np.mean([v['beta'] for v in objects.values()],axis=0) if region=='pooled' else objects[region]['beta']
            for inference in ['two_way','three_way']:
                V=sum(v[inference] for v in objects.values())/9 if region=='pooled' else objects[region][inference]
                for j,outcome in enumerate(OUTCOMES):
                    for label,c in contrasts.items():
                        b=float(c@B[j]);se=float(np.sqrt(max(0,c@V[j]@c)))
                        rows.append(dict(sample=sample,region=region,inference=inference,outcome=outcome,contrast=label,
                            estimate=b,se=se,ci_low=b-1.95996398454*se,ci_high=b+1.95996398454*se,
                            ci90_low=b-1.64485362695*se,ci90_high=b+1.64485362695*se,p=float(2*stats.norm.sf(abs(b/se))),
                            tost_p=float(max(stats.norm.sf((b+.02)/se),stats.norm.cdf((b-.02)/se)))))
        for reg,v in objects.items():
            for p,n in zip(PAIRS,v['counts']):counts.append(dict(sample=sample,region=reg,pair='|'.join(p),n_records=int(n)))
    frame=pd.DataFrame(rows)
    mask=frame.region.eq('pooled')&frame.contrast.isin(['support_jungle_mean','support_jungle_minus_others'])
    for _,ix in frame[mask].groupby(['sample','inference']).groups.items():frame.loc[ix,'p_holm']=holm(frame.loc[ix,'p'])
    frame.to_csv(WORK/'tables/role_pair_results.csv',index=False)
    pd.DataFrame(counts).to_csv(WORK/'tables/role_pair_counts.csv',index=False)
    log('Directed role contrasts complete')

def cache_signature():
    paths=[DATA/'player_listing.parquet',DATA/'focal_valid_matches.parquet',
        DATA/'holdout/primary/holdout_player_matches.parquet',
        ROOT/'timeline/data/canonical_focal_links.parquet',
        DATA.parent/'tables/study2/discovery/standardization.json',
        WORK/'code/common.py',Path(__file__)]
    for folder in [DATA/'c1',DATA/'c1_holdout',REV/'data/peer/strict']:
        paths += [folder/'focal_types.parquet']+[folder/f'outcome_rows_{region}.parquet' for region in REGIONS]
    paths += [REV/'code/peer'/name for name in ['study2_analysis.py','twoway.py','cluster_sensitivity.py','config.py']]
    return {str(path.relative_to(ROOT)):sha(path) for path in paths}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--refit',action='store_true',help='Re-estimate all nine region-by-sample models.');args=parser.parse_args()
    signature=cache_signature();stamp=WORK/'qa/role_cache_signature.json'
    existing=list((WORK/'data').glob('roles_*.npz'))
    if existing and not args.refit:
        if not stamp.exists() or json.loads(stamp.read_text(encoding='utf-8'))!=signature:
            raise RuntimeError('Role cache provenance is missing or has changed. Run with --refit to rebuild the estimates.')
    for sample in ['discovery','holdout','strict']:
        for region in REGIONS:
            if args.refit or not (WORK/f'data/roles_{sample}_{region}.npz').exists():compute_region(sample,region)
    collect()
    dump(stamp,signature)
