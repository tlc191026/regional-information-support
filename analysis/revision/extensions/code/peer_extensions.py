"""Cross-activity, role robustness, and complete-team count models."""
from common_revision import *
import argparse, gc
from scipy import sparse, linalg
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(REV/'code/peer'))
import study2_analysis as analysis
from cluster_sensitivity import MultiwayCodes, COMMON
from twoway import ClusterCodes

ORIGINAL_FOLDER=analysis.c1_folder
PAIRCODES=['|'.join(p) for p in PAIRS]

def residualize_sparse(values,cats,conts):
    """Residualize against categorical and continuous controls using sparse cross-products."""
    n=len(values);blocks=[]
    for j,cat in enumerate(cats):
        codes,levels=pd.factorize(cat);assert (codes>=0).all()
        a=sparse.csr_matrix((np.ones(n),(np.arange(n),codes)),shape=(n,len(levels)))
        blocks.append(a if j==0 else a[:,1:])
    if conts is not None and conts.shape[1]:blocks.append(sparse.csr_matrix(conts))
    x=sparse.hstack(blocks,format='csr');del blocks
    gram=(x.T@x).toarray();cross=x.T@values
    coef,_,rank,_=linalg.lstsq(gram,cross,cond=1e-12,lapack_driver='gelsy')
    out=np.empty_like(values)
    for start in range(0,n,100000):out[start:start+100000]=values[start:start+100000]-x[start:start+100000]@coef
    err=float(np.max(np.abs(x.T@out))/max(1,n))
    if err>1e-6:raise ValueError(f'Control residualization error {err}')
    return out,dict(control_columns=int(x.shape[1]),control_rank=int(rank),residual_crossproduct_max_per_row=err)

def joint_oriv(Y,A,B,clusters):
    """Fit multiple outcomes with identical split-half instruments and joint covariance."""
    aa=A.T@A;bb=B.T@B;ab=A.T@B
    F1=B@np.linalg.solve(bb,ab.T);F2=A@np.linalg.solve(aa,ab)
    gram=A.T@F1+B.T@F2;inv=np.linalg.inv(gram)
    beta=inv@(F1.T@Y+F2.T@Y)
    n,k=A.shape;q=Y.shape[1];psi=np.empty((n,k*q))
    for j in range(q):
        psi[:,j*k:(j+1)*k]=(F1*(Y[:,j]-A@beta[:,j])[:,None]+F2*(Y[:,j]-B@beta[:,j])[:,None])@inv.T
    covs=clusters.covariance(psi)
    if isinstance(covs,np.ndarray):covs={'two_way':(covs,float(np.linalg.eigvalsh(covs).min()))}
    return beta.T,covs,dict(condition_number=float(np.linalg.cond(gram)))

def load(sample,region):
    base='discovery' if sample=='discovery' else 'holdout'
    analysis.c1_folder=(lambda name:REV/'data/peer/strict') if sample=='strict' else ORIGINAL_FOLDER
    types=analysis.load_types(base,10)
    oldcols=analysis.ROW_COLS
    analysis.ROW_COLS=oldcols+(['champion_id'] if 'champion_id' not in oldcols else [])
    rows=analysis.load_rows(base,region,types,{},None,None)
    analysis.ROW_COLS=oldcols
    before=len(rows)
    rows=rows[~rows.mnum.astype(str).str.startswith('TW')].copy()
    servers=region_check(rows)
    dump(WORK/f'qa/peer_server_audit_{sample}_{region}.json',dict(listing_region=region,excluded_tw_records=before-len(rows),match_server_counts=servers,region_definition='focal player listing region; complete match prefixes retained'))
    return rows,types

def fit_roles(rows,types,exposure,outcomes,spec,clusters):
    labels=rows.focal_position.astype(str)+'|'+rows.team_position.astype(str)
    pc=pd.Categorical(labels,categories=PAIRCODES).codes
    extra=[]
    if spec in ['recipient_champion','both_champions']:
        rows=rows.copy();rows['recipient_champion_role']=rows.champion_id.astype(str)+'|'+rows.team_position.astype(str);extra.append('recipient_champion_role')
    if spec in ['focal_champion','both_champions']:extra.append('focal_champion')
    cats,conts,tv=analysis.controls(rows,types,extra_cats=tuple(extra),exposure_cols=(f'TA_{exposure}',f'TB_{exposure}'))
    n=len(rows);q=len(outcomes);values=np.zeros((n,q+40))
    values[:,:q]=rows[[f'Y_{o}' for o in outcomes]].to_numpy(float)
    values[np.arange(n),q+pc]=tv[f'TA_{exposure}'].to_numpy()
    values[np.arange(n),q+20+pc]=tv[f'TB_{exposure}'].to_numpy()
    res,diag=residualize_sparse(values,cats,conts);del values,tv,cats,conts;gc.collect()
    beta,cov,condition=joint_oriv(res[:,:q],res[:,q:q+20],res[:,q+20:],clusters)
    del res;gc.collect()
    return beta,cov,{**diag,**condition}

def save_role_fit(sample,region,spec,exposure,outcomes,beta,cov,diag,rows):
    dest=WORK/f'data/peer_{sample}_{region}_{spec}_{exposure}.npz'
    np.savez_compressed(dest,beta=beta,two_way=cov['two_way'][0],three_way=cov['three_way'][0],pairs=np.asarray(PAIRS),outcomes=np.asarray(outcomes))
    dump(dest.with_suffix('.json'),dict(sample=sample,region=region,spec=spec,exposure=exposure,n_records=len(rows),n_focal=int(rows.focal_pid.nunique()),n_recipient=int(rows.pid.nunique()),n_matches=int(rows.mnum.nunique()),min_eigen_two=cov['two_way'][1],min_eigen_three=cov['three_way'][1],**diag))

def complete_teams(rows,types,sample,region):
    """Estimate total placement association on intact five-member focal-match units."""
    dest=WORK/f'data/team_{sample}_{region}.npz'
    if dest.exists():return
    r=rows[rows.relation.isin(['self','teammate'])].copy()
    g=r.groupby(['focal_pid','mnum'],observed=True)
    valid=g.size().eq(5)&g.pid.nunique().eq(5)&g.team_position.nunique().eq(5)
    idx=pd.MultiIndex.from_frame(r[['focal_pid','mnum']]);r=r[idx.isin(valid.index[valid])]
    own=r[r.relation.eq('self')].copy()
    totals=r.groupby(['focal_pid','mnum'],observed=True)['Y_wards_placed'].sum().rename('team_placements')
    own=own.join(totals,on=['focal_pid','mnum']);own['peer_placements']=own.team_placements-own.Y_wards_placed
    assert own.groupby(['focal_pid','mnum']).size().max()==1
    cats,conts,tv=analysis.controls(own,types,self_rows=True,exposure_cols=('TA_wards_placed','TB_wards_placed'))
    pc=pd.Categorical(own.focal_position,categories=POSITIONS).codes;n=len(own)
    values=np.zeros((n,13));values[:,:3]=own[['Y_wards_placed','peer_placements','team_placements']].to_numpy()
    values[np.arange(n),3+pc]=tv.TA_wards_placed.to_numpy();values[np.arange(n),8+pc]=tv.TB_wards_placed.to_numpy()
    res,diag=residualize_sparse(values,cats,conts)
    cs=ClusterCodes(own.focal_pid.to_numpy(),own.mnum.to_numpy())
    beta,cov,condition=joint_oriv(res[:,:3],res[:,3:8],res[:,8:13],cs)
    np.savez_compressed(dest,beta=beta,two_way=cov['two_way'][0],roles=np.asarray(POSITIONS),outcomes=['own','four_teammates','whole_team'])
    dump(dest.with_suffix('.json'),dict(n_units=n,n_focal=int(own.focal_pid.nunique()),n_matches=int(own.mnum.nunique()),**diag,**condition))
    assert np.allclose(beta[0]+beta[1],beta[2],atol=1e-8)
    log(f'{sample} {region}: complete-team model {n:,} units')

def run(sample):
    for region in REGIONS:
        log(f'{sample} {region}: loading records')
        allrows,types=load(sample,region)
        complete_teams(allrows,types,sample,region)
        raw=allrows[allrows.relation.isin(['self','teammate'])]
        unit_sizes=raw.groupby(['focal_pid','mnum']).size()
        intact=unit_sizes.index[unit_sizes.eq(5)]
        rows=allrows[allrows.relation.eq('teammate')&allrows.focal_position.ne(allrows.team_position)].reset_index(drop=True)
        del allrows,raw;gc.collect()
        assert not rows[['focal_pid','pid','mnum']].duplicated().any()
        cs=MultiwayCodes(rows.focal_pid.to_numpy(),rows.pid.to_numpy(),rows.mnum.to_numpy())
        for exposure in BEHAVIOURS:
            dest=WORK/f'data/peer_{sample}_{region}_base_{exposure}.npz'
            if dest.exists():continue
            log(f'{sample} {region}: cross-activity {exposure}')
            beta,cov,diag=fit_roles(rows,types,exposure,BEHAVIOURS,'base',cs)
            save_role_fit(sample,region,'base',exposure,BEHAVIOURS,beta,cov,diag,rows)
        for spec in ['recipient_champion','focal_champion','both_champions','complete_units']:
            dest=WORK/f'data/peer_{sample}_{region}_{spec}_wards_placed.npz'
            if dest.exists():continue
            log(f'{sample} {region}: placement robustness {spec}')
            if spec=='complete_units':
                key=pd.MultiIndex.from_frame(rows[['focal_pid','mnum']]);r=rows[key.isin(intact)].reset_index(drop=True)
                clusters=MultiwayCodes(r.focal_pid.to_numpy(),r.pid.to_numpy(),r.mnum.to_numpy())
            else:r=rows;clusters=cs
            beta,cov,diag=fit_roles(r,types,'wards_placed',['wards_placed'],spec,clusters)
            save_role_fit(sample,region,spec,'wards_placed',['wards_placed'],beta,cov,diag,r)
        # Role-specific scales use unique focal-role records and observed recipient outcomes.
        scale=[]
        for role in POSITIONS:
            ids=rows.loc[rows.focal_position.eq(role),'focal_pid'].unique();tt=types.loc[ids]
            for o in BEHAVIOURS:
                tcov=np.cov(tt[f'TA_{o}'],tt[f'TB_{o}'])[0,1]
                scale.append(dict(sample=sample,region=region,role=role,outcome=o,sd_t_reliable=float(np.sqrt(max(tcov,0))),sd_y=float(rows.loc[rows.team_position.eq(role),f'Y_{o}'].std())))
        pd.DataFrame(scale).to_csv(WORK/f'tables/role_scales_{sample}_{region}.csv',index=False)
        del rows,types,cs;gc.collect()
        log(f'{sample} {region}: complete')

def collect():
    records=[];teams=[];diagnostics=[]
    for sample in SAMPLES:
        for spec in ['base','recipient_champion','focal_champion','both_champions','complete_units']:
            for exposure in (BEHAVIOURS if spec=='base' else ['wards_placed']):
                objects=[np.load(WORK/f'data/peer_{sample}_{r}_{spec}_{exposure}.npz') for r in REGIONS]
                beta=np.mean([z['beta'] for z in objects],axis=0);outcomes=list(objects[0]['outcomes'])
                for inference in ['two_way','three_way']:
                    V=sum(z[inference] for z in objects)/9
                    for j,outcome in enumerate(outcomes):
                        for contrast,c in role_contrasts().items():
                            v=c@V[j*20:(j+1)*20,j*20:(j+1)*20]@c;b=float(c@beta[j])
                            for unit,factor in [('natural',1.),('common_sd',COMMON['pooled']['pooled'][exposure]['sd_t_reliable']/COMMON['pooled']['pooled'][outcome]['sd_y'])]:
                                records.append(dict(sample=sample,spec=spec,exposure=exposure,outcome=outcome,inference=inference,contrast=contrast,unit=unit,**estimate_record(b*factor,v*factor**2)))
        objs=[np.load(WORK/f'data/team_{sample}_{r}.npz') for r in REGIONS]
        B=np.mean([z['beta'] for z in objs],axis=0);V=sum(z['two_way'] for z in objs)/9
        for j,outcome in enumerate(['own','four_teammates','whole_team']):
            for k,role in enumerate(POSITIONS):teams.append(dict(sample=sample,outcome=outcome,role=role,**estimate_record(B[j,k],V[j*5+k,j*5+k])))
        for k,role in enumerate(POSITIONS):
            den=B[0,k];num=B[2,k];g=np.zeros(15);g[10+k]=1/den;g[k]=-num/den**2
            teams.append(dict(sample=sample,outcome='team_to_own_ratio',role=role,**estimate_record(num/den,g@V@g)))
    f=pd.DataFrame(records)
    primary=(f.spec.eq('base')&f.exposure.eq('wards_placed')&f.outcome.isin(['control_wards','wards_killed'])&f.contrast.isin(['JUNGLE|UTILITY','non_jungle_to_support']))
    for _,ix in f[primary].groupby(['sample','inference','unit']).groups.items():f.loc[ix,'p_holm_primary']=holm(f.loc[ix,'p'])
    core=f.contrast.isin(['to_support','from_support','non_support','support_jungle_minus_other_support'])
    for _,ix in f[core].groupby(['sample','spec','exposure','inference','unit']).groups.items():f.loc[ix,'p_holm_structure']=holm(f.loc[ix,'p'])
    f.to_csv(WORK/'tables/peer_extensions.csv',index=False);pd.DataFrame(teams).to_csv(WORK/'tables/team_placement_associations.csv',index=False)
    for p in (WORK/'data').glob('peer_*.json'):diagnostics.append(json.loads(p.read_text()))
    pd.DataFrame(diagnostics).to_csv(WORK/'qa/peer_model_diagnostics.csv',index=False)
    log('Peer extension tables assembled')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--sample',choices=SAMPLES);ap.add_argument('--collect',action='store_true');args=ap.parse_args()
    with threadpool_limits(limits=4):
        if args.sample:run(args.sample)
        if args.collect:collect()
