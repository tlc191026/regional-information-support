"""Estimate common direction and tier convergence using fixed discovery scores."""
from common import *
import pandas as pd
from scipy import sparse, special
import itertools, gc

PARAMS={}
MEANS=[];SHARES=[];DISTANCES=[];TESTS=[];DIRECTION=[];ROLES=[];AUDIT=[]

def scores(frame,fit=False):
    audit=pd.read_csv(ROOT/'inputs/context/tables/context_score_transformation_audit.csv').set_index('metric')
    sub={}
    for name,metrics in GROUPS.items():
        z=[]
        for metric in metrics:
            a=audit.loc[metric+'_pm']
            logged=np.log1p(frame[metric].to_numpy(float)/frame.duration_min.to_numpy(float))
            z.append(((np.clip(logged,a.winsor_log1p_p01,a.winsor_log1p_p99)-a.pre_z_mean)/a.pre_z_sd).astype(np.float32))
        raw=np.mean(np.column_stack(z),axis=1)
        if fit:PARAMS[name]=(float(raw.mean(dtype=np.float64)),float(raw.std(dtype=np.float64)))
        mu,sd=PARAMS[name];sub[name]=((raw-mu)/sd).astype(np.float32)
    raw=np.mean(np.column_stack([sub[x] for x in ['state','request','action']]),axis=1)
    if fit:PARAMS['signalling']=(float(raw.mean(dtype=np.float64)),float(raw.std(dtype=np.float64)))
    mu,sd=PARAMS['signalling']
    return np.column_stack([sub['maintenance'],((raw-mu)/sd).astype(np.float32)])

def sufficient(frame,Y,tiers,sample):
    patches=sorted(frame.patch.astype(str).unique(),key=lambda s:tuple(int(v) for v in s.split('.')))
    pc=pd.Categorical(frame.patch.astype(str),categories=patches).codes.astype(np.int64)
    rc=pd.Categorical(frame.team_position,categories=POSITIONS).codes.astype(np.int64)
    levels=list(itertools.product(REGIONS,tiers,POSITIONS,patches))
    C=5*len(patches);F=13
    values=np.column_stack([np.ones(len(frame)),(frame.duration_min-frame.duration_min.mean())/frame.duration_min.std(ddof=0)])
    d=values[:,1]
    values=np.column_stack([values,d*d,Y,d[:,None]*Y,*[frame[fields].sum(axis=1).to_numpy(float) for fields in FAMILIES.values()]])
    assert values.shape==(len(frame),F)
    stats_all=np.zeros((REPS+1,len(levels),F))
    rng=np.random.default_rng(SEED)
    for r,region in enumerate(REGIONS):
        for t,tier in enumerate(tiers):
            ix=np.flatnonzero((frame.region.eq(region)&frame.tier.eq(tier)).to_numpy())
            player,code=np.unique(frame.puuid.iloc[ix].astype(str),return_inverse=True)
            cell=rc[ix]*len(patches)+pc[ix]
            rows=np.repeat(code,F);cols=(cell[:,None]*F+np.arange(F)).ravel()
            a=sparse.csr_matrix((values[ix].ravel(),(rows,cols)),shape=(len(player),C*F))
            w=np.vstack([np.ones(len(player)),rng.multinomial(len(player),np.full(len(player),1/len(player)),size=REPS)])
            ss=np.asarray(a.T@w.T).T.reshape(REPS+1,C,F)
            start=(r*len(tiers)+t)*C;stats_all[:,start:start+C]=ss
            AUDIT.append(dict(sample=sample,region=region,tier=tier,n_players=len(player),n_records=len(ix)))
            log(f'{sample}: bootstrap sufficient statistics {region} {tier}')
    reference={p:float(np.mean(pc==j)) for j,p in enumerate(patches)}
    return stats_all,levels,patches,reference

def fit_models(S,levels,patches,reference,tiers,sample):
    meta=pd.DataFrame(levels,columns=['region','tier','position','patch'])
    for context in ['regional','tier','position']:
        names=['intercept'];columns=[np.ones(len(meta))]
        for k,levs in [('region',REGIONS),('tier',tiers),('position',POSITIONS),('patch',patches)]:
            for lev in levs[1:]:names.append(k+'|'+lev);columns.append(meta[k].eq(lev).to_numpy(float))
        if context!='regional':
            levs=tiers if context=='tier' else POSITIONS
            for reg in REGIONS[1:]:
                for lev in levs[1:]:
                    names.append('interaction|'+reg+'|'+lev);columns.append((meta.region.eq(reg)&meta[context].eq(lev)).to_numpy(float))
        Z=np.column_stack(columns);p=Z.shape[1]+1
        levs=['all'] if context=='regional' else (tiers if context=='tier' else POSITIONS)
        L=[];keys=[]
        for reg in REGIONS:
            for lev in levs:
                mask=meta.region.eq(reg)
                if context!='regional':mask &= meta[context].eq(lev)
                weights=meta.loc[mask,'patch'].map(reference).to_numpy(float)
                weights=weights/weights.sum()
                L.append(np.r_[weights@Z[mask],0]);keys.append((reg,lev))
        L=np.asarray(L);draws=np.empty((REPS+1,len(keys),2));fail=0
        for start in range(0,REPS+1,25):
            a=S[start:start+25];n=len(a);gram=np.zeros((n,p,p));rhs=np.zeros((n,p,2))
            gram[:,:-1,:-1]=np.einsum('ci,bc,cj->bij',Z,a[:,:,0],Z,optimize=True)
            gram[:,-1,:-1]=a[:,:,1]@Z;gram[:,:-1,-1]=gram[:,-1,:-1]
            gram[:,-1,-1]=a[:,:,2].sum(axis=1)
            rhs[:,:-1]=np.einsum('ci,bcq->biq',Z,a[:,:,3:5],optimize=True)
            rhs[:,-1]=a[:,:,5:7].sum(axis=1)
            for j in range(n):
                ev,Q=np.linalg.eigh((gram[j]+gram[j].T)/2)
                keep=ev>ev[-1]*1e-12
                if keep.all():b=np.linalg.solve(gram[j],rhs[j])
                else:
                    b=(Q[:,keep]/ev[keep])@(Q[:,keep].T@rhs[j]);fail+=1
                draws[start+j]=L@b
        np.savez_compressed(WORK/f'data/{sample}_{context}_draws.npz',values=draws,keys=np.array(keys),processes=PROCESSES)
        for j,(reg,lev) in enumerate(keys):
            for k,measure in enumerate(PROCESSES):MEANS.append(dict(sample=sample,context=context,region=reg,level=lev,measure=measure,**summary(draws[0,j,k],draws[1:,j,k])))
        AUDIT.append(dict(sample=sample,model=context,bootstrap_pseudoinverse_solves=fail,n_parameters=p,maximum_absolute_adjusted_mean=float(np.max(np.abs(draws)))))
        if context=='regional':direction(draws,sample)
        elif context=='tier':convergence(draws.reshape(REPS+1,3,len(tiers),2),'level',tiers,sample)
        else:role_order(draws,sample)
        log(f'{sample}: exact bootstrap OLS {context} complete')

def direction(v,sample):
    def determinant(a):
        d=a[:,2]-a[:,0];e=a[:,1]-a[:,0]
        return d[:,0]*e[:,1]-d[:,1]*e[:,0]
    result=summary(determinant(v)[0],determinant(v)[1:])
    centered=v-v.mean(axis=1,keepdims=True)
    u,s,vh=np.linalg.svd(centered,full_matrices=False)
    explained=s[:,0]**2/(s*s).sum(axis=1)
    projected=(centered@vh[:,0,:,None])*vh[:,0,None,:]
    residual=centered-projected
    DIRECTION.append(dict(sample=sample,test='common_direction_determinant',**result,
        explained_fraction=float(explained[0]),explained_low=float(np.quantile(explained[1:],.025)),explained_high=float(np.quantile(explained[1:],.975)),
        rms_orthogonal_residual=float(np.sqrt(np.mean(np.sum(residual[0]**2,axis=1))))))
    for r,region in enumerate(REGIONS):
        DIRECTION.append(dict(sample=sample,test='region_projection',region=region,x=v[0,r,0],y=v[0,r,1],
            projected_x=projected[0,r,0]+v[0,:,0].mean(),projected_y=projected[0,r,1]+v[0,:,1].mean(),
            residual_x=residual[0,r,0],residual_y=residual[0,r,1]))

def distance(p,q,kind):
    if kind=='level':return np.sqrt(np.mean((p-q)**2,axis=-1))
    m=(p+q)/2
    return np.sqrt(np.maximum(0,(special.rel_entr(p,m).sum(axis=-1)+special.rel_entr(q,m).sum(axis=-1))/(2*np.log(2))))

def convergence(v,kind,tiers,sample):
    pairs=[(2,0),(2,1),(1,0)]
    ds=[]
    for a,b in pairs:
        d=distance(v[:,a],v[:,b],kind);ds.append(d)
        for j,tier in enumerate(tiers):DISTANCES.append(dict(sample=sample,kind=kind,contrast=REGIONS[a]+'-'+REGIONS[b],tier=tier,**summary(d[0,j],d[1:,j])))
    all_d=np.mean(ds,axis=0)
    for j,tier in enumerate(tiers):DISTANCES.append(dict(sample=sample,kind=kind,contrast='mean_three_pairs',tier=tier,**summary(all_d[0,j],all_d[1:,j])))
    for start in [tiers[0]]+(['BRONZE'] if tiers[0]=='IRON' else []):
        j=tiers.index(start);change=all_d[:,-1]-all_d[:,j]
        TESTS.append(dict(sample=sample,kind=kind,contrast='DIAMOND-'+start,**summary(change[0],change[1:])))
    np.savez_compressed(WORK/f'data/{sample}_{kind}_distances.npz',distances=np.array(ds),mean=all_d,tiers=tiers)

def role_order(v,sample):
    expected={'maintenance':['UTILITY','JUNGLE','MIDDLE','BOTTOM','TOP'],'signalling':['JUNGLE','MIDDLE','UTILITY','BOTTOM','TOP']}
    for k,measure in enumerate(PROCESSES):
        order=[POSITIONS.index(x) for x in expected[measure]]
        for r,reg in enumerate(REGIONS):
            a=v[:,r*5:(r+1)*5,k]
            agreement=np.all(np.diff(a[1:,order],axis=1)<0,axis=1).mean()
            for i in range(4):
                d=a[:,order[i]]-a[:,order[i+1]]
                ROLES.append(dict(sample=sample,region=reg,measure=measure,contrast=expected[measure][i]+'-'+expected[measure][i+1],rank_agreement=agreement,**summary(d[0],d[1:])))

def content(S,tiers,patches,sample):
    counts=S[:,:,7:13].reshape(REPS+1,3,len(tiers),5,len(patches),6).sum(axis=4)
    assert (counts.sum(axis=-1)>0).all()
    shares=(counts/counts.sum(axis=-1,keepdims=True)).mean(axis=3)
    assert np.allclose(shares.sum(axis=-1),1)
    for r,reg in enumerate(REGIONS):
        for t,tier in enumerate(tiers):
            for k,family in enumerate(FAMILIES):SHARES.append(dict(sample=sample,region=reg,tier=tier,family=family,**summary(shares[0,r,t,k],shares[1:,r,t,k])))
    np.savez_compressed(WORK/f'data/{sample}_content_draws.npz',shares=shares,tiers=tiers,families=list(FAMILIES))
    convergence(shares,'content',tiers,sample)

def main():
    for sample,file,tiers in [('discovery','study1_focal_rows.parquet',TIERS),('holdout','study1_holdout_rows_primary.parquet',TIERS[1:]),('strict','study1_holdout_rows_strict.parquet',TIERS[1:])]:
        log('Read '+sample);f=pd.read_parquet(DATA/file)
        Y=scores(f,fit=(sample=='discovery'))
        if sample=='discovery':
            orig=pd.read_parquet(ROOT/'inputs/context/data/context_unresidualized_participant_scores.parquet',columns=['match_id','puuid','shared_information_maintenance','team_signalling'])
            assert not orig.duplicated(['puuid','match_id']).any() and not f.duplicated(['puuid','match_id']).any()
            orig=orig.set_index(['puuid','match_id']).reindex(pd.MultiIndex.from_frame(f[['puuid','match_id']]))
            assert orig.notna().all().all()
            err=np.abs(Y-orig[['shared_information_maintenance','team_signalling']].to_numpy()).max()
            assert err<2e-5,err
            dump(WORK/'qa/score_parameters.json',dict(subscales=PARAMS,maximum_reconstruction_error=float(err)))
            Y=orig[['shared_information_maintenance','team_signalling']].to_numpy(float)
            del orig
        valid=f.team_position.isin(POSITIONS)&f.tier.isin(tiers)
        f=f.loc[valid].reset_index(drop=True);Y=Y[valid]
        S,levels,patches,reference=sufficient(f,Y,tiers,sample)
        dump(WORK/f'qa/{sample}_reference.json',dict(tiers=tiers,patch_weights=reference,n_records=len(f),n_players=f.puuid.nunique(),n_unique_matches=f.match_id.nunique()))
        content(S,tiers,patches,sample)
        fit_models(S,levels,patches,reference,tiers,sample)
        del f,Y,S;gc.collect()
    for name,rows in [('adjusted_means',MEANS),('signal_tier_shares',SHARES),('tier_distances',DISTANCES),('common_direction',DIRECTION),('role_order',ROLES)]:
        pd.DataFrame(rows).to_csv(WORK/f'tables/{name}.csv',index=False)
    tests=pd.DataFrame(TESTS)
    for _,ix in tests.groupby(['sample','contrast']).groups.items():tests.loc[ix,'p_holm']=holm(tests.loc[ix,'p'])
    tests.to_csv(WORK/'tables/convergence_tests.csv',index=False)
    pd.DataFrame(AUDIT).to_csv(WORK/'qa/regional_estimation.csv',index=False)
    log('Regional tests complete')

if __name__=='__main__':main()
