"""Estimate early-window associations by the timing of subsequent objectives."""
from common_revision import *
import argparse,gc,warnings
import pyarrow.parquet as pq
from scipy import special
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

STATE=['gold_at_window_diff','xp_at_window_diff','level_at_window_diff','kills_pre_diff','neutral_objectives_pre_diff','turret_plates_pre_diff']
MAP=['forward_diff','enemy_half_diff','jungle_cs_diff']
COMPONENTS=['valid_wards_placed','valid_ward_kills','control_ward_purchases']
WINDOWS={'w0_5':list(range(1,6)),'w5_8':list(range(6,9)),'w8_10':[9,10]}
OUTCOME='first_post_neutral_blue'

def aggregate(source):
    path=WORK/f'data/minute_teams_{source}.parquet'
    if path.exists():return pd.read_parquet(path)
    folder=DATA/'timeline_participants'/('fixed' if source=='discovery' else 'holdout')
    counts=[];audit=[]
    cols=['match_id','team_id','window_valid','cutoff_ms']+[f'{c}_m{k:02d}' for c in COMPONENTS for k in range(1,11)]
    for j,file in enumerate(sorted(folder.glob('*.parquet'))):
        f=pd.read_parquet(file,columns=cols);f=f[f.window_valid&f.match_id.str.split('_').str[0].isin(REGIONS)].copy()
        for c in COMPONENTS:
            for w,ks in WINDOWS.items():f[f'{c}_{w}']=f[[f'{c}_m{k:02d}' for k in ks]].sum(axis=1)
        countcols=[f'{c}_{w}' for c in COMPONENTS for w in WINDOWS]
        g=f.groupby(['match_id','team_id'],observed=True)[countcols].sum().unstack('team_id')
        n=f.groupby(['match_id','team_id'],observed=True).size().unstack('team_id')
        good=n.eq(5).all(axis=1);g=g.loc[good]
        out=pd.DataFrame({'match_id':g.index})
        for c in countcols:
            out['blue_'+c]=g[(c,100)].to_numpy();out['red_'+c]=g[(c,200)].to_numpy()
        cutoff=f.groupby('match_id',observed=True).cutoff_ms.first()
        out['participant_cutoff_ms']=cutoff.reindex(out.match_id).to_numpy();counts.append(out)
        audit.append(dict(file=file.name,complete_matches=len(out),incomplete_matches=int((~good).sum())))
        if j%60==0:log(f'{source}: minute aggregation {j+1} shards')
    out=pd.concat(counts,ignore_index=True);assert not out.match_id.duplicated().any()
    out.to_parquet(path,index=False);pd.DataFrame(audit).to_csv(WORK/f'qa/timeline_shards_{source}.csv',index=False)
    return out

def objective_times(source):
    dest=WORK/f'data/objective_times_{source}.parquet'
    if dest.exists():return pd.read_parquet(dest)
    cols=['match_id','first_post_neutral_ms','analysis_cutoff_ms']
    if source=='discovery':f=pd.read_parquet(ROOT/'timeline/data/timeline_features_0_10.parquet',columns=cols)
    else:
        frames=[]
        for p in sorted((DATA/'timeline_team_holdout').glob('*.parquet')):
            names=pq.ParquetFile(p).schema.names;selected=[c for c in cols if c in names]
            a=pd.read_parquet(p,columns=selected+['window'],filters=[('window','==','0_10')])
            frames.append(a.drop(columns='window'))
        f=pd.concat(frames,ignore_index=True)
    assert not f.match_id.duplicated().any();f.to_parquet(dest,index=False);return f

def prepare(sample):
    tag='holdout_strict' if sample=='strict' else sample
    source='discovery' if sample=='discovery' else 'holdout'
    f=pd.read_parquet(REV/f'data/regional_performance/performance_{tag}.parquet')
    original=len(f)
    prefix=f.match_id.str.split('_').str[0]
    excluded=int((~prefix.isin(REGIONS)).sum())
    reassigned=int((f.region.ne(prefix)&prefix.isin(REGIONS)).sum())
    f=f[prefix.isin(REGIONS)].copy();f['region']=f.match_id.str.split('_').str[0]
    g=aggregate(source);t=objective_times(source)
    f=f.merge(g,on='match_id',how='inner',validate='one_to_one').merge(t,on='match_id',how='inner',validate='one_to_one')
    if 'analysis_cutoff_ms' not in f:f['analysis_cutoff_ms']=f.participant_cutoff_ms
    delta=(f.participant_cutoff_ms-f.analysis_cutoff_ms).abs()
    if delta.max()>1:raise ValueError('Incompatible timeline boundaries')
    f['delay_ms']=f.first_post_neutral_ms-f.analysis_cutoff_ms
    assert (f.delay_ms>0).all()
    f['delay_group']=pd.cut(f.delay_ms,bins=[0,60000,180000,np.inf],labels=['near','middle','far']).astype(str)
    checks=[]
    for c in COMPONENTS:
        diff=np.zeros(len(f))
        for w in WINDOWS:
            col=f'{c}_{w}_diff';f[col]=f[f'blue_{c}_{w}'].astype(float)-f[f'red_{c}_{w}'].astype(float);diff+=f[col]
        error=diff-f[c+'_diff']
        checks.append(dict(component=c,n_different=int((error!=0).sum()),max_abs=float(abs(error).max())))
    dump(WORK/f'qa/timeline_input_{sample}.json',dict(original_n=original,excluded_other_servers=excluded,reassigned_to_actual_server=reassigned,analysis_n=len(f),boundary_max_difference_ms=float(delta.max()),count_checks=checks,delay_counts=f.delay_group.value_counts().to_dict(),servers=f.match_id.str.split('_').str[0].value_counts().to_dict()))
    # Minute bins use nominal minute ends; any boundary-only discrepancy is retained and reported.
    scale_path=WORK/'data/window_scales.json'
    if sample=='discovery':
        scale={}
        for w in WINDOWS:
            scale[w]={};zs=[]
            for side in ['blue','red']:
                parts=[]
                for c in COMPONENTS:
                    if c not in scale[w]:
                        a=np.log1p(np.concatenate([f[f'blue_{c}_{w}'],f[f'red_{c}_{w}']]))
                        lo,hi=np.quantile(a,[.01,.99]);a=np.clip(a,lo,hi)
                        scale[w][c]=[float(lo),float(hi),float(a.mean()),float(a.std())]
                    lo,hi,mu,sd=scale[w][c]
                    parts.append((np.clip(np.log1p(f[f'{side}_{c}_{w}']),lo,hi)-mu)/sd)
                zs.append(np.mean(parts,axis=0))
            d=zs[0]-zs[1];scale[w]['diff_mean']=float(d.mean());scale[w]['diff_sd']=float(d.std())
        dump(scale_path,scale)
    else:scale=json.loads(scale_path.read_text())
    for w in WINDOWS:
        zs=[]
        for side in ['blue','red']:
            parts=[]
            for c in COMPONENTS:
                lo,hi,mu,sd=scale[w][c];parts.append((np.clip(np.log1p(f[f'{side}_{c}_{w}']),lo,hi)-mu)/sd)
            zs.append(np.mean(parts,axis=0))
        f['score_'+w]=(zs[0]-zs[1]-scale[w]['diff_mean'])/scale[w]['diff_sd']
    f.to_parquet(WORK/f'data/timeline_windows_{sample}.parquet',index=False)
    return f

def design(f,predictors,delay=False):
    columns=[np.ones(len(f))];names=['intercept']
    cats=['region','tier','patch']+(['delay_group'] if delay else [])
    for c in cats:
        values=f[c].astype(str)
        for level in sorted(values.unique())[1:]:columns.append(values.eq(level).to_numpy(float));names.append(c+'_'+level)
    for c in STATE+MAP:
        a=f[c].to_numpy(float);columns.append((a-a.mean())/(a.std() or 1));names.append(c)
    for c in predictors:columns.append(f[c].to_numpy(float));names.append(c)
    return np.column_stack(columns),names

def fit(f,predictors,sample,spec,interaction=False):
    log(f'{sample}: {spec} {len(f):,} matches')
    x,names=design(f,predictors,interaction);y=f[OUTCOME].to_numpy(float)
    model=LogisticRegression(C=1e8,solver='newton-cholesky',fit_intercept=False,max_iter=150,tol=1e-9)
    with warnings.catch_warnings(record=True) as ww:
        warnings.simplefilter('always');model.fit(x,y)
    beta=model.coef_[0];eta=x@beta;p=special.expit(eta);r=y-p
    h=(x*(p*(1-p))[:,None]).T@x;bread=np.linalg.pinv(h)
    cov=bread@((x*(r*r)[:,None]).T@x)@bread
    rows=[];grads=[]
    for pred in predictors:
        j=names.index(pred)
        group=pred.rsplit('__',1)[1] if interaction else 'all'
        keep=f.delay_group.eq(group).to_numpy() if interaction else np.ones(len(f),bool)
        xx=x[keep];pp=p[keep];p1=special.expit(eta[keep]+beta[j]);w1=p1*(1-p1)
        grad=xx.T@(w1-pp*(1-pp))/len(xx);grad[j]+=w1.mean()
        rec=estimate_record(float((p1-pp).mean())*100,float(grad@cov@grad)*10000)
        rows.append(dict(sample=sample,spec=spec,predictor=pred,delay_group=group,n_matches=len(f),n_evaluation=int(keep.sum()),beta=float(beta[j]),beta_se=float(np.sqrt(cov[j,j])),odds_ratio=float(np.exp(beta[j])),**rec));grads.append(grad)
    gradients=np.array(grads);amecov=gradients@cov@gradients.T*10000
    np.savez_compressed(WORK/f'data/timeline_model_{sample}_{spec}.npz',beta=beta,covariance=cov,names=np.asarray(names),predictors=np.asarray(predictors),ame_covariance=amecov)
    dump(WORK/f'qa/timeline_model_{sample}_{spec}.json',dict(n=len(f),iterations=model.n_iter_.tolist(),score_max=float(np.max(abs(x.T@r))/len(f)),warnings=[str(w.message) for w in ww],n_columns=x.shape[1]))
    tests=[]
    if interaction:
        for w in WINDOWS:
            js=[names.index(f'score_{w}__{g}') for g in ['near','middle','far']]
            L=np.zeros((2,len(names)));L[0,js[0]]=1;L[0,js[1]]=-1;L[1,js[0]]=1;L[1,js[2]]=-1
            d=L@beta;v=L@cov@L.T;chi=float(d@np.linalg.pinv(v)@d)
            tests.append(dict(sample=sample,spec=spec,test='delay_heterogeneity',window=w,chi_square=chi,df=2,p=float(stats.chi2.sf(chi,2))))
        for g in ['near','middle','far']:
            for a,b in [('w8_10','w0_5'),('w8_10','w5_8'),('w5_8','w0_5')]:
                ia=predictors.index(f'score_{a}__{g}');ib=predictors.index(f'score_{b}__{g}')
                d=rows[ia]['estimate']-rows[ib]['estimate'];v=amecov[ia,ia]+amecov[ib,ib]-2*amecov[ia,ib]
                tests.append(dict(sample=sample,spec=spec,test='window_difference',window=a+'-'+b,delay_group=g,**estimate_record(d,v)))
    del x,model;gc.collect()
    return rows,tests

def run(sample):
    cached=WORK/f'data/timeline_windows_{sample}.parquet'
    f=pd.read_parquet(cached) if cached.exists() else prepare(sample)
    scores=['score_'+w for w in WINDOWS];records=[];tests=[]
    specs=[('joint_windows',f,scores,False),('exclude_60_seconds',f[f.delay_ms>60000],scores,False)]
    for w in WINDOWS:
        for g in ['near','middle','far']:f[f'score_{w}__{g}']=f['score_'+w]*f.delay_group.eq(g)
    ip=[f'score_{w}__{g}' for w in WINDOWS for g in ['near','middle','far']]
    specs.append(('window_by_delay',f,ip,True))
    natural=[f'{c}_{w}_diff' for c in COMPONENTS for w in WINDOWS]
    specs.append(('natural_components',f,natural,False))
    if sample=='discovery':
        ids=pd.read_parquet(ROOT/'timeline/data/all_participant_disjoint_matches.parquet').match_id
        sub=f[f.match_id.isin(ids)]
        specs += [('participant_disjoint_windows',sub,scores,False),('participant_disjoint_delay',sub,ip,True)]
    for name,data,preds,interaction in specs:
        rec,tt=fit(data,preds,sample,name,interaction);records+=rec;tests+=tt
    r=pd.DataFrame(records)
    for _,ix in r.groupby('spec').groups.items():r.loc[ix,'p_holm']=holm(r.loc[ix,'p'])
    r.to_csv(WORK/f'tables/timeline_extensions_{sample}.csv',index=False)
    t=pd.DataFrame(tests)
    for _,ix in t.groupby(['spec','test']).groups.items():t.loc[ix,'p_holm']=holm(t.loc[ix,'p'])
    t.to_csv(WORK/f'tables/timeline_contrasts_{sample}.csv',index=False)
    log(f'{sample}: timeline extensions complete')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--sample',choices=SAMPLES,default='discovery');a=ap.parse_args()
    with threadpool_limits(limits=4):run(a.sample)
