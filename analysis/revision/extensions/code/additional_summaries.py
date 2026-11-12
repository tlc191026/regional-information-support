"""Export role-scaled associations and the prespecified cross-activity comparisons."""
from common_revision import *
f=pd.read_csv(WORK/'tables/peer_extensions.csv',keep_default_na=False)
team=pd.read_csv(WORK/'tables/team_placement_associations.csv')
primary=[]
for sample in SAMPLES:
    z=f[(f['sample']==sample)&(f.spec=='base')&(f.exposure=='wards_placed')&(f.outcome!='wards_placed')&(f.contrast.isin(['JUNGLE|UTILITY','non_jungle_to_support']))&(f.inference=='two_way')&(f.unit=='common_sd')].copy()
    z['test']=z.outcome+' / '+z.contrast
    t=team[(team['sample']==sample)&(team.outcome=='four_teammates')&team.role.isin(['UTILITY','JUNGLE'])].copy();t['test']='four teammates / '+t.role
    t['unit']='events per 10 min per one-unit tendency difference'
    g=pd.concat([z,t],ignore_index=True);g['p_holm_six']=holm(g.p);primary.append(g)
pd.concat(primary,ignore_index=True).to_csv(WORK/'tables/primary_extension_tests.csv',index=False)
rows=[]
for sample in SAMPLES:
    for exposure in BEHAVIOURS:
        coefs=[];vs=[]
        for region in REGIONS:
            z=np.load(WORK/f'data/peer_{sample}_{region}_base_{exposure}.npz')
            s=pd.read_csv(WORK/f'tables/role_scales_discovery_{region}.csv').set_index(['role','outcome'])
            factor=np.asarray([[s.loc[(a,exposure),'sd_t_reliable']/s.loc[(b,o),'sd_y'] for a,b in PAIRS] for o in BEHAVIOURS])
            coefs.append(z['beta']*factor);vs.append(z['two_way']*factor.ravel()[:,None]*factor.ravel()[None,:])
        beta=np.mean(coefs,axis=0);v=sum(vs)/9
        for j,o in enumerate(BEHAVIOURS):
            for name,c in role_contrasts().items():
                rows.append(dict(sample=sample,exposure=exposure,outcome=o,contrast=name,**estimate_record(c@beta[j],c@v[j*20:(j+1)*20,j*20:(j+1)*20]@c)))
pd.DataFrame(rows).to_csv(WORK/'tables/role_specific_scales.csv',index=False)
print('Additional summaries complete')
