"""Compute regional and role contrasts from the existing joint estimates."""
from common_revision import *

def bootstrap_record(x):
    low,high=np.quantile(x[1:],[.025,.975]); out=estimate_record(x[0],np.var(x[1:],ddof=1))
    out.update(ci_low=float(low),ci_high=float(high)); return out

def main():
    regions=[];tiers=[];roles=[];positions=[]
    scale=json.loads((DATA.parent/'tables/study2/discovery/standardization.json').read_text())['pooled']['pooled']
    for sample in SAMPLES:
        z=np.load(OLD/f'data/{sample}_regional_draws.npz');v=z['values'];ind={k[0]:i for i,k in enumerate(z['keys'])}
        r=(v[:,ind['EUW1']]-v[:,ind['NA1']])/(v[:,ind['KR']]-v[:,ind['NA1']])
        for j,p in enumerate(['maintenance','signalling']): regions.append(dict(sample=sample,measure=p,**bootstrap_record(r[:,j])))
        regions.append(dict(sample=sample,measure='signalling_minus_maintenance',**bootstrap_record(r[:,1]-r[:,0])))
        z=np.load(OLD/f'data/{sample}_tier_draws.npz');v=z['values'];ind={tuple(k):i for i,k in enumerate(z['keys'])}
        for a,b in [('KR','NA1'),('KR','EUW1'),('EUW1','NA1')]:
            low=v[:,ind[a,'BRONZE']]-v[:,ind[b,'BRONZE']];high=v[:,ind[a,'DIAMOND']]-v[:,ind[b,'DIAMOND']]
            for j,p in enumerate(['maintenance','signalling']):tiers.append(dict(sample=sample,contrast=a+'-'+b,process=p,bronze=float(low[0,j]),diamond=float(high[0,j]),**bootstrap_record(high[:,j]-low[:,j])))
        z=np.load(OLD/f'data/{sample}_position_draws.npz');v=z['values'];ind={tuple(k):i for i,k in enumerate(z['keys'])}
        for a,b in [('KR','NA1'),('KR','EUW1'),('EUW1','NA1')]:
            for role in POSITIONS:
                d=v[:,ind[a,role]]-v[:,ind[b,role]]
                for j,p in enumerate(['maintenance','signalling']):positions.append(dict(sample=sample,contrast=a+'-'+b,role=role,process=p,**bootstrap_record(d[:,j])))
        objects={r:np.load(OLD/f'data/roles_{sample}_{r}.npz') for r in REGIONS}
        for inference in ['two_way','three_way']:
            beta=np.mean([z['beta'] for z in objects.values()],axis=0);cov=sum(z[inference] for z in objects.values())/9
            for j,outcome in enumerate(BEHAVIOURS):
                for label,c in role_contrasts().items():
                    for unit,factor in [('common_sd',1.),('natural',scale[outcome]['sd_y']/scale[outcome]['sd_t_reliable'])]:
                        roles.append(dict(sample=sample,inference=inference,outcome=outcome,contrast=label,unit=unit,**estimate_record(float(c@beta[j])*factor,float(c@cov[j]@c)*factor**2)))
    for name,rows in [('regional_positions',regions),('tier_process_changes',tiers),('role_structure',roles),('position_differences',positions)]:
        f=pd.DataFrame(rows)
        if name=='tier_process_changes':
            for _,ix in f.groupby('sample').groups.items():f.loc[ix,'p_holm']=holm(f.loc[ix,'p'])
        if name=='role_structure':
            keep=['to_support','from_support','non_support','support_jungle_minus_other_support']
            for _,ix in f[f.contrast.isin(keep)].groupby(['sample','inference','unit']).groups.items():f.loc[ix,'p_holm']=holm(f.loc[ix,'p'])
        f.to_csv(WORK/f'tables/{name}.csv',index=False)
    log('Regional positions, process gradients, and role-structure contrasts saved')

if __name__=='__main__':main()
