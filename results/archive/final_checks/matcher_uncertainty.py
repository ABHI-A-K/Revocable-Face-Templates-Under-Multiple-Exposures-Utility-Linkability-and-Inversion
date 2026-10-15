"""Run after run_validation.py. Conditional test-person bootstrap for CFP matcher rates."""
import argparse,zipfile,io,pathlib
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
threadpool_limits(2)
p=argparse.ArgumentParser();p.add_argument('--cfp',required=True);p.add_argument('--results',required=True);args=p.parse_args();out=pathlib.Path(args.results)
z=zipfile.ZipFile(args.cfp)
def read(n):return pd.read_csv(io.BytesIO(z.read(n)),float_precision='round_trip')
df=read('frozen_subset.csv');parts=read('identity_partitions.csv');ids=np.sort(parts[parts.partition=='test'].identity.unique());ix={v:i for i,v in enumerate(ids)}
rng=np.random.default_rng(2026092402);W=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=2000)
scores=np.load(out/'cfp_matcher_scores.npz');thresholds=pd.read_csv(out/'threshold_utility_detail.csv',float_precision='round_trip');results=[]
for (target,bits,method),g in thresholds.groupby(['target_far','bits','method']):
 rep={'tar':[],'far':[]};point={'tar':[],'far':[]}
 for split,t in g.groupby('split_seed'):
  pairs=read(f'pairs_s{split}_test.csv');p1=df.iloc[pairs.a].identity.to_numpy();p2=df.iloc[pairs.b].identity.to_numpy()
  acceptance=np.mean([scores[f's{split}_b{bits}_{method}_q{int(r.query_seed)}_test']>=r.threshold for r in t.itertuples()],axis=0)
  for metric,label in [('tar',1),('far',0)]:
   m=pairs.label.to_numpy()==label;ii=np.array([ix[v] for v in p1[m]]);jj=np.array([ix[v] for v in p2[m]])
   weights=W[:,ii].astype(float)
   if label==0:weights*=W[:,jj] # Both people determine an impostor comparison.
   denom=weights.sum(1);assert (denom>0).all();rep[metric].append((weights@acceptance[m])/denom);point[metric].append(acceptance[m].mean())
 for metric in rep:
  lo,hi=np.quantile(np.mean(rep[metric],axis=0),[.025,.975]);results.append(dict(target_far=target,bits=bits,method=method,metric=metric,mean=np.mean(point[metric]),ci95_low=lo,ci95_high=hi,scope='fixed_validation_thresholds_and_templates; global_test_identity_bootstrap',zero_event_warning=bool(metric=='far' and lo==hi==0)))
 print(target,bits,method,flush=True)
pd.DataFrame(results).to_csv(out/'matcher_identity_intervals.csv',index=False)
