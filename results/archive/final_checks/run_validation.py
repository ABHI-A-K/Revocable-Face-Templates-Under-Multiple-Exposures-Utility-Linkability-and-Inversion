"""Offline post-hoc validation; no network calls. Usage: python run_validation.py --e06 E06.zip --cfp CFP.zip --lfw LFW.zip --cache CACHE.zip --out results"""
import argparse,io,json,zipfile,hashlib,pathlib
import numpy as np
import pandas as pd
from scipy.linalg import hadamard
from threadpoolctl import threadpool_limits
threadpool_limits(2)
p=argparse.ArgumentParser();p.add_argument('--e06',required=True);p.add_argument('--cfp',required=True);p.add_argument('--lfw',required=True);p.add_argument('--cache',required=True);p.add_argument('--out',required=True);args=p.parse_args()
OUT=pathlib.Path(args.out);OUT.mkdir(parents=True,exist_ok=True)
SIX=zipfile.ZipFile(args.e06);C=zipfile.ZipFile(args.cfp);L=zipfile.ZipFile(args.lfw);Z=zipfile.ZipFile(args.cache)
def read(z,n):return pd.read_csv(io.BytesIO(z.read(n)),compression='gzip' if n.endswith('.gz') else None,float_precision='round_trip')
def sha(b):return hashlib.sha256(b).hexdigest()
assert all(sha(C.read(r['path']))==r['sha256'] for r in json.loads(C.read('output_manifest.json')))
a=read(C,'per_identity_attacks.csv.gz');l=read(L,'per_image_inversion_results.csv.gz');df=read(C,'frozen_subset.csv');part=read(C,'identity_partitions.csv')
assert len(a)==432000
assert np.array_equal(a.fresh_asr,(a.fresh_score>=a.fresh_threshold).astype(int))
assert np.array_equal(a.raw_asr,(a.raw_score>=a.raw_threshold).astype(int))
assert part.groupby(['split_seed','identity']).size().eq(1).all()
assert (a.exposed_row!=a.enrollment_row).all()
for seed,d in a.groupby('split_seed'):
 assert set(d.identity)==set(part.query('split_seed==@seed and partition=="test"').identity)
assert np.array_equal(df.set_index('row').loc[a.exposed_row,'identity'].to_numpy(),a.identity.to_numpy())
assert np.array_equal(df.set_index('row').loc[a.enrollment_row,'identity'].to_numpy(),a.identity.to_numpy())
schedule=json.loads(C.read('key_schedule.json'));assert all(r['query_seed'] not in r['exposure_seeds'] for r in schedule)
# Preserve all repeats of a person together. Global multinomial weights are shared across splits.
# This estimates target-person uncertainty CONDITIONAL on fitted models, partitions, keys and thresholds.
BOOT=2000;bootstrap_rows=[];contrast_rows=[]
def bootstrap_dataset(d,dataset):
 ids=np.sort(d.identity.unique());rng=np.random.default_rng(2026092401)
 W=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=BOOT)
 def estimate(t,metric):
  reps=[];points=[]
  for sp,g in t.groupby('split_seed'):
   v=g.groupby('identity')[metric].mean().reindex(ids)
   mask=v.notna().to_numpy();values=v.fillna(0).to_numpy()
   denom=W@mask.astype(float);assert (denom>0).all()
   reps.append((W@values)/denom);points.append(v.mean())
  return float(np.mean(points)),np.mean(reps,axis=0)
 metrics=['fresh_asr','raw_asr','exposed_cosine','cross_cosine']
 for key,g in d.groupby(['bits','method','K','attack']):
  for metric in metrics:
   mean,b=estimate(g,metric);lo,hi=np.quantile(b,[.025,.975])
   bootstrap_rows.append(dict(dataset=dataset,bits=key[0],method=key[1],K=key[2],attack=key[3],metric=metric,mean=mean,ci95_low=lo,ci95_high=hi,unique_people=g.identity.nunique(),bootstrap_replicates=BOOT,scope='conditional_on_fitted_models_keys_partitions_thresholds'))
 for key,g in d.groupby(['bits','method','attack']):
  p1,b1=estimate(g[g.K==1],'fresh_asr');p10,b10=estimate(g[g.K==10],'fresh_asr')
  lo,hi=np.quantile(b10-b1,[.025,.975]);contrast_rows.append(dict(dataset=dataset,bits=key[0],method=key[1],attack=key[2],contrast='K10_minus_K1',difference=p10-p1,ci95_low=lo,ci95_high=hi))
 for key,g in d[d.K==10].groupby(['bits','attack']):
  pr,br=estimate(g[g.method=='RO'],'fresh_asr')
  for m in ['SRHT','PCA_GUIDED_RO']:
   pm,bm=estimate(g[g.method==m],'fresh_asr');lo,hi=np.quantile(bm-br,[.025,.975])
   contrast_rows.append(dict(dataset=dataset,bits=key[0],method=m,attack=key[1],contrast='K10_method_minus_RO',difference=pm-pr,ci95_low=lo,ci95_high=hi))
 print('Identity bootstrap complete:',dataset,len(ids),'unique test people',flush=True)
bootstrap_dataset(a,'CFP_FP_SCREENED')
ll=[]
for variant in ['analytical','calibrated']:
 t=l[['method','exposure_count','fold','identity']].rename(columns={'exposure_count':'K','fold':'split_seed'}).copy();t['bits']=128;t['attack']=variant
 for dest,src in [('fresh_asr','cross_image_fresh_accept'),('raw_asr','cross_image_raw_accept'),('exposed_cosine','exposed_image_raw_cosine'),('cross_cosine','cross_image_raw_cosine')]:t[dest]=l[variant+'_'+src]
 ll.append(t)
bootstrap_dataset(pd.concat(ll,ignore_index=True),'LFW')
pd.DataFrame(bootstrap_rows).to_csv(OUT/'identity_bootstrap_intervals.csv',index=False)
pd.DataFrame(contrast_rows).to_csv(OUT/'paired_identity_contrasts.csv',index=False)
# Cache is cryptographically bound to frozen E06/E07 subset.
cache_name=next(n for n in Z.namelist() if pathlib.PurePosixPath(n).name=='aligned_pair_side_embeddings.npy')
b=Z.read(cache_name);assert sha(b)=='a899bf866695c2e470d4ec039b2918dbc764c5c44955241fd62ff93c6c8d6824'
E=np.load(io.BytesIO(b),allow_pickle=False);X=E[df.bin_index.to_numpy()].astype(float)
def unit(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
def ns(seed,method):return seed if method=='RO' else int.from_bytes(hashlib.sha256(f'CFP-E06|{method}|{seed}'.encode()).digest()[:8],'little')
def qr(dim,seed):
 q,r=np.linalg.qr(np.random.default_rng(seed).standard_normal((dim,dim)));return np.ascontiguousarray(q*np.where(np.diag(r)<0,-1.,1.))
def threshold(s,target):
 v=np.sort(s);budget=int(np.floor(target*len(v)));return float(np.nextafter(v[len(v)-budget-1],np.inf))
H=hadamard(512).astype(float)/np.sqrt(512);keys=list(range(24001,24021));ro=[qr(512,k) for k in keys]
assert SIX.read('frozen_subset.csv')==C.read('frozen_subset.csv')
old=read(C,'matcher_utility.csv');op=read(C,'operator_hashes.csv');rates=[.0001,.0005,.001,.005,.01]
utility=[];attack=[];resolution=[];score_blobs={}
for seed in sorted(a.split_seed.unique()):
 vp=read(C,f'pairs_s{seed}_validation.csv');tp=read(C,f'pairs_s{seed}_test.csv');vy=vp.label.to_numpy();ty=tp.label.to_numpy()
 pca=np.load(io.BytesIO(C.read(f'pca_{seed}.npz')));mu,V=pca['mean'],pca['components']
 for bits,method in [(0,'RAW')]+[(b,m) for b in [512,128] for m in ['RO','SRHT','PCA_GUIDED_RO']]:
  pdim=256 if bits==128 else 512;source=unit((X-mu)@V[:,:pdim]) if method=='PCA_GUIDED_RO' else X
  for ki,key in enumerate([0] if bits==0 else keys):
   if bits==0:
    vs=np.sum(X[vp.a]*X[vp.b],axis=1);ts=np.sum(X[tp.a]*X[tp.b],axis=1)
   else:
    if ki==0:
     packed=np.load(io.BytesIO(SIX.read(f'templates_s{seed}_{method}_b{bits}.npz')))['packed']
     templates=np.unpackbits(packed,axis=2,count=bits,bitorder='big').astype(bool)
    B=templates[ki];vs=(B[vp.a.to_numpy()]==B[vp.b.to_numpy()]).mean(1);ts=(B[tp.a.to_numpy()]==B[tp.b.to_numpy()]).mean(1)
   original=old.query('split_seed==@seed and bits==@bits and method==@method and query_seed==@key').iloc[0]
   th0=threshold(vs[vy==0],.001)
   assert th0==original.threshold
   assert np.isclose((ts[ty==1]>=th0).mean(),original.tar) and np.isclose((ts[ty==0]>=th0).mean(),original.far)
   score_blobs[f's{seed}_b{bits}_{method}_q{key}_val']=vs
   score_blobs[f's{seed}_b{bits}_{method}_q{key}_test']=ts
   for target in rates:
    th=threshold(vs[vy==0],target);vf=int(sum(vs[vy==0]>=th));tf=int(sum(ts[ty==0]>=th));ta=int(sum(ts[ty==1]>=th))
    utility.append(dict(split_seed=seed,bits=bits,method=method,query_seed=key,target_far=target,threshold=th,
        validation_impostors=int(sum(vy==0)),validation_false_accepts=vf,validation_far=vf/sum(vy==0),
        test_impostors=int(sum(ty==0)),test_genuine=int(sum(ty==1)),false_accepts=tf,genuine_accepts=ta,
        far=tf/sum(ty==0),tar=ta/sum(ty==1),validation_false_accept_budget=int(np.floor(target*sum(vy==0)))))
    # Raw ASR is reported by the originating template family as well.
    g=a[a.split_seed==seed] if bits==0 else a[(a.split_seed==seed)&(a.bits==bits)&(a.method==method)&(a.query_seed==key)]
    g=g.copy();g['success']=(g.raw_score if bits==0 else g.fresh_score)>=th
    for group,v in g.groupby(['bits','method','K','attack']):
     attack.append(dict(split_seed=seed,bits=group[0],method=group[1],K=group[2],attack=group[3],query_seed=key,
                        target_far=target,matcher='RAW' if bits==0 else 'PROTECTED',asr=v.success.mean()))
  print('Thresholds',seed,bits,method,flush=True)
u=pd.DataFrame(utility);aa=pd.DataFrame(attack)
u.to_csv(OUT/'threshold_utility_detail.csv',index=False);aa.to_csv(OUT/'threshold_attack_detail.csv',index=False)
u.groupby(['target_far','bits','method'],as_index=False)[['tar','far','validation_far']].mean().to_csv(OUT/'threshold_utility_summary.csv',index=False)
aa.groupby(['target_far','matcher','bits','method','K','attack'],as_index=False).asr.mean().to_csv(OUT/'threshold_attack_summary.csv',index=False)
np.savez_compressed(OUT/'cfp_matcher_scores.npz',**score_blobs)
# Source correspondence audit: internal checks are not promoted to independent verification.
m=read(C,'mapping_evidence.csv');src=read(C,'source_content_inventory.csv')
assert m.status.eq('SCREEN_SUPPORTED').sum()==5704
assert src.groupby('source_pixel_sha256').identity.nunique().max()==1
review=df[['encoded_sha256','identity','pose','source_path','source_pixel_sha256']].copy()
review['independent_evidence_type']='';review['independent_evidence_reference']='';review['reviewer']='';review['decision']='PENDING'
review.to_csv(OUT/'independent_source_review_required.csv',index=False)
report=dict(status='PARTIAL_CLOSURE',cfp_identity='NOT_INDEPENDENTLY_VERIFIED',cfp_rows=len(a),lfw_rows=len(l),
 cfp_unique_test_people=int(a.identity.nunique()),lfw_unique_test_people=int(l.identity.nunique()),retained_cfp_crops=len(df),
 uncertainty='2000 global identity-cluster bootstrap draws; shared person weights across splits; equal split/fold weighting; conditional on fixed fitted models, keys and thresholds; not full pipeline uncertainty',
 threshold_analysis='CFP complete: 5 validation-selected FAR targets; original operating points reproduced exactly; no test tuning',
 lfw_threshold_analysis='NOT_RUN: raw matcher calibration scores/embedding cache absent; existing per-attack scores alone cannot select new matcher thresholds',
 identity_blocker='Need original CFP source images and frozen cfp_fp.bin or authoritative generation mapping to independently establish crop-to-source correspondence. Existing geometric evidence and hashes alone cannot certify it.',
 input_sha256={pathlib.Path(v).name:sha(pathlib.Path(v).read_bytes()) for v in [args.cfp,args.lfw,args.cache,args.e06]})
(OUT/'validation_status.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
