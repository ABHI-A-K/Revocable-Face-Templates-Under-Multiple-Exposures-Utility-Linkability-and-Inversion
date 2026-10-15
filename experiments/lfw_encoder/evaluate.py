from pathlib import Path
import json,hashlib,time
import numpy as np,pandas as pd
from scipy.linalg import hadamard
R=Path(__file__).resolve().parent.parent;O=R/'lfw_encoder';OUT=O/'results';OUT.mkdir(exist_ok=True)
meta=pd.read_csv(O/'frozen_metadata.csv');pairs=pd.read_csv(O/'pair_manifest.csv');X0=np.load(O/'frozen_mbf.npy');X1=np.load(O/'r50_embeddings.npy')
lookup={v:i for i,v in enumerate(meta.image_path)};ia=np.array([lookup[v] for v in pairs.image_a]);ib=np.array([lookup[v] for v in pairs.image_b]);y=pairs.label.to_numpy();folds=pairs.fold.to_numpy();valid=meta.status.eq('valid').to_numpy();pv=valid[ia]&valid[ib];assert pv.sum()==5965
assert np.array_equal(np.isfinite(X1).all(1),valid) and X1.shape==X0.shape
SEEDS=list(range(2026081101,2026081106));RO={}
for seed in SEEDS:
 w=np.load(O/f'ro_projection_seed_{seed}.npy');RO[seed]=w
 assert np.array_equal(np.packbits(X0[valid]@w>=0,axis=1),np.load(O/f'biohashes_packed_seed_{seed}.npy')[valid])
def threshold(v):
 a=np.sort(v);budget=int(np.floor(.001*len(a)));return float(np.nextafter(a[-budget-1],np.inf))
def ns(seed,method):return int.from_bytes(hashlib.sha256(f'LFW-ENC-E01|{method}|{seed}'.encode()).digest()[:8],'little')
def orthogonal(d,b,seed):
 q,r=np.linalg.qr(np.random.default_rng(seed).standard_normal((d,b)),mode='reduced');return q*np.where(np.diag(r)<0,-1.,1.)
H=hadamard(512).astype(float)/np.sqrt(512);SH={}
for seed in SEEDS:
 rng=np.random.default_rng(ns(seed,'SRHT'));SH[seed]=rng.choice([-1.,1.],512)[:,None]*H[:,rng.permutation(512)[:128]]
records=[];decisions=[];pca_audit=[];start=time.time()
for enc,X in [('MobileFaceNet',X0),('ResNet50_ArcFace',X1)]:
 for method in ['RAW','RO','SRHT','PCA_GUIDED_RO']:
  seeds=[0] if method=='RAW' else SEEDS;scores_by_key={}
  if method=='RAW':scores_by_key[0]=np.array([float(np.dot(X[a],X[b])) for a,b in zip(ia,ib)])
  if method in ['RO','SRHT']:
   for seed in seeds:
    B=X@(RO[seed] if method=='RO' else SH[seed])>=0;scores_by_key[seed]=(B[ia]==B[ib]).mean(1)
  for fold in sorted(np.unique(folds)):
   train=folds!=fold;test=~train
   if method=='PCA_GUIDED_RO':
    testids=set(meta.iloc[np.unique(np.r_[ia[test],ib[test]])].identity);fit=np.unique(np.r_[ia[train],ib[train]]);fit=fit[valid[fit]&~meta.iloc[fit].identity.isin(testids).to_numpy()]
    assert len(fit)>256 and set(meta.iloc[fit].identity).isdisjoint(testids)
    Z=X[fit].astype(float);mu=Z.mean(0);_,V=np.linalg.eigh((Z-mu).T@(Z-mu));V=V[:,::-1][:,:256];signs=np.sign(V[np.argmax(np.abs(V),axis=0),np.arange(256)]);V*=np.where(signs==0,1,signs)
    Z=(X.astype(float)-mu)@V;Z/=np.linalg.norm(Z,axis=1,keepdims=True)
    np.savez_compressed(O/f'pca_{enc}_fold{fold}.npz',mean=mu,components=V,fit_rows=fit)
    pca_audit.append(dict(encoder=enc,fold=int(fold),fit_images=len(fit),fit_identities=meta.iloc[fit].identity.nunique(),dimension=256,test_identity_overlap=0))
    for seed in seeds:
     B=Z@orthogonal(256,128,ns(seed,'PCA_GUIDED_RO'))>=0;scores_by_key[seed]=(B[ia]==B[ib]).mean(1)
   for seed in seeds:
    scores=scores_by_key[seed]
    for view in ['end_to_end_operational','matcher_only']:
     tr=train if view=='end_to_end_operational' else train&pv;te=test if view=='end_to_end_operational' else test&pv;ss=np.where(pv,scores,-np.inf);th=threshold(ss[tr&(y==0)]);accept=pv&(scores>=th);g=te&(y==1);im=te&(y==0)
     records.append(dict(encoder=enc,method=method,bits=0 if method=='RAW' else 128,key_seed=seed,fold=int(fold),evaluation=view,threshold=th,genuine_pairs=int(g.sum()),impostor_pairs=int(im.sum()),tp=int(accept[g].sum()),fp=int(accept[im].sum()),training_far=float((ss[tr&(y==0)]>=th).mean())))
     for j in np.flatnonzero(te):decisions.append(dict(encoder=enc,method=method,key_seed=seed,fold=int(fold),evaluation=view,pair_id=int(pairs.iloc[j].pair_id),label=int(y[j]),valid=bool(pv[j]),score=float(scores[j]) if pv[j] else None,threshold=th,accept=bool(accept[j])))
  print('Evaluated',enc,method,'sec',round(time.time()-start),flush=True)
a=pd.DataFrame(records);assert len(a)==640 and (a.training_far<=.001).all();a.to_csv(OUT/'fold_results.csv',index=False);pd.DataFrame(decisions).to_csv(OUT/'out_of_fold_decisions.csv',index=False);pd.DataFrame(pca_audit).to_csv(OUT/'pca_fit_audit.csv',index=False)
for method,stage in [('RAW','E01'),('RO','E02')]:
 ref=pd.read_csv(O/f'reference_{stage}_fold_results.csv').rename(columns={'seed':'key_seed'});q=a.query('encoder=="MobileFaceNet" and method==@method');keys=['evaluation','fold']+(['key_seed'] if method=='RO' else []);b=q.merge(ref,on=keys,suffixes=('_new','_ref'),validate='one_to_one');assert len(b)==len(q)==len(ref)
 for field in ['tp','fp','genuine_pairs','impostor_pairs']:assert np.array_equal(b[field+'_new'],b[field+'_ref'])
g=a.groupby(['encoder','method','bits','key_seed','evaluation'],as_index=False)[['genuine_pairs','impostor_pairs','tp','fp']].sum();g['tar']=g.tp/g.genuine_pairs;g['far']=g.fp/g.impostor_pairs;g.to_csv(OUT/'seed_summary.csv',index=False)
s=g.groupby(['encoder','method','bits','evaluation'],as_index=False).agg(tar=('tar','mean'),far=('far','mean'),tar_seed_sd=('tar','std'));s.to_csv(OUT/'summary.csv',index=False)
expected={'MobileFaceNet':{'RAW':98.66,'RO':82.99,'SRHT':84.48,'PCA_GUIDED_RO':85.22},'ResNet50_ArcFace':{'RAW':99.03,'RO':92.28,'SRHT':92.10,'PCA_GUIDED_RO':94.66}}
for r in s.query('evaluation=="matcher_only"').itertuples():assert round(r.tar*100,2)==expected[r.encoder][r.method],('Logged result changed',r)
status={'status':'PASS','experiment':'LFW-ENC-E01','recovery_reproduced_logged_TAR':True,'frozen_RAW_RO_fold_counts_reproduced':True,'frozen_RO_templates_reproduced':True,'fold_rows':len(a),'valid_pairs':5965,'PCA_fit_test_identity_overlap':0,'alignment':json.loads((O/'alignment_gate.json').read_text())}
(OUT/'run_status.json').write_text(json.dumps(status,indent=2));print(s.to_string(index=False));print('PASS full evaluation',flush=True)
