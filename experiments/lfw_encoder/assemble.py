from pathlib import Path
import numpy as np,pandas as pd,json
O=Path(__file__).resolve().parent;X=np.full((7701,512),np.nan,np.float32);seen=[];rows=[];original=np.load(O/'frozen_mbf.npy')
for p in sorted((O/'chunks').glob('s[0-3]_????.npz')):
 with np.load(p,allow_pickle=False) as z:
  ids=z['indices'];seen.extend(ids.tolist());X[ids]=z['r50'];r=json.loads(str(z['audit']));rows+=r
  assert np.array_equal(z['mbf'],original[ids],equal_nan=True)
assert len(seen)==7701 and len(set(seen))==7701
meta=pd.read_csv(O/'frozen_metadata.csv');assert np.array_equal(np.isfinite(X).all(1),meta.status.eq('valid'));assert np.max(abs(np.linalg.norm(X[np.isfinite(X).all(1)],axis=1)-1))<1e-5
np.save(O/'r50_embeddings.npy',X);a=pd.DataFrame(rows).sort_values('image_id');a.to_csv(O/'alignment_audit.csv',index=False);assert a.max_abs_error.max()==0
(O/'alignment_gate.json').write_text(json.dumps({'status':'PASS','images':7701,'valid_images':7682,'detection_failures':19,'max_abs_error':0,'complete_recovery':True},indent=2));print('PASS assembled all image records')
