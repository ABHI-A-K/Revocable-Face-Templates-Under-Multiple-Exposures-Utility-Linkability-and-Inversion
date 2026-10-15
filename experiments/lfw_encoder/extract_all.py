from pathlib import Path
import sys,json,time,zipfile,hashlib
import numpy as np,pandas as pd,cv2,onnxruntime as ort
from insightface.model_zoo.scrfd import SCRFD
from insightface.model_zoo.arcface_onnx import ArcFaceONNX
from insightface.utils.face_align import norm_crop
R=Path(__file__).resolve().parent.parent;O=R/'lfw_encoder';shard=int(sys.argv[1]);C=O/'chunks';C.mkdir(exist_ok=True)
def sha(b):return hashlib.sha256(b).hexdigest()
locks={'det_500m.onnx':'5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a','w600k_mbf.onnx':'9cc6e4a75f0e2bf0b1aed94578f14475f357bdc05e815e5c4a02b319eb4f','w600k_r50.onnx':'4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43'}
locks['w600k_mbf.onnx']='9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f'
assert cv2.__version__.startswith('4.13.') and ort.__version__=='1.28.0'
cv2.setNumThreads(1);models={}
for name,h in locks.items():
 p=R/'models'/name;assert sha(p.read_bytes())==h
 opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1;opts.add_session_config_entry('session.intra_op.allow_spinning','0');opts.add_session_config_entry('session.inter_op.allow_spinning','0')
 sess=ort.InferenceSession(str(p),sess_options=opts,providers=['CPUExecutionProvider']);cls=SCRFD if name.startswith('det') else ArcFaceONNX;models[name]=cls(model_file=str(p),session=sess)
det=models['det_500m.onnx'];det.prepare(ctx_id=-1,det_thresh=.5,input_size=(640,640));mbf=models['w600k_mbf.onnx'];r50=models['w600k_r50.onnx'];mbf.prepare(ctx_id=-1);r50.prepare(ctx_id=-1)
meta=pd.read_csv(O/'frozen_metadata.csv');X=np.load(O/'frozen_mbf.npy');inds=np.arange(len(meta))[shard::4]
z=zipfile.ZipFile(R/'datasets/lfw-dataset.zip');lookup={'/'.join(Path(n).parts[-2:]):n for n in z.namelist() if n.endswith('.jpg')};start=time.time()
for base in range(0,len(inds),100):
 ids=inds[base:base+100];dest=C/f's{shard}_{base:04d}.npz'
 if dest.exists():continue
 crops=np.zeros((len(ids),112,112,3),np.uint8);V=np.full((len(ids),512),np.nan,np.float32);M=V.copy();rows=[]
 for j,i in enumerate(ids):
  rowmeta=meta.iloc[i];key='/'.join(Path(rowmeta.image_path).parts[-2:]);raw=z.read(lookup[key]);im=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR);boxes,kps=det.detect(im,max_num=0)
  ok=len(boxes)>0;assert ok==(rowmeta.status=='valid') and len(boxes)==int(rowmeta.faces_detected),(int(i),'detection mismatch')
  a={'image_id':int(i),'status':'valid' if ok else 'detection_failure','faces':len(boxes),'image_sha256':sha(raw)}
  if ok:
   k=int(np.argmax((boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])));crop=norm_crop(im,landmark=kps[k],image_size=112);crops[j]=crop
   v=mbf.get_feat(crop).ravel().astype(np.float32);v/=np.linalg.norm(v);err=float(np.max(abs(v-X[i])));assert err==0,(int(i),'MBF not exact',err)
   M[j]=v;r=r50.get_feat(crop).ravel().astype(np.float32);assert np.isfinite(r).all() and np.linalg.norm(r)>0;V[j]=r/np.linalg.norm(r)
   a.update(max_abs_error=err,crop_sha256=sha(crop.tobytes()),bbox=boxes[k].tolist(),landmarks=kps[k].tolist())
  rows.append(a)
 temp=dest.with_suffix('.tmp.npz');np.savez_compressed(temp,indices=ids,crops=crops,mbf=M,r50=V,audit=np.array(json.dumps(rows)));temp.replace(dest)
 print('PASS block',shard,base+len(ids),'/',len(inds),'sec',round(time.time()-start),flush=True)
(O/f'extract_{shard}.done.json').write_text(json.dumps({'status':'PASS','shard':shard,'rows':len(inds)}));print('PASS shard',shard,flush=True)
