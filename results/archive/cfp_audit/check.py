import cv2,numpy as np,pandas as pd,zipfile,pickle,hashlib,io,json,time
from pathlib import Path
cv2.setNumThreads(2);cv2.setRNGSeed(20260924)
OUT=Path('correspondence_audit');z=zipfile.ZipFile('upload/CFP-FP-source-evidence.zip');raw=Path('upload/cfp_fp.bin').read_bytes();assert hashlib.sha256(raw).hexdigest()=='76306c783c2ef59c8569ebdcdd2f529f450bcc3fad57c94a5fc2b91df0f10370'
bins,labels=pickle.loads(raw,encoding='bytes');mapping=pd.read_csv('final_audit/input/frozen_subset.csv');candidate=pd.read_csv('final_audit/input/candidate_mapping.csv')
assert [hashlib.sha256(b).hexdigest() for b in bins]==candidate.encoded_sha256.tolist()
sift=cv2.SIFT_create(nfeatures=1500,contrastThreshold=.01);bf=cv2.BFMatcher()
def gray(data):return cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_GRAYSCALE)
def source(path):
 im=gray(z.read('original/Images/'+path.split('/Images/')[1]));scale=min(1,640/max(im.shape));return cv2.resize(im,None,fx=scale,fy=scale) if scale<1 else im

def verify(t,s):
 kt,dt=sift.detectAndCompute(t,None);ks,ds=sift.detectAndCompute(s,None)
 result=dict(matches=0,inliers=0,coverage=0.,correlation=-1.,valid_fraction=0.,pass_geometry=False)
 if dt is None or ds is None or len(ds)<2:return result
 matches=[m for m,n in bf.knnMatch(dt,ds,k=2) if m.distance<.7*n.distance]
 # Repeated SIFT orientations must not create duplicate geometry support.
 unique={ (tuple(np.round(kt[m.queryIdx].pt,1)),tuple(np.round(ks[m.trainIdx].pt,1))):m for m in matches};matches=list(unique.values());result['matches']=len(matches)
 if len(matches)<8:return result
 a=np.float32([ks[m.trainIdx].pt for m in matches]);b=np.float32([kt[m.queryIdx].pt for m in matches])
 A,mask=cv2.estimateAffine2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2.,maxIters=3000,confidence=.999)
 if A is None:return result
 mask=mask.ravel().astype(bool);ni=int(mask.sum());result['inliers']=ni
 if ni<8:return result
 coverage=cv2.contourArea(cv2.convexHull(b[mask]))/(t.shape[0]*t.shape[1]);result['coverage']=float(coverage)
 warped=cv2.warpAffine(s,A,(t.shape[1],t.shape[0]),flags=cv2.INTER_LINEAR)
 valid=cv2.warpAffine(np.ones(s.shape,np.uint8),A,(t.shape[1],t.shape[0]),flags=cv2.INTER_NEAREST)>0
 valid[:4]=False;valid[-4:]=False;valid[:,:4]=False;valid[:,-4:]=False
 result['valid_fraction']=float(valid.mean())
 if valid.sum()<100:return result
 corr=float(np.corrcoef(t[valid],warped[valid])[0,1]);result['correlation']=corr if np.isfinite(corr) else -1.
 result['pass_geometry']=bool(ni>=8 and coverage>=.10 and result['valid_fraction']>=.6 and corr>=.95 and np.std(t[valid])>=10)
 return result
records=[];start=time.time()
import sys
limit=int(sys.argv[1]) if len(sys.argv)>1 else len(mapping)
for i,r in enumerate(mapping.head(limit).itertuples()):
 t=cv2.resize(gray(bins[r.bin_index]),(224,224));result=verify(t,source(r.source_path))
 records.append(dict(encoded_sha256=r.encoded_sha256,identity=int(r.identity),pose=r.pose,source_path=r.source_path,**result))
 if (i+1)%100==0:print(i+1,'supported',sum(r['pass_geometry'] for r in records),'seconds',round(time.time()-start),flush=True)
 pd.DataFrame(records).to_csv(OUT/'independent_sift_affine.csv',index=False) if (i+1)%500==0 else None
pd.DataFrame(records).to_csv(OUT/'independent_sift_affine.csv',index=False)
print('COMPLETE',len(records),sum(r['pass_geometry'] for r in records),flush=True)
