from pathlib import Path
exec(Path('correspondence_audit/check.py').read_text().split('records=[];start=')[0])
d=pd.read_csv(OUT/'independent_sift_affine.csv');pending=d[~((d.inliers>=8)&(d.coverage>=.1)&(d.correlation>=.95))];records=[]
for r in pending.itertuples():
 entry=mapping[mapping.encoded_sha256==r.encoded_sha256].iloc[0];t=cv2.resize(gray(bins[int(entry.bin_index)]),(224,224));s=source(r.source_path)
 kt,dt=sift.detectAndCompute(t,None);ks,ds=sift.detectAndCompute(s,None);ms=[m for m,n in bf.knnMatch(dt,ds,k=2) if m.distance<.7*n.distance]
 a=np.float32([ks[m.trainIdx].pt for m in ms]);b=np.float32([kt[m.queryIdx].pt for m in ms]);A,mask=cv2.estimateAffine2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2.,maxIters=3000,confidence=.999)
 status='ECC_FAILED';corr=-1.;fraction=0.
 try:
  inv=cv2.invertAffineTransform(A).astype(np.float32)
  cc,inv=cv2.findTransformECC(t,s,inv,cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,200,1e-6),None,5)
  warped=cv2.warpAffine(s,inv,(224,224),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
  valid=cv2.warpAffine(np.ones(s.shape,np.uint8),inv,(224,224),flags=cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP)>0
  valid[:4]=False;valid[-4:]=False;valid[:,:4]=False;valid[:,-4:]=False
  corr=float(np.corrcoef(t[valid],warped[valid])[0,1]);fraction=float(valid.mean());status='REFINED_CORRESPONDENCE' if corr>=.98 and fraction>=.4 and np.std(t[valid])>=10 else 'REVIEW_REQUIRED'
 except cv2.error:pass
 records.append(dict(encoded_sha256=r.encoded_sha256,correlation=corr,valid_fraction=fraction,status=status))
pd.DataFrame(records).to_csv(OUT/'supplemental_ecc.csv',index=False);print(pd.DataFrame(records).to_string(index=False))
