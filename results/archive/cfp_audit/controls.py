from pathlib import Path
exec(Path('correspondence_audit/check.py').read_text().split('records=[];start=')[0])
rng=np.random.default_rng(2026092403);rows=mapping.iloc[np.sort(rng.choice(len(mapping),500,replace=False))];records=[]
allpaths=sorted(n for n in z.namelist() if n.lower().endswith('.jpg'))
for idx,r in enumerate(rows.itertuples()):
 target=cv2.resize(gray(bins[r.bin_index]),(224,224));orig='original/Images/'+r.source_path.split('/Images/')[1]
 same=[p for p in allpaths if p.split('/')[2]==f'{r.identity:03d}' and p.split('/')[3]==r.pose and p!=orig]
 other=[p for p in allpaths if p.split('/')[2]!=f'{r.identity:03d}' and p.split('/')[3]==r.pose]
 for kind,choices in [('same_identity_other_photo',same),('different_identity',other)]:
  name=str(rng.choice(choices));s=gray(z.read(name));scale=min(1,640/max(s.shape));s=cv2.resize(s,None,fx=scale,fy=scale) if scale<1 else s
  # Same original pixels are not a valid negative photo control.
  actual=gray(z.read(orig))
  if s.shape==actual.shape and np.array_equal(s,actual):continue
  result=verify(target,s);records.append(dict(encoded_sha256=r.encoded_sha256,kind=kind,control_path=name,**result))
 if (idx+1)%100==0:print(idx+1,flush=True)
pd.DataFrame(records).to_csv(OUT/'negative_controls.csv',index=False)
print(pd.DataFrame(records).groupby('kind').pass_geometry.agg(['sum','count']).to_string())
