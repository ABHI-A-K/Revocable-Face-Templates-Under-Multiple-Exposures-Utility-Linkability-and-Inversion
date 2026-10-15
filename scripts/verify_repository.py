from pathlib import Path
import hashlib,json,sys,subprocess,re
import pandas as pd
import nbformat
P=Path(__file__).resolve().parents[1];checks={}
f=P/'MANIFEST_SHA256.json'
if f.exists():
 manifest=json.loads(f.read_text())
 for n,h in manifest.items():
  q=P/n;assert q.is_file() and hashlib.sha256(q.read_bytes()).hexdigest()==h,('hash mismatch',n)
 checks['integrity_files']=len(manifest)
count=0
for p in (P/'notebooks').glob('*.ipynb'):
 n=nbformat.read(p,as_version=4);nbformat.validate(n);assert all(not c.get('outputs') for c in n.cells if c.cell_type=='code');count+=1
checks['clean_notebooks']=count
count=0
for p in (P/'results/archive').rglob('key_trial_schedule.csv'):
 d=pd.read_csv(p);cols=[c for c in d if c.startswith('exposure_seed_')]
 if 'query_seed' in d and cols:
  assert not d[cols].eq(d.query_seed,axis=0).any().any(),p
  assert all(len(set(row))==len(row) for row in d[cols].itertuples(index=False,name=None)),p
  count+=len(d)
assert count>0;checks['query_key_exclusion_rows']=count
count=0
for p in (P/'results/archive').rglob('identity_partitions.csv'):
 d=pd.read_csv(p)
 if {'split_seed','identity','partition'}<=set(d):
  assert d.groupby(['split_seed','identity']).partition.nunique().max()==1,p
  if 'cfp_e06' in str(p):
   for seed,g in d.groupby('split_seed'):assert sorted(g.groupby('partition').identity.nunique())==[75,75,100,250],p
  count+=len(d)
assert count>0;checks['identity_partition_rows']=count
for p in (P/'results/archive').rglob('identity_fold_assignment.csv'):
 d=pd.read_csv(p);assert d.groupby('identity').fold.nunique().max()==1,p
locks=json.loads((P/'models/model_locks.json').read_text());assert all(re.fullmatch('[0-9a-f]{64}',v['sha256']) for v in locks.values());checks['model_hash_records']=len(locks)
for name in ['editorial_metrics.py','validate_encoder_evidence.py']:subprocess.run([sys.executable,str(P/'reproduction'/name)],cwd=P/'reproduction',check=True)
checks['numerical_consistency']='PASS';checks['full_image_to_attack_rerun']='NOT PERFORMED';checks['independent_reannotation']='NOT PERFORMED'
print(json.dumps(checks,indent=2))
