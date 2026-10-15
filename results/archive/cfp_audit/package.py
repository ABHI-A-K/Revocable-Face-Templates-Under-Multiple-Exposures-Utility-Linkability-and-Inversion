from pathlib import Path
import pandas as pd,numpy as np,zipfile,pickle,hashlib,json,io,base64,html,shutil
from PIL import Image
P=Path('correspondence_audit');d=pd.read_csv(P/'independent_sift_affine.csv');e=pd.read_csv(P/'supplemental_ecc.csv');ctrl=pd.read_csv(P/'negative_controls.csv');m=pd.read_csv('final_audit/input/frozen_subset.csv')
strict=d.pass_geometry;overlap=(d.inliers>=8)&(d.coverage>=.1)&(d.correlation>=.95)
refined=set(e.loc[e.status=='REFINED_CORRESPONDENCE','encoded_sha256'])
d['audit_tier']=np.where(strict,'STRICT_FIRST_PASS',np.where(overlap,'POSTHOC_OVERLAP_SUPPORTED',np.where(d.encoded_sha256.isin(refined),'POSTHOC_ECC_SUPPORTED','REVIEW_REQUIRED')))
d.to_csv(P/'final_correspondence_status.csv',index=False);pending=d[d.audit_tier=='REVIEW_REQUIRED'];pending.to_csv(P/'review_required_26.csv',index=False)
assert len(pending)==26
# Quantify exposure in existing models; do not filter old test outcomes and present it as retraining.
rows=set(m[m.encoded_sha256.isin(pending.encoded_sha256)].row)
z=zipfile.ZipFile('upload/CFP-FP-E07_INV-results.zip');a=pd.read_csv(io.BytesIO(z.read('per_identity_attacks.csv.gz')),compression='gzip');affected=a[a.exposed_row.isin(rows)|a.enrollment_row.isin(rows)]
part=pd.read_csv('final_audit/input/identity_partitions.csv');impact=[]
for seed,g in part.groupby('split_seed'):
 for r in g.itertuples():
  count=int(((m.identity==r.identity)&m.row.isin(rows)).sum())
  if count:impact.append(dict(split_seed=seed,partition=r.partition,identity=r.identity,unresolved_images=count))
pd.DataFrame(impact).to_csv(P/'unresolved_partition_impact.csv',index=False)
# Offline review document embeds only unresolved photo pairs; it makes no identity decisions.
raw=Path('upload/cfp_fp.bin').read_bytes();bins,_=pickle.loads(raw,encoding='bytes');src=zipfile.ZipFile('upload/CFP-FP-source-evidence.zip')
def pic(data):
 im=Image.open(io.BytesIO(data)).convert('RGB');im.thumbnail((450,450));out=io.BytesIO();im.save(out,format='PNG');return 'data:image/png;base64,'+base64.b64encode(out.getvalue()).decode()
parts=['<!doctype html><meta charset="utf-8"><title>CFP photographic correspondence review</title><style>body{font:16px sans-serif;margin:30px;max-width:1100px}article{border-top:1px solid #aaa;padding:20px 0}img{max-width:43%;vertical-align:top;margin:10px}code{word-break:break-all}</style><h1>26 unresolved crop-to-source correspondences</h1><p>Private offline review. Compare photographic content, not a person’s real-world identity. Dataset folder labels are inherited annotations. No decision is prefilled. Record SAME_SOURCE, DIFFERENT_SOURCE or UNCERTAIN with evidence and reviewer name in review_decisions.csv. Cropping, scaling and padding may differ. Decisions require subsequent technical review before changing any experiment.</p>']
for n,r in enumerate(pending.itertuples(),1):
 row=m[m.encoded_sha256==r.encoded_sha256].iloc[0];source='original/Images/'+r.source_path.split('/Images/')[1]
 parts.append(f'<article><h2>{n}. Dataset folder {r.identity}, {r.pose}</h2><p><code>{r.encoded_sha256}</code></p><p>Frozen crop (left); proposed original (right). Affine inliers: {r.inliers}; overlapping-pixel correlation: {r.correlation:.4f}.</p><img src="{pic(bins[int(row.bin_index)])}"><img src="{pic(src.read(source))}"><p>Source path: {html.escape(source)}</p></article>')
(P/'review_26_photographic_pairs.html').write_text('\n'.join(parts))
review=pending[['encoded_sha256','identity','pose','source_path']].copy();review['decision']='PENDING';review['reviewer']='';review['evidence_note']='';review.to_csv(P/'review_decisions.csv',index=False)
counts=d.audit_tier.value_counts().to_dict();summary=dict(status='AUDIT_COMPLETE_WITH_26_UNRESOLVED_CORRESPONDENCES',counts=counts,retained_crops=5704,negative_controls=1000,negative_controls_accepted=0,affected_E07_attack_records=len(affected),affected_E07_unique_people=int(affected.identity.nunique()),both_bin_uploads_identical=True,bin_sha256=hashlib.sha256(raw).hexdigest(),limitations=['Post-hoc overlap and ECC tiers are exploratory corroboration, not preregistered acceptance','Negative controls are sampled, not an exhaustive uniqueness search','No human review or real-world identity certification','Dataset identity annotations are inherited','No original results overwritten; no automatic relabeling; affected training partitions require refitting if labels or inclusion change'])
(P/'audit_summary.json').write_text(json.dumps(summary,indent=2))
text=f'''# CFP photographic correspondence audit

Status: **audit complete; 26 correspondences remain unresolved**. No more file uploads are needed for this review.

Both bin uploads have the expected SHA-256 and are identical. All 14,000 encoded crop hashes match the earlier cache metadata. The previous source check matched all 5,901 source references by decoded pixel hash. Together these checks bind the audit to the actual experiment inputs.

## Results
- Strict first-pass criterion: {counts.get('STRICT_FIRST_PASS',0)} / 5,704.
- Follow-up overlap-only support: {counts.get('POSTHOC_OVERLAP_SUPPORTED',0)}. These satisfy the feature and correlation criteria but fail the initial 60% whole-crop overlap requirement. The change in interpretation is explicitly post hoc; it is not a silently relaxed original pass gate.
- Follow-up ECC registration support: {counts.get('POSTHOC_ECC_SUPPORTED',0)} / 29 inspected exceptions.
- Still unresolved: 26. A failed registration criterion is not proof of a wrong identity.
- Negative controls: 0 / 500 sampled different-identity photographs and 0 / 500 sampled same-identity different photographs passed the strict check. None passed the follow-up feature/correlation rule either. ECC was applied to proposed positives only, so it has no separately measured control error rate.

## Method
SIFT local image features (not face-recognition embeddings), ratio matching at 0.7, deduplicated coordinate pairs, affine RANSAC at 2 pixels with 3,000 iterations. Strict support requires ≥8 inliers, convex-hull coverage ≥10% of the crop, overlapping-pixel correlation ≥0.95, overlap ≥60%, and overlapping target standard deviation ≥10. Crop images are resized to 224×224; sources capped at 640 pixels. Follow-up overlap support omits the 60% criterion and is separately labeled. Three ECC-supported cases require correlation ≥0.98, overlap ≥40% and target standard deviation ≥10. These are analyst-selected, post-hoc diagnostics, not externally validated identity-certification thresholds.

## What remains
Open `review_26_photographic_pairs.html` locally and review photographic correspondence for the 26 cases. Fill `review_decisions.csv` with SAME_SOURCE, DIFFERENT_SOURCE or UNCERTAIN, reviewer and evidence. Do not infer real-world identity from the face. Original dataset annotation is the identity source once photographic correspondence is established.

The unresolved images touch {len(affected):,} saved E07 attack rows involving {affected.identity.nunique()} distinct test identities. `unresolved_partition_impact.csv` also records their occurrence in calibration, training and validation partitions. If any source mapping or inclusion changes, rerun affected fits and evaluation; merely deleting affected test scores does not validate the original pipeline.

The 5,678 supported correspondences are technical corroboration, not a blanket statement that every identity has been independently certified. No identity labels, templates, scores or manuscript conclusions were changed. The earlier statistical and threshold analyses remain conditional on the existing screened mapping.

## Reproducibility
The scripts use relative paths matching the existing workspace: `upload/cfp_fp.bin`, `upload/CFP-FP-source-evidence.zip`, and `final_audit/input/` manifests. Included input manifests reproduce those references. Use Python with numpy, pandas, Pillow and OpenCV (run used opencv-python-headless 5.0.0.93). Run `check.py`, then `controls.py`, then `refine.py`. The first 100 rows were used to exercise the implementation before full execution. All outputs and analyst-selected follow-up rules are retained.
'''
(P/'READ_ME.md').write_text(text)
(P/'tracker_row.tsv').write_text('CFP-FINAL-ID-AUDIT\tDo frozen crops correspond to their proposed original photographs?\tHash binding; local-feature geometry; pixel correlation; negative controls\t5704 audited: 4295 strict, 1380 post-hoc overlap-supported, 3 ECC-supported, 26 unresolved; 0/1000 control acceptances\tTechnical correspondence corroboration; 26 require review; no automatic relabeling or blanket identity certification\n')
# Include exact referenced manifests for reproducibility.
for n in ['frozen_subset.csv','candidate_mapping.csv','source_content_inventory.csv','identity_partitions.csv']:shutil.copy2(Path('final_audit/input')/n,P/n)
files=[p for p in P.iterdir() if p.is_file() and p.suffix not in ['.log']]
manifest=[dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files];(P/'manifest.json').write_text(json.dumps(manifest,indent=2))
with zipfile.ZipFile('CFP-FP-correspondence-audit.zip','w',zipfile.ZIP_DEFLATED) as archive:
 for p in files+[P/'manifest.json']:archive.write(p,p.name)
print(json.dumps(summary,indent=2))
