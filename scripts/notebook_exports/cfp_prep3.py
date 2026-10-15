# Original code export; run with IPython after configuring inputs.
# Not an independently rerun or refactored implementation.
# Cell 1
from pathlib import Path
import io, re, json, hashlib, pickle, zipfile, shutil, platform
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
from IPython.display import display, FileLink

INPUT_ROOT = Path('/kaggle/input')
WORK_ROOT = Path('/kaggle/working') if Path('/kaggle').exists() else Path.cwd()
# Set explicit paths if more than one matching input exists. ZIP paths are accepted.
CACHE_INPUT = None
REFERENCE_INPUT = None   # directory or ZIP of person-labelled CFP images
BIN_INPUT = None         # optional exact cfp_fp.bin
REFERENCE_PROVENANCE = 'Kaggle chinafax/cfpw-dataset; source identity folders and FP protocol' # e.g. provider URL, release name and how obtained
# Standard CFP layout; change only if the source documents a different layout.
IDENTITY_PATTERN = r'(?:^|/)(?P<identity>\d{3})/(?P<pose>frontal|profile)/[^/]+$'
from datetime import datetime, timezone
OUT = WORK_ROOT / ('CFP-FP-E05-PREP-R3-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
OUT.mkdir(parents=True, exist_ok=True)
# Remove stale success outputs from an earlier run, never user inputs.
for name in ['verified_mapping.csv', 'identity_folds.csv']:
    (OUT / name).unlink(missing_ok=True)
issues = []
input_records = []

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def record(path):
    input_records.append({'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size})

def unpack(path):
    path = Path(path)
    if path.is_dir(): return path
    if path.suffix.lower() != '.zip': raise ValueError('Expected directory or ZIP: '+str(path))
    record(path)
    dest = WORK_ROOT / 'mapping_inputs' / sha(path)[:16]
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as z:
        if sum(i.file_size for i in z.infolist()) > 20*1024**3:
            raise ValueError('Archive exceeds 20 GiB extraction limit.')
        for item in z.infolist():
            target = (dest / item.filename).resolve()
            if not target.is_relative_to(dest.resolve()) or ((item.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError('Unsafe ZIP entry: '+item.filename)
        z.extractall(dest)
    return dest

def pixel_hash(data):
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert('RGB')
        return hashlib.sha256(str(im.size).encode()+b'RGB'+im.tobytes()).hexdigest()

print('Python', platform.python_version(), '| NumPy', np.__version__, '| pandas', pd.__version__)

DOWNLOAD_BASELINE_BIN = True
EXPECTED_BIN_SHA256 = '76306c783c2ef59c8569ebdcdd2f529f450bcc3fad57c94a5fc2b91df0f10370'
BASELINE_BIN_URL = ('https://huggingface.co/datasets/Icar/val_sets/resolve/'
                    '1f7800d9d5ba295554573a4e292e870b64a49b74/cfp_fp.bin')
# Downloads are checksum-checked before deserialization.
print('Results directory:', OUT)


# Cell 3
if REFERENCE_INPUT is None:
    refs = sorted({p.parent for p in INPUT_ROOT.rglob('Pair_list_F.txt')
                   if (p.parent / 'Pair_list_P.txt').is_file()
                   and (p.parent / 'Split' / 'FP').is_dir()})
    if len(refs) != 1:
        raise RuntimeError(f'Expected one CFP Protocol directory; found {refs}. Set REFERENCE_INPUT to the CFP dataset directory.')
    protocol = refs[0]
    REFERENCE_INPUT = protocol.parent
else:
    refroot = unpack(REFERENCE_INPUT)
    refs = [p.parent for p in refroot.rglob('Pair_list_F.txt')
            if (p.parent/'Split'/'FP').is_dir()]
    if len(refs) != 1: raise RuntimeError(f'Expected one FP protocol: {refs}')
    protocol = refs[0]
print('Reference root:', REFERENCE_INPUT)
print('FP protocol:', protocol/'Split'/'FP')


# Cell 5
if CACHE_INPUT is None:
    direct = sorted(p for p in INPUT_ROOT.rglob('aligned_pair_side_metadata.csv') if (p.parent/'aligned_pair_side_embeddings.npy').is_file()) if INPUT_ROOT.exists() else []
    archives = []
    if not direct and INPUT_ROOT.exists():
        for p in INPUT_ROOT.rglob('*.zip'):
            try:
                with zipfile.ZipFile(p) as z:
                    if {'aligned_pair_side_metadata.csv','aligned_pair_side_embeddings.npy'}.issubset({Path(x).name for x in z.namelist()}): archives.append(p)
            except zipfile.BadZipFile: pass
    choices = [p.parent for p in direct] or archives
    if len(choices) != 1:
        raise RuntimeError(f'Set CACHE_INPUT to the one R4 cache directory/ZIP. Candidates: {choices}')
    CACHE_INPUT = choices[0]
root = unpack(CACHE_INPUT)
paths = list(root.rglob('aligned_pair_side_metadata.csv'))
if len(paths) != 1: raise RuntimeError('Cache must contain exactly one pair-side metadata file.')
meta_path = paths[0]
meta = pd.read_csv(meta_path)
emb_path = meta_path.parent / 'aligned_pair_side_embeddings.npy'
record(meta_path); record(emb_path)
emb = np.load(emb_path, mmap_mode='r', allow_pickle=False)
required = {'bin_index','pair_id','side','fold','label','encoded_sha256','status'}
assert required.issubset(meta.columns), f'Missing columns: {required-set(meta.columns)}'
assert len(meta) == 14000 and emb.shape == (14000,512), 'Wrong R4 cache dimensions'
assert np.array_equal(meta.bin_index.to_numpy(), np.arange(14000)), 'Cache rows are not in bin order'
assert meta.status.eq('valid').all() and np.isfinite(emb).all(), 'Invalid embeddings'
assert np.allclose(np.linalg.norm(emb,axis=1),1,atol=1e-4), 'Non-normalized embeddings'
assert np.array_equal(meta.pair_id.to_numpy(),np.arange(14000)//2+1)
assert meta.side.tolist() == ['a','b']*7000
assert set(meta.label.unique()) == {0,1}
assert np.array_equal(meta.label.to_numpy()[::2],meta.label.to_numpy()[1::2])
assert meta.encoded_sha256.astype(str).str.fullmatch(r'[0-9a-f]{64}').all()
print(f'Cache valid: {len(meta):,} occurrences; {meta.encoded_sha256.nunique():,} unique encoded images.')


# Cell 7
# Candidate mapping only: passing these checks does not yet verify
# that each aligned crop visually corresponds to its proposed source image.
from pathlib import Path
import re
import numpy as np
import pandas as pd

def read_image_list(path):
    result = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        number, relative_path = line.strip().split(maxsplit=1)
        number = int(number)
        if number in result:
            raise ValueError(f"Duplicate image index in {path}: {number}")
        result[number] = (path.parent / relative_path).resolve()
    return result

frontal = read_image_list(protocol / "Pair_list_F.txt")
profile = read_image_list(protocol / "Pair_list_P.txt")

rows = []
for fold in range(1, 11):
    for filename, label in [("same.txt", 1), ("diff.txt", 0)]:
        pair_file = protocol / "Split" / "FP" / f"{fold:02d}" / filename

        record(pair_file)
        pair_lines = [x for x in pair_file.read_text().splitlines() if x.strip()]
        if len(pair_lines) != 350: raise ValueError(f'Expected 350 pairs: {pair_file}')
        for line in pair_lines:
            if not line.strip():
                continue
            f_index, p_index = map(int, re.split(r"[,\s]+", line.strip()))

            for side, path in [
                ("a", frontal[f_index]),
                ("b", profile[p_index]),
            ]:
                if not path.is_file():
                    raise FileNotFoundError(path)

                rows.append({
                    "bin_index": len(rows),
                    "candidate_fold": fold,
                    "candidate_label": label,
                    "candidate_side": side,
                    "source_path": str(path),
                    "identity": path.parent.parent.name,
                    "pose": path.parent.name,
                })

candidate = pd.DataFrame(rows)
cached = meta.sort_values("bin_index").reset_index(drop=True)

assert len(candidate) == len(cached) == 14000
order_checks = {
    'bin_index': bool(np.array_equal(candidate.bin_index, cached.bin_index)),
    'fold': bool(np.array_equal(candidate.candidate_fold, cached.fold)),
    'label': bool(np.array_equal(candidate.candidate_label, cached.label)),
    'side': bool(np.array_equal(candidate.candidate_side, cached.side)),
}
order_ok = all(order_checks.values())
print('Proposed ordering checks:', order_checks)
record(protocol/'Pair_list_F.txt'); record(protocol/'Pair_list_P.txt')
candidate["encoded_sha256"] = cached.encoded_sha256.to_numpy()

# Same original image should consistently correspond to the same crop hash.
path_counts = candidate.groupby("source_path").encoded_sha256.nunique()

# Same cached crop must not be assigned to multiple original images.
hash_counts = candidate.groupby("encoded_sha256").source_path.nunique()

path_conflicts = int((path_counts > 1).sum())
hash_conflicts = int((hash_counts > 1).sum())

# Check the source identities against every genuine/impostor label.
a = candidate.iloc[::2].reset_index(drop=True)
b = candidate.iloc[1::2].reset_index(drop=True)
identity_conflicts = int(
    ((a.identity == b.identity).astype(int) != a.candidate_label).sum()
)

print("Original images referenced:", candidate.source_path.nunique())
print("Unique cached images:", candidate.encoded_sha256.nunique())
print("Original images assigned multiple crop hashes:", path_conflicts)
print("Crop hashes assigned multiple original images:", hash_conflicts)
print("Pairs with inconsistent identity labels:", identity_conflicts)

output = OUT / 'candidate_mapping.csv'
candidate.to_csv(output, index=False)
print("Candidate audit saved:", output)

print('Candidate saved; duplicate and identity audits follow. No identity approval is made here.')


# Cell 9
# Inspect the 11 shared crop hashes without approving the mapping.
import hashlib
from pathlib import Path
from PIL import Image
import pandas as pd

unique_links = candidate[
    ["encoded_sha256", "source_path", "identity", "pose"]
].drop_duplicates()

shared_hashes = (
    unique_links.groupby("encoded_sha256").source_path.nunique()
)
shared_hashes = shared_hashes[shared_hashes > 1].index

audit_rows = []

for crop_hash in shared_hashes:
    group = unique_links[
        unique_links.encoded_sha256 == crop_hash
    ]

    source_byte_hashes = set()
    source_pixel_hashes = set()

    for source_path in group.source_path.unique():
        path = Path(source_path)
        source_byte_hashes.add(
            hashlib.sha256(path.read_bytes()).hexdigest()
        )

        with Image.open(path) as image:
            image = image.convert("RGB")
            fingerprint = (
                str(image.size).encode() + b"RGB" + image.tobytes()
            )
            source_pixel_hashes.add(
                hashlib.sha256(fingerprint).hexdigest()
            )

    audit_rows.append({
        "crop_hash": crop_hash,
        "source_files": group.source_path.nunique(),
        "person_ids": ",".join(sorted(group.identity.astype(str).unique())),
        "person_count": group.identity.nunique(),
        "poses": ",".join(sorted(group.pose.unique())),
        "identical_source_bytes": len(source_byte_hashes) == 1,
        "identical_source_pixels": len(source_pixel_hashes) == 1,
        "source_paths": " | ".join(group.source_path.unique()),
    })

audit = pd.DataFrame(audit_rows, columns=['crop_hash','source_files','person_ids','person_count','poses','identical_source_bytes','identical_source_pixels','source_paths'])
output = OUT / 'shared_crop_audit.csv'
audit.to_csv(output, index=False)

print("Shared crop groups:", len(audit))
print("Groups involving multiple person IDs:",
      int((audit.person_count > 1).sum()))
print("Groups with identical original pixels:",
      int(audit.identical_source_pixels.sum()))

display(audit[[
    "person_ids", "source_files", "poses",
    "identical_source_bytes", "identical_source_pixels"
]])

print("Saved:", output)
print("E05-ID remains blocked pending mapping verification.")


# Cell 11
pixel_by_encoded = {}
crop_bytes = {}
bin_verified = False
try:
    if BIN_INPUT is None:
        found = sorted(INPUT_ROOT.rglob('cfp_fp.bin'))
        if len(found) > 1: raise ValueError('Multiple bins attached; set BIN_INPUT explicitly.')
        if found: BIN_INPUT = found[0]
    if BIN_INPUT is None and DOWNLOAD_BASELINE_BIN:
        from urllib.request import urlopen
        destination = WORK_ROOT / 'cfp_fp_verified_source' / 'cfp_fp.bin'
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or sha(destination) != EXPECTED_BIN_SHA256:
            temporary = destination.with_suffix('.download')
            with urlopen(BASELINE_BIN_URL, timeout=60) as response, open(temporary,'wb') as f:
                shutil.copyfileobj(response, f)
            if sha(temporary) != EXPECTED_BIN_SHA256:
                temporary.unlink(missing_ok=True)
                raise ValueError('Downloaded bin checksum mismatch; file rejected.')
            temporary.replace(destination)
        BIN_INPUT = destination
    if BIN_INPUT is None: raise FileNotFoundError('No baseline bin selected; crop inspection unavailable.')
    path = Path(BIN_INPUT)
    if sha(path) != EXPECTED_BIN_SHA256: raise ValueError('Bin is not the exact R4 baseline file.')
    record(path)
    # Loading is permitted only after the exact baseline fingerprint passes.
    with open(path,'rb') as f: bins, bin_labels = pickle.load(f, encoding='bytes')
    if len(bins)!=14000 or len(bin_labels)!=7000: raise ValueError('Unexpected bin dimensions')
    if not np.array_equal(np.asarray(bin_labels,dtype=int),meta.label.to_numpy()[::2]):
        raise ValueError('Bin/cache labels differ')
    pending_pixels, pending_crops = {}, {}
    for idx, data in enumerate(bins):
        h = hashlib.sha256(data).hexdigest()
        if h != meta.encoded_sha256.iloc[idx]: raise ValueError(f'Bin/cache mismatch at occurrence {idx}')
        if h not in pending_pixels:
            pending_pixels[h] = pixel_hash(data)
            pending_crops[h] = data
    pixel_by_encoded, crop_bytes = pending_pixels, pending_crops
    bin_verified = True
    del bins, pending_pixels, pending_crops
    print('Exact R4 bin verified against all 14,000 cache rows.')
except Exception as exc:
    pixel_by_encoded, crop_bytes = {}, {}
    print('Bin unavailable or rejected:', type(exc).__name__, str(exc))
    issues.append('Baseline crop restoration: '+str(exc))


# Cell 13
import cv2, time
from functools import lru_cache
cv2.setNumThreads(1)
RULES = dict(ratio=0.72, min_inliers=10, min_inlier_fraction=0.60,
             min_hull_fraction=0.08, min_overlap=0.45,
             min_correlation=0.90, max_mean_absolute_error=0.12,
             max_median_reprojection=1.5)
print('OpenCV:',cv2.__version__)
print('Fixed screening rules:',RULES)
if not hasattr(cv2,'SIFT_create'):
    raise RuntimeError('OpenCV SIFT is unavailable. Use a standard Kaggle Python CPU image with OpenCV 4.x.')
sift = cv2.SIFT_create(nfeatures=1500, contrastThreshold=0.02)
matcher = cv2.BFMatcher(cv2.NORM_L2)

def features(gray):
    kp, desc = sift.detectAndCompute(gray,None)
    return np.asarray([k.pt for k in kp],dtype=np.float32).reshape(-1,2), desc

@lru_cache(maxsize=128)
def original_features(path):
    im = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    if im is None: raise ValueError('Cannot decode '+path)
    factor = min(1.0,600/max(im.shape))
    if factor<1: im=cv2.resize(im,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
    xy,desc=features(im)
    return im,xy,desc,factor

def assess(original, crop, seed=20260921):
    gray, xy1, d1, factor = original
    target, xy2, d2 = crop
    out=dict(supported=False,reason='insufficient_features',matches=0,inliers=0,
             inlier_fraction=0.,hull_fraction=0.,overlap=0.,correlation=0.,
             mean_absolute_error=1.,median_reprojection=999.,source_resize_factor=factor,
             transform_json='')
    if d1 is None or d2 is None or len(d1)<2 or len(d2)<2: return out
    pairs=matcher.knnMatch(d2,d1,k=2)
    good=[pair[0] for pair in pairs if len(pair)==2 and pair[0].distance<RULES['ratio']*pair[1].distance]
    # One source feature cannot support multiple crop features.
    good=sorted(good,key=lambda m:m.distance)
    seen=set(); good=[m for m in good if not (m.trainIdx in seen or seen.add(m.trainIdx))]
    out['matches']=len(good)
    if len(good)<RULES['min_inliers']: return out
    src=np.float32([xy1[m.trainIdx] for m in good]); dst=np.float32([xy2[m.queryIdx] for m in good])
    cv2.setRNGSeed(seed)
    matrix,mask=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,
        ransacReprojThreshold=2.,maxIters=5000,confidence=0.999,refineIters=10)
    if matrix is None or mask is None: out['reason']='no_transform'; return out
    take=mask.ravel().astype(bool)
    count=int(take.sum()); h,w=target.shape
    out.update(inliers=count,inlier_fraction=count/len(good),transform_json=json.dumps(matrix.tolist()))
    if count<3: return out
    projected=cv2.transform(src.reshape(-1,1,2),matrix).reshape(-1,2)
    out['median_reprojection']=float(np.median(np.linalg.norm(projected[take]-dst[take],axis=1)))
    out['hull_fraction']=float(cv2.contourArea(cv2.convexHull(dst[take]))/(w*h))
    warped=cv2.warpAffine(gray,matrix,(w,h),flags=cv2.INTER_LINEAR)
    support=cv2.warpAffine(np.ones_like(gray,dtype=np.uint8),matrix,(w,h),flags=cv2.INTER_NEAREST)
    support=cv2.erode(support,np.ones((5,5),np.uint8)).astype(bool)
    out['overlap']=float(support.mean())
    a=warped[support].astype(float)/255; b=target[support].astype(float)/255
    if len(a)>20 and a.std()>0.02 and b.std()>0.02:
        out['correlation']=float(np.corrcoef(a,b)[0,1])
        out['mean_absolute_error']=float(np.mean(np.abs(a-b)))
    out['supported']=bool(count>=RULES['min_inliers'] and out['inlier_fraction']>=RULES['min_inlier_fraction']
        and out['hull_fraction']>=RULES['min_hull_fraction'] and out['overlap']>=RULES['min_overlap']
        and out['correlation']>=RULES['min_correlation'] and out['mean_absolute_error']<=RULES['max_mean_absolute_error']
        and out['median_reprojection']<=RULES['max_median_reprojection'])
    out['reason']='supported_by_screen' if out['supported'] else 'needs_review'
    return out


# Cell 15
geo_rows=[]; control_rows=[]
structural_ok = bool(order_ok and path_conflicts==0 and identity_conflicts==0
                     and (audit.person_count>1).sum()==0)
if bin_verified and structural_ok:
    links=unique_links.sort_values(['encoded_sha256','source_path']).reset_index(drop=True).copy()
    print('Hashing source pixels to exclude duplicate-photo controls...')
    source_hash={p:pixel_hash(Path(p).read_bytes()) for p in links.source_path.unique()}
    links['source_pixel_sha256']=links.source_path.map(source_hash)
    pool=links.drop_duplicates('source_pixel_sha256').reset_index(drop=True)
    links.to_csv(OUT/'source_content_inventory.csv',index=False)
    started=time.perf_counter()
    groups=list(links.groupby('encoded_sha256',sort=True))
    for index,(crop_hash, group) in enumerate(groups):
        im=cv2.imdecode(np.frombuffer(crop_bytes[crop_hash],np.uint8),cv2.IMREAD_GRAYSCALE)
        if im is None: raise ValueError('Cannot decode verified baseline crop')
        xy,desc=features(im); crop=(im,xy,desc)
        for _,row in group.iterrows():
            result=assess(original_features(row.source_path),crop)
            geo_rows.append(dict(encoded_sha256=crop_hash,identity=str(row.identity),pose=row.pose,
                                 source_path=row.source_path,**result))
        first=group.iloc[0]; excluded=set(group.source_pixel_sha256)
        for kind in ['same_person_other_photo','different_person']:
            eligible=pool[~pool.source_pixel_sha256.isin(excluded)]
            eligible=eligible[eligible.identity.eq(first.identity) if kind=='same_person_other_photo' else ~eligible.identity.eq(first.identity)]
            pose_pool=eligible[eligible.pose.eq(first.pose)]
            if len(pose_pool):eligible=pose_pool
            if not len(eligible):
                control_rows.append(dict(encoded_sha256=crop_hash,kind=kind,available=False,supported=False));continue
            other=eligible.iloc[index%len(eligible)]
            result=assess(original_features(other.source_path),crop)
            control_rows.append(dict(encoded_sha256=crop_hash,kind=kind,available=True,
                source_path=other.source_path,control_identity=str(other.identity),**result))
        if (index+1)%100==0:
            pd.DataFrame(geo_rows).to_csv(OUT/'geometry_scores.csv',index=False)
            pd.DataFrame(control_rows).to_csv(OUT/'control_scores.csv',index=False)
            print(f'{index+1}/{len(groups)} crops; {(time.perf_counter()-started)/60:.1f} minutes',flush=True)
else:
    print('Geometry not executed: bin verification or structural prerequisites failed. Report will explain.')
geometry=pd.DataFrame(geo_rows)
controls=pd.DataFrame(control_rows)
geometry.to_csv(OUT/'geometry_scores.csv',index=False)
controls.to_csv(OUT/'control_scores.csv',index=False)


# Cell 17
summary_rows=[]
if len(geometry):
    for h,g in geometry.groupby('encoded_sha256',sort=True):
        c=controls[controls.encoded_sha256.eq(h)]
        confounded=bool(c.supported.any())
        supported=bool(g.supported.all() and not confounded)
        summary_rows.append(dict(encoded_sha256=h,identity=g.identity.iloc[0],pose=g.pose.iloc[0],
            aliases=len(g),all_aliases_supported=bool(g.supported.all()),control_pass=confounded,
            status='SCREEN_SUPPORTED' if supported else 'REVIEW_REQUIRED',
            min_correlation=float(g.correlation.min()),source_path=g.source_path.iloc[0]))
evidence=pd.DataFrame(summary_rows)
evidence.to_csv(OUT/'mapping_evidence.csv',index=False)
if len(evidence):
    evidence[evidence.status.eq('REVIEW_REQUIRED')].to_csv(OUT/'unresolved_mapping.csv',index=False)
    from PIL import ImageOps,ImageDraw
    review=evidence[evidence.status.eq('REVIEW_REQUIRED')].head(48)
    # Include examples of supported mappings as a separate deterministic sample.
    ok=evidence[evidence.status.eq('SCREEN_SUPPORTED')]
    review=pd.concat([review,ok.sample(min(12,len(ok)),random_state=20260921)],ignore_index=True)
    review.to_csv(OUT/'review_index.csv',index=False)
    review_dir=OUT/'review_sheets';review_dir.mkdir(exist_ok=True)
    for start in range(0,len(review),6):
        section=review.iloc[start:start+6];canvas=Image.new('RGB',(760,240*len(section)),'white');draw=ImageDraw.Draw(canvas)
        for j,(_,row) in enumerate(section.iterrows()):
            y=j*240
            with Image.open(row.source_path) as im: a=ImageOps.contain(im.convert('RGB'),(300,190))
            with Image.open(io.BytesIO(crop_bytes[row.encoded_sha256])) as im: b=ImageOps.contain(im.convert('RGB'),(190,190))
            canvas.paste(a,(0,y+40));canvas.paste(b,(360,y+40))
            draw.text((5,y+3),f'{row.status} | candidate {row.identity} | {row.pose}',fill='black')
            draw.text((360,y+20),'Baseline crop',fill='black')
        canvas.save(review_dir/f'page_{start//6+1:03d}.jpg',quality=90)
    print(evidence.status.value_counts())


# Cell 18
n_supported=int(evidence.status.eq('SCREEN_SUPPORTED').sum()) if len(evidence) else 0
n_control_pass=int(controls.supported.sum()) if len(controls) else 0
complete=bool(len(evidence)==meta.encoded_sha256.nunique() and n_supported==len(evidence) and structural_ok)
report=dict(experiment='CFP-FP-E05-PREP-R3',run_completed=True,
    status='ALL_SCREEN_SUPPORTED_PENDING_REVIEW' if complete else 'REVIEW_REQUIRED',
    total_cached_crops=int(meta.encoded_sha256.nunique()),crops_evaluated=len(evidence),
    crops_screen_supported=n_supported,crops_not_supported=int(meta.encoded_sha256.nunique())-n_supported,
    negative_controls_evaluated=int(controls.available.sum()) if len(controls) else 0,
    negative_controls_passing=n_control_pass,structural_checks_passed=structural_ok,
    baseline_bin_verified=bin_verified,rules=RULES,opencv=cv2.__version__,numpy=np.__version__,
    source_inputs=input_records,issues=issues,attacker_executed=False,
    interpretation='Engineering image-correspondence screen, not a validated face-recognition or security metric.',
    next_step='Review geometric evidence and all unresolved cases before finalising identity mapping; do not treat candidate labels as verified automatically.',
    future_E05=dict(primary_bits=512,comparison_bits=128,exposures=[1,2,5,10]))
(OUT/'geometry_report.json').write_text(json.dumps(report,indent=2))
row=['CFP-FP-E05-PREP-R3','Check aligned crop correspondence to original photos',
     'Geometric support; mismatched-image controls',
     f'{n_supported}/{meta.encoded_sha256.nunique()} crops screen-supported; {n_control_pass} negative controls passed',
     'Mapping evidence collected; final review pending; attacker not run']
(OUT/'tracker_row.tsv').write_text('\t'.join(row)+'\n')
manifest=[dict(path=p.relative_to(OUT).as_posix(),sha256=sha(p),bytes=p.stat().st_size)
          for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='output_manifest.json']
(OUT/'output_manifest.json').write_text(json.dumps(manifest,indent=2))
archive=Path(shutil.make_archive(str(WORK_ROOT/'CFP-FP-E05-PREP-R3-results'),'zip',OUT))
print({k:v for k,v in report.items() if k not in ['source_inputs','rules']})
print('\t'.join(row))
print('Download:',archive)
import os
display(FileLink(os.path.relpath(archive,Path.cwd())))
