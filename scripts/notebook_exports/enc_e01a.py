# Original code export; run with IPython after configuring inputs.
# Not an independently rerun or refactored implementation.
# Cell 2
import sys, subprocess, importlib.metadata as im
if 'onnxruntime' in sys.modules:
    raise RuntimeError('Restart this Kaggle session, then Run All from the beginning.')
def installed(name):
    try: return im.version(name)
    except im.PackageNotFoundError: return None
if installed('onnxruntime'):
    subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', 'onnxruntime'])
if installed('onnxruntime-gpu') != '1.26.0':
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', 'onnxruntime-gpu==1.26.0'])
print('Environment installation complete.')


# Cell 4
import os, io, json, hashlib, pickle, platform, shutil, time, zipfile, urllib.request
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import cv2
import torch
import onnxruntime as ort
from sklearn.metrics import roc_auc_score, roc_curve
from IPython.display import display, FileLink

RUN = 'CFP-ENC-E01A'
BASE = Path('/kaggle/working') / RUN
OUT, CACHE, PUBLIC = BASE/'results', BASE/'caches', BASE/'public_inputs'
for p in [OUT, CACHE, PUBLIC]: p.mkdir(parents=True, exist_ok=True)
BIN_PATH = None                      # Optional exact cfp_fp.bin path
R50_EXPECTED_SHA256 = None            # Paste first run's recorded hash for later runs
BATCH_SIZE = 64
TARGET_FAR = 0.001
BIN_SHA256 = '76306c783c2ef59c8569ebdcdd2f529f450bcc3fad57c94a5fc2b91df0f10370'
MBF_SHA256 = '9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f'
BIN_URL = ('https://huggingface.co/datasets/Icar/val_sets/resolve/'
           '1f7800d9d5ba295554573a4e292e870b64a49b74/cfp_fp.bin')
MODEL_RELEASE = 'https://github.com/deepinsight/insightface/releases/download/v0.7/'
MODEL_SPECS = {'MobileFaceNet': ('buffalo_sc', 'w600k_mbf.onnx', MBF_SHA256),
               'ResNet50_ArcFace': ('buffalo_l', 'w600k_r50.onnx', R50_EXPECTED_SHA256)}

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def save_json(path, obj):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)
def download(url, dest):
    if dest.exists(): return dest
    tmp = dest.with_suffix(dest.suffix+'.part')
    req = urllib.request.Request(url, headers={'User-Agent':'CFP-research/1.0'})
    with urllib.request.urlopen(req, timeout=120) as src, open(tmp, 'wb') as dst:
        shutil.copyfileobj(src, dst)
    tmp.replace(dest)
    return dest
if not torch.cuda.is_available():
    raise RuntimeError('Select a Kaggle T4 GPU and restart the session.')
if hasattr(ort, 'preload_dlls'): ort.preload_dlls()
if 'CUDAExecutionProvider' not in ort.get_available_providers():
    raise RuntimeError('CUDA provider missing. Restart the fresh GPU session after installation.')
runtime = {'python':sys.version, 'platform':platform.platform(),
           'gpu':torch.cuda.get_device_name(0), 'torch':torch.__version__,
           'cuda_build':torch.version.cuda, 'providers':ort.get_available_providers(),
           'packages':{x:installed(x) for x in ['numpy','pandas','opencv-python','scikit-learn','onnxruntime-gpu']}}
save_json(OUT/'runtime.json', runtime)
print('GPU:', runtime['gpu'])


# Cell 6
# NumPy pickles may use either the NumPy 1.x or 2.x module name.
try:
    from numpy._core.multiarray import _reconstruct, scalar
except ImportError:
    from numpy.core.multiarray import _reconstruct, scalar
import codecs

class DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ('numpy', 'ndarray'): np.ndarray,
            ('numpy', 'dtype'): np.dtype,
            ('_codecs', 'encode'): codecs.encode,
        }
        for prefix in ('numpy.core.multiarray', 'numpy._core.multiarray'):
            allowed[(prefix, '_reconstruct')] = _reconstruct
            allowed[(prefix, 'scalar')] = scalar
        if (module, name) in allowed:
            return allowed[(module, name)]
        raise pickle.UnpicklingError(f'Unexpected pickle global: {module}.{name}')

if BIN_PATH is None:
    candidates = sorted(Path('/kaggle/input').rglob('cfp_fp.bin'))
    if len(candidates)>1: raise RuntimeError('Multiple cfp_fp.bin files: set BIN_PATH explicitly.')
    source = candidates[0] if candidates else download(BIN_URL, PUBLIC/'cfp_fp.bin')
else: source = Path(BIN_PATH)
if sha256(source) != BIN_SHA256:
    raise RuntimeError('CFP source differs from the frozen R4 checksum. Do not substitute data.')
with open(source, 'rb') as f:
    bins, original_labels = DataOnlyUnpickler(f, encoding='bytes').load()
if len(bins)!=14000 or len(original_labels)!=7000:
    raise RuntimeError('Expected 14000 images and 7000 pair labels.')
if not all(x in (True, False, 0, 1) for x in original_labels):
    raise RuntimeError('Non-binary labels.')
labels = np.asarray(original_labels, dtype=np.int8)
folds = np.arange(7000)//700+1
for fold in range(1,11):
    if labels[folds==fold].sum()!=350: raise RuntimeError(f'Fold {fold} is not 350G/350I.')
def encoded_image_bytes(value):
    # Preserve compressed image bytes; do not cast numeric values or re-encode images.
    if isinstance(value, bytes):
        result = value
    elif isinstance(value, (bytearray, memoryview)):
        result = bytes(value)
    elif isinstance(value, np.ndarray) and value.dtype in (np.dtype('uint8'), np.dtype('int8')):
        result = value.tobytes(order='C')
    else:
        raise RuntimeError(
            f'Unsupported encoded image buffer: type={type(value).__name__}, '
            f'dtype={getattr(value, "dtype", None)}, shape={getattr(value, "shape", None)}'
        )
    if not result:
        raise RuntimeError('Empty encoded image buffer.')
    return result

rows, unique_bins, lookup, first = [], [], {}, []
for i, encoded in enumerate(bins):
    encoded = encoded_image_bytes(encoded)
    h = hashlib.sha256(encoded).hexdigest()
    if h not in lookup:
        lookup[h]=len(unique_bins); unique_bins.append(encoded); first.append(i)
    rows.append({'bin_index':i, 'pair_id':i//2+1, 'side':'a' if i%2==0 else 'b',
                 'fold':int(folds[i//2]), 'label':int(labels[i//2]),
                 'encoded_sha256':h, 'unique_image_id':lookup[h], 'status':'valid'})
meta = pd.DataFrame(rows)
if len(unique_bins)!=5890: raise RuntimeError('Expected 5890 unique encoded R4 crops.')
meta.to_csv(CACHE/'aligned_pair_side_metadata.csv', index=False)
unique_meta = meta.iloc[first].reset_index(drop=True)
unique_meta.to_csv(CACHE/'unique_aligned_metadata.csv', index=False)
occurrence_to_unique = meta.unique_image_id.to_numpy()
np.save(CACHE/'occurrence_to_unique.npy', occurrence_to_unique)

def decode(encoded):
    img = cv2.imdecode(np.frombuffer(encoded,dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None or img.shape!=(112,112,3):
        raise RuntimeError('Invalid 112x112 aligned crop; automatic resize is disabled.')
    return img
# All images are decoded once for validation before starting either encoder.
for b in unique_bins: decode(b)
source_audit = {'sha256':BIN_SHA256, 'path':str(source), 'pairs':7000,
                'pair_sides':14000, 'unique_crops':len(unique_bins),
                'fold_rule':'10 contiguous folds, 700 pairs each, 350G/350I',
                'detector':False,'flip':False,'resize':False}
save_json(OUT/'source_audit.json', source_audit)
print('PASS: frozen R4 source, all 5890 unique crops and 7000 pairs validated.')
print('Encoded buffer normalization preserves compressed bytes; no resizing or re-encoding.')


# Cell 8
model_paths, model_audits = {}, {}
for encoder, (pack, filename, expected_hash) in MODEL_SPECS.items():
    destination = PUBLIC/filename
    lock_path = PUBLIC/(filename+'.lock.json')
    if expected_hash:
        candidates = sorted(Path('/kaggle/input').rglob(filename))
        matches = [p for p in candidates if sha256(p)==expected_hash]
        if matches and not destination.exists(): shutil.copy2(matches[0],destination)
    # Never trust an unpinned/unlocked local R50 model simply because its name matches.
    if destination.exists() and not expected_hash and not lock_path.exists():
        raise RuntimeError(f'Unverified local model {filename}: remove it and rerun to use official download.')
    if not destination.exists():
        archive = download(MODEL_RELEASE+pack+'.zip', PUBLIC/(pack+'.zip'))
        with zipfile.ZipFile(archive) as z:
            members=[x for x in z.infolist() if Path(x.filename).name==filename]
            if len(members)!=1: raise RuntimeError(f'Missing/ambiguous {filename} in official archive.')
            tmp=destination.with_suffix('.part')
            with z.open(members[0]) as src, open(tmp,'wb') as dst: shutil.copyfileobj(src,dst)
            tmp.replace(destination)
        downloaded_from = MODEL_RELEASE+pack+'.zip'
    else: downloaded_from = 'local copy with checksum/lock validation'
    digest=sha256(destination)
    if expected_hash and digest!=expected_hash: raise RuntimeError(f'Checkpoint checksum mismatch: {filename}')
    if lock_path.exists() and json.loads(lock_path.read_text())['sha256']!=digest:
        raise RuntimeError(f'Checkpoint differs from first-run lock: {filename}')
    lock={'filename':filename,'sha256':digest,'official_pack':pack,
          'source_url':MODEL_RELEASE+pack+'.zip','bytes':destination.stat().st_size,
          'hash_validation':'prior known checksum' if expected_hash else 'official source; first-run hash lock'}
    save_json(lock_path,lock)
    model_paths[encoder]=destination; model_audits[encoder]=lock
save_json(OUT/'model_locks.json',model_audits)
print(json.dumps(model_audits,indent=2))


# Cell 10
extraction_rows=[]
for encoder,path in model_paths.items():
    d=CACHE/encoder; d.mkdir(exist_ok=True)
    binding={'source_sha256':BIN_SHA256,'model_sha256':model_audits[encoder]['sha256'],
             'preprocessing':'BGR decode -> RGB NCHW float32 (x-127.5)/127.5; no flip',
             'dimension':512,'unique_crops':len(unique_bins),'runtime':runtime}
    binding_path=d/'cache_binding.json'; features_path=d/'unique_aligned_embeddings.npy'
    if binding_path.exists() and json.loads(binding_path.read_text())!=binding:
        raise RuntimeError(f'Cache identity changed for {encoder}; use a new output directory.')
    if features_path.exists() and not binding_path.exists(): raise RuntimeError('Unbound cache file.')
    save_json(binding_path,binding)
    if not features_path.exists():
        f=np.lib.format.open_memmap(features_path,mode='w+',dtype='float32',shape=(len(unique_bins),512))
        f[:]=np.nan; f.flush(); del f
    X=np.load(features_path,mmap_mode='r+')
    if X.shape!=(len(unique_bins),512) or X.dtype!=np.float32: raise RuntimeError('Invalid cache format.')
    session=ort.InferenceSession(str(path),providers=['CUDAExecutionProvider','CPUExecutionProvider'])
    if 'CUDAExecutionProvider' not in session.get_providers(): raise RuntimeError('CUDA session unavailable.')
    inp=session.get_inputs()[0]; outputs=session.get_outputs()
    if len(session.get_inputs())!=1 or inp.shape[1:]!=[3,112,112] or inp.type!='tensor(float)':
        raise RuntimeError(f'Unexpected input for {encoder}: {inp.shape}, {inp.type}')
    if len(outputs)!=1 or outputs[0].shape[-1]!=512: raise RuntimeError('Expected 512-D recognition output.')
    fixed_batch=inp.shape[0] if isinstance(inp.shape[0],int) and inp.shape[0]>0 else None
    bs=fixed_batch or BATCH_SIZE
    def infer(images):
        blob=cv2.dnn.blobFromImages(images,1/127.5,(112,112),(127.5,127.5,127.5),swapRB=True)
        raw=session.run([outputs[0].name],{inp.name:blob})[0].astype(np.float32)
        norm=np.linalg.norm(raw,axis=1,keepdims=True)
        if raw.shape!=(len(images),512) or not np.isfinite(raw).all() or (norm<=1e-12).any():
            raise RuntimeError('Non-finite/zero/incorrect embedding output.')
        return raw/norm
    warm=[decode(unique_bins[0]) for _ in range(bs)]
    a=infer(warm); b=infer(warm)
    if not np.allclose(a,b,atol=1e-5): raise RuntimeError('Repeat-inference smoke test failed.')
    pending=np.flatnonzero(~np.isfinite(X).all(axis=1))
    start=time.perf_counter()
    for offset in range(0,len(pending),bs):
        ids=pending[offset:offset+bs]; imgs=[decode(unique_bins[int(j)]) for j in ids]
        if fixed_batch and len(imgs)<fixed_batch: imgs+= [imgs[-1]]*(fixed_batch-len(imgs))
        X[ids]=infer(imgs)[:len(ids)]; X.flush()
        if offset%(bs*10)==0: print(encoder, min(offset+bs,len(pending)), '/',len(pending))
    elapsed=time.perf_counter()-start
    if not np.isfinite(X).all() or not np.allclose(np.linalg.norm(X,axis=1),1,atol=1e-4):
        raise RuntimeError('Incomplete/non-unit cache; metrics blocked.')
    np.save(d/'aligned_pair_side_embeddings.npy',np.asarray(X[occurrence_to_unique]))
    # Reuse established R4 metadata filenames for later adapters.
    meta.to_csv(d/'aligned_pair_side_metadata.csv',index=False)
    unique_meta.to_csv(d/'unique_aligned_metadata.csv',index=False)
    save_json(d/'model_lock.json',model_audits[encoder])
    extraction_rows.append({'encoder':encoder,'coverage':1.,'unique_crops':len(X),
                            'pair_sides':14000,'newly_computed':len(pending),
                            'extraction_seconds_this_run':elapsed,
                            'checkpoint_bytes':path.stat().st_size,'raw_embedding_bytes':512*4})
    del session,X
pd.DataFrame(extraction_rows).to_csv(OUT/'extraction_audit.csv',index=False)
print('PASS: complete caches for both encoders.')


# Cell 12
def accuracy_threshold(scores,y):
    # Equivalent to the R4 highest-threshold tie rule without an N x N allocation.
    values,inverse=np.unique(scores,return_inverse=True)
    pos=np.bincount(inverse,weights=(y==1),minlength=len(values))
    neg=np.bincount(inverse,weights=(y==0),minlength=len(values))
    candidates=np.r_[values,np.nextafter(values[-1],np.inf)]
    correct=np.r_[0,np.cumsum(neg)] + np.sum(pos)-np.r_[0,np.cumsum(pos)]
    return float(candidates[np.flatnonzero(correct==correct.max())[-1]])
def far_threshold(scores,y,target=TARGET_FAR):
    neg=np.sort(scores[y==0])[::-1]
    allowed=int(np.floor(target*len(neg)))
    t=float(np.nextafter(neg[allowed],np.inf))
    if np.mean(neg>=t)>target: raise RuntimeError('Calibration FAR violation.')
    return t

def roc_stats(y,s):
    fpr,tpr,_=roc_curve(y,s,drop_intermediate=False); d=fpr-(1-tpr)
    crosses=np.flatnonzero(d[:-1]*d[1:]<=0)
    if len(crosses):
        i=min(crosses,key=lambda j:abs(d[j])+abs(d[j+1]))
        w=.5 if d[i+1]==d[i] else -d[i]/(d[i+1]-d[i])
        eer=float(fpr[i]+w*(fpr[i+1]-fpr[i]))
    else:
        i=np.argmin(abs(d)); eer=float((fpr[i]+1-tpr[i])/2)
    return float(roc_auc_score(y,s)),eer

def evaluate_encoder(encoder,E,y,fold_ids):
    s=np.sum(E[0::2]*E[1::2],axis=1).astype(np.float64)
    if len(s)!=len(y) or not np.isfinite(s).all(): raise RuntimeError('Invalid pair scores.')
    decisions=[]; fold_rows=[]
    for fold in sorted(np.unique(fold_ids)):
        tr=fold_ids!=fold; te=~tr
        for rule,select in [('accuracy',accuracy_threshold),('far_0.001',far_threshold)]:
            threshold=select(s[tr],y[tr]); pred=s[te]>=threshold; yt=y[te]
            tp=int(np.sum(pred & (yt==1))); fp=int(np.sum(pred & (yt==0)))
            tn=int(np.sum(~pred & (yt==0))); fn=int(np.sum(~pred & (yt==1)))
            fold_rows.append({'encoder':encoder,'rule':rule,'fold':int(fold),'threshold':threshold,
                              'training_far':float(np.mean(s[tr][y[tr]==0]>=threshold)),
                              'tp':tp,'fp':fp,'tn':tn,'fn':fn,
                              'tar':tp/(tp+fn),'far':fp/(fp+tn),'accuracy':(tp+tn)/len(yt)})
            for j,p in zip(np.flatnonzero(te),pred):
                decisions.append({'encoder':encoder,'rule':rule,'pair_id':int(j+1),
                                  'fold':int(fold),'label':int(y[j]),'score':float(s[j]),
                                  'threshold':threshold,'accepted':int(p)})
    f=pd.DataFrame(fold_rows); auc,eer=roc_stats(y,s); summary=[]
    for rule,g in f.groupby('rule'):
        tp,fp,tn,fn=[int(g[k].sum()) for k in ['tp','fp','tn','fn']]
        summary.append({'encoder':encoder,'rule':rule,'pairs':len(y),
                        'accuracy':(tp+tn)/len(y),'tar':tp/(tp+fn),'observed_far':fp/(fp+tn),
                        'frr':fn/(tp+fn),'auc':auc,'eer':eer,'tp':tp,'fp':fp,'tn':tn,'fn':fn,
                        'raw_embedding_bytes':2048})
    return f,pd.DataFrame(decisions),pd.DataFrame(summary)

all_folds=[]; all_decisions=[]; all_summaries=[]
for encoder in MODEL_SPECS:
    E=np.load(CACHE/encoder/'aligned_pair_side_embeddings.npy')
    if E.shape!=(14000,512): raise RuntimeError('Pair-side shape mismatch.')
    f,d,s=evaluate_encoder(encoder,E,labels,folds)
    all_folds.append(f); all_decisions.append(d); all_summaries.append(s)
fold_results=pd.concat(all_folds,ignore_index=True)
decisions=pd.concat(all_decisions,ignore_index=True)
summary=pd.concat(all_summaries,ignore_index=True)
fold_results.to_csv(OUT/'fold_results.csv',index=False)
decisions.to_csv(OUT/'out_of_fold_decisions.csv',index=False)
summary.to_csv(OUT/'recognition_summary.csv',index=False)
display(summary)
comparison=[]
for rule in ['accuracy','far_0.001']:
    g=summary[summary.rule==rule].set_index('encoder')
    for metric in ['accuracy','tar','observed_far','auc','eer']:
        comparison.append({'rule':rule,'metric':metric,
                           'MobileFaceNet':g.loc['MobileFaceNet',metric],
                           'ResNet50_ArcFace':g.loc['ResNet50_ArcFace',metric],
                           'R50_minus_MBF_percentage_points':100*(g.loc['ResNet50_ArcFace',metric]-g.loc['MobileFaceNet',metric])})
pd.DataFrame(comparison).to_csv(OUT/'paired_protocol_descriptive_comparison.csv',index=False)
display(pd.DataFrame(comparison))


# Cell 14
payload=pd.DataFrame([{'representation':'512-D float32','bytes':2048,'reduction_factor':1,'saved_percent':0},
                      {'representation':'128-bit packed (planned)','bytes':16,'reduction_factor':128,'saved_percent':99.21875},
                      {'representation':'512-bit packed (planned)','bytes':64,'reduction_factor':32,'saved_percent':96.875}])
payload.to_csv(OUT/'template_payload_arithmetic.csv',index=False)
status={'status':'PASS','experiment':RUN,'completed_utc':datetime.now(timezone.utc).isoformat(),
        'completed_scope':'CFP preparation and paired raw recognition only',
        'pending_scope':['LFW preparation','RO/SRHT/PCA utility','renewal/linkage','inversion','identity-bootstrap and threshold comparison'],
        'source':source_audit,'models':model_audits,
        'limitations':['Official folds are not identity-disjoint.',
                      'New R50 hash is source-recorded, not independently authenticated against a published digest.',
                      'Pretrained-model comparison; not a controlled training ablation.',
                      'No protected utility or security measurements in this notebook.',
                      'No measured edge-device latency, memory, energy, or significance inference.']}
save_json(OUT/'run_status.json',status)
save_json(CACHE/'run_provenance.json',status)
r=summary[summary.rule=='far_0.001'].set_index('encoder')
row='\t'.join([RUN,'CFP-FP official aligned; paired raw encoders','PASS',
    '7000 pairs; 14000 pair sides; 5890 unique crops',
    f"MBF TAR {100*r.loc['MobileFaceNet','tar']:.4f}%; observed FAR {100*r.loc['MobileFaceNet','observed_far']:.4f}%",
    f"R50 TAR {100*r.loc['ResNet50_ArcFace','tar']:.4f}%; observed FAR {100*r.loc['ResNet50_ArcFace','observed_far']:.4f}%",
    'Training FAR target 0.1%; 9/1 folds; raw cosine; 2048 B; protection pending'])
(OUT/'tracker_row_no_headers.tsv').write_text(row+'\n')
print('Paste-ready tracker row (no column names):\n'+row)
for root in [OUT,CACHE]:
    records=[{'file':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':sha256(p)}
             for p in sorted(root.rglob('*')) if p.is_file() and p.name!='file_manifest.csv']
    pd.DataFrame(records).to_csv(root/'file_manifest.csv',index=False)
for name,root in [('results',OUT),('caches',CACHE)]:
    z=Path(shutil.make_archive('/kaggle/working/'+RUN+'-'+name,'zip',root_dir=root))
    print(z.name, round(z.stat().st_size/1024**2,2),'MiB')
    display(FileLink(str(z)))
print('PASS — CFP-ENC-E01A complete. Return the results ZIP; retain the caches ZIP for the next run.')
