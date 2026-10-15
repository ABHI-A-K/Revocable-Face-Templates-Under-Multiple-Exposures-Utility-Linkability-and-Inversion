"""Offline E03-template to E04 reproduction. Requires numpy and pandas."""
import io, json, zipfile, hashlib, platform
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

def check(condition, message):
    if not condition: raise ValueError(message)

def digest(raw): return hashlib.sha256(raw).hexdigest()

def verified_zip(raw):
    z = zipfile.ZipFile(io.BytesIO(raw))
    check(len(z.namelist()) == len(set(z.namelist())), 'Duplicate archive entries')
    manifest = json.loads(z.read('manifest.json'))
    check({i.filename for i in z.infolist() if not i.is_dir()} == {e['path'] for e in manifest} | {'manifest.json'}, 'Manifest coverage')
    for e in manifest:
        data = z.read(e['path'])
        check(len(data) == e['bytes'] and digest(data) == e['sha256'], 'Manifest mismatch: '+e['path'])
    return z, len(manifest)

def csv(z, name, **kw):
    return pd.read_csv(io.BytesIO(z.read(name)), float_precision='round_trip', **kw)

def reproduce(e03_path, e04_path, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    e03raw=Path(e03_path).read_bytes(); e04raw=Path(e04_path).read_bytes()
    e03=zipfile.ZipFile(io.BytesIO(e03raw)); e04=zipfile.ZipFile(io.BytesIO(e04raw))
    rr=e03.read('CFP-E03-DIM-results.zip'); cr=e03.read('CFP-E03-DIM-protected-template-cache.zip')
    er=e04.read('CFP-E04-REV-results.zip')
    results,nr=verified_zip(rr); cache,nc=verified_zip(cr); saved,ne=verified_zip(er)
    record=json.loads(saved.read('experiment_record.json'))
    check(record['inputs']['e03_results_sha256']==digest(rr), 'E03 results differ from E04 source')
    check(record['inputs']['e03_cache_sha256']==digest(cr), 'E03 templates differ from E04 source')
    source=json.loads(results.read('experiment_record.json'))
    for name, field in [('aligned_pair_side_metadata.csv','pair_metadata_sha256'),('unique_aligned_metadata.csv','unique_metadata_sha256')]:
        check(digest(cache.read(name))==source['cache_audit'][field], 'Metadata provenance')
    meta=csv(cache,'aligned_pair_side_metadata.csv')
    check(len(meta)==14000, 'Pair metadata size')
    check(np.array_equal(meta.bin_index,np.arange(14000)), 'Pair ordering')
    check(np.array_equal(meta.pair_id,np.repeat(np.arange(1,7001),2)), 'Pair identifiers')
    check(np.array_equal(meta.side,np.tile(['a','b'],7000)), 'Side ordering')
    a=meta.iloc[::2]; b=meta.iloc[1::2]
    check(np.array_equal(a.label,b.label) and np.array_equal(a.fold,b.fold), 'Side labels/folds')
    y=a.label.to_numpy(); folds=a.fold.to_numpy(); ids=a.pair_id.to_numpy()
    ia=a.unique_image_id.to_numpy(); ib=b.unique_image_id.to_numpy()
    check(min(ia.min(),ib.min())>=0 and max(ia.max(),ib.max())<5890, 'Template indices')
    for f in range(1,11):
        check(((folds==f)&(y==1)).sum()==350 and ((folds==f)&(y==0)).sum()==350, 'Fold protocol')
    seeds=list(range(24001,24011)); controls=csv(saved,'legitimate_new_key_fold_controls.csv')
    check(source['key_seeds']==seeds and record['seeds']==seeds,'Seed ordering')
    pop=np.array([int(i).bit_count() for i in range(256)],dtype=np.uint8)
    # Independent implementation: XOR packed bytes and count differing bits.
    def score(x,z,bits): return 1.0-pop[np.bitwise_xor(x,z)].sum(axis=1)/bits
    checks=[]; summaries=[]; calibration=[]
    for bits in [512,128]:
        cfg=json.loads(cache.read(f'bits_{bits}/config.json'))
        check(cfg['seeds']==seeds and cfg['bits']==bits and cfg['bitorder']=='big','Template configuration')
        p=np.load(io.BytesIO(cache.read(f'bits_{bits}/templates_packed.npy')),allow_pickle=False)
        check(p.dtype==np.uint8 and p.shape==(10,5890,bits//8),'Template shape')
        ref=csv(results,f'bits_{bits}/same_key_pair_scores_all_keys.csv')
        cal=csv(results,f'bits_{bits}/tar_at_far_1e-3_fold_results_all_keys.csv')
        thresholds=np.empty((10,10))
        for k,seed in enumerate(seeds):
            s=score(p[k,ia],p[k,ib],bits)
            expected=ref[ref.key_seed==seed].sort_values('pair_id')
            check(np.array_equal(expected.pair_id,ids) and np.array_equal(expected.label,y) and np.array_equal(expected.fold,folds),'E03 pair alignment')
            check(np.array_equal(s,expected.hamming_similarity),'E03 score mismatch')
            for f in range(1,11):
                imp=s[(folds!=f)&(y==0)]
                # Select the fourth largest impostor score; reject its whole tie group.
                budget=int(np.floor(.001*len(imp)))
                boundary=np.partition(imp,len(imp)-budget-1)[len(imp)-budget-1]
                t=np.nextafter(boundary,np.inf); thresholds[k,f-1]=t
                test=folds==f; pred=s[test]>=t
                row=dict(bits=bits,key=seed,fold=f,threshold=float(t),training_fp=int((imp>=t).sum()),training_impostors=len(imp),tp=int((pred&(y[test]==1)).sum()),genuine=int((y[test]==1).sum()),fp=int((pred&(y[test]==0)).sum()),impostors=int((y[test]==0).sum()))
                old=controls[(controls.bits==bits)&(controls.key==seed)&(controls.fold==f)]
                check(len(old)==1,'Control identity')
                for col,value in row.items(): check(old.iloc[0][col]==value,'E04 control mismatch: '+col)
                prior=cal[(cal.key_seed==seed)&(cal.fold==f)]
                check(len(prior)==1,'E03 calibration identity')
                for col in ['threshold','tp','fp']: check(prior.iloc[0][col]==row[col],'E03 calibration mismatch: '+col)
                calibration.append(row)
        df=csv(saved,f'revoked_scores_{bits}.csv.gz',compression='gzip')
        check(len(df)==1260000 and not df.duplicated(['old_key','new_key','scenario','pair_id']).any(),'E04 coverage/duplicates')
        g=y==1; ga=ia[g]; gb=ib[g]; gp=ids[g]; gf=folds[g]
        groups=df.groupby(['old_key','new_key','scenario'])
        scenarios={'primary_old_b_new_a':(gb,ga),'reverse_capture_old_a_new_b':(ga,gb),'same_image_a':(ga,ga),'same_image_b':(gb,gb)}
        check(len(groups)==360,'Transition/scenario coverage')
        for old,os in enumerate(seeds):
            for new,ns in enumerate(seeds):
                if old==new: continue
                for name,(oi,ni) in scenarios.items():
                    r=groups.get_group((os,ns,name)).sort_values('pair_id')
                    check(np.array_equal(r.pair_id,gp) and np.array_equal(r.fold,gf),'E04 pair alignment')
                    s=score(p[old,oi],p[new,ni],bits); t=thresholds[new,gf-1]; accept=s>=t
                    check(np.array_equal(s,r.similarity),'Regenerated E04 score mismatch')
                    check(np.array_equal(t,r.threshold),'Regenerated E04 threshold mismatch')
                    check(np.array_equal(accept,r.accepted),'Regenerated E04 decision mismatch')
                    summaries.append(dict(bits=bits,scenario=name,old_key=os,new_key=ns,accepted=int(accept.sum()),trials=len(s)))
        checks.append(dict(bits=bits,same_key_scores_exact=70000,thresholds_exact=100,revoked_scores_exact=len(df),revoked_decisions_exact=len(df)))
        print(f'{bits} bits: 70,000 same-key scores, 100 thresholds and 1,260,000 renewal scores/decisions match exactly.',flush=True)
    transitions=pd.DataFrame(summaries)
    summary=transitions.groupby(['bits','scenario'],as_index=False)[['accepted','trials']].sum()
    summary['acceptance_rate']=summary.accepted/summary.trials
    summary.to_csv(output/'regenerated_summary.csv',index=False)
    transitions.to_csv(output/'regenerated_transitions.csv',index=False)
    pd.DataFrame(calibration).to_csv(output/'regenerated_controls.csv',index=False)
    report=dict(status='PASS',completed_utc=datetime.now(timezone.utc).isoformat(),execution='Local CPU; independent packed-byte XOR/popcount implementation',python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,inputs=dict(e03_package_sha256=digest(e03raw),e04_package_sha256=digest(e04raw),e03_results_sha256=digest(rr),e03_cache_sha256=digest(cr)),manifest_files_verified=dict(e03_results=nr,e03_cache=nc,e04_results=ne),source_hashes_match_E04=True,checks=checks,limitations=['Reproduction starts from cached protected templates; raw embeddings and images were not regenerated.','Same dataset, pairs and keys: replication is not additional independent statistical evidence.','No identity-disjoint attacker, inversion, authenticated cloud deployment or zero-risk claim.'])
    (output/'reproduction_audit.json').write_text(json.dumps(report,indent=2))
    return report,summary

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('e03');p.add_argument('e04');p.add_argument('--out',required=True)
    args=p.parse_args(); report,summary=reproduce(args.e03,args.e04,args.out)
    print(summary.to_string(index=False)); print('PASS: cached-template reproduction complete.')
