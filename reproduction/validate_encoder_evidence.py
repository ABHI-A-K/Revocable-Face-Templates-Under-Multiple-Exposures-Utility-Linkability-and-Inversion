"""Validate supplied aggregates without rerunning CFP attacks or bootstraps."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
D=Path(__file__).resolve().parent/'data/encoder_extension'
e3=pd.read_csv(D/'E03_attack_summary.csv').set_index(['encoder','bits','method','K','attack'])
ci=pd.read_csv(D/'E04_identity_bootstrap_intervals.csv')
metrics=['fresh_asr','same_image_fresh_asr','raw_asr','baseline_fresh_asr','exposed_cosine','source_cosine','cross_cosine']
n=0
for row in ci.query('target_far==.001 and K>0').itertuples():
 if row.metric in metrics:
  assert np.isclose(row.estimate,e3.loc[(row.encoder,row.bits,row.method,row.K,row.attack),row.metric],atol=1e-12,rtol=0)
  n+=1
assert n==672
paired=pd.read_csv(D/'E04_paired_encoder_intervals.csv').query('K==10 and attack=="calibrated" and target_far==.001 and metric=="fresh_asr"')
assert len(paired)==6 and (paired.ci_low>0).all()
for r in paired.itertuples():
 a=ci.query('method==@r.method and bits==@r.bits and K==10 and attack=="calibrated" and target_far==.001 and metric=="fresh_asr"').set_index('encoder')
 assert np.isclose(a.loc['ResNet50_ArcFace','estimate']-a.loc['MobileFaceNet','estimate'],r.difference,atol=1e-12,rtol=0)
assert json.loads((D/'lfw/run_status.json').read_text())['status']=='PASS'
print('PASS: 672 CFP aggregate estimates, six paired differences, and completed LFW validation status')
