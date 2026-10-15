from pathlib import Path
import pandas as pd
P=Path(__file__).resolve().parents[1]
p=list((P/'results/archive').glob('lfw_e01/**/out_of_fold_decisions.csv'));assert len(p)==1
x=pd.read_csv(p[0],float_precision='round_trip');assert len(x)==6000
v=x.matcher_evaluable.astype(str).str.lower().eq('true');assert v.sum()==5965
for name,mask in [('matcher',v),('operational',pd.Series(True,index=x.index))]:
 score='matcher_similarity' if name=='matcher' else 'operational_score'
 pred=(x[score]>=x[name+'_threshold']) & v
 assert (pred[mask]==x[name+'_prediction'].fillna(0).astype(bool)[mask]).all()
 print(name,'saved-threshold TAR',pred[mask & x.label.eq(1)].mean(),'FAR',pred[mask & x.label.eq(0)].mean())
print('PASS: saved E01 decisions; these are not substituted for separately calibrated TAR-at-FAR results')
