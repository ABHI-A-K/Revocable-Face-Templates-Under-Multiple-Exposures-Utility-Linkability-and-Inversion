"""Rebuild publication figures/tables from archived, aggregate measurements.
Run: python build_figures.py (requires numpy, pandas, matplotlib).
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
D=ROOT/'data'; F=ROOT/'figures'
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.6,'lines.linewidth':1.2,'pdf.fonttype':42,'ps.fonttype':42,'savefig.pad_inches':.03})
styles={'RO':('#0072B2','o','-'),'SRHT':('#D55E00','s','--'),'PCA_GUIDED_RO':('#009E73','^',':')}
names={'RO':'RO','SRHT':'SRHT','PCA_GUIDED_RO':'PCA-guided RO'}
a=pd.read_csv(D/'identity_bootstrap_intervals.csv')
fig,axs=plt.subplots(1,3,figsize=(7.16,2.45),sharey=True)
for ax,(ds,b,title) in zip(axs,[('LFW',128,'(a) LFW, 128 bits'),('CFP_FP_SCREENED',128,'(b) CFP screened, 128 bits'),('CFP_FP_SCREENED',512,'(c) CFP screened, 512 bits')]):
 for m,(c,mk,ls) in styles.items():
  v=a.query('dataset==@ds and bits==@b and method==@m and attack=="calibrated" and metric=="fresh_asr"').sort_values('K')
  ax.errorbar(v.K,v['mean']*100,yerr=np.vstack([v['mean']-v.ci95_low,v.ci95_high-v['mean']])*100,color=c,marker=mk,linestyle=ls,markersize=3.5,capsize=2,label=names[m])
 ax.set_title(title);ax.set_xticks([1,2,5,10]);ax.set_xlabel('Exposed templates, K');ax.set_ylim(0,100);ax.grid(axis='y',color='.86',linewidth=.5);ax.spines[['top','right']].set_visible(False)
axs[0].set_ylabel('Fresh-key attack success (%)');fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.02),ncol=3,frameon=False)
fig.tight_layout(w_pad=1,rect=[0,0,1,.90]);fig.savefig(F/'exposure.pdf');plt.close(fig)
u=pd.read_csv(D/'threshold_utility_summary.csv'); t=pd.read_csv(D/'threshold_attack_summary.csv')
fig,axs=plt.subplots(1,3,figsize=(7.16,2.3))
for m,(c,mk,ls) in styles.items():
 v=u.query('bits==512 and method==@m').sort_values('target_far');w=t.query('bits==512 and method==@m and K==10 and attack=="calibrated" and matcher=="PROTECTED"').sort_values('target_far')
 for ax,y in zip(axs,[v.tar*100,v.far*100,w.asr*100]):
  ax.plot(v.target_far,y,color=c,marker=mk,linestyle=ls,markersize=3.5,label=names[m]);ax.set_xscale('log');ax.set_xlabel('Validation FAR target');ax.set_xticks([.0001,.001,.01]);ax.grid(axis='y',color='.86',linewidth=.5);ax.spines[['top','right']].set_visible(False)
for ax,title in zip(axs,['(a) Genuine acceptance (%)','(b) Observed test FAR (%)','(c) Fresh-key ASR, K=10 (%)']):ax.set_title(title)
axs[1].plot([.0001,.01],[.01,1.0],color='.65',linestyle='-.',linewidth=.7,label='Target reference')
fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.03),ncol=3,frameon=False);axs[1].legend(handles=[axs[1].lines[-1]],loc='upper left',frameon=False,fontsize=6);fig.tight_layout(w_pad=1.2,rect=[0,0,1,.9]);fig.savefig(F/'thresholds.pdf');plt.close(fig)
# Compact numerical table generated directly from the analysis output.
rows=[]
for ds,b in [('LFW',128),('CFP_FP_SCREENED',128),('CFP_FP_SCREENED',512)]:
 for m in styles:
  v=a.query('dataset==@ds and bits==@b and method==@m and attack=="calibrated" and metric=="fresh_asr"').set_index('K')
  c=a.query('dataset==@ds and bits==@b and method==@m and K==10 and attack=="calibrated" and metric=="exposed_cosine"').iloc[0]
  raw=a.query('dataset==@ds and bits==@b and method==@m and K==10 and attack=="calibrated" and metric=="raw_asr"').iloc[0]
  label='LFW' if ds=='LFW' else 'CFP screened'
  rows.append(f"{label} & {b} & {names[m]} & {v.loc[1,'mean']*100:.2f} & {v.loc[10,'mean']*100:.2f} [{v.loc[10,'ci95_low']*100:.2f}, {v.loc[10,'ci95_high']*100:.2f}] & {c['mean']:.4f} & {raw['mean']*100:.2f} \\\\")
(ROOT/'data'/'attack_rows.tex').write_text('\n'.join(rows)+'\n')

# Keep the manuscript table synchronized with the generated rows.
p=ROOT/'sections'/'03_results.tex'
if p.exists():
 import re
 text=p.read_text()
 text=re.sub(r'% BEGIN GENERATED ATTACK ROWS.*?% END GENERATED ATTACK ROWS',lambda _: '% BEGIN GENERATED ATTACK ROWS\n'+'\n'.join(rows)+'\n% END GENERATED ATTACK ROWS',text,flags=re.S)
 p.write_text(text)
