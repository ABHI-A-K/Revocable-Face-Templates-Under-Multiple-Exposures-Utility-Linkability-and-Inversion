from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
P=Path(__file__).resolve().parent;D=P/'data/encoder_extension';F=P/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
encs=['MobileFaceNet','ResNet50_ArcFace']; names=['MobileFaceNet','ResNet50 ArcFace'];colors=['#1764a0','#bd4b12']; methods=['RO','SRHT','PCA_GUIDED_RO'];ml=['RO','SRHT','PCA-guided RO'];markers=['o','s']
d=pd.read_csv(D/'E04_identity_bootstrap_intervals.csv');saved=[]
def save(fig,name):
 fig.savefig(F/(name+'.pdf'),bbox_inches='tight');fig.savefig(F/(name+'.png'),dpi=170,bbox_inches='tight');plt.close(fig);saved.append(name)
def ci(ax,x,r,c,marker,label=None):
 y=100*r.estimate.to_numpy();ax.errorbar(x,y,yerr=np.maximum(0,np.array([y-100*r.ci_low,100*r.ci_high-y])),fmt=marker+'-',color=c,ms=3,lw=1.1,capsize=2,label=label)
# Recognition on matched CFP screened subset.
fig,axs=plt.subplots(1,2,figsize=(7.1,2.8),sharey=True)
for ax,b in zip(axs,[128,512]):
 for e,en in enumerate(encs):
  r=d.query('encoder==@en and bits==@b and metric=="recognition_tar" and target_far==.001').set_index('method').loc[methods];assert len(r)==3
  ci(ax,np.arange(3)+(e-.5)*.07,r,colors[e],markers[e],names[e])
 ax.set_xticks(range(3),ml);ax.set_title(f'{b} bits');ax.grid(axis='y',alpha=.2);ax.set_ylim(0,100)
axs[0].set_ylabel('Genuine TAR (%)');axs[1].legend(loc='lower right');fig.tight_layout();save(fig,'cfp_recognition_ci')
# LFW 128 recognition - seed SD is explicitly not CI.
l=pd.read_csv(D/'lfw/summary.csv');fig,ax=plt.subplots(figsize=(3.4,2.05))
for e,en in enumerate(encs):
 r=l.query('encoder==@en and evaluation=="matcher_only" and bits==128').set_index('method').loc[methods]
 ax.errorbar(np.arange(3)+(e-.5)*.08,100*r.tar,yerr=100*r.tar_seed_sd,fmt=markers[e]+'-',color=colors[e],label=names[e],capsize=3)
ax.set_xticks(range(3),['RO','SRHT','PCA-guided']);ax.set_ylim(75,100);ax.set_ylabel('LFW matcher-only TAR (%)');ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();save(fig,'lfw_recognition_sd')
# Paired encoder and exposure effects.
for source,name,kind in [('E04_paired_encoder_intervals.csv','paired_encoder_effects','enc'),('E04_paired_K10_minus_K1_intervals.csv','paired_exposure_effects','exp')]:
 a=pd.read_csv(D/source);fig,axs=plt.subplots(1,2,figsize=(7.1,2.4),sharey=True)
 for ax,b in zip(axs,[128,512]):
  for e in range(2):
   if kind=='enc':
    metric=['recognition_tar','fresh_asr'][e];r=a.query('bits==@b and metric==@metric and target_far==.001');r=r[r.K.eq(0 if e==0 else 10)];r=r[r.attack.eq('none' if e==0 else 'calibrated')];lab=['Genuine TAR','K=10 attack ASR'][e]
   else:
    en=encs[e];r=a.query('encoder==@en and bits==@b and metric=="fresh_asr" and attack=="calibrated" and target_far==.001');lab=names[e]
   r=r.set_index('method').loc[methods];assert len(r)==3
   y=np.arange(3)+(e-.5)*.16;x=100*r.difference.to_numpy();ax.errorbar(x,y,xerr=np.array([x-100*r.ci_low,100*r.ci_high-x]),fmt=markers[e],color=colors[e],capsize=3,label=lab)
  ax.axvline(0,color='black',ls=':',lw=.8);ax.set_yticks(range(3),ml);ax.set_title(f'{b} bits');ax.set_xlabel('Difference (percentage points)');ax.grid(axis='x',alpha=.2)
 axs[0].invert_yaxis();axs[1].legend(loc='lower right');fig.tight_layout();save(fig,name)
# Source vs cross-image cosine.
fig,axs=plt.subplots(2,3,figsize=(7.1,3.6),sharex=True,sharey=True)
for row,b in enumerate([128,512]):
 for j,method in enumerate(methods):
  ax=axs[row,j]
  for e,en in enumerate(encs):
   for metric,style in [('exposed_cosine','-'),('cross_cosine','--')]:
    r=d.query('encoder==@en and bits==@b and method==@method and metric==@metric and attack=="calibrated" and target_far==.001').sort_values('K');assert len(r)==4
    ax.plot(r.K,r.estimate,color=colors[e],ls=style,marker=markers[e],ms=2.5,label=names[e]+(' / source' if style=='-' else ' / cross-image'))
  ax.set_title(f'{ml[j]}, {b} bits');ax.set_xticks([1,2,5,10]);ax.set_ylim(0,1);ax.grid(alpha=.2)
  if j==0:ax.set_ylabel('Cosine similarity')
  if row==1:ax.set_xlabel('Exposures, K')
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False);fig.tight_layout(rect=(0,0,1,.88));save(fig,'reconstruction_quality')
# Learned linkage, no invented intervals.
a=pd.read_csv(D/'E02_linkage_summary.csv');fig,axs=plt.subplots(1,2,figsize=(7.1,2.8),sharey=True)
for ax,b in zip(axs,[128,512]):
 for e,en in enumerate(encs):
  r=a.query('encoder==@en and bits==@b and K==10').set_index('method').loc[methods];ax.plot(range(3),r.auc,markers[e]+'-',color=colors[e],label=names[e])
 ax.axhline(.5,color='black',ls=':',lw=.8);ax.set_xticks(range(3),ml);ax.set_ylim(.49,.59);ax.set_title(f'{b} bits');ax.grid(axis='y',alpha=.2)
axs[0].set_ylabel('Template-only linkage AUC');axs[1].legend();fig.tight_layout();save(fig,'linkage_comparison')
# Storage model, deterministic.
fig,ax=plt.subplots(figsize=(3.4,2.1));n=np.array([1000,10000,100000,1000000])
for b,label,c,mk in [(2048,'Raw 512-D float32','#444444','^'),(64,'512-bit reference','#bd4b12','s'),(16,'128-bit reference','#1764a0','o')]:ax.plot(n,n*b/2**20,marker=mk,label=label,color=c)
ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Number of stored references');ax.set_ylabel('Template payload (MiB)');ax.grid(alpha=.2,which='major');ax.legend();fig.tight_layout();save(fig,'storage_scaling')
# Regenerate the current monochrome methodology separately.
import runpy
runpy.run_path(str(P/'build_methodology_bw.py'),run_name='__main__')
(P/'EXTENDED_FIGURE_MANIFEST.txt').write_text('\n'.join(saved+['methodology'])+'\n')
print('Generated evidence plots and monochrome methodology')

runpy.run_path(str(P/'build_compact_comparisons.py'),run_name='__main__')
