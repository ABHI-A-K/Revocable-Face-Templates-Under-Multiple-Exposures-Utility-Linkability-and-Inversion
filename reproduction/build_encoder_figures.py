from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
P=Path(__file__).resolve().parent;D=P/'data/encoder_extension';OUT=P/'figures';d=pd.read_csv(D/'E04_identity_bootstrap_intervals.csv');encs=['MobileFaceNet','ResNet50_ArcFace'];labs=['MobileFaceNet','ResNet50 ArcFace'];methods=['RO','SRHT','PCA_GUIDED_RO'];colors=['#1764a0','#b44710']
plt.rcParams.update({'font.size':9,'axes.titlesize':10,'pdf.fonttype':42})
fig,axs=plt.subplots(2,3,figsize=(7.16,3.7),sharex=True,sharey=True)
for row,bits in enumerate([128,512]):
 for col,method in enumerate(methods):
  ax=axs[row,col]
  for e,enc in enumerate(encs):
   a=d.query("encoder==@enc and method==@method and bits==@bits and attack=='calibrated' and metric=='fresh_asr' and target_far==.001").sort_values('K');assert a.K.tolist()==[1,2,5,10]
   y=100*a.estimate.to_numpy();ax.errorbar(a.K,y,yerr=np.array([y-100*a.ci_low,100*a.ci_high-y]),color=colors[e],marker=['o','s'][e],linestyle=['-','--'][e],label=labs[e],capsize=2,ms=3,lw=1.3)
  ax.set_title(f"{method.replace('PCA_GUIDED_RO','PCA-guided RO')}, {bits} bits");ax.set_xticks([1,2,5,10]);ax.set_ylim(0,100);ax.grid(alpha=.2)
  if row==1:ax.set_xlabel('Exposed templates, K')
  if col==0:ax.set_ylabel('Fresh-key ASR (%)')
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False);fig.tight_layout(rect=(0,0,1,.93))
for ext in ['pdf','png']:fig.savefig(OUT/f'encoder_exposure.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
for bits in [128,512]:
 fig,axs=plt.subplots(1,3,figsize=(7.16,2.5));mc=['#1764a0','#b44710','#33734c']
 for j,metric in enumerate(['recognition_tar','recognition_far','fresh_asr']):
  for k,method in enumerate(methods):
   for e,enc in enumerate(encs):
    a=d.query('encoder==@enc and bits==@bits and method==@method and metric==@metric');a=a.query("K==10 and attack=='calibrated'") if metric=='fresh_asr' else a.query("K==0 and attack=='none'");a=a.sort_values('target_far');assert len(a)==3 and np.allclose(a.target_far,[.0001,.001,.01])
    axs[j].errorbar(a.target_far,100*a.estimate,yerr=np.maximum(0,np.array([100*(a.estimate-a.ci_low),100*(a.ci_high-a.estimate)])),color=mc[k],linestyle=['-','--'][e],marker=['o','s'][e],ms=3,capsize=2,lw=1)
  axs[j].set_xscale('log');axs[j].set_xlabel('Validation FAR target');axs[j].set_ylabel(['Genuine TAR (%)','Observed test FAR (%)','Fresh-key ASR, K=10 (%)'][j]);axs[j].grid(alpha=.2)
 handles=[Line2D([0],[0],color=mc[k],label=m.replace('PCA_GUIDED_RO','PCA-guided RO')) for k,m in enumerate(methods)]+[Line2D([0],[0],color='black',linestyle=['-','--'][e],marker=['o','s'][e],label=labs[e]) for e in range(2)]
 fig.legend(handles=handles,ncol=3,loc='upper center',frameon=False);fig.tight_layout(rect=(0,0,1,.82))
 for ext in ['pdf','png']:fig.savefig(OUT/f'encoder_threshold_targets_{bits}.{ext}',dpi=180,bbox_inches='tight')
 plt.close(fig)
c=pd.read_csv(D/'E04_threshold_sweep.csv')
for bits in [128,512]:
 fig,axs=plt.subplots(1,3,figsize=(10.5,3.5),sharey=True)
 for j,method in enumerate(methods):
  for e,enc in enumerate(encs):
   a=c.query("encoder==@enc and bits==@bits and method==@method and K==10 and attack=='calibrated' and matcher=='protected'").sort_values('threshold');assert len(a)>100 and not a.threshold.duplicated().any()
   axs[j].plot(a.threshold,100*a.tar,color=colors[e],label=labs[e]+' TAR');axs[j].plot(a.threshold,100*a.asr,color=colors[e],ls='--',label=labs[e]+' ASR')
  axs[j].set_title(method.replace('PCA_GUIDED_RO','PCA-guided RO'));axs[j].set_xlim(.5,1.01);axs[j].set_ylim(0,100);axs[j].set_xlabel('Agreement threshold');axs[j].grid(alpha=.2)
 axs[0].set_ylabel('Acceptance (%)');fig.legend(*axs[0].get_legend_handles_labels(),ncol=4,loc='upper center',frameon=False,fontsize=8);fig.tight_layout(rect=(0,0,1,.82))
 for ext in ['pdf','png']:fig.savefig(OUT/f'encoder_discrete_sweep_{bits}.{ext}',dpi=180,bbox_inches='tight')
 plt.close(fig)
print('PASS corrected encoder-specific figures')
