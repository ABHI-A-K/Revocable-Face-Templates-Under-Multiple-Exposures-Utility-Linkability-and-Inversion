"""Reproduce the monochrome methodology flowchart as a vector PDF."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, Polygon
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(7.16,5.65))
ax.set(xlim=(0,10),ylim=(0,9)); ax.axis('off')
def node(x,y,w,h,text,shape='process'):
 if shape=='terminal': patch=Ellipse((x,y),w,h)
 elif shape=='decision': patch=Polygon([(x-w/2,y),(x,y+h/2),(x+w/2,y),(x,y-h/2)],closed=True)
 elif shape=='io': patch=Polygon([(x-w/2+.18,y+h/2),(x+w/2,y+h/2),(x+w/2-.18,y-h/2),(x-w/2,y-h/2)],closed=True)
 else: patch=Rectangle((x-w/2,y-h/2),w,h)
 patch.set(facecolor='white',edgecolor='black',linewidth=.85);ax.add_patch(patch)
 ax.text(x,y,text,ha='center',va='center',fontsize=8.7,linespacing=1.25)
def arrow(points,label=None,loc=None):
 if len(points)>2: ax.plot(*zip(*points[:-1]),color='black',lw=.85)
 ax.annotate('',xy=points[-1],xytext=points[-2],arrowprops=dict(arrowstyle='-|>',lw=.85,color='black',shrinkA=0,shrinkB=1))
 if label: ax.text(*loc,label,fontsize=8,ha='center',va='bottom',bbox=dict(facecolor='white',edgecolor='none',pad=1))
node(5,8.6,5.8,.48,'Study repeated exposure and edge storage','terminal')
node(5,7.8,5,.65,'LFW images / CFP aligned crops','io')
node(5,6.76,5,.75,'Extract normalized 512-D embeddings\nMobileFaceNet; ResNet50 extension')
node(5,5.55,3.25,.95,'Evaluation\nprotocol?','decision')
arrow([(5,8.36),(5,8.13)]);arrow([(5,7.47),(5,7.14)]);arrow([(5,6.38),(5,6.03)])
node(2.35,4.05,4.05,1.08,'Official pairs and fold thresholds\nLFW encoder utility; baseline replay\nVary bit length; count payload bytes')
node(7.65,4.05,4.05,1.08,'Identity-disjoint attack studies\nRO / SRHT / PCA-guided RO\nCFP: two encoders; LFW: baseline')
arrow([(3.375,5.55),(2.35,5.55),(2.35,4.59)],'Official pairs',(2.35,5.63))
arrow([(6.625,5.55),(7.65,5.55),(7.65,4.59)],'Custom attacks',(7.85,5.63))
node(7.65,2.52,4.05,1.18,'1, 2, 5, or 10 old exposures\nUtility, linkage, and inversion\nFresh operator: verifier only')
arrow([(7.65,3.51),(7.65,3.12)])
node(5,1.16,5.75,.74,'Report template bytes / TAR / FAR / AUC / ASR\nRelate compactness to utility and exposure risk','io')
arrow([(2.35,3.51),(2.35,1.88),(3.6,1.88),(3.6,1.54)])
arrow([(7.65,1.93),(7.65,1.74),(6.4,1.74),(6.4,1.54)])
node(5,.27,6.3,.5,'Assess storage--recognition--security trade-off','terminal')
arrow([(5,.79),(5,.53)])
fig.subplots_adjust(left=.015,right=.985,bottom=.02,top=.985)
fig.savefig(Path(__file__).resolve().parent/'figures/methodology.pdf')
