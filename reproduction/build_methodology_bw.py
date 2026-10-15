from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch
P=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(3.45,2.85))
ax.set_xlim(0,3.45); ax.set_ylim(0,3.5); ax.axis('off')
left,width,height=.18,3.09,.48
labels=['Aligned face images','Frozen feature extraction\nMobileFaceNet / ResNet50','Keyed binary templates\nRO / SRHT / PCA-guided RO','Evaluate recognition, security\nand template storage','Joint interpretation\nUtility and exposure risk']
for i,label in enumerate(labels):
    y=2.96-i*.69
    if i in (0,4):
        patch=Polygon([(left+.10,y),(left+width,y),(left+width-.10,y+height),(left,y+height)],closed=True,ec='black',fc='white',lw=.8)
    else:
        patch=Rectangle((left,y),width,height,ec='black',fc='white',lw=.8)
    ax.add_patch(patch)
    ax.text(1.725,y+height/2,label,ha='center',va='center',fontsize=8)
    if i<4:
        ax.add_patch(FancyArrowPatch((1.725,y),(1.725,y-.21),arrowstyle='-|>',mutation_scale=8,lw=.8,color='black',shrinkA=1,shrinkB=1))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
for ext in ('pdf','png'):
    fig.savefig(P/f'figures/methodology.{ext}',dpi=220,bbox_inches='tight',pad_inches=.015)
plt.close(fig)
