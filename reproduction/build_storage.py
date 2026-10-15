"""Calculate template-payload storage; this is not a hardware benchmark."""
from pathlib import Path
import csv
out=Path(__file__).resolve().parent/'data'/'template_storage_accounting.csv'
raw=512*4
rows=[]
for representation,bits in [('Raw 512-D float32',raw*8),('Protected 128-bit',128),('Protected 512-bit',512)]:
 size=bits//8
 rows.append({'representation':representation,'payload_bytes':size,'reduction_factor':raw/size,'saved_percent':100*(1-size/raw),'scope':'template payload only; excludes models, operators, keys, metadata and working memory'})
with out.open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
