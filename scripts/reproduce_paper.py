from pathlib import Path
import subprocess,sys
P=Path(__file__).resolve().parents[1]/'reproduction'
for name in ['editorial_metrics.py','validate_encoder_evidence.py','build_figures.py','build_storage.py','build_encoder_figures.py','build_extended_figures.py']:
 print('Running',name,flush=True);subprocess.run([sys.executable,str(P/name)],cwd=P,check=True)
print('PASS: manuscript numerical checks and figure regeneration')
