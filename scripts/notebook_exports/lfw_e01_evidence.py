# Original code export; run with IPython after configuring inputs.
# Not an independently rerun or refactored implementation.
# Cell 1
from pathlib import Path
import sys, subprocess
root=Path.cwd()
if not (root/"scripts").exists() and (root.parent/"scripts").exists(): root=root.parent
assert (root/"scripts").exists()
subprocess.run([sys.executable,str(root/"scripts/verify_lfw_baseline.py")],cwd=root,check=True)
