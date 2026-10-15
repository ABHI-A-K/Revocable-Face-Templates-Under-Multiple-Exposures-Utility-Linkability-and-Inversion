import sys,hashlib
from pathlib import Path
for n in sys.argv[1:]:
 p=Path(n);print(hashlib.sha256(p.read_bytes()).hexdigest(),p.name)
