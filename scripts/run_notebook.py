import argparse
from pathlib import Path
import nbformat
from nbclient import NotebookClient
p=argparse.ArgumentParser();p.add_argument('notebook');p.add_argument('--output',required=True);p.add_argument('--timeout',type=int,default=7200);a=p.parse_args()
n=nbformat.read(a.notebook,as_version=4)
NotebookClient(n,timeout=a.timeout,kernel_name='python3',allow_errors=False,resources={'metadata':{'path':str(Path.cwd())}}).execute()
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);nbformat.write(n,out)
