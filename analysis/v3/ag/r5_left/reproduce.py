import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import compute
for k, v in compute().items():
    print(k, v)
