import os
import json,glob
for p in sorted(glob.glob(os.path.join(os.environ.get('QP_ROOT', '.'), 'cache/tmp/ag_*_patch_test.json'))):
    try: J=json.load(open(p))
    except: continue
    for k in ['2473','2479','2485','2508','2512','2516']:
        if k in J: print(p.split('/')[-1],k,J[k])
