import os, sys, pandas as pd
R = os.getcwd()
OUT = 'output/work/v15r03/meas'
os.makedirs(OUT, exist_ok=True)
split, tracks, FIXS = sys.argv[1], sys.argv[2], sys.argv[3]
ENV = dict(QP_REJ_LO='0.2', QP_REJ_HI='5', QP_PRIOR_DRIFT='1.5', QP_PRIOR_ACCGUARD='3', QP_SCENE_HEAD='1', QP_TSHIFT='1',
           QP_DEPTH_DEDUP='1', QP_SCENE_SA2VA='1', QP_SECOND_FIX='1', QP_SAMECLASS='1', QP_SCENE_RULER='1', QP_FF_ROT='1',
           QP_SIDE_FIX='1', QP_FLOOR='1', QP_STATIC_CAP='25', QP_FF_SUB='1', QP_FF_SUB_SRC='simulation', QP_PARSE2='1',
           QP_NEAREST='1', QP_BELT='1', QP_FPERSON='1', QP_FPERSON_MINN='2', CUDA_VISIBLE_DEVICES='', QP_ROOT=R)
os.environ.update(ENV)
sys.path.insert(0, R + '/geo')
import measure_v3 as M
_to_csv = pd.DataFrame.to_csv
def to_csv(self, path=None, *a, **k):
    if isinstance(path, str) and (path.startswith(R + '/geo/out/') or path.startswith('geo/out/')):
        path = OUT + '/' + os.path.basename(path)
    return _to_csv(self, path, *a, **k)
pd.DataFrame.to_csv = to_csv
TAG = f'_v15_{tracks}_{FIXS}'
sys.argv = ['measure_v3.py', '--split', split, '--fix', FIXS, '--tag', TAG, 
            '--moge', '--moge_opts', 'fbrej', '--tracks', tracks]
M.main()
