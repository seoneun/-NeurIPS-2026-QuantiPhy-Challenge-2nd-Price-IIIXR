import pandas as pd, numpy as np
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
prod={'m':'v3m_rrj5sQ','v':'v3v_rntrj5sQ','mu':'v3m_rrj5sQu','vu':'v3v_rntrj5sQu'}
for V,t in prod.items():
    P=pd.read_csv(f'{R}/geo/out/meas_test_v2_{t}.csv').set_index('id')
    A=pd.read_csv(f'{R}/cache/tmp/ag_p1pf_meas_test_{V}_ref.csv').set_index('id')
    B=pd.read_csv(f'{R}/cache/tmp/ag_p1pf_meas_test_{V}_new.csv').set_index('id')
    P=P.reindex(A.index)
    same=np.isclose(P.value.astype(float),A.value.astype(float),rtol=1e-6,equal_nan=True)
    print(V,'rows',len(A),'ref==prod value',int(same.sum()),'mode same',int((P['mode'].astype(str)==A['mode'].astype(str)).sum()))
    if (~same).any(): print(pd.DataFrame({'prod':P.value,'ref':A.value,'pm':P['mode'],'rm':A['mode']})[~same].head(10))
    ch=~np.isclose(A.value.astype(float),B.value.astype(float),rtol=1e-6,equal_nan=True)
    D=pd.DataFrame({'vid':A.video_id,'kind':A.kind,'ref':A.value,'new':B.value,'ratio':B.value/A.value,'rmode':A['mode'],'nmode':B['mode'],'why':A.why})
    pd.set_option('display.width',250)
    print(D[ch].to_string())
