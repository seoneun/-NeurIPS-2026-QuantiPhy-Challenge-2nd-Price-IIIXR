import sys, json, os
sys.path.insert(0,os.path.join(os.environ.get('QP_ROOT', '.'), 'analysis/v3/ag/rul193'))
os.environ['TL_SUB']='sub_xt_Nzz27'
import pandas as pd
pd.set_option('display.width',250)
import diag_rule as T
T.SUB='sub_xt_Nzz27'; T.OUT='ag_c89tlight3diag'
R=T.R
det=pd.read_csv(f'{R}/submissions/sub_xt_Nzz27_detail.csv',index_col=0)
P=T.parse('test')
for i in [1287,1300,1313,1325]:
    j=P[i]; print(i, det.loc[i,['video_id','question','cat','answer','answer_source','g','cs','mode']].to_dict()); print('  q=',j.get('q'), 'target=',j.get('target'))
df,out=T.run('test'); print(df.drop(columns=['q']).to_string())
t2=json.load(open(f'{R}/cache/tmp/ag_tlight2_patch_test.json')); print('rul436', {k:v['value'] for k,v in t2.items()})
for k,v in t2.items(): print(k, v['why'][:300])
raw=pd.read_csv(f'{R}/preds/test_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv',index_col=0)
print(raw.loc[[1287,1300,1313,1325],['meas_why','meas_mode','geo_value','vlm_value']].to_string())
