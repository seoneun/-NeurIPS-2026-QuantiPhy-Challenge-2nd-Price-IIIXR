import json,sys,pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',120)
det=pd.read_csv(f'{R}/submissions/sub_xt_Nzz24_detail.csv',index_col=0)
raw=pd.read_csv(f'{R}/preds/test_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv',index_col=0)
print(raw.columns.tolist())
for i in [495,498,518,526,506,515,524]:
    r=det.loc[i]; w=raw.loc[i]
    print(i,r.video_id,r['cat'],r.answer,r.answer_source,r.g,r.cs,'|',w.meas_why,w.meas_mode,w.meas_mpp,w.meas_tracks,w.vlm_value)
    print('   Q:',w.question); print('   prior:',w.ground_truth_prior)
vids=det.loc[[495,498,518,526],'video_id'].unique()
s=det[det.video_id.isin(vids)]
for i,r in s.iterrows():
    print(i,r.video_id,r['cat'],round(r.answer,5),r.answer_source,r.g,r.cs,raw.loc[i,'meas_why'],raw.loc[i,'meas_mode'],raw.loc[i,'meas_mpp'],raw.loc[i,'meas_tracks'],'|',raw.loc[i,'question'][:110])
