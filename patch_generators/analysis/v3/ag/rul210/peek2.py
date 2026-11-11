import pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
for s in ['test','val']:
    raw=pd.read_csv(R+'/preds/%s_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv'%s)
    m=raw.video_id.str.startswith('captured')&(raw.meas_kind=='speed')
    print(s, raw[m].meas_tracks.value_counts().to_dict(), raw[m].merge_src.value_counts().to_dict(), raw[m].meas_mode.astype(str).str[:12].value_counts().to_dict())
