import pandas as pd, json, re
pat=re.compile(r'parking (space|spot|bay)|box formed by the white lines|marked .* on the (ground|floor)|painted (box|rectangle)',re.I)
for sp in ['test','val']:
    raw=pd.read_csv(f'preds/{sp}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv')
    if sp=='test': print(raw.columns.tolist())
    qc=[c for c in raw.columns if 'question' in c.lower()][0]
    m=raw[qc].astype(str).str.contains(pat)
    print(sp,'noun gate hits',m.sum())
    for i in raw.index[m]: print(sp,i,raw.loc[i,'video_path'],raw.loc[i,qc][:150])
    vids=set(raw.loc[m,'video_path'])
    sub=raw[raw.video_path.isin(vids)]
    cols=[c for c in ['video_path','ground_truth_prior','meas_why','meas_mode','meas_mpp','geo_value','vlm_value'] if c in raw.columns]
    pd.set_option('display.width',250); pd.set_option('display.max_colwidth',80)
    print(sub[cols+[qc]].to_string())
