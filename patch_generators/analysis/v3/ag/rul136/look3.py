import pandas as pd, re, glob
for sp in ['test','val']:
    raw=pd.read_csv(f'preds/{sp}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv')
    m=raw.ground_truth_prior.astype(str).str.contains('parking|white line|robot',case=False)|raw.question.astype(str).str.contains('parking|white line',case=False)
    print(raw.loc[m,['video_id','ground_truth_prior','question']].to_string()[:6000])
