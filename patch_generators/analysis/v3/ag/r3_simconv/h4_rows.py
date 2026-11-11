"""H4 (ffbd) row table: every question of the drop clips simulation_0196-0211 with scored flag (3010 mask), best21v value,
and the patch tags (cache/tmp/ag_*_patch_test.json) that touch the row. Also lists kinoffset rows and their videos."""
import json, glob, os, re
import pandas as pd
R = '/data/02_Work/QuantiPhy-Challenge-2026'
A = pd.read_csv(f'{R}/analysis/v3/calib/audit_test.csv')
M = pd.read_csv(f'{R}/analysis/v3/calib/r3/inputs/scored_mask_3010.csv').set_index('id0')
b21 = pd.read_csv(f'{R}/submissions/sub_xt_best21v_id1.csv').parsed_value.values
T = pd.read_parquet(f'{R}/data/test/test_dataset.parquet')
tags = {}
for p in glob.glob(f'{R}/cache/tmp/ag_*_patch_test.json'):
    t = os.path.basename(p)[3:-16]
    try: js = json.load(open(p))
    except Exception: continue
    for k in js: tags.setdefault(int(k), []).append(t)
vids = [f'simulation_0{n}' for n in range(196, 212)]
for _, r in A[A.video.isin(vids)].iterrows():
    i = int(r.id0)
    print(i, r.video, r['cat'], 'sc' if M.loc[i, 'scored'] else 'NS', 'fps=%g' % T.fps.iloc[i], r.kind, '|', str(r.question)[:100],
          '| prior:', str(r.prior)[:80], '| b21=%.4g' % b21[i], '|', r.source, '|', ','.join(sorted(tags.get(i, []))))
ko = json.load(open(f'{R}/cache/tmp/ag_kinoffset_patch_test.json'))
kv = A.set_index('id0').loc[[int(k) for k in ko]]
print('\nkinoffset rows', len(ko), 'scored(3010):', int(M.loc[[int(k) for k in ko], 'scored'].sum()))
print(kv.groupby(['video']).size().to_string())
print('why example:', list(ko.items())[0])
