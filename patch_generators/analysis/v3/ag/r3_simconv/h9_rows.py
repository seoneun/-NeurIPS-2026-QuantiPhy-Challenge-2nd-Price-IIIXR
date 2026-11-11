"""H9 row table: questions of the X-named simulation families with scored flag (3010 mask), best21v value, source and
patch tags. Families: 0045-0050 (street), 0088-0091 (birds), 0121, 0253, and every clip whose question/prior says 'beach'."""
import json, glob, os, re, sys
import pandas as pd
R = '/data/02_Work/QuantiPhy-Challenge-2026'
A = pd.read_csv(f'{R}/analysis/v3/calib/audit_test.csv')
M = pd.read_csv(f'{R}/analysis/v3/calib/r3/inputs/scored_mask_3010.csv').set_index('id0')
b21 = pd.read_csv(f'{R}/submissions/sub_xt_best21v_id1.csv').parsed_value.values
tags = {}
for p in glob.glob(f'{R}/cache/tmp/ag_*_patch_test.json'):
    t = os.path.basename(p)[3:-16]
    try: js = json.load(open(p))
    except Exception: continue
    for k in js: tags.setdefault(int(k), []).append(t)
base = A.video.str.replace('_segmented', '', regex=False)
fam = {'street': [f'simulation_00{n}' for n in range(45, 51)], 'birds': [f'simulation_00{n}' for n in range(88, 92)],
       'e0121': ['simulation_0121'], 'e0253': ['simulation_0253']}
beach = sorted(set(A.video[A.question.str.contains('beach', case=False) | A.prior.str.contains('beach', case=False, na=False)]))
fam['beach'] = sorted(set(b.replace('_segmented', '') for b in beach))
for f, vids in fam.items():
    print(f'\n##### {f}: {vids}')
    for v in vids:
        print('  prior', v, ':', A.prior[base == v].iloc[0] if (base == v).any() else 'NA', '| depth:', str(A.depth[base == v].iloc[0])[:150] if (base == v).any() else '')
    for _, r in A[base.isin(vids)].iterrows():
        i = int(r.id0)
        print(i, r.video, r['cat'], 'sc' if M.loc[i, 'scored'] else 'NS', r.kind, '|', str(r.question)[:95], '| b21=%.4g' % b21[i],
              '|', r.source, '|', ','.join(sorted(tags.get(i, []))))
