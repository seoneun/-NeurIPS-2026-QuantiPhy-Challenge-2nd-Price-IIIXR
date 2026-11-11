"""rul281 builder (copied from analysis/v3/ag/tri87/filter_r100.py, rul280 part only; rul441 part removed so
that tag's files are never touched).  Label-free.  Inputs: parse_{split}_v2, cache/tmp/ag_echo2_patch_{split}.json
(rul280 values + why strings), geo column g of submissions/sub_xt_Nz_{detail,valdetail}.csv.
rul281 = rul280 minus
  (A) same-object other-dimension size rows: q.kind==size, prior kind size, target phrase == prior_obj phrase,
      prior dim != q dim, q dim not in {thickness, depth} -> a CaW value repeating another dimension of the same
      object is a legitimate aspect-ratio-1 guess, not an echo.
  (B) speed spikes: q.kind==speed and the row's g > 1.5 x max g over the same video's other speed rows on the same
      target phrase; plus any 'segmented copy follows its plain recording' row whose plain base row
      (split_variant base, same question) was dropped.
writes cache/tmp/ag_echo2r100_patch_{test,val}.json"""
import json, sys
import numpy as np, pandas as pd
TAG = 'rul281'
SRC = 'rul280'
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + '/analysis/v3')
from variant_keys import split_variant
SPIKE = 1.5
def load_parse(split):
    P = {}
    for l in open(f'{R}/geo/out/parse_{split}_v2.jsonl'):
        j = json.loads(l); P[j['id']] = j
    return P
def ph(e): return (e or {}).get('phrase') if isinstance(e, dict) else e
for split in ('test', 'val'):
    P = load_parse(split)
    det = pd.read_csv(f"{R}/submissions/sub_xt_Nz_{'detail' if split == 'test' else 'valdetail'}.csv", index_col=0)
    E = json.load(open(f'{R}/cache/tmp/ag_{SRC}_patch_{split}.json')); out = {}; drop = {}
    # (A) same-object other-dimension size rows
    for k, v in E.items():
        i = int(k); p = P[i]; q = p['q']; pr = (p.get('priors') or [{}])[0]
        if q.get('kind') == 'size' and pr.get('kind') == 'size' and ph(p.get('target')) == ph(p.get('prior_obj')) \
           and pr.get('dim') and q.get('dim') and pr['dim'] != q['dim'] and q['dim'] not in ('thickness', 'depth'):
            drop[i] = 'A same-object other-dim'; continue
        out[k] = v
    # (B) speed spikes on plain rows
    for k in list(out):
        i = int(k); p = P[i]; q = p['q']
        if q.get('kind') != 'speed' or 'follows its plain' in out[k]['why']: continue
        sib = [j for j, pj in P.items() if j != i and pj['video_id'] == p['video_id'] and pj['q'].get('kind') == 'speed'
               and ph(pj.get('target')) == ph(p.get('target'))]
        gs = [det.loc[j, 'g'] for j in sib if j in det.index and np.isfinite(det.loc[j, 'g'])]
        g = det.loc[i, 'g']
        if gs and g > SPIKE * max(gs): drop[i] = f'B speed spike g={g:.3g} max_sib={max(gs):.3g}'; del out[k]
    # segmented followers of dropped plain rows
    dropped_plain = {(P[i]['video_id'], P[i]['question']) for i in drop}
    for k in list(out):
        i = int(k)
        if 'follows its plain' in out[k]['why']:
            base = split_variant(P[i]['video_id'])[0]
            if any(v == base and qq == P[i]['question'] for v, qq in dropped_plain):
                drop[i] = 'B follower of dropped plain row'; del out[k]
    json.dump(out, open(f'{R}/cache/tmp/ag_{TAG}_patch_{split}.json', 'w'), indent=1)
    print(split, SRC, len(E), '->', len(out), 'dropped', dict(sorted(drop.items())))
