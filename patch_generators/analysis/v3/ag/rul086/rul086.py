"""rul086: coverage gap of the scene-layout noun list (geo/measure.py SCENE has court/field/lane/track/pool but not
'pitch'), so 'width of football pitch' was measured as an OBJECT from a frame-filling, border-cut detector box.
Rule: a size row whose parsed target (or question) names a sports pitch -- '(football|soccer|rugby|cricket|hockey|
sports|playing|grass) pitch(es)', or a target whose head noun is 'pitch(es)' with dim length/width/area -- is routed like
every other scene_target row: CaW alone, times the pipeline's existing CaW-alone shrink for scene_target rows of that
category (S* = 1.0, D* = 0.81, read from the current build's own caw_nogeo/scene_target rows, not hard-coded per row).
Speed / accel questions ('speed of the pitch' in baseball) never fire (kind must be size). Label-free; no literal ids.
Check (internet_0031, label-free, painted standard markings on frame 0): touchlines y=42/786 (744 px); penalty area 467.5 px
= 40.32 m and goal area 212 px = 18.32 m both give 0.0863 m/px -> pitch width 64.3 m; geo's 31.06 was the clipped box's
horizontal extent (392 px of a 480 px frame), i.e. the wrong axis of a frame-cut box. CaW says 65.
usage: tools/run_limited.sh --io python analysis/v3/ag/rul086/rul086.py [sub_name] [write]"""
import sys, json, re, numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sub = next((a for a in sys.argv[1:] if a.startswith('sub_')), 'sub_xt_Nzz5')
WRITE = 'write' in sys.argv[1:]
PITCH = re.compile(r"\b(football|soccer|rugby|cricket|hockey|sports?|playing|grass)\s+pitch(es)?\b|\bpitch(es)?\s+(width|length)\b")
HEAD = re.compile(r"\bpitch(es)?$")
for split in ('test', 'val'):
    D = pd.read_csv(f"{R}/submissions/{sub}_{'detail' if split == 'test' else 'valdetail'}.csv", index_col=0)
    raw = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", index_col=0)
    P = {j['id']: j for j in map(json.loads, open(f"{R}/geo/out/parse_{split}_v2.jsonl"))}
    # the pipeline's own CaW-alone shrink on scene_target rows, per category letter (S / D)
    st = [(str(D['cat'][i])[0], D.answer[i] / D.cs[i]) for i in D.index
          if str(raw.meas_why.get(i)) == 'scene_target' and str(D.answer_source[i]) == 'caw_nogeo' and D.cs[i] > 0]
    K = {c: float(np.median([r for cc, r in st if cc == c])) for c in set(c for c, _ in st)}
    out = {}
    for i in D.index:
        p = P.get(i, {}); q = p.get('q') or {}
        if q.get('kind') != 'size' or str(raw.meas_why.get(i)) == 'scene_target':
            continue
        tg = ((p.get('target') or {}).get('raw') or (p.get('target') or {}).get('phrase') or '').lower().strip()
        ql = str(raw.question[i]).lower()
        if not (PITCH.search(tg) or PITCH.search(ql) or (HEAD.search(tg) and q.get('dim') in ('width', 'length', 'area', 'side', None))):
            continue
        cs = D.cs[i]
        if not (cs == cs and cs > 0):
            continue
        k = K.get(str(D['cat'][i])[0], 1.0)
        out[str(i)] = {'value': float(cs * k), 'why': f"rul086: sports pitch is scene layout (noun missing from SCENE); "
                       f"scene_target route = CaW {cs:.4g} x {k:.2f}; was {D.answer[i]:.4g} ({D.answer_source[i]}, geo {D.g[i]:.4g})"}
        print(split, i, D.video_id[i], D['cat'][i], ql[:60], '|', D.answer[i], '->', out[str(i)]['value'])
    print(split, 'scene shrink by cat', K, 'rows changed', len(out))
    if WRITE:
        json.dump(out, open(f"{R}/cache/tmp/ag_c49pitch_patch_{split}.json", 'w'), indent=1)
