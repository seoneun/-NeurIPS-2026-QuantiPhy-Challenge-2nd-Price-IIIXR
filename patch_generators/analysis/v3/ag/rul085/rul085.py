"""rul085: the static-track gate (meas_why 'target_static_track(Npx<T)') uses a robust motion range, so a BRIEF
excursion (a toast popping up for ~6 of 60 frames) is called static and a 'peak displacement' question falls to CaW,
which echoes the prior number. Rule: for kind=disp questions asking a peak/maximum vertical displacement whose target
track was gated static, read the excursion of the target's leading (top) edge from its own track: rest = median top
edge, peak = min top edge, kept only if the excursion is >= 3x the gate threshold and lasts >= 3 consecutive frames.
Ruler = the prior object's own track extent in the stated dimension (height -> box height). Label-free, no literal ids.
usage: python rul085.py <sub_name> [write]"""
import sys, json, re, os, numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sub = sys.argv[1] if len(sys.argv) > 1 else 'sub_xt_Nzz5'
WRITE = len(sys.argv) > 2 and sys.argv[2] == 'write'
def track(split, tv, vid, phrase):
    d = {'v1': f'tracks_{split}', 'v2': f'tracks_{split}_v2'}.get(tv, f'tracks_{split}_{tv}')
    f = f"{R}/geo/out/{d}/{vid}.json"
    if not os.path.exists(f):
        return None
    j = json.load(open(f))
    q = [q for q in j.get('queries', []) if q.get('phrase') == phrase]
    if not q:
        return None
    o = q[0]['frames'] if 'frames' in q[0] else j.get('objs', {}).get(str(q[0].get('obj')))
    if not o:
        return None
    A = np.array([tuple(x['st']['box']) for x in o if x and x.get('st')], float)
    return A if len(A) >= 10 else None
for split in ('test', 'val'):
    D = pd.read_csv(f"{R}/submissions/{sub}_{'detail' if split == 'test' else 'valdetail'}.csv").set_index('Unnamed: 0')
    raw = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv")
    P = {}
    for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        j = json.loads(l); P[j['id']] = j
    out = {}
    assert len(raw) == len(D)
    for k, i in enumerate(D.index):
        r = raw.iloc[k]; p = P.get(i, {}); q = p.get('q') or {}
        m = re.match(r'target_static_track\((\d+)px<(\d+)\)', str(r.meas_why))
        if not m or q.get('kind') != 'disp':
            continue
        ql = str(r.question).lower()
        if not (re.search(r'\b(peak|maximum|max|highest)\b', ql) and ('vertical' in ql or q.get('vertical'))):
            continue
        thr = float(m.group(2))
        pri = (p.get('priors') or [{}])[0]
        if pri.get('kind') != 'size' or pri.get('dim') not in ('height', 'length', 'width', 'diameter'):
            continue
        T = track(split, str(r.meas_tracks), r.video_id, (p.get('target') or {}).get('phrase'))
        Pt = track(split, str(r.meas_tracks), r.video_id, (p.get('prior_obj') or {}).get('phrase'))
        if T is None or Pt is None:
            continue
        top = T[:, 1]; rest = np.median(top); exc = rest - top
        if exc.max() < 3 * thr:
            continue
        big = exc > 0.5 * exc.max(); run = best = 0
        for b in big:
            run = run + 1 if b else 0; best = max(best, run)
        if best < 3:
            continue
        ext = np.median(Pt[:, 3] - Pt[:, 1]) if pri['dim'] == 'height' else np.median(Pt[:, 2] - Pt[:, 0])
        val = float(exc.max() * pri['value_si'] / ext)
        f = 1.0
        u = str(q.get('unit') or 'm').lower()
        if u in ('cm',): f = 100.0
        elif u in ('mm',): f = 1000.0
        out[str(i)] = {'value': val * f, 'why': f"rul085: static gate ({m.group(0)}) missed a brief excursion; top-edge "
                       f"peak {exc.max():.0f}px over {best} frames x prior ruler {pri['value_si']}/{ext:.0f}px"}
        print(split, i, r.video_id, 'ans', D.answer[i], '->', val * f, out[str(i)]['why'])
    print(split, 'rows changed', len(out))
    if WRITE:
        json.dump(out, open(f"{R}/cache/tmp/ag_c49peakx_patch_{split}.json", 'w'), indent=1)
