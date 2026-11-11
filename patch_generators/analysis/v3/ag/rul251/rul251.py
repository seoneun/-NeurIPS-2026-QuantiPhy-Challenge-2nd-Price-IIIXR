"""rul251: planar simulation SIZE rows (S2/D2, length/height, unit m) whose geo was refused only for a low target
detector score (meas_why target_low_det*) and answered caw_nogeo. Exactly one prior (speed or size). Target and prior objects
from the track json's queries[].phrase == parse target / prior_obj phrase (tracks dir from the raw's own meas_tracks).
Different objs, each with states in >= 80 % of fidx. Speed prior: m/px = v / net centre px speed (first-to-last / dt, json fps);
target net px speed >= 0.3x prior's (moving target only). Size prior: m/px = value / median prior extent along the prior dim.
Value = median target box extent over non-border states, along the target's net motion axis (length) or vertical (height).
Gate 0.25 <= value/CaW <= 10. Fusion: within 2x of CaW -> geomean (agree), else the 2d_ratio value (strong mode).
No ids, no labels in the logic (labels joined at the end for evaluation only).
usage: tools/run_limited.sh --io python analysis/v3/ag/rul251/rul251.py [SUB] [write]"""
import sys, json, os
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + '/cache/tmp'); sys.path.insert(0, R + '/caw')
import analyze_caw as A
from lab_rows import labelled
pd.set_option('display.width', 400); pd.set_option('display.max_colwidth', 60); pd.set_option('display.max_rows', 3000)
TAG = 'rul251'
args = [a for a in sys.argv[1:] if a != 'write']
SUB = args[0] if args else 'sub_xt_Nzz40'
RAWN = 'geo_v4ntrj5sQ2srngsr_d_vw'
WRITE = 'write' in sys.argv[1:]


def states(tr, o):
    s = tr['objs'].get(str(o)) or []
    return [(k, x['st']) for k, x in enumerate(s) if x and x.get('st')]


def findobj(tr, phrase):
    ph = str(phrase or '').strip().lower()
    for q in tr.get('queries') or []:
        if str(q.get('phrase', '')).strip().lower() == ph and q.get('obj') is not None:
            return q['obj']
    return None


def netspeed(tr, st):
    fidx, fps = tr['fidx'], float(tr['fps'])
    (k0, a), (k1, b) = st[0], st[-1]
    dt = (fidx[k1] - fidx[k0]) / fps
    dx, dy = b['c'][0] - a['c'][0], b['c'][1] - a['c'][1]
    return (np.hypot(dx, dy) / dt if dt > 0 else np.nan), dx, dy


def ext(st, axis):  # axis 0 = width, 1 = height
    return np.array([x['box'][2] - x['box'][0] if axis == 0 else x['box'][3] - x['box'][1] for _, x in st], float)


out = {'test': {}, 'val': {}}
diag = []
for sp, suf in (('test', 'detail'), ('val', 'valdetail')):
    raw = pd.read_csv(f"{R}/preds/{sp}_{RAWN}_raw.csv", index_col=0)
    det = pd.read_csv(f"{R}/submissions/{SUB}_{suf}.csv", index_col=0)
    assert (raw.video_id.astype(str).values == det.video_id.reindex(raw.index).astype(str).values).all()
    PP = {p['id']: p for p in map(json.loads, open(f"{R}/geo/out/parse_{sp}_v2.jsonl"))}
    for i in raw.index:
        p, q = PP[i], PP[i]['q']
        vid = str(raw.video_id[i])
        if not (det['cat'][i] in ('S2', 'D2') and vid.startswith('simulation') and q.get('kind') == 'size'
                and q.get('dim') in ('length', 'height') and q.get('unit') == 'm'):
            continue
        why = str(raw.meas_why[i])
        if not why.startswith('target_low_det'):
            continue
        src = str(det.answer_source[i])
        d = dict(sp=sp, rid=int(i), vid=vid, cat=det['cat'][i], question=raw.question[i], prior=raw.ground_truth_prior[i],
                 src=src, ans=float(det.answer[i]), cs=det.cs[i], why=why, fail='')
        diag.append(d)
        if src != 'caw_nogeo':
            d['fail'] = 'src'; continue
        pr = p['priors']
        if len(pr) != 1 or pr[0].get('kind') not in ('speed', 'size'):
            d['fail'] = 'prior'; continue
        pr = pr[0]
        fn = f"{R}/geo/out/tracks_{sp}_{raw.meas_tracks[i]}/{vid}.json"
        if not os.path.exists(fn):
            d['fail'] = 'notrk'; continue
        tr = json.load(open(fn))
        to = findobj(tr, (p.get('target') or {}).get('phrase'))
        po = findobj(tr, (p.get('prior_obj') or {}).get('phrase'))
        if to is None or po is None or to == po:
            d['fail'] = f'obj {to} {po}'; continue
        n = len(tr['fidx'])
        ts, ps = states(tr, to), states(tr, po)
        if len(ts) < 0.8 * n or len(ps) < 0.8 * n:
            d['fail'] = f'cov {len(ts)} {len(ps)} /{n}'; continue
        tv, tdx, tdy = netspeed(tr, ts)
        if pr['kind'] == 'speed':
            pv, _, _ = netspeed(tr, ps)
            if not (pv > 0):
                d['fail'] = 'pv0'; continue
            mpp = float(pr['value_si']) / pv
            d['pv'] = pv
            if not (tv >= 0.3 * pv):
                d['fail'] = f'static {tv:.1f}/{pv:.1f}'; continue
        else:
            pdim = pr.get('dim')
            pax = 1 if pdim == 'height' else 0
            if pdim == 'length':  # along prior's own motion axis
                _, pdx, pdy = netspeed(tr, ps)
                pax = 0 if abs(pdx) >= abs(pdy) else 1
            mpp = float(pr['value_si']) / float(np.median(ext(ps, pax)))
        tax = (0 if abs(tdx) >= abs(tdy) else 1) if q['dim'] == 'length' else 1
        nb = [(k, x) for k, x in ts if not x.get('border')]
        if len(nb) < 3:
            d['fail'] = 'border'; continue
        epx = float(np.median(ext(nb, tax)))
        val = epx * mpp
        cw = float(det.cs[i]) if det.cs[i] == det.cs[i] else np.nan
        d.update(tv=tv, epx=epx, mpp=mpp, val=val)
        if not (cw > 0 and 0.25 <= val / cw <= 10):
            d['fail'] = f'gate {val:.3g}/{cw}'; continue
        r = val / cw
        fin = float(np.sqrt(val * cw)) if 0.5 <= r <= 2 else val
        d['new'] = fin
        out[sp][str(int(i))] = dict(value=round(fin, 5), why=f"{TAG}: {epx:.0f}px x {mpp:.4g} m/px ({pr['kind']} prior), caw {cw:.3g}"
                                    + (" agree->gm" if 0.5 <= r <= 2 else " 2d_ratio"))
D = pd.DataFrame(diag)
print(D.to_string())
for sp in out:
    print(sp, len(out[sp]), out[sp])
L = labelled(SUB)
print('Lcols', L.columns.tolist(), L.index.name)
for sp in ('val', 'test'):
    for rid, v in out[sp].items():
        idx = int(rid)
        hit = L[(L.split == sp) & (L.index == idx)] if 'rid' not in L.columns else L[(L.split == sp) & (L.rid == idx)]
        if len(hit):
            y = float(hit.y.iloc[0])
            print('LAB', sp, rid, 'y', y, 'old', hit.m.iloc[0], 'new', A.row_mra(np.array([v['value']]), np.array([y])))
        else:
            print('nolab', sp, rid)
if WRITE:
    for sp in out:
        json.dump(out[sp], open(f"{R}/cache/tmp/ag_{TAG}_patch_{sp}.json", 'w'), indent=1)
    print('written')
