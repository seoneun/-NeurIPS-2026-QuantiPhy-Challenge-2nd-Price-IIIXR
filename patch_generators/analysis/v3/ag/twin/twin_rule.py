"""A1 time-window rule (rule twin).
Finds rows whose question names a time window the parser left out of q (t1/t2), or left inside the target phrase,
plus "<X>'s moving distance between A s to B s" rows that QP_PARSE2 turned into a path of a garbled phrase, and the
two-window acceleration "from A-B s to C-D s" = (vbar[C,D] - vbar[A,B]) / (mid2 - mid1).
Re-measures those rows with geo/measure_v3.measure_video (chain rj5sQ environment, detector tracks v2sc -> v2 -> v1)
and refuses whenever the cleaned target phrase has no track.  Writes cache/tmp/ag_twin_meas_{split}.csv
usage: python analysis/v3/ag/twin/twin_rule.py test|val
"""
import os, sys, re, json, copy, math
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
ENV = dict(QP_DEPTH_DEDUP='1', QP_SCENE_SA2VA='1', QP_SECOND_FIX='1', QP_SAMECLASS='1', QP_SCENE_RULER='1', QP_FF_ROT='1',
           QP_SIDE_FIX='1', QP_FLOOR='1', QP_STATIC_CAP='25', QP_FF_SUB='1', QP_FF_SUB_SRC='simulation', QP_PARSE2='1',
           QP_NEAREST='1', QP_BELT='1', RL_SELF_GRAV='1', RL_SELF_GRAV_MODE='2d_gself', QP_FPERSON='1', QP_FPERSON_MINN='2',
           QP_REJ_LO='0.2', QP_REJ_HI='5', QP_PRIOR_DRIFT='1.5', QP_PRIOR_ACCGUARD='3', QP_SCENE_HEAD='1', QP_TSHIFT='1')
os.environ.update(ENV)
sys.path.insert(0, R + '/geo')
import numpy as np, pandas as pd
import measure_v3 as MV
import measure as M1
from rules import UNIT_SI

split = sys.argv[1] if len(sys.argv) > 1 else 'test'
NUM = r'(\d+(?:\.\d+)?)'
SS = r'\s*(?:s|sec|secs|seconds)'
DASH = '[-–]'
TWO = re.compile(r'\bfrom\s+' + NUM + r'(?:' + SS + r')?\s*' + DASH + r'\s*' + NUM + SS + r'\s+to\s+' + NUM + r'(?:' + SS + r')?\s*' + DASH + r'\s*' + NUM + SS + r'\b')
WIN = re.compile(r'\b(?:over|from|between|during)\s+(?:the\s+)?(?:time\s+)?(?:t\s*=\s*)?' + NUM + r'(?:' + SS + r')?\s*(?:to|' + DASH + r'|and|until)\s*(?:t\s*=\s*)?' + NUM + SS + r'\b')
START = re.compile(r'\bfrom\s+(?:the\s+)?(?:r?om\s+)?(?:start|beginning)(?:\s+of\s+the\s+(?:video|clip))?\s+(?:to|until)\s+(?:t\s*=\s*)?' + NUM + SS + r'\b')
END = re.compile(r'\bfrom\s+(?:t\s*=\s*)?' + NUM + r'(?:' + SS + r')?\s+(?:to|until)\s+the\s+end\b')
MOVD = re.compile(r"\b([a-z][a-z\-]*(?: [a-z][a-z\-]*){0,2})(?:['’]s)? moving distance between " + NUM + r'\s*s?\s*(?:to|and|-)\s*' + NUM + r'\s*s\b')
TKINDS = ('avgspeed', 'speed', 'accel', 'disp', 'path', 'hvel')


def clean_phrase(p):
    """drop a time window left inside a target phrase: 'boat over 0-7s', 'boat to 2s-4.5s', 'toilet paper roll over 0.5s to 1.73s'"""
    p2 = re.sub(r'\s+(?:over|from|between|during|to|until|at)\s+(?:t\s*=\s*)?\d.*$', '', p)
    p2 = re.sub(r'\s+\d+(?:\.\d+)?\s*s\b.*$', '', p2)
    return p2.strip()


def find_rows(byv, orig):
    """the rule's trigger: question text + parse only"""
    plan = {}
    for v, rs in byv.items():
        for r in rs:
            q = r['q']; ql = r['question'].lower(); kind = q.get('kind')
            o = orig[r['id']]; oq = o['q']
            tp0 = ((r.get('target') or {}).get('phrase') or '').strip().lower()
            m2 = TWO.search(ql)
            fix = None
            if m2 and kind == 'accel':
                a, b, c, d_ = map(float, m2.groups())
                fix = dict(rule='two_window', w1=(a, b), w2=(c, d_))
            elif kind in ('path', 'dist2') and MOVD.search(ql) and not ((o.get('target2') or {}).get('phrase')) \
                    and oq.get('t1') is not None and oq.get('t2') is not None:
                ph = ((o.get('target') or {}).get('phrase') or '').strip().lower()      # the parser's own target (QP_PARSE2 garbles it)
                if ph and tp0 != ph:
                    fix = dict(rule='d2path', t1=float(oq['t1']), t2=float(oq['t2']), phrase=ph)
            elif kind in TKINDS:
                t1 = t2 = None; nm = None
                for nm_, rx in (('win', WIN), ('start', START), ('end', END)):
                    m = rx.search(ql)
                    if m:
                        nm = nm_
                        if nm_ == 'win':
                            t1, t2 = float(m.group(1)), float(m.group(2))
                        elif nm_ == 'start':
                            t1, t2 = 0.0, float(m.group(1))
                        else:
                            t1, t2 = float(m.group(1)), None
                        break
                if nm and q.get('t') is None:
                    same = (q.get('t1') is not None and abs(q['t1'] - t1) < 1e-6 and
                            ((t2 is None and q.get('t2') is None) or (t2 is not None and q.get('t2') is not None and abs(q['t2'] - t2) < 1e-6)))
                    if not same and (t2 is None or t2 > t1):
                        fix = dict(rule=nm, t1=t1, t2=t2)
            # a time window left inside the target phrase (any kind)
            if re.search(r'\d', tp0):
                cp = clean_phrase(tp0)
                if cp and cp != tp0:
                    fix = dict(fix or dict(rule='strip'), phrase=cp)
            if fix:
                plan[r['id']] = fix
    return plan


def main():
    MV.TRACK_VER = 'v2sc'; MV.FIX = set('abcdef'); MV.HALF_V, MV.HALF_A = 0.25, 0.4
    MV.FOCAL_MIX = ''; MV.PARSE_TAG = ''; MV.AGG3D_P = None; MV.STATIC_FRAC = 0.25
    MV.MEASURE_SCENE = False; MV.WINGED_LEN_SHORT = False; MV.MOGE_OPTS = {'fbrej'}
    fn_ = f'{R}/geo/out/scene_ruler_{split}.json'
    if os.path.exists(fn_):
        MV.SCENE_RULER.update({k: v for k, v in json.load(open(fn_)).items() if v and v.get('ok')})
    fn_ = f'{R}/geo/out/person_focal_{split}.json'
    if os.path.exists(fn_):
        MV.FPERSON.update(json.load(open(fn_)))
    M1.SCALE_DEPTH = True
    cfg = 'e'
    tp = f'{R}/geo/out/owl_threshold_v2.json'
    if os.path.exists(tp):
        MV.T_OWL = json.load(open(tp))['T_owl']
    orig = {}
    for l in open(f'{R}/geo/out/parse_{split}_v2.jsonl'):
        j = json.loads(l); orig[j['id']] = j
    byv, vjs = MV.load_split_tracks(split, cfg)
    fmed = json.load(open(f'{R}/geo/out/focal_medians_test.json'))['f_over_W']
    allf = float(np.median(list(fmed.values())))
    ref = pd.read_csv(f'{R}/geo/out/meas_{split}_v2_v3m_rrj5sQ.csv').set_index('id')
    plan = find_rows(byv, orig)
    print('planned', len(plan), plan, flush=True)
    vids = sorted({r['video_id'] for v, rs in byv.items() for r in rs if r['id'] in plan})
    for v in vids:
        p = f'{R}/geo/out/moge_{split}/{v}.npz'
        if os.path.exists(p):
            z = np.load(p)
            MV.MOGE[v] = dict(f=float(np.median(z['fx'])), J=[int(j) for j in z['j']], depth=z['depth'], W=int(z['W']))
    out = []
    for v in vids:
        rs = byv[v]; vj = vjs.get(v)
        if vj is None:
            for r in rs:
                if r['id'] in plan:
                    out.append(dict(id=r['id'], video_id=v, rule=plan[r['id']]['rule'], ok=False, why='no_track_file'))
            continue
        W = vj['W']; dur = (max(vj['fidx']) + 1) / float(vj['fps'])
        fdd = {r['video_source']: fmed.get(r['video_source'], allf) * W for r in rs}
        o0, V0 = MV.measure_video(copy.deepcopy(rs), vj, fdd, cfg)
        base = {x['id']: x for x in o0}
        rs2 = copy.deepcopy(rs); extra = {}
        for r in rs2:
            if r['id'] not in plan:
                continue
            f = plan[r['id']]
            if f.get('phrase'):
                ph = f['phrase']
                r['target'] = dict(r['target'], phrase=ph, raw=ph, where=[])
                r['q'] = dict(r['q'], target=ph)
            if f['rule'] in ('win', 'start', 'end', 'd2path'):
                t2 = f['t2'] if f['t2'] is not None else dur
                kind = r['q']['kind']
                if f['rule'] == 'd2path':
                    kind = 'path'
                elif kind == 'speed':
                    kind = 'avgspeed'
                r['q'] = dict(r['q'], t=None, t1=f['t1'], t2=t2, kind=kind)
                if f['rule'] == 'd2path':
                    r['target2'] = None; r['target2_dkeys'] = []
                if kind == 'accel':
                    # the window's mean speed: the kinematic bound |v(b)-v(a)| <= 2 vbar for a CaW plausibility check
                    rr = copy.deepcopy(r); rr['id'] = f"{r['id']}_vb"
                    rr['q'] = dict(rr['q'], kind='avgspeed', unit='m/s')
                    extra[rr['id']] = rr
            if f['rule'] == 'two_window':
                for k_, (a, b) in (('w1', f['w1']), ('w2', f['w2'])):
                    rr = copy.deepcopy(r); rr['id'] = f"{r['id']}_{k_}"
                    rr['q'] = dict(rr['q'], kind='avgspeed', t=None, t1=a, t2=b, unit='m/s')
                    extra[rr['id']] = rr
        o1, V1 = MV.measure_video(rs2 + list(extra.values()), vj, fdd, cfg)
        new = {x['id']: x for x in o1}
        for r in rs:
            i = r['id']
            b0 = base[i]; u = r['q']['unit']
            v0 = b0['val_si'] / UNIT_SI[u] if b0.get('ok') else np.nan
            refv = ref.loc[i, 'value'] if i in ref.index else np.nan
            if i not in plan:
                n1 = new[i]; v1 = n1['val_si'] / UNIT_SI[u] if n1.get('ok') else np.nan
                if not (np.isnan(v0) and np.isnan(v1)) and not np.isclose(v0, v1, rtol=1e-6):
                    print('WARN unplanned row changed', i, v0, v1)
                continue
            f = plan[i]
            rec = dict(id=i, video_id=v, rule=f['rule'], phrase=f.get('phrase', ''), t1=f.get('t1'), t2=f.get('t2'),
                       dur=round(dur, 3), base=v0, ref=refv, mode0=b0.get('mode'), why0=b0.get('why'))
            if f['rule'] == 'two_window':
                x1, x2 = new[f'{i}_w1'], new[f'{i}_w2']
                rec.update(w1=f['w1'], w2=f['w2'])
                if f['w2'][0] >= dur - 0.1:
                    rec.update(ok=False, why=f'window2 past clip end {dur:.2f}')
                elif x1.get('ok') and x2.get('ok'):
                    m1_ = sum(f['w1']) / 2
                    m2_ = (f['w2'][0] + min(f['w2'][1], dur)) / 2
                    acc = (x2['val_si'] - x1['val_si']) / (m2_ - m1_)
                    rec.update(ok=True, val=abs(acc) / UNIT_SI[u], v1=x1['val_si'], v2=x2['val_si'], signed=acc, mode=x1['mode'], why='two_window',
                               span=m2_ - m1_, vref=max(x1['val_si'], x2['val_si']), bound=1.5)
                else:
                    rec.update(ok=False, why=f"w1 {x1.get('why')} / w2 {x2.get('why')}")
            else:
                n1 = new[i]
                rec.update(ok=bool(n1.get('ok')), val=n1['val_si'] / UNIT_SI[u] if n1.get('ok') else np.nan, mode=n1.get('mode'), why=n1.get('why'))
                xb = new.get(f'{i}_vb')
                if xb is not None and xb.get('ok'):
                    t2_ = f['t2'] if f.get('t2') is not None else dur
                    rec.update(span=min(t2_, dur) - f['t1'], vref=xb['val_si'], bound=3.0)
            if rec.get('ok'):
                k = {"m": 0, "cm": 0, "mm": 0, "km": 0}.get(u, 2 if "^2" in u else 1)
                lo, hi = MV.PLAUS[k]
                if not (lo <= rec['val'] * UNIT_SI[u] <= hi):
                    rec.update(ok=False, why='implausible')
            out.append(rec)
            print(rec, flush=True)
    df = pd.DataFrame(out)
    df.to_csv(f'{R}/cache/tmp/ag_twin_meas_{split}.csv', index=False)
    print(df.to_string())


if __name__ == '__main__':
    main()
