"""rul238: rul163 (template-stated time-less speed fact -> at-t / interval / whole rows) with its own-base guard lifted
when the template members that STATE the fact show visible rates (centre px speed / box height over [1,2] s) spread >= 1.5x and the
receiver lies inside that range: the template value does not follow visible motion, so an off-template own geo is no evidence.
Labels only evaluate. No ids in the logic."""
"""c82fact: coverage gaps of the template-stated time-less SPEED facts (rul034 / rul018 / rul053 family).
Two gaps, two tags (patches written separately):
 rul164 : object identity. The parser drops a leading participle from PRIOR objects ('speed of the walking person = 1.236'
               -> obj 'person') but keeps it in question targets ('walking person'), and 'bike' never meets 'bicycle', so the
               rul034 linkage (exact nobj) misses them. Here both sides are normalised: rul067's strip_ing (leading -ing
               words, not ring/king/wing/...) and a small synonym map (bike->bicycle, motorbike->motorcycle, aeroplane/airplane
               ->plane). Witness = rul034's: own geo within 1.25x of v.
 rul163  : witness. rul018's A2 template-geo witness (>= 3 speed rows about X in OTHER members of the template, median
               geo/v within 1.15x and >= 2/3 within 1.25x; guard: no second own-base speed row about X confirms the row's own
               off-template geo) is used only for whole-clip forms; at-t / interval rows whose own geo fails had no witness.
               Applied to at-t (t >= 0.1 s) and interval rows, with the noun normalisation above.
Gates kept from rul034: linked facts agree within 2 % (>= 2-question component, else >= 3); own base states no speed and no
acceleration of X; no template member states an acceleration of X; skip 'initial' / t < 0.1; skip rows already within 2 % of v,
covered (scene_prior / xrow_prior / patch:rul018 / patch:rul020) or set by any patch. Output v (m/s); _segmented rows with the
identical question copy the value. Labels joined at the end for evaluation only. No ids in the logic.
usage: tools/run_limited.sh --io python analysis/v3/ag/rul238/rul238.py sub_xt_Nzz38 [write]"""
import sys, re, json, collections
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + '/cache/tmp'); sys.path.insert(0, R + '/caw')
import analyze_caw as A
from lab_rows import labelled
pd.set_option('display.width', 400); pd.set_option('display.max_colwidth', 60); pd.set_option('display.max_rows', 3000)
SUB = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != 'write' else 'sub_xt_Nzz38'
WRITE = 'write' in sys.argv[1:]
STOP = {"the", "a", "an", "of", "in", "at", "is", "what", "meters", "meter", "cm", "m", "mm", "s", "total", "each", "one"}
KEEPING = {'ring','king','wing','string','spring','swing','building','ceiling','painting','railing','ping','thing'}
SYN = {'bike': 'bicycle', 'bikes': 'bicycle', 'motorbike': 'motorcycle', 'aeroplane': 'plane', 'airplane': 'plane'}
def nobj(x):
    return " ".join(w for w in re.sub(r"[^a-z ]", " ", str(x).lower()).split() if w not in STOP)
def nobj2(x):
    w = [SYN.get(z, z) for z in nobj(x).split()]
    while len(w) > 1 and w[0].endswith('ing') and w[0] not in KEEPING: w = w[1:]
    return " ".join(w)
TIMED = r"\d\s*s\b|t\s*=|from|between|before|after|\bat \d"
# --- rul238 helper: visible rate (centre px speed / box height, 1/s) over [1,2] s on measure_v3's own track chain
import os as _os, json as _json
_RT = {}
_FPS = {}
for _sp in ('test', 'val'):
    _dd = pd.read_csv(f"{R}/preds/{_sp}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", usecols=["video_id", "fps"])
    for _v, _f in zip(_dd.video_id, _dd.fps): _FPS[(_sp, str(_v))] = float(_f)
def _ekey(ent):
    if not ent or not ent.get('phrase'): return None
    rel = ent.get('rel')
    return ent['phrase'] + '|' + ','.join(ent.get('where') or []) + ('|' + rel[0] + ':' + rel[1] if rel else '')
def vis_rate(sp, v, ent):
    k = _ekey(ent)
    if (sp, v, k) in _RT: return _RT[(sp, v, k)]
    res = np.nan
    fps = _FPS.get((sp, v))
    for d in ('_v2sc', '_v2', ''):
        p = f"{R}/geo/out/tracks_{sp}{d}/{v}.json"
        if not _os.path.exists(p) or k is None or not fps: continue
        j = _json.load(open(p))
        if 'objs' in j:
            q = next((q for q in (j.get('queries_d') or []) + (j.get('queries') or []) if isinstance(q, dict) and q.get('key') == k and q.get('obj') is not None), None)
            fr = j['objs'][str(q['obj'])] if q else None
        else:
            q = next((q for q in (j.get('queries') or []) if q.get('key') == k and q.get('frames') is not None), None)
            fr = q['frames'] if q else None
        if fr is None: break
        C = {f: (np.array(e['st']['c'], float), e['st']['box'][3] - e['st']['box'][1], e['st'].get('border', 0)) for f, e in zip(j['fidx'], fr) if e and e.get('st')}
        h = max(1, int(round(0.25 * fps))); rr = []
        for f in sorted(C):
            if not (1.0 - 1e-9 <= f / fps <= 2.0 + 1e-9): continue
            if f - h in C and f + h in C and C[f][2] == 0 and C[f - h][2] == 0 and C[f + h][2] == 0 and C[f][1] > 4:
                rr.append(float(np.hypot(*(C[f + h][0] - C[f - h][0]))) / (2 * h / fps) / C[f][1])
        if len(rr) >= 8: res = float(np.median(rr))
        break
    _RT[(sp, v, k)] = res
    return res
PO = {}
for _sp in ('test', 'val'):
    for _l in open(f"{R}/geo/out/parse_{_sp}_v2.jsonl"):
        _j = _json.loads(_l); PO.setdefault(str(_j['video_id']), (_sp, _j.get('prior_obj')))
def lift_wit(sp, vid, ent, bases):
    """template-stating members' visible rates of X vs the receiver's: lift the own-geo guard only if the stating members' visible
    rates spread >= 1.5x (the template value does not follow visible motion) and the receiver's rate lies inside their range (x1.15)"""
    r0 = vis_rate(sp, vid, ent)
    rs = []
    for b in sorted(set(bases)):
        for vv in (b, b + '_segmented'):
            if vv in PO:
                s_, pe = PO[vv]; x = vis_rate(s_, vv, pe)
                if np.isfinite(x): rs.append(x); break
    ok = np.isfinite(r0) and len(rs) >= 2 and max(rs) / min(rs) >= 1.5 and min(rs) / 1.15 <= r0 <= max(rs) * 1.15
    return bool(ok), r0, rs

rows = []
for sp, suf in (('test', 'detail'), ('val', 'valdetail')):
    raw = pd.read_csv(f"{R}/preds/{sp}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", index_col=0)
    det = pd.read_csv(f"{R}/submissions/{SUB}_{suf}.csv", index_col=0)
    PP = {p['id']: p for p in map(json.loads, open(f"{R}/geo/out/parse_{sp}_v2.jsonl"))}
    for i in raw.index:
        p = PP[i]
        rows.append(dict(sp=sp, rid=int(i), vid=str(raw.video_id[i]), question=str(raw.question[i]), q=p['q'], priors=p['priors'], tent=p.get('target'),
                         cat=det['cat'][i], ans=float(det.answer[i]), src=str(det.answer_source[i]).replace('mo_guard:', ''), g=det.g[i], cs=det.cs[i], mm=str(raw.meas_mode[i])))
d = pd.DataFrame(rows)
d['ql'] = d.question.str.strip().str.lower(); d['base'] = d.vid.str.replace("_segmented", "", regex=False)
d = d[d.base.str.startswith('simulation_')].reset_index(drop=True)
qs = d.groupby('base').ql.apply(set); sb = list(qs.index)
def components(minq):
    lk = {b: [c for c in sb if c != b and len(qs[b] & qs[c]) >= minq] for b in sb}
    comp = {}
    for b0 in sb:
        if b0 in comp: continue
        st, seen = [b0], {b0}
        while st:
            x = st.pop()
            for y in lk[x]:
                if y not in seen: seen.add(y); st.append(y)
        for x in seen: comp[x] = frozenset(seen)
    return comp
COMPS = {2: components(2), 3: components(3)}
def pkey(x):
    lhs = x['raw'].lower().split('=')[0]
    if x['kind'] == 'speed': return ('speed', None if not re.search(TIMED, lhs) else 'timed')
    if x['kind'] == 'accel': return ('grav', None) if (x.get('gravity') or not x['obj']) else ('accel', None)
    return (x['kind'], 'other')
F = []
for r in d.itertuples():
    for x in r.priors:
        if x.get('value_si') is None: continue
        k, tk = pkey(x)
        F.append((r.base, nobj(x['obj'] or ''), nobj2(x['obj'] or ''), k, tk, float(abs(x['value_si'])), x['raw'].strip(), r.sp))
F = pd.DataFrame(F, columns=['base', 'obj', 'obj2', 'kind', 'tk', 'v', 'raw', 'fsp']).drop_duplicates(['base', 'raw', 'fsp'])
own2 = collections.defaultdict(set)
for f in F.itertuples(): own2[(f.base, f.obj2)].add(f.kind)
SPF = {b: g for b, g in F[(F.kind == 'speed') & F.tk.isna()].groupby('base')}
# speed rows with normalised object (for template-geo witness)
d['o2'] = [nobj2(q.get('target') or '') for q in d.q]
d['o1'] = [nobj(q.get('target') or '') for q in d.q]
d['isspd'] = [(q.get('kind') in ('speed', 'avgspeed') and q.get('unit') == 'm/s') for q in d.q]
L = labelled(SUB); Lt = L[L.split == 'test']; Lt = Lt[~Lt.index.duplicated()]; Lv = L[L.split == 'val']
ACC2 = set(zip(F[F.kind == 'accel'].base, F[F.kind == 'accel'].obj2))
COVER = ('scene_prior', 'xrow_prior', 'patch:rul018', 'patch:rul020')
d['t_'] = [q.get('t') for q in d.q]
spd = d[d.isspd & ~d.ql.str.contains('initial') & ~(d.t_.fillna(1.0) < 0.1)]
out = []
for r in spd.itertuples():
    q = r.q; o2 = r.o2
    if not o2:
        continue
    t, t1, t2 = q.get('t'), q.get('t1'), q.get('t2')
    form = 'at' if t is not None else ('int' if (t1 is not None or t2 is not None) else 'whole')
    if {'speed', 'accel'} & own2[(r.base, o2)]:
        continue
    src = None; agree = False
    for lev in (2, 3):
        s_ = [(c, f.v, f.raw, f.obj) for c in sorted(COMPS[lev][r.base] - {r.base}) if c in SPF for f in SPF[c].itertuples() if f.obj2 == o2]
        if not s_:
            break
        src, used = s_, lev; vals = [z[1] for z in src]; agree = max(vals) / min(vals) <= 1.02
        if agree:
            break
    if not src or not agree:
        continue
    exact = any(z[3] == r.o1 for z in src)
    v = float(np.median(vals)); comp = COMPS[used][r.base]
    if any((c, o2) in ACC2 for c in comp):
        continue
    W = spd[(spd.o2 == o2) & spd.base.isin(comp - {r.base}) & (spd.g > 0)].drop_duplicates(['vid', 'ql'])
    rr = (W.g / v).values
    tw = len(rr) >= 3 and abs(np.log(np.median(rr))) <= np.log(1.15) and (np.abs(np.log(rr)) <= np.log(1.25)).mean() >= 2 / 3
    gw = r.g == r.g and r.g > 0 and v / 1.25 <= r.g <= 1.25 * v
    ownr = spd[(spd.base == r.base) & (spd.o2 == o2) & (spd.rid != r.rid) & (spd.sp == r.sp) & (spd.g > 0)]
    side = np.sign(np.log(r.g / v)) if (r.g == r.g and r.g > 0) else 0
    guard = bool(((np.abs(np.log(ownr.g / v)) > np.log(1.25)) & (np.sign(np.log(ownr.g / v)) == side)).any()) if side != 0 else False
    lift = (False, np.nan, [])
    if guard:
        lift = lift_wit(r.sp, r.vid, r.tent, [z[0] for z in src])
        print('LIFT', r.sp, r.rid, r.vid, form, 'v', v, 'g', r.g, 'ok', lift[0], 'r0 %.3f' % lift[1], 'stating', [round(x, 3) for x in lift[2]], sorted(set(z[0] for z in src)))
        guard_lifted = lift[0]
    else:
        guard_lifted = False
    have2 = abs(r.ans - v) <= 0.02 * v
    covd = r.src in COVER; patched = r.src.startswith('patch') or r.src == 'patch_copy'
    tag = None
    if not have2 and not covd and not patched:
        if form in ('at', 'int') and not exact and gw:
            tag = 'rul164'
        elif form in ('at', 'int') and not gw and tw and not guard:
            tag = 'rul163'
        elif form in ('at', 'int', 'whole') and not gw and tw and guard and guard_lifted:
            tag = 'rul238'
    y = (Lv.y.get(r.rid, np.nan) if r.sp == 'val' else Lt.y.get(r.rid, np.nan))
    out.append(dict(sp=r.sp, rid=r.rid, vid=r.vid, cat=r.cat, form=form, o=o2, exact=exact, v=v, ans=r.ans, src=r.src, g=r.g, cs=r.cs,
                    gw=gw, tw=tw, nW=len(rr), wmed=float(np.median(rr)) if len(rr) else np.nan, guard=guard, have2=have2, covd=covd,
                    patched=patched, tag=tag, lr0=lift[1], lrs=','.join('%.2f' % x for x in lift[2]), y=y, q=r.question, fsrc=','.join(sorted(set(z[0][-4:] for z in src))), fraw=src[0][2]))
X = pd.DataFrame(out)
cols = ['sp', 'rid', 'vid', 'cat', 'form', 'o', 'exact', 'v', 'ans', 'src', 'g', 'cs', 'gw', 'tw', 'nW', 'wmed', 'guard', 'have2', 'patched', 'tag', 'y', 'fsrc']
print('== candidates not already within 2%:')
print(X[~X.have2][cols].sort_values(['sp', 'vid']).round(3).to_string())
print('== labelled rows among all linked candidates (any gate state):')
print(X[X.y == X.y][cols].round(3).to_string())
FI = X[X.tag.notna()]
for sp, suf in (('test', 'detail'), ('val', 'valdetail')):
    det = pd.read_csv(f"{R}/submissions/{SUB}_{suf}.csv", index_col=0)
    raw = pd.read_csv(f"{R}/preds/{sp}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", index_col=0)
    for tg in ('rul238',):
        P = {}
        for r in FI[(FI.sp == sp) & (FI.tag == tg)].itertuples():
            if tg == 'rul164':
                wit = 'object match after participle/synonym normalisation; own geo %.3g within 1.25x' % r.g
            else:
                wit = 'own geo %.3g off; template-geo witness: %d other-member rows, median geo/v %.2f; own-base guard lifted: stating members visible rates [%s]/s spread >= 1.5x, receiver %.2f/s inside' % (r.g, r.nW, r.wmed, r.lrs, r.lr0)
            why = f"{tg}: template ({r.fsrc}) states {r.fraw!r} (time-less, linked facts agree, no template acceleration of '{r.o}'); {wit}; was {r.ans:.4g} ({r.src})"
            P[str(int(r.rid))] = {'value': float(r.v), 'why': why}
            qq = str(raw.question[r.rid]).strip().lower()
            tw_ = raw[(raw.video_id.astype(str) == r.vid + '_segmented') & (raw.question.astype(str).str.strip().str.lower() == qq)]
            for j in tw_.index:
                P[str(int(j))] = {'value': float(r.v), 'why': why + f' [_segmented copy of {r.rid}]'}
        print(sp, tg, 'rows', len(P), sorted(P, key=int))
        for k, e in P.items():
            i = int(k); y = (Lv.y.get(i, np.nan) if sp == 'val' else Lt.y.get(i, np.nan))
            ev = '' if y != y else '| MRA %.2f -> %.2f' % (A.row_mra(np.array([det.answer[i]]), np.array([y]))[0], A.row_mra(np.array([e['value']]), np.array([y]))[0])
            print('   ', k, det.video_id[i], det['cat'][i], str(raw.question[i])[:70], '| %.4g -> %.4g' % (det.answer[i], e['value']), '| y', y, ev)
        if WRITE:
            json.dump(P, open(f"{R}/cache/tmp/ag_{tg}_patch_{sp}.json", 'w'), indent=1)
            print('wrote', f"cache/tmp/ag_{tg}_patch_{sp}.json")
