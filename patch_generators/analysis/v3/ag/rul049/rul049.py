"""rul049: rul034 (template-stated time-less speed -> 'speed of X at t') extended to rows whose question the pipeline
parse left as kind 'other' although the answer unit is a speed and a time is given (a missing noun, e.g.
'What is the average of the person at 1.0s in m/s?').

Re-derivation (rul327's rule): kind == 'other', unit is a speed (*/s, */h, not ^2), target parsed ->
  a t1/t2 window -> avgspeed;  a single t -> speed.
The c24tf template machinery (analysis/v3/ag/c24scan/c24tf.py) runs UNMODIFIED via runpy with only json.loads wrapped
so the re-derived kind is seen on load.  Every rul034 gate is kept:
  template = simulation bases sharing >= 2 lower-cased questions (transitive closure, test + val inputs);
  fact     = a time-less 'speed of X = v' stated by another member, all such facts within 2 %;
  own base states nothing (speed / accel) about X;  no template member states an acceleration of X;
  sibling at-t speed rows of X in the same recording (with geo) within 1.25x of v;
  answer not already within 2 % of v;  not answered by scene_prior / xrow_prior / rul018 / rul020.
Witness: rul034's own-geo witness (geo within 1.25x) when the row has geo; for re-derived rows WITHOUT geo (the parse
refused them, so they were never measured) the witness is CaW within 1.25x of v (rul032 / rul033 already accept CaW).
Rows set by any of the existing --patch_json tags (cache/tmp/sr81.sh) are skipped.  Output: v.
Writes cache/tmp/ag_c35tat_patch_{test,val}.json (FINAL answers).  Labels joined only for evaluation."""
import sys, os, re, json, runpy
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
SUB = 'sub_xt_Nzv'
os.environ['C24_SUB'] = SUB
os.environ.pop('C24_WRITE', None)
_loads = json.loads
SPEED_U = re.compile(r"/\s*(s|sec|h|hr)$")


def rederive(q):
    u = str(q.get('unit') or '').strip()
    if q.get('kind') != 'other' or not q.get('target') or not SPEED_U.search(u) or '^2' in u:
        return None
    if q.get('t1') is not None or q.get('t2') is not None:
        return 'avgspeed'
    if q.get('t') is not None:
        return 'speed'
    return None


REP = {}


def loads2(s, *a, **k):
    o = _loads(s, *a, **k)
    if isinstance(o, dict) and isinstance(o.get('q'), dict) and 'id' in o and 'video_id' in o:
        kd = rederive(o['q'])
        if kd:
            o['q'] = dict(o['q'], kind=kd)
            REP[(str(o['video_id']), int(o['id']))] = (kd, o.get('question'))
    return o


json.loads = loads2
# c24tf pickles its receiver table to cache/tmp/ag_c24scan_tf<MINQ>.pkl; redirect that to our own file
_to_pickle = pd.DataFrame.to_pickle
pd.DataFrame.to_pickle = lambda self, path, *a, **k: _to_pickle(self, f"{R}/cache/tmp/ag_c35tat_tf.pkl", *a, **k)
argv0 = sys.argv
sys.argv = [f"{R}/analysis/v3/ag/c24scan/c24tf.py"]
try:
    G = runpy.run_path(f"{R}/analysis/v3/ag/c24scan/c24tf.py", run_name="c24tf_c35")
finally:
    sys.argv = argv0
    json.loads = _loads
    pd.DataFrame.to_pickle = _to_pickle
X = G['X'].copy()
pd.set_option('display.width', 330); pd.set_option('display.max_colwidth', 70); pd.set_option('display.max_rows', 500)
print('\n================ rul049 ================')
print('re-derived parse rows (test+val):', len(REP))
for k_, v_ in sorted(REP.items()):
    print('  ', k_, v_)
X['rep'] = [(v, int(i)) in REP for v, i in zip(X.vid, X.rid)]
# existing patches (113 tags of the current build)
sr = open(f"{R}/cache/tmp/sr81.sh").read()
TAGS = re.findall(r'"Nzv\|\|--patch_json ([^"]+)"', sr)[0].split(',')
print('existing patch tags:', len(TAGS))
PSET = {'test': {}, 'val': {}}
for t_ in TAGS:
    for sp in ('test', 'val'):
        f_ = f"{R}/cache/tmp/ag_{t_}_patch_{sp}.json"
        if os.path.exists(f_):
            for k_ in json.load(open(f_)):
                PSET[sp][int(k_)] = t_
X['ptag'] = [PSET[s].get(int(i), '') for s, i in zip(X.sp, X.rid)]
T = X[(X.kind == 'rul034') & X.rep].copy()
hasg = T.g == T.g
T['wit35'] = np.where(hasg, T.gw, T.cw) & ~T.tacc & ~T.sibbad
T['fire35'] = T.agree & T.wit35 & ~T.have2 & ~T.covd & (T.ptag == '')
cols = ['sp', 'rid', 'vid', 'cat', 'obj', 'v', 'ans', 'src', 'g', 'cs', 'agree', 'gw', 'cw', 'tacc', 'sibbad', 'have2', 'covd', 'ptag',
        'fire35', 'srcb', 'raw0', 'y', 'q']
print('re-derived rows that became rul034 receivers:', len(T))
print(T[cols].round(3).to_string())
print('re-derived rows NOT receivers (no template fact for the object / own base states it / unit):',
      sorted(set(REP) - set(zip(T.vid, T.rid.astype(int)))))
# template evidence: every row of the same template asking the speed of X (any form), its answer / label
P0 = {sp: json.load(open(f"{R}/cache/tmp/ag_c24tat_patch_{sp}.json")) for sp in ('test', 'val')}
for r in T[T.fire35].itertuples():
    sib = X[(X.kind == 'rul034') & (X.obj == r.obj) & ((X.v - r.v).abs() <= 0.02 * r.v)]
    print(f'  {r.sp} {r.rid} {r.vid}: same template fact receivers:')
    print(sib[['sp', 'rid', 'vid', 'cat', 'ans', 'src', 'g', 'cs', 'gw', 'fire', 'y', 'q']].round(3).to_string())
    print('   already set to v by rul034:', [(s, i) for s, i in zip(sib.sp, sib.rid) if str(int(i)) in P0[s]])
# labelled evidence: rows of these templates whose OWN question asks the object's speed and whose label exists
L = G['L']; Lv = L[L.split == 'val']; Lt = L[L.split == 'test']; Lt = Lt[~Lt.index.duplicated()]
d = G['d']; comp = G['comp']
ev = []
for r in T[T.fire35].itertuples():
    members = comp[r.base]
    for z in d[d.base.isin(members)].itertuples():
        qk = z.q
        if qk.get('kind') in ('speed', 'avgspeed') and G['nobj'](qk.get('target') or '') == r.obj:
            y = (Lv.y.get(z.rid, np.nan) if z.sp == 'val' else Lt.y.get(z.rid, np.nan))
            ev.append(dict(sp=z.sp, rid=z.rid, vid=z.vid, q=z.question[:60], ans=z.ans, src=z.src, g=z.g, cs=z.cs, v=r.v, y=y))
if ev:
    E = pd.DataFrame(ev).drop_duplicates(['sp', 'rid'])
    print('== template rows asking the same object speed (labels for evaluation only)')
    print(E.round(3).to_string())
    El = E[E.y == E.y]
    if len(El):
        import analyze_caw as A
        print('labelled: n=%d  |v-y|/y max %.3f; MRA(v) %s  MRA(ans) %s' % (len(El), ((El.v - El.y).abs() / El.y).max(),
              A.row_mra(El.v.values, El.y.values).round(2).tolist(), A.row_mra(El.ans.values, El.y.values).round(2).tolist()))
# write
for sp in ('test', 'val'):
    Z = T[T.fire35 & (T.sp == sp)]
    P = {str(int(r.rid)): {'value': float(r.v), 'why': (f"rul049: parse kind 'other' with a speed unit and a time -> speed of "
                                                       f"'{r.obj}' at t; template ({r.srcb}) states {r.raw0!r} (time-less, uniform motion); own base silent; "
                                                       + (f"geo {r.g:.3g}" if r.g == r.g else "no geo (parse refused)")
                                                       + f", CaW {r.cs:.3g} within 1.25x; was {r.ans:.4g} ({r.src})")}
         for r in Z.itertuples()}
    json.dump(P, open(f"{R}/cache/tmp/ag_c35tat_patch_{sp}.json", "w"), indent=1)
    print('wrote rul049', sp, len(P), {k: v['value'] for k, v in P.items()})
    print('  by cat:', Z.groupby('cat').size().to_dict())
    lab = Z[Z.y == Z.y]
    print('  labelled rows touched:', len(lab), lab[['rid', 'y', 'ans', 'v', 'm0', 'm1']].to_dict('records') if len(lab) else '')
# all-237 check: any labelled row (val or test copy) whose id is patched
Pt = json.load(open(f"{R}/cache/tmp/ag_c35tat_patch_test.json")); Pv = json.load(open(f"{R}/cache/tmp/ag_c35tat_patch_val.json"))
hit = [('test', int(k)) for k in Pt if int(k) in Lt.index] + [('val', int(k)) for k in Pv if int(k) in Lv.index]
print('all-237 labelled rows touched:', hit, ' (labelled set size %d)' % len(L))
# macro before / after on labelled rows (val, all-237); the patch replaces m by the row MRA of the patched value
import analyze_caw as A
L2 = L.copy()
for sp, P in (('test', Pt), ('val', Pv)):
    for k, e in P.items():
        msk = (L2.split == sp) & (L2.index == int(k))
        if msk.any():
            L2.loc[msk, 'm'] = A.row_mra(np.array([e['value']] * int(msk.sum())), L2.loc[msk, 'y'].values)
for nm, M0, M1 in (('val', L[L.split == 'val'], L2[L2.split == 'val']), ('all-237', L, L2)):
    print(f'{nm}: n={len(M0)} macro before {A.macro(M0.m, M0.cat)[0]:.5f} after {A.macro(M1.m, M1.cat)[0]:.5f}')
