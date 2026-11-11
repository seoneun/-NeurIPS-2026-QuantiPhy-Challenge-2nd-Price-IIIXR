"""rul052: do not shrink the CaW-alone size of the object that is the video's own ruler.

A 2D video whose motion rows are measured with the target's own CaW size as the ruler (ruler_build.py stage m2,
merge_src 'selfruler', modes 2d_gself / 2d_selfruler) fixes that object's size at CaW's reading. When the same video
also asks the size of that object and the answer is CaW-alone (caw_nogeo / dis_caw_routed / caw_c1) with the
label-tuned x0.9 / x0.81 shrink, the video answers two different sizes for one object. This rule restores the
unshrunk CaW size (cs) on such rows, so the size row uses the same ruler as the video's own motion rows.
Inputs only: raw geo csv (merge_src / meas_mode), parse (target phrase keys), ruler map + CaW ruler answers,
detail (answer_source, answer, cs). Labels are only used for evaluation at the end.
"""
import sys, re, json, os
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + '/cache/tmp'); sys.path.insert(0, R)
TAG = 'rul052'
SUB = os.environ.get('C36_SUB', 'sub_xt_Nzw')
RAWT = os.environ.get('C36_RAW', 'geo_v4ntrj5sQ2srngsr_d_vw')
CAW_ALONE = {'caw_nogeo', 'dis_caw_routed', 'caw_c1'}
SELF_MODES = {'2d_gself', '2d_selfruler'}
UNIT = {'m': 1.0, 'meters': 1.0, 'meter': 1.0, 'cm': 0.01, 'centimeters': 0.01, 'mm': 0.001, 'km': 1000.0}


def key(e):
    if not e or not e.get('phrase'):
        return None
    rel = e.get('rel')
    return e['phrase'] + '|' + ','.join(e.get('where') or []) + ('|' + rel[0] + ':' + rel[1] if rel else '')


def ruler_answers(sp):
    a = {}
    f = f'{R}/caw/sc/out/{sp}_ruler/g16.jsonl'
    if not os.path.exists(f):
        return a
    for ln in open(f, encoding='utf-8'):
        x = json.loads(ln)
        t = (x['responses'] or [''])[0]
        p = [z for z in x['parsed'] if z is not None and np.isfinite(z) and z > 0]
        if not p:
            continue
        v = float(p[0])
        u = re.search(r'\d\s*(km|cm|mm|m)\b', t.lower())
        if u:
            v *= {'km': 1000.0, 'cm': 0.01, 'mm': 0.001, 'm': 1.0}[u.group(1)]
        a[int(x['original_id'])] = v
    return a


def unit_si(q, p):
    u = (p.get('q') or {}).get('unit')
    if u and str(u).lower() in UNIT:
        return UNIT[str(u).lower()]
    m = re.search(r'\bin (cm|mm|km|meters?|m)\b', str(q).lower())
    return UNIT.get(m.group(1), 1.0) if m else 1.0


def run(sp):
    raw = pd.read_csv(f'{R}/preds/{sp}_{RAWT}_raw.csv').set_index('Unnamed: 0')
    D = pd.read_csv(f'{R}/submissions/{SUB}_' + ('detail' if sp == 'test' else 'valdetail') + '.csv').set_index('Unnamed: 0')
    P = {}
    for ln in open(f'{R}/geo/out/parse_{sp}_v2.jsonl'):
        r = json.loads(ln); P[r['id']] = r
    mp = f'{R}/analysis/v3/ruler_{sp}.json'
    M = {int(k): v for k, v in json.load(open(mp)).items()} if os.path.exists(mp) else {}
    A = ruler_answers(sp)
    # 1) the video's own-ruler objects: the target of a selfruler motion row, with the CaW size that scaled it
    own = {}
    S = raw[(raw.merge_src.astype(str) == 'selfruler') & raw.meas_mode.astype(str).isin(SELF_MODES)]
    for i, r in S.iterrows():
        m = M.get(int(i)); p = P.get(int(i))
        if not m or not p:
            continue
        tk = key(p.get('target'))
        for z in m['rulers']:
            if z['key'] == tk and z['qid'] in A:
                own.setdefault(r.video_id, {}).setdefault(tk, []).append((A[z['qid']], z['dim'], r.meas_mode, int(i)))
                break
    patch, info = {}, []
    for i, r in D.iterrows():
        v = r.video_id
        if v not in own:
            continue
        p = P.get(int(i))
        if not p or (p.get('q') or {}).get('kind') != 'size':
            continue
        if str(r.answer_source) not in CAW_ALONE:
            continue
        cs, ans = float(r.cs), float(r.answer)
        if not (np.isfinite(cs) and cs > 0 and np.isfinite(ans) and ans > 0):
            continue
        if ans / cs > 0.99:               # only rows where the CaW-alone shrink was applied
            continue
        tk = key(p.get('target'))
        hit = own[v].get(tk)
        if hit is None:                   # same phrase with no spatial qualifier on either side
            ph = (p.get('target') or {}).get('phrase')
            for k2, vv in own[v].items():
                if k2.split('|')[0] == ph and not (p.get('target') or {}).get('where') and not k2.split('|')[1]:
                    hit = vv; break
        if hit is None:
            continue
        rv = float(np.median([h[0] for h in hit]))    # the ruler's CaW size (SI)
        csi = cs * unit_si(r.get('question', ''), p)
        if abs(np.log(csi / rv)) > np.log(1.25):     # cs must be the same reading as the ruler size
            continue
        patch[str(int(i))] = dict(value=cs, why=f'{TAG}: target is the own ruler of {len(hit)} {hit[0][2]} motion rows '
                                                 f'(CaW {rv:g} m); undo CaW-alone shrink {ans:.4g}->{cs:.4g}')
        info.append(dict(id=int(i), video=v, cat=r.get('cat'), q=str(r.get('question', ''))[:60], old=ans, new=cs, ruler_m=rv,
                         motion_rows=[h[3] for h in hit]))
    json.dump(patch, open(f'{R}/cache/tmp/ag_{TAG}_patch_{sp}.json', 'w'), indent=1)
    print(f'[{sp}] own-ruler videos {len(own)} {sorted(own)}; patched rows {len(patch)}')
    for x in info:
        print('  ', x)
    return patch


if __name__ == '__main__':
    PT = run('test'); PV = run('val')
    # ---- evaluation only (labels never feed the patch) ----
    from lab_rows import labelled
    import analyze_caw as AC
    L = labelled(SUB)
    for sp, PP in (('test', PT), ('val', PV)):
        Ls = L[L.split == sp] if 'split' in L.columns else L
        hit = [int(k) for k in PP if int(k) in Ls.index]
        print(f'labelled rows changed ({sp}):', len(hit))
        for k in hit:
            y = Ls.loc[k].y; o = Ls.loc[k].answer; n = PP[str(k)]['value']
            print('   ', k, y, o, n, AC.row_mra(np.array([o]), np.array([y]))[0], AC.row_mra(np.array([n]), np.array([y]))[0])
    # scenario table for the changed rows: score of the old vs new answer over possible truths
    D = pd.read_csv(f'{R}/submissions/{SUB}_detail.csv').set_index('Unnamed: 0')
    ys = np.array([18, 19.44, 20, 21, 21.6, 22, 23, 23.9, 24.3, 25, 27.1, 28.6])
    for k, x in list(PT.items())[:1]:
        o = float(D.loc[int(k)].answer); n = x['value']
        print('scenario row', k, 'truth', list(ys))
        print('   old', o, list(np.round(AC.row_mra(np.full(len(ys), o), ys), 2)))
        print('   new', n, list(np.round(AC.row_mra(np.full(len(ys), n), ys), 2)))
