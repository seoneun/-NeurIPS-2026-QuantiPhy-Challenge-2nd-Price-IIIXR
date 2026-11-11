"""rul109: fast pass-by mover speed (and gated accel) from horizontal patch flow,
scaled by a stated part-size prior detected on the same mover.
Writes cache/tmp/ag_c63passflow_patch_{split}.json (+ _scan_{split}.csv)."""
import sys, json, re, math
import numpy as np, pandas as pd, cv2
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + '/geo'); import track_common as tc

STOP = set('the a an of on in this that these those at to with and or not its it is are'.split())


def toks(s):
    out = []
    for w in re.findall(r'[a-z]+', str(s or '').lower()):
        if w in STOP:
            continue
        if len(w) > 3 and w.endswith('s') and not w.endswith('ss'):
            w = w[:-1]
        out.append(w)
    return out


def cv(a):
    a = np.asarray(a, float)
    return float(a.std() / a.mean()) if len(a) and a.mean() > 0 else np.nan


def prior_scale(tj, raw_phrase, value, W):
    """mpp from prior-role detections whose phrase tokens all appear in the raw prior phrase."""
    rt = set(toks(raw_phrase)); ws = []; used = []
    for qs in ('queries', 'queries_d'):
        for q in tj.get(qs) or []:
            if 'prior' not in q.get('roles', []):
                continue
            pt = toks(q.get('phrase', ''))
            if not pt or not set(pt) <= rt:
                continue
            used.append(qs + ':' + q['key'])
            for s, b in zip(q.get('det_scores') or [], q.get('det_boxes') or []):
                if b is None or s is None or s < 0.7:
                    continue
                w = b[2] - b[0]
                if w > 0.5 * W or b[0] <= 2 or b[2] >= W - 2:   # too wide / cut by the frame edge
                    continue
                ws.append(w)
    if len(ws) < 3:
        return np.nan, dict(n=len(ws), used=used)
    m = np.median(ws); inl = [w for w in ws if abs(w - m) <= 0.10 * m]
    c = cv(inl)
    ok = len(inl) >= 3 and len(inl) >= 0.5 * len(ws) and c < 0.15
    return (value / np.median(inl) if ok else np.nan), dict(n=len(ws), ninl=len(inl), wmed=float(np.median(inl)), cv=round(c, 3), used=used)


def ncc_shift(A, B, y0, x0, h, w, rng=250):
    T = A[y0:y0 + h, x0:x0 + w]
    if T.std() < 3:
        return None
    lo = max(0, x0 - rng); hi = min(B.shape[1], x0 + w + rng)
    res = cv2.matchTemplate(B[y0:y0 + h, lo:hi], T, cv2.TM_CCOEFF_NORMED)[0]
    k = int(np.argmax(res))
    return (lo + k - x0, float(res[k]))


NPATCH = int(__import__("os").environ.get("NPATCH", 15))


def flow(G, band, rng=250):
    """Per pair and patch NCC curves over shifts [-rng, rng]; a global constant-acceleration shift model
    s(f) = a + b (f - fc) is fitted to the summed curves (robust to periodic-structure aliasing),
    then each pair is accepted when >= 2 patches peak (NCC >= 0.8) within 3 px of the model and of each other."""
    H, W = G[0].shape
    yc = int(round((band[0] + band[1]) / 2)); y0 = int(np.clip(yc - 50, 0, H - 100))
    xs = [int(x) for x in np.linspace(0.05 * W, 0.95 * W - 100, NPATCH)]
    ry0 = 4; rh = int(0.15 * H) - ry0
    rx = sorted(range(0, W - 100, 50), key=lambda x: -np.median([g[ry0:ry0 + rh, x:x + 100].std() for g in G]))[:2]
    nF = len(G) - 1; S = 2 * rng + 1
    C = np.full((nF, len(xs), S), -1.0, np.float32); mov = np.zeros((nF, len(xs)), bool); ref = []
    for f in range(nF):
        for k, x0 in enumerate(xs):
            T = G[f][y0:y0 + 100, x0:x0 + 100]
            if T.std() < 3:
                continue
            lo = max(0, x0 - rng); hi = min(W, x0 + 100 + rng)
            res = cv2.matchTemplate(G[f + 1][y0:y0 + 100, lo:hi], T, cv2.TM_CCOEFF_NORMED)[0]
            C[f, k, lo - x0 + rng: lo - x0 + rng + len(res)] = res
            mov[f, k] = C[f, k, rng] < 0.95          # not static background
        rr = [ncc_shift(G[f], G[f + 1], ry0, x, rh, 100) for x in rx]
        rr = [o for o in rr if o is not None and o[1] >= 0.8]
        ref.append(min(abs(o[0]) for o in rr) if rr else None)
    from scipy.ndimage import maximum_filter1d
    Cm = maximum_filter1d(np.clip(C, 0, 1), 5, axis=2)
    M = (Cm * mov[:, :, None]).sum(1)                     # (nF, S)
    fr = np.arange(nF); act = mov.sum(1) >= 2
    fc = fr[act].mean() if act.any() else nF / 2
    best = {1: (-1, None), -1: (-1, None)}
    for b in np.arange(-8, 8.01, 0.25):
        for a in range(-rng, rng + 1):
            if abs(a) <= 20:
                continue
            s = np.rint(a + b * (fr - fc)).astype(int)
            ok = act & (np.abs(s) <= rng)
            sc = M[fr[ok], s[ok] + rng].sum()
            sg = 1 if a > 0 else -1
            if sc > best[sg][0]:
                best[sg] = (sc, (a, b))
    sg = 1 if best[1][0] >= best[-1][0] else -1
    a, b = best[sg][1]; margin = best[-sg][0] / max(best[sg][0], 1e-6)
    pairs = []
    for f in range(nF):
        sm = a + b * (f - fc); pos = []
        for k in range(len(xs)):
            if not mov[f, k]:
                continue
            lo = int(round(sm)) - 3 + rng; hi = int(round(sm)) + 4 + rng
            if lo < 0 or hi > S:
                continue
            w = C[f, k, lo:hi]; j = int(np.argmax(w))
            # must be a local peak of this patch's curve and strong
            if w[j] >= 0.8 and 0 < j + lo < S - 1 and C[f, k, j + lo] >= C[f, k, j + lo - 1] and C[f, k, j + lo] >= C[f, k, j + lo + 1]:
                pos.append(j + lo - rng)
        agree = [p for p in pos if sum(abs(p - q) <= 3 for q in pos) >= 2]
        pairs.append(dict(f=f, acc=len(agree) >= 2, shift=float(np.median(agree)) if len(agree) >= 2 else np.nan,
                          ref=ref[f], model=round(sm, 1), npos=len(pos)))
    return pairs, dict(y0=y0, xs=xs, rx=rx, a=a, b=b, fc=float(fc), margin=round(float(margin), 3))


def main(split):
    P = {}
    for l in open(f'{R}/geo/out/parse_{split}_v2.jsonl'):
        j = json.loads(l); P[j['id']] = j
    det = pd.read_csv(f'{R}/submissions/sub_xt_Nzz13_' + ('detail' if split == 'test' else 'valdetail') + '.csv', index_col=0)
    raw = pd.read_csv(f'{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv', index_col=0)
    try:
        neg = set(int(k) for k in json.load(open(f'{R}/cache/tmp/ag_negprior_patch_{split}.json')))
    except Exception:
        neg = set()
    rows = []; patch = {}; vcache = {}
    for i, j in P.items():
        q = j['q']; cat = j.get('cat') or det.loc[i, 'cat']
        if q.get('kind') not in ('speed', 'accel') or q.get('t') is None or q.get('t1') is not None:
            continue
        if j.get('target_dkeys') or str(cat)[1:] != '2':                       # G1
            continue
        pr = (j.get('priors') or [{}])[0]
        if pr.get('kind') != 'size' or pr.get('dim') not in ('width', 'length') or not pr.get('value_si'):
            continue
        po = j.get('prior_obj') or {}
        rawp = po.get('raw') or pr.get('obj') or ''          # c63priorphrase: always the raw prior phrase
        tgt = toks(q.get('target') or (j.get('target') or {}).get('phrase'))
        if not tgt or tgt[-1] not in toks(rawp) or toks(rawp) == tgt:      # G2: prior is a part of the target
            continue
        vid = j['video_id']; fps = float(raw.loc[i, 'fps']); t = float(q['t'])
        tj = json.load(open(f'{R}/geo/out/tracks_{split}_v2sc/{vid}.json'))
        W = tj['W']; H = tj['H']
        tq = [qq for qq in tj['queries'] if 'target' in qq.get('roles', []) and toks(qq.get('phrase')) == tgt]
        tq = tq[0] if tq else None
        wcv = np.nan
        if tq is not None and tq.get('obj') is not None:
            tr = tj['objs'][str(tq['obj'])]
            ws = [s['st']['box'][2] - s['st']['box'][0] for fi, s in zip(tj['fidx'], tr)
                  if s is not None and s.get('st') and abs(fi / fps - t) <= 0.5]
            wcv = cv(ws) if len(ws) >= 2 else np.nan
        geo = raw.loc[i, 'geo_value']; why = str(raw.loc[i, 'meas_why'])
        g3a = bool(wcv > 0.5); g3b = bool(pd.isna(geo) and why not in ('nan', '', 'ok')); g3c = i in neg
        g3 = g3a or g3b or g3c
        rec = dict(id=i, vid=vid, cat=cat, kind=q['kind'], t=t, prior=rawp[:40], val=pr['value_si'], wcv=wcv,
                   g3=f"{int(g3a)}{int(g3b)}{int(g3c)}", why=why[:30], geo=geo,
                   old=det.loc[i, 'answer'], src=det.loc[i, 'answer_source'])
        if not g3:
            rows.append(rec); continue
        mpp, minfo = prior_scale(tj, rawp, pr['value_si'], W); rec.update(mpp=mpp, minfo=json.dumps(minfo)[:300])
        if vid not in vcache:
            Fr = tc.load_video(raw.loc[i, 'video_path']); Fr = Fr[0] if isinstance(Fr, tuple) else Fr
            G = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32) for f in Fr]
            bx = [b for b in (tq or {}).get('det_boxes') or [] if b is not None]
            band = (np.median([b[1] for b in bx]), np.median([b[3] for b in bx])) if bx else (0.3 * H, 0.7 * H)
            vcache[vid] = flow(G, band)
        pairs = vcache[vid][0]
        acc = [p for p in pairs if p['acc']]
        sg = np.sign([p['shift'] for p in acc]) if acc else np.array([])
        maj = 1 if (sg > 0).sum() >= (sg < 0).sum() else -1
        A = [p for p in acc if np.sign(p['shift']) == maj]
        refok = [p['ref'] is not None and p['ref'] < 2 for p in A]
        margin = vcache[vid][1]['margin']
        g4 = bool(len(A) >= 6 and len(A) >= 0.8 * len(acc) and np.median([abs(p['shift']) for p in A]) > 20
                  and len(refok) and np.mean(refok) >= 0.8 and margin <= 0.8)
        rec.update(margin=margin, nacc=len(acc), nA=len(A), refok=float(np.mean(refok)) if refok else np.nan, g4=g4,
                   shifts=' '.join(f"{p['f']}:{p['shift']:.0f}" for p in A))
        if g4 and np.isfinite(mpp):
            x = np.array([p['f'] + 0.5 for p in A]); y = np.array([abs(p['shift']) for p in A])
            (b, a), C = np.polyfit(x, y, 1, cov=True); seb = math.sqrt(C[0, 0])
            ft = t * fps
            inrange = bool((x.min() - 0.5 * fps) <= ft <= (x.max() + 0.5 * fps))
            v = abs(a + b * ft) * fps * mpp; acc_v = abs(b) * fps * fps * mpp
            rec.update(slope=b, se=seb, v=v, a=acc_v, inrange=inrange)
            new = None
            if q['kind'] == 'speed' and inrange:
                new = v
            if q['kind'] == 'accel' and inrange and len(A) >= 8 and abs(b) / seb >= 2:
                new = acc_v
            if new is not None:
                rec['new'] = new
                patch[str(i)] = dict(value=float(new), why=f"rul109: {q['kind']} from patch flow {len(A)} pairs, "
                                     f"shift {a + b * ft:.0f}px/f slope {b:.2f}+-{seb:.2f}, mpp {mpp:.4f} ({rawp})")
        rows.append(rec)
    df = pd.DataFrame(rows); df.to_csv(f'{R}/cache/tmp/ag_c63passflow_scan_{split}.csv', index=False)
    json.dump(patch, open(f'{R}/cache/tmp/ag_c63passflow_patch_{split}.json', 'w'), indent=1)
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 50)
    print(split, 'G1+G2 rows', len(df), 'G3', int((df.g3 != '000').sum()) if len(df) else 0, 'patched', len(patch))
    if len(df):
        print(df.drop(columns=[c for c in ('minfo', 'shifts') if c in df]).to_string())
        for c in ('minfo', 'shifts'):
            if c in df:
                print(df[['id', c]].dropna().to_string())
    for k, v in patch.items():
        print(k, v)
    for vid, (pr, geom) in vcache.items():
        print(vid, geom)
        for p in pr:
            print('  ', p)


if __name__ == '__main__':
    main(sys.argv[1])
