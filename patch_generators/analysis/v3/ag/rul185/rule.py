"""rul185: two same-class movers too small for the detector (dist2 '#second' rows, target_low_det /
target_no_track, no geo) -> measure the pair distance from motion blobs in the full-res frames, scaled by the scene ruler.
Gates (all code conditions on inputs):
  q.kind == dist2 and q.target2 == '#second'; meas_why starts with target_low_det / target_no_track; geo NaN;
  answer_source == caw_nogeo; priors[0].kind == size; geo/out/scene_ruler_<split>.json has ok=True for the video;
  asked time (q.t, or 0 when q.when == 'start') inside the clip (N-1)/fps (dataset fps).
Measurement: frames registered to frame 0 (ECC euclidean at 1/4 scale), background = median of registered frames
(stride 4), foreground = max-channel |f - bg| > 40, 4 % border masked, open 2x2 / dilate 3x3, blobs >= 15 px.
Blobs on every STEP-th frame are linked by nearest neighbour (gate JUMP px); tracks seen in >= 25 % of the sampled
frames whose position range >= MOVE px are movers.  Exactly two movers present at the asked time (+-3 samples) ->
distance = median over those samples of |c1 - c2| (px) * prior_value / ruler_px.  Otherwise no fire.
usage: python rule.py [sub] [split ...]
"""
import sys, json, os, numpy as np, pandas as pd, cv2
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
SUB = sys.argv[1] if len(sys.argv) > 1 else 'sub_xt_Nzz27'
SPLITS = sys.argv[2:] or ['val', 'test']
TAG = 'rul185'
RAW = 'geo_v4ntrj5sQ2srngsr_d_vw'
STEP, THR, MINA, JUMP, MOVE, PERSIST = 3, 40, 15, 25, 12, 0.25
cv2.setNumThreads(4)


def load_frames(path):
    cap = cv2.VideoCapture(path); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    return fr


def movers(path):
    fr = load_frames(path)
    H, W = fr[0].shape[:2]
    ref = cv2.cvtColor(fr[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
    s = 0.25; refs = cv2.resize(ref, None, fx=s, fy=s)

    def reg(f):
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32), None, fx=s, fy=s)
        w = np.eye(2, 3, dtype=np.float32)
        try:
            _, w = cv2.findTransformECC(refs, g, w, cv2.MOTION_EUCLIDEAN,
                                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5), None, 5)
            w[:, 2] /= s
        except cv2.error:
            pass
        return cv2.warpAffine(f, w, (W, H), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    idx = list(range(0, len(fr), STEP))
    RG = {i: reg(fr[i]) for i in idx}
    bg = np.median(np.stack([RG[i] for i in range(0, len(fr), 4) if i in RG] or [RG[i] for i in idx]), 0).astype(np.float32)
    mx, my = int(0.04 * W), int(0.04 * H)
    det = {}
    for i in idx:
        d = np.abs(RG[i].astype(np.float32) - bg).max(2)
        m = (d > THR).astype(np.uint8); m[:my] = 0; m[-my:] = 0; m[:, :mx] = 0; m[:, -mx:] = 0
        m = cv2.dilate(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)), np.ones((3, 3), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        det[i] = [cen[k] for k in range(1, n) if st[k, 4] >= MINA]
    # nearest-neighbour linking
    tracks = []   # list of dict(f -> c)
    for i in idx:
        used = set()
        for tr in tracks:
            lf = max(tr)
            if i - lf > 3 * STEP: continue
            c0 = tr[lf]; best, bd = None, JUMP * (i - lf) / STEP
            for k, c in enumerate(det[i]):
                if k in used: continue
                dd = np.hypot(*(c - c0))
                if dd <= bd: best, bd = k, dd
            if best is not None:
                tr[i] = det[i][best]; used.add(best)
        for k, c in enumerate(det[i]):
            if k not in used: tracks.append({i: c})
    mv = []
    for tr in tracks:
        P = np.array(list(tr.values()))
        rng = np.hypot(*(P.max(0) - P.min(0)))
        if len(tr) >= PERSIST * len(idx) and rng >= MOVE:
            mv.append(tr)
    return mv, len(fr), W, H


def run(split):
    D = pd.read_csv(f'{R}/submissions/{SUB}_{"valdetail" if split == "val" else "detail"}.csv', index_col=0)
    raw = pd.read_csv(f'{R}/preds/{split}_{RAW}_raw.csv', index_col=0)
    P = {int(json.loads(l)['id']): json.loads(l) for l in open(f'{R}/geo/out/parse_{split}_v2.jsonl')}
    SR = json.load(open(f'{R}/geo/out/scene_ruler_{split}.json')) if os.path.exists(f'{R}/geo/out/scene_ruler_{split}.json') else {}
    cache, patch, rep = {}, {}, []
    for i in D.index:
        p = P[i]; q = p['q']; pr = p['priors'][0] if p['priors'] else {}
        why = str(raw.meas_why.get(i, ''))
        if not (q.get('kind') == 'dist2' and str(q.get('target2')) == '#second'): continue
        if not (why.startswith('target_low_det') or why.startswith('target_no_track')): continue
        if np.isfinite(D.g.get(i, np.nan)) or str(D.answer_source[i]) != 'caw_nogeo': continue
        if pr.get('kind') != 'size' or not pr.get('value_si'): continue
        vid = D.video_id[i]; sr = SR.get(vid)
        if not (sr and sr.get('ok')): rep.append((i, 'no_ruler')); continue
        t = q.get('t')
        if t is None and q.get('when') == 'start': t = 0.0
        if t is None: rep.append((i, 'no_time')); continue
        vp = str(raw.video_path[i]).replace(f'/{split}_480p/', f'/{split}/')
        if not os.path.exists(vp): vp = str(raw.video_path[i])
        if vp not in cache: cache[vp] = movers(vp)
        mv, N, W, H = cache[vp]
        fps = float(p.get('fps') or 30.0)
        if t > (N - 1) / fps + 1e-6: rep.append((i, 'beyond_clip')); continue
        fi = t * fps
        ds = []
        for f in sorted({f for tr in mv for f in tr if abs(f - fi) <= 3 * STEP}):
            here = [tr[f] for tr in mv if f in tr]
            if len(here) == 2 and np.hypot(*(here[0] - here[1])) > 3: ds.append(np.hypot(*(here[0] - here[1])))
        if len(mv) < 2 or not ds:
            rep.append((i, f'movers={len(mv)}')); continue
        if len({id(tr) for tr in mv if any(abs(f - fi) <= 3 * STEP for f in tr)}) != 2:
            rep.append((i, f'not_two_movers({len(mv)})')); continue
        px = float(np.median(ds))
        ruler_px = sr['px'] * (W / sr['W'])
        val = px * pr['value_si'] / ruler_px
        u = str(q.get('unit') or 'm')
        val = val / {'m': 1, 'cm': 0.01, 'mm': 0.001, 'km': 1000}.get(u, 1)
        patch[str(i)] = {'value': float(val), 'why': f'{TAG}: motion-blob pair {px:.1f}px / ruler {ruler_px:.1f}px x {pr["value_si"]} (CaW {D.cs[i]:.4g} dropped)'}
        rep.append((i, f'fire {val:.3f} (old {D.answer[i]:.3f}, movers {len(mv)}, n {len(ds)})'))
    json.dump(patch, open(f'{R}/cache/tmp/ag_{TAG}_patch_{split}.json', 'w'), indent=0)
    print(split, 'patch rows', len(patch)); [print('  ', r) for r in rep]


if __name__ == '__main__':
    for sp in SPLITS: run(sp)
