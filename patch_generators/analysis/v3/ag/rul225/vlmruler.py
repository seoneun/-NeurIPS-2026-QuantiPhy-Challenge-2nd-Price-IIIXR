"""rul225: copy of rul175 with the mode gate extended to vlm_2d_ratio / 2d_ratio_scene / 2d_vlmruler (c85 gated meas_mode == 2d_ratio only). prior-speed ruler corrected for occlusion clipping (2D).
A 2d_ratio row whose ruler is a stated SPEED of a moving object takes mpp = v_prior / v_px with v_px the prior track's
centre speed. When the prior object is partly hidden (passes under/behind an occluder) for part of the track, its box is
clipped: one edge stalls, the box shrinks along the motion axis and the centre moves slower than the object. Gates (code only,
the prior object's own track): (1) mode 2d_ratio, prior kind speed without time; (2) motion axis = dominant axis of the LS
centre displacement; ext = box extent along it; >= 8% of frames have ext < 0.85*median while neither axis edge touches the
frame border; (3) r = (faster edge LS slope)/(centre LS slope) >= 1.12; (4) v_fit = centre LS slope on frames with
ext >= 0.9*median, |v_fit/v_edge - 1| < 0.06. Then k = v_used/v_fit (v_used = prior/meas_mpp) rescales every 2d_ratio row of
the video on that ruler (same meas_mpp); re-fused by the pipeline rule.
usage: python occvel.py -> cache/tmp/ag_d03occ2_vlmruler_patch_{test,val}.json"""
import sys, json, os
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
RAW = 'geo_v4ntrj5sQ2srngsr_d_vw'
SUB = 'sub_xt_Nzz36'
W_SIZE = 0.65
VERB = os.environ.get('VERB')


def load_parse(split):
    P = {}
    for l in open(f'{R}/geo/out/parse_{split}_v2.jsonl'):
        j = json.loads(l); P[j['id']] = j
    return P


def tdir(split, mt):
    mt = str(mt)
    return f'{R}/geo/out/tracks_{split}' + ('' if mt in ('v1', 'nan', '') else '_' + mt)


def find_query(vj, phrase):
    qs = vj.get('queries') or []
    for q in qs:
        if q.get('phrase') == phrase and not q.get('where'):
            return q
    for q in qs:
        if q.get('phrase') == phrase:
            return q
    return None


def series(vj, q):
    fps = float(vj['fps']); fidx = vj['fidx']
    t, b = [], []
    frames = q.get('frames')
    if frames is None:
        frames = (vj.get('objs') or {}).get(str(q.get('obj'))) or []
    for f, fr in zip(fidx, frames):
        st = fr.get('st') if fr else None
        if st and st.get('box'):
            t.append(f / fps); b.append(st['box'])
    return np.array(t, float), np.array(b, float)


def analyse(vj, q):
    t, b = series(vj, q)
    if len(t) < 12:
        return dict(ok=False, why='few')
    H, W = vj['H'], vj['W']
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    vx, vy = np.polyfit(t, cx, 1)[0], np.polyfit(t, cy, 1)[0]
    ax = 0 if abs(vx) >= abs(vy) else 1
    lo, hi = b[:, ax], b[:, ax + 2]
    lim = W if ax == 0 else H
    c = (lo + hi) / 2
    ext = hi - lo
    em = float(np.median(ext))
    border = (lo <= 2) | (hi >= lim - 3)
    clip = (ext < 0.85 * em) & ~border
    fclip = float(clip.mean())
    sc = np.polyfit(t, c, 1)[0]
    sl, sh = np.polyfit(t, lo, 1)[0], np.polyfit(t, hi, 1)[0]
    sgn = np.sign(sc) if sc != 0 else 1.0
    v_c = abs(sc)
    v_edge = max(sl * sgn, sh * sgn)
    r = v_edge / v_c if v_c > 0 else np.nan
    full = ext >= 0.9 * em
    v_fit = abs(np.polyfit(t[full], c[full], 1)[0]) if full.sum() >= 6 else np.nan
    agree = abs(v_fit / v_edge - 1) if v_edge > 0 else np.nan
    ok = fclip >= 0.08 and r >= 1.12 and agree < 0.06
    return dict(ok=bool(ok), ax=ax, n=len(t), fclip=fclip, v_c=v_c, v_edge=v_edge, r=r, v_fit=v_fit, agree=agree, em=em,
                nborder=int(border.sum()), why='')


def fuse(g, cs, kind):
    """pipeline fusion: size agree (0.5-2x) -> g^0.65 c^0.35 (agree_w size:0.65:strong), size disagree -> gm
    (dis_size_gm all); dist2/avgspeed -> geo (geo_kinds); other agree -> gm; other disagree -> geo (2d_ratio strong)"""
    if not np.isfinite(cs) or cs <= 0:
        return g, 'geo'
    if kind in ('dist2', 'avgspeed'):
        return g, 'geo_kind'
    rat = g / cs
    if 0.5 <= rat <= 2:
        if kind == 'size':
            return g ** W_SIZE * cs ** (1 - W_SIZE), 'agree_wgm'
        return float(np.sqrt(g * cs)), 'agree_gm'
    if kind == 'size':
        return float(np.sqrt(g * cs)), 'dis_size_gm'
    return g, 'dis_geo'


FUSABLE = ('agree_wgm', 'agree_gm', 'dis_geo', 'geo', 'dis_size_gm', 'agree_geo_kind', 'geo_only', 'meas')


def run(split):
    raw = pd.read_csv(f'{R}/preds/{split}_{RAW}_raw.csv').set_index('Unnamed: 0')
    det = pd.read_csv(f'{R}/submissions/{SUB}_{"detail" if split == "test" else "valdetail"}.csv').set_index('Unnamed: 0')
    P = load_parse(split)
    patch = {}
    cache = {}
    m2 = raw.meas_mode.astype(str).isin(['vlm_2d_ratio', '2d_ratio_scene', '2d_vlmruler'])
    for i in raw.index[m2]:
        p = P.get(int(i))
        if not p or not p.get('priors'):
            continue
        pr = p['priors'][0]
        if pr.get('kind') != 'speed' or pr.get('t') is not None or pr.get('t1') is not None or pr.get('gravity'):
            continue
        mpp = float(raw.meas_mpp.get(i)); g0 = float(raw.geo_value.get(i))
        if not (np.isfinite(mpp) and mpp > 0 and np.isfinite(g0) and g0 > 0):
            continue
        vid = raw.video_id[i]; mt = raw.meas_tracks.get(i)
        phrase = (p.get('prior_obj') or {}).get('phrase') or pr.get('obj')
        key = (vid, str(mt), phrase)
        if key not in cache:
            fn = f'{tdir(split, mt)}/{vid}.json'
            if not os.path.exists(fn):
                cache[key] = dict(ok=False, why='notrack')
            else:
                vj = json.load(open(fn)); q = find_query(vj, phrase)
                cache[key] = analyse(vj, q) if q else dict(ok=False, why='noquery')
            a = cache[key]
            if VERB and (a['ok'] or a.get('r', 0) > 1.05 or a.get('fclip', 0) >= 0.08):
                print(f'[{split}] PRIOR {vid} {mt} {phrase} ' + ' '.join(f'{k}={v:.3g}' if isinstance(v, float) else f'{k}={v}' for k, v in a.items()))
        a = cache[key]
        if not a['ok']:
            continue
        v_used = pr['value_si'] / mpp
        k = v_used / a['v_fit']
        kind = p['q']['kind']
        g1 = g0 * k
        cs = float(det.cs.get(i)) if pd.notna(det.cs.get(i)) else np.nan
        src = str(det.answer_source.get(i)); old = float(det.answer.get(i))
        # the CaW value the fusion actually used (the detail's cs column can differ from the fused cs): recover it from the
        # current answer of a fused row, then re-fuse with the corrected geo
        if src == 'agree_wgm':
            cs = (old / g0 ** W_SIZE) ** (1 / (1 - W_SIZE))
        elif src in ('agree_gm', 'dis_size_gm'):
            cs = old ** 2 / g0
        ans, how = fuse(g1, cs, kind)
        if src == "agree_gm" and how == "agree_wgm":   # vlm_2d_ratio rows fuse size with a plain gm: keep the row's own fusion
            ans, how = float(np.sqrt(g1 * cs)), "agree_gm"
        fusable = src in FUSABLE
        print(f"[{split}] {i} {vid} {kind} {p['question'][:60]} | v_used {v_used:.2f} v_c {a['v_c']:.2f} v_edge {a['v_edge']:.2f} "
              f"v_fit {a['v_fit']:.2f} r {a['r']:.3f} fclip {a['fclip']:.2f} agree {a['agree']:.3f} k {k:.3f} | g {g0:.4g}->{g1:.4g} "
              f"cs {cs} src {src} old {old:.4g} -> {ans:.4g} ({how}) {'PATCH' if fusable else 'SKIP(src)'}")
        if fusable and abs(ans / old - 1) > 1e-6:
            patch[str(i)] = dict(value=float(ans), why=f"rul225: prior speed ruler occlusion-clipped (edge/centre {a['r']:.2f}, "
                                                         f"{a['fclip']:.0%} clipped frames), k={k:.3f}, geo {g0:.3g}->{g1:.3g}, {how}")
    print(f'[{split}] prior-speed rulers analysed {len(cache)}, patched {len(patch)}')
    return patch


if __name__ == '__main__':
    for sp in ('test', 'val'):
        p = run(sp)
        json.dump(p, open(f'{R}/cache/tmp/ag_d03occ2_vlmruler_patch_{sp}.json', 'w'), indent=1)
