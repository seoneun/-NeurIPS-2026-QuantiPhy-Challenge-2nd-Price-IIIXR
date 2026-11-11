"""rul076: facade 'width' read on the rotated-rect MINOR axis, which for a wider-than-tall (or top/bottom-clipped)
facade box is the VERTICAL extent, i.e. geo reports the house height (or the clipped visible height) as its width.

geo (Track.ext_px) uses width = rr minor axis. For a facade object seen face-on, the asked width is the horizontal extent.
Gates (all label-free):
  - size row, parse q.dim == 'width'; question has none of short/thick/depth/deep/long/length;
  - facade noun: target head noun in FAC, or the question mentions eave(s);
  - the video asks no LENGTH of the same target phrase (rul291 gate 6);
  - box axis-aligned: |rr_long - bw| <= 10% bw;  minor axis IS the vertical extent: |rr_short - bh| <= 10% bh;
  - and one of: (eave question and bw >= 1.05 bh) | bw >= 1.3 bh | (>= 80% frames cut only at top/bottom and bw >= 1.3 rr_short);
  - simulation or internet rows only (no captured), not lane, current answer not a patch.
Action: g' = g * median(bw) / median(rr_short), re-fused like the row's own route:
  dis_geo / geo-only           -> g'
  agree_wgm (D2 size)          -> g'^0.7 c^0.3 if within 2x else g' (strong mode)
  agree_gm / agree_geo_kind    -> sqrt(g' c) if within 2x else g' (strong mode)
  dis_size_gm / d3size_*_gm / d3size_dimgm -> sqrt(g' c)
  var_ans_plain                -> the plain row's new answer when the plain row fires, else skip.
usage: tools/run_limited.sh --io python analysis/v3/ag/rul076/rul076.py [build]
writes cache/tmp/ag_c45wspan_patch_{test,val}.json"""
import sys, os, re, json
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
BUILD = sys.argv[1] if len(sys.argv) > 1 else 'sub_xt_Nzz2'
FAC = {'house', 'building', 'barn', 'shed', 'cabin', 'cottage', 'hut', 'garage', 'facade', 'wall', 'fence', 'gate'}
NEG = re.compile(r'\b(short|shorter|thick|thickness|depth|deep|long|longer|length)\b', re.I)
STRONG = ('2d_ratio', '3d_calib')


def head(ph):
    w = re.findall(r"[a-z]+", (ph or '').lower())
    return w[-1] if w else ''


def run(split):
    det = pd.read_csv(f'{R}/submissions/{BUILD}_{"detail" if split == "test" else "valdetail"}.csv', index_col=0)
    raw = pd.read_csv(f'{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv', index_col=0)
    P = {}
    for ln in open(f'{R}/geo/out/parse_{split}_v2.jsonl', encoding='utf-8'):
        j = json.loads(ln); P[int(j['id'])] = j
    lenq = {}
    for i, pj in P.items():
        q = pj.get('q') or {}
        if q.get('kind') == 'size' and q.get('dim') == 'length':
            vid = str(raw.video_id.get(i, ''))
            lenq.setdefault(vid, set()).add(((pj.get('target') or {}).get('phrase') or '').lower())
    cache, out, newplain = {}, {}, {}
    order = sorted(det.index, key=lambda i: ('_segmented' in str(det.video_id[i]), i))
    for i in order:
        pj = P.get(int(i)) or {}; q = pj.get('q') or {}
        if q.get('kind') != 'size' or q.get('dim') != 'width': continue
        vid = str(det.video_id[i]); qtext = str(raw.question.get(i, ''))
        if vid.startswith('captured') or 'lane' in qtext.lower() or NEG.search(qtext): continue
        src = str(det.answer_source[i])
        if src.startswith('patch'): continue
        ph = ((pj.get('target') or {}).get('phrase') or '')
        eave = bool(re.search(r'\beaves?\b', qtext, re.I))
        if not (head(ph) in FAC or eave): continue
        if ph.lower() in lenq.get(vid, set()): continue
        if src == 'var_ans_plain':
            base = vid.replace('_segmented', '')
            ql = qtext.strip().lower()
            hit = [k for k, (b, qq) in newplain.items() if b == base and qq == ql]
            if hit:
                out[str(int(i))] = {'value': out[hit[0]]['value'], 'why': f'rul076: copy of plain row {hit[0]}; was var_ans_plain {det.answer[i]:.4g}'}
                print(split, i, vid, qtext[:60], '| copy', hit[0], '->', round(out[hit[0]]['value'], 4))
            continue
        g = float(pd.to_numeric(raw.geo_value.get(i), errors='coerce')); c = float(det.cs[i])
        if not (np.isfinite(g) and g > 0): continue
        tv = str(raw.meas_tracks.get(i))
        cands = [f'tracks_{split}'] if tv == 'v1' else [f'tracks_{split}_v2sc', f'tracks_{split}_v2', f'tracks_{split}']
        J = None
        for dd in cands:
            p = f'{R}/geo/out/{dd}/{vid}.json'
            if os.path.exists(p):
                if p not in cache: cache[p] = json.load(open(p))
                J = cache[p]; break
        if J is None: continue
        wh = (pj.get('target') or {}).get('where') or []
        qs = {qq['key']: qq for qq in J.get('queries') or []}
        qq = qs.get(f'{ph}|{",".join(wh)}') or next((qs[k] for k in qs if k.split('|')[0] == ph), None)
        if qq is None: continue
        if qq.get('obj') is not None: lst = J['objs'][str(qq['obj'])]
        elif qq.get('frames') is not None: lst = qq['frames']          # v1 track files keep frames on the query
        else: continue
        st = [e['st'] for e in lst if e and e.get('st') and e['st'].get('rr')]
        if len(st) < 5: continue
        W, H = J['W'], J['H']
        bx = np.array([s_['box'] for s_ in st], float); rr = np.array([s_['rr'][:2] for s_ in st], float)
        bw = float(np.median(bx[:, 2] - bx[:, 0])); bh = float(np.median(bx[:, 3] - bx[:, 1]))
        rl = float(np.nanmedian(np.maximum(rr[:, 0], rr[:, 1]))); rs = float(np.nanmedian(np.minimum(rr[:, 0], rr[:, 1])))
        cv = ((bx[:, 1] <= 1.5) | (bx[:, 3] >= H - 2)).mean(); ch = ((bx[:, 0] <= 1.5) | (bx[:, 2] >= W - 2)).mean()
        if abs(rl - bw) > 0.10 * bw or abs(rs - bh) > 0.10 * bh: continue
        if not ((eave and bw >= 1.05 * bh) or bw >= 1.3 * bh or (cv >= 0.8 and ch <= 0.2 and bw >= 1.3 * rs)): continue
        g2 = g * bw / rs
        mode = str(det['mode'][i]); ok_c = np.isfinite(c) and c > 0
        within = ok_c and max(g2, c) / min(g2, c) <= 2.0
        if src in ('dis_geo', 'geo_cawinvalid', 'sruler_geo'):
            new = g2
        elif src == 'agree_wgm':
            new = g2 ** 0.7 * c ** 0.3 if within else (g2 if mode.startswith(STRONG) else None)
        elif src in ('agree_gm', 'agree_geo_kind'):
            new = np.sqrt(g2 * c) if within else (g2 if mode.startswith(STRONG) else None)
        elif src in ('dis_size_gm', 'd3size_unstable_gm', 'd3size_dimgm') and ok_c:
            new = np.sqrt(g2 * c)
        else:
            new = None
        if new is None or abs(np.log(new / det.answer[i])) < np.log(1.02): continue
        out[str(int(i))] = {'value': float(new), 'why': f'rul076: facade width read on the minor axis (= vertical extent {rs:.0f}px); horizontal span {bw:.0f}px -> geo {g:.4g}->{g2:.4g}; {src} re-fused; was {det.answer[i]:.4g}'}
        newplain[str(int(i))] = (vid.replace('_segmented', ''), qtext.strip().lower())
        print(split, i, vid, qtext[:60], '|', src, round(det.answer[i], 4), '->', round(new, 4), '| g', round(g, 3), '->', round(g2, 3), 'c', c, 'bw/bh/rs', bw, bh, round(rs, 1), 'cv', round(cv, 2))
    json.dump(out, open(f'{R}/cache/tmp/ag_c45wspan_patch_{split}.json', 'w'), indent=1)
    print(split, 'rows', len(out))


for sp in ('test', 'val'):
    run(sp)
