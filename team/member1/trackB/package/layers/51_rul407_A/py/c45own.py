import sys, os, re, json
import numpy as np, pandas as pd
R = os.path.abspath(os.environ["RA_Q"]); WO = os.path.abspath(os.environ["RA_OUT"])
BUILD = sys.argv[1]
FAC = {'house', 'building', 'barn', 'shed', 'cabin', 'cottage', 'hut', 'garage', 'facade', 'wall', 'fence', 'gate'}
NEG = re.compile(r'\b(short|shorter|thick|thickness|depth|deep|long|longer|length)\b', re.I)
STRONG = ('2d_ratio', '3d_calib')


def head(ph):
    w = re.findall(r"[a-z]+", (ph or '').lower())
    return w[-1] if w else ''


def run(split):
    det = pd.read_csv(f'{R}/submissions/{BUILD}_detail.csv', index_col=0)
    raw = pd.read_csv(f'{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv', index_col=0)
    P = {}
    for ln in open(f'{R}/geo/out/parse_{split}_v2.jsonl', encoding='utf-8'):
        j = json.loads(ln); P[int(j['id'])] = j
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
        if src == 'var_ans_plain':
            print(split, i, vid, 't162'); continue
            base = vid.replace('_segmented', '')
            ql = qtext.strip().lower()
            hit = [k for k, (b, qq) in newplain.items() if b == base and qq == ql]
            if hit:
                out[str(int(i))] = {'value': out[hit[0]]['value'], 'why': 't163'}
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
        elif qq.get('frames') is not None: lst = qq['frames']
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
        elif src in ('dis_size_gm', 'rul263_unstable_gm', 'rul263_dimgm') and ok_c:
            new = np.sqrt(g2 * c)
        else:
            new = None
        if new is None or abs(np.log(new / det.answer[i])) < np.log(1.02): continue
        out[str(int(i))] = {'value': float(new), 'why': 't161'}
        newplain[str(int(i))] = (vid.replace('_segmented', ''), qtext.strip().lower())
        print(split, i, vid, qtext[:60], '|', src, round(det.answer[i], 4), '->', round(new, 4), '| g', round(g, 3), '->', round(g2, 3), 'c', c, 'bw/bh/rs', bw, bh, round(rs, 1), 'cv', round(cv, 2))
    json.dump(out, open(f'{WO}/c45own_{split}.json', 'w'), indent=1)
    print(split, 'rows', len(out))


run('test')
