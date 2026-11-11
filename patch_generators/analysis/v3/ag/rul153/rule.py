"""rul153: wrong-axis ruler for a stated LENGTH prior on a masted vessel (sailboat/yacht/...).

A sailboat's box is taller than wide (the mast), so a 2D ruler that maps the stated hull LENGTH onto the box height
over-scales metres-per-pixel by h/w.  When the vessel sails horizontally, its hull length lies along the box width.
Gates: (1) prior dim=length and prior/target phrase names a masted vessel (sail*, yacht, schooner, sloop, ketch);
(2) meas_mode 2d_ratio/vlm_2d_ratio with finite meas_mpp; (3) prior/mpp == median box height within 3% and h>1.1w;
(4) prior track moves horizontally: sum|dx| > 3 sum|dy|.
Output: k=h/w; every same-video row on this ruler (mpp within 1%) gets g*k, re-fused (speed: dis_geo if >2x from CaW
else gm; size: agree_wgm w=0.7 within 2x else geo). _segmented var_ans_plain copies follow their plain row.
"""
import os, sys, json, re
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
SUB = sys.argv[1] if len(sys.argv) > 1 else 'sub_xt_Nzz24'
RAW = 'geo_v4ntrj5sQ2srngsr_d_vw'
MAST = re.compile(r'\b(sail\w*|yacht|schooner|sloop|ketch)\b', re.I)
TDIRS = {'v1': ['', '_v1'], 'v2': ['_v2sc', '_v2'], 'v3': ['_v3nt', '_v3'], 'v3nt': ['_v3nt']}
SPEEDK = ('avgspeed', 'speed', 'instspeed', 'maxspeed', 'velocity')


def track_states(j, phrase):
    qs = j.get('queries')
    if isinstance(qs, list):
        for q in qs:
            if q.get('phrase') == phrase or q.get('key', '').split('|')[0] == phrase:
                if 'obj' in q and q['obj'] is not None:
                    L = j.get('objs', {}).get(str(q['obj'])) or []
                    return [x['st'] for x in L if x and x.get('st')]
                return [x['st'] for x in q.get('frames', []) if x and x.get('st')]
        return None
    if isinstance(qs, dict):
        for key, o in qs.items():
            if key.split('|')[0] == phrase:
                L = j.get('objs', {}).get(str(o)) or j.get('objs', {}).get(o) or []
                return [x['st'] for x in L if x and x.get('st')]
    return None


def fuse(kind, g, cs):
    if not (pd.notna(cs) and cs > 0):
        return g, 'geo'
    r = max(g / cs, cs / g)
    if kind == 'size':
        if r <= 2:
            return float(np.exp(0.7 * np.log(g) + 0.3 * np.log(cs))), 'agree_wgm'
        return g, 'dis_geo'
    if r > 2:
        return g, 'dis_geo'
    return float(np.sqrt(g * cs)), 'agree_gm'


def run(split):
    det = pd.read_csv(f'{R}/submissions/{SUB}_{"detail" if split == "test" else "valdetail"}.csv', index_col=0)
    raw = pd.read_csv(f'{R}/preds/{split}_{RAW}_raw.csv', index_col=0)
    P = {}
    for l in open(f'{R}/geo/out/parse_{split}_v2.jsonl'):
        j = json.loads(l); P[j['id']] = j
    out, log = {}, []
    for v, grp in det.groupby('video_id'):
        ids = [i for i in grp.index if i in P and i in raw.index]
        done = set()
        for i in ids:
            if str(det.loc[i, 'answer_source']) == 'var_ans_plain':
                continue
            pr = (P[i].get('priors') or [None])[0]
            if not pr or pr.get('kind') != 'size' or pr.get('dim') != 'length':
                continue
            ph = (P[i].get('prior_obj') or {}).get('phrase') or ''
            tg = (P[i].get('target') or {}).get('phrase') or ''
            if not (MAST.search(ph) or MAST.search(tg)):
                continue
            mode, mpp, ver = raw.loc[i, 'meas_mode'], raw.loc[i, 'meas_mpp'], str(raw.loc[i, 'meas_tracks'])
            rec = dict(split=split, v=v, id=int(i), ph=ph, tg=tg, mode=mode, mpp=mpp, ver=ver)
            if mode not in ('2d_ratio', 'vlm_2d_ratio') or not (pd.notna(mpp) and float(mpp) > 0):
                log.append(rec); continue
            key = (ph, ver, round(float(mpp), 6))
            if key in done:
                continue
            done.add(key)
            S = float(pr['value_si']); hpx = S / float(mpp)
            best = None
            for suf in TDIRS.get(ver, ['_' + ver]):
                p = f'{R}/geo/out/tracks_{split}{suf}/{v}.json'
                if not os.path.exists(p):
                    continue
                st = track_states(json.load(open(p)), ph)
                if not st or len(st) < 5:
                    continue
                b = np.array([s['box'] for s in st], float)
                bw, bh = np.median(b[:, 2] - b[:, 0]), np.median(b[:, 3] - b[:, 1])
                c = np.array([s['c'] for s in st], float)
                dx, dy = np.abs(np.diff(c[:, 0])).sum(), np.abs(np.diff(c[:, 1])).sum()
                cand = dict(suf=suf, n=len(st), bw=round(bw, 1), bh=round(bh, 1), hpx=round(hpx, 1), dx=round(dx, 1), dy=round(dy, 1))
                if abs(bh / hpx - 1) < 0.03:
                    best = cand; break
                rec.setdefault('cands', []).append(cand)
            rec['track'] = best
            log.append(rec)
            if best is None:
                continue
            g3 = best['bh'] > 1.1 * best['bw']
            g4 = best['dx'] > 3 * best['dy']
            rec.update(g3=g3, g4=g4)
            if not (g3 and g4):
                continue
            k = best['bh'] / best['bw']
            for r in ids:
                if str(det.loc[r, 'answer_source']) == 'var_ans_plain':
                    continue
                if raw.loc[r, 'meas_mode'] not in ('2d_ratio', 'vlm_2d_ratio'):
                    continue
                m2 = raw.loc[r, 'meas_mpp']
                if not (pd.notna(m2) and abs(float(m2) / float(mpp) - 1) < 0.01):
                    continue
                g = det.loc[r, 'g']
                if not (pd.notna(g) and g > 0):
                    continue
                kd = str(det.loc[r, 'kind'])
                kind = 'size' if kd == 'size' else ('speed' if kd in SPEEDK else kd)
                if kind not in ('size', 'speed'):
                    continue
                g2 = float(g) * k
                val, src = fuse(kind, g2, det.loc[r, 'cs'])
                out[int(r)] = dict(value=val, why=f'rul153: length prior on box height h{best["bh"]:.0f}/w{best["bw"]:.0f}; geo {g:.4g}*k{k:.3f}={g2:.4g} -> {src}',
                                   old=float(det.loc[r, 'answer']), src=str(det.loc[r, 'answer_source']), video=v, q=det.loc[r, 'question'], cat=det.loc[r, 'cat'])
    # _segmented var_ans_plain copies follow their plain row
    for r in det.index[det['answer_source'].astype(str) == 'var_ans_plain']:
        v = str(det.loc[r, 'video_id'])
        if not v.endswith('_segmented'):
            continue
        pv, q = v[:-len('_segmented')], str(det.loc[r, 'question']).strip()
        m = [p for p in list(out) if str(det.loc[p, 'video_id']) == pv and str(det.loc[p, 'question']).strip() == q]
        if len(m) == 1 and abs(det.loc[r, 'answer'] / out[m[0]]['old'] - 1) < 1e-6:
            d = dict(out[m[0]])
            d.update(why='rul153 copy of plain row: ' + d['why'], old=float(det.loc[r, 'answer']), src='var_ans_plain', video=v)
            out[int(r)] = d
    return out, log


if __name__ == '__main__':
    sys.path.insert(0, R + '/cache/tmp')
    from lab_rows import labelled
    import analyze_caw as A
    L = labelled(SUB)
    for split in ('test', 'val'):
        out, log = run(split)
        print(split, 'masted length-prior rulers scanned', len(log))
        for l in log:
            print('  ', l)
        for r, d in sorted(out.items()):
            print(f"  CHANGE {split} {r} {d['video']} {d['cat']} | {d['q']} | {d['src']} {d['old']:.4g} -> {d['value']:.4g}")
        Ls = L[L['split'] == split]
        hit = [r for r in out if r in Ls.index]
        print(split, 'labelled rows changed:', hit)
        for r in hit:
            y = Ls.loc[r, 'y']
            print('   LAB', r, y, A.row_mra(np.array([out[r]['old']]), np.array([y])), A.row_mra(np.array([out[r]['value']]), np.array([y])))
        json.dump({str(r): dict(value=d['value'], why=d['why']) for r, d in out.items()},
                  open(f'{R}/cache/tmp/ag_c80mastlen_patch_{split}.json', 'w'), indent=1)
