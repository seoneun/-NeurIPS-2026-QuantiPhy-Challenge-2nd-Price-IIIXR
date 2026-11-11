import os, sys, json, re
import numpy as np, pandas as pd
R = os.getcwd()
SUB = sys.argv[1] if len(sys.argv) > 1 else 'pred_t107'
RAW = 'geo_v4ntrj5sQ2srngsr_d_vw'
TDIR = {'v1': '', 'v2': '_v2', 'v3': '_v3'}


def load_tracks(split, ver, v):
    suf = TDIR.get(ver, '_' + str(ver))
    p = f'{R}/geo/out/tracks_{split}{suf}/{v}.json'
    return json.load(open(p)) if os.path.exists(p) else None




def run(split):
    det = pd.read_csv(f'{R}/submissions/{SUB}_{"detail" if split == "test" else "valdetail"}.csv', index_col=0)
    raw = pd.read_csv(f'{R}/preds/{split}_{RAW}_raw.csv', index_col=0)
    P = {}
    for l in open(f'{R}/geo/out/parse_{split}_v2.jsonl'):
        j = json.loads(l); P[j['id']] = j
    out, log = {}, []
    for v, grp in det.groupby('video_id'):
        if not str(v).startswith('simulation'):
            continue
        ids = [i for i in grp.index if i in P and i in raw.index]
        done = set()
        for i in ids:
            pr = (P[i].get('priors') or [None])[0]
            if not pr or pr.get('kind') != 'size' or pr.get('dim') != 'width':
                continue
            ph = (P[i].get('prior_obj') or {}).get('phrase')
            ver = raw.loc[i, 'meas_tracks']
            if not ph or (ph, ver) in done:
                continue
            done.add((ph, ver))
            S = float(pr['value_si'])
            j = load_tracks(split, ver, v)
            st = track_states(j, ph) if j else None
            if not st or len(st) < 5:
                continue
            b = np.array([s['box'] for s in st], float)
            bw, bh = np.median(b[:, 2] - b[:, 0]), np.median(b[:, 3] - b[:, 1])
            rs = float(np.median([s['rr'][1] for s in st]))
            bord = float(np.mean([s.get('border', 0) != 0 for s in st]))
            g2 = bw > 1.3 * bh and abs(rs / bh - 1) < 0.08
            g4 = bord <= 0.2
            echo = [e for e in ids if (P[e].get('q') or {}).get('kind') == 'size' and (P[e].get('q') or {}).get('dim') == 'height'
                    and ((P[e].get('target') or {}).get('phrase') == ph) and raw.loc[e, 'meas_tracks'] == ver
                    and pd.notna(raw.loc[e, 'geo_value']) and abs(raw.loc[e, 'geo_value'] / S - 1) < 0.03]
            log.append(dict(split=split, v=v, ph=ph, ver=ver, bw=bw, bh=bh, rs=rs, bord=bord, g2=g2, g4=g4, echo=echo))
            if not (g2 and g4 and echo):
                continue
            k = rs / bw
            for r in ids:
                if r in echo:
                    continue
                if raw.loc[r, 'meas_mode'] not in ('2d_ratio', 'vlm_2d_ratio'):
                    continue
                pr_ = (P[r].get('priors') or [None])[0]
                if not pr_ or pr_.get('kind') != 'size' or pr_.get('dim') != 'width' or (P[r].get('prior_obj') or {}).get('phrase') != ph \
                        or raw.loc[r, 'meas_tracks'] != ver or abs(float(pr_['value_si']) / S - 1) > 1e-6:
                    continue
                src = str(det.loc[r, 'answer_source'])
                if not (src in ('dis_geo', 'geo') or src.startswith('agree')):
                    continue
                g = det.loc[r, 'g']
                if not (pd.notna(g) and g > 0):
                    continue
                g2_ = float(g) * k
                cs = det.loc[r, 'cs']
                if pd.notna(cs) and cs > 0 and max(g2_ / cs, cs / g2_) <= 2.0:
                    val, why = float(np.sqrt(g2_ * cs)), f'rul151: geo {g:.4g}*k{k:.3f}={g2_:.4g} agrees with CaW {cs:.4g} -> gm'
                else:
                    val, why = g2_, f'rul151: prior width on short side (rs{rs:.0f}/bw{bw:.0f}) geo {g:.4g}*k{k:.3f}'
                out[int(r)] = dict(value=val, why=why, old=float(det.loc[r, 'answer']), src=src, video=v, q=det.loc[r, 'question'], cat=det.loc[r, 'cat'])
    return out, log


if __name__ == '__main__':
    for split in ('test', 'val'):
        out, log = run(split)
        nw = [l for l in log if l['g2'] or l['echo']]
        print(split, 'width-prior objects scanned', len(log), '| with g2 or echo:')
        for l in nw:
            print('  ', l)
        for r, d in sorted(out.items()):
            print(f"  CHANGE {split} {r} {d['video']} {d['cat']} | {d['q']} | {d['src']} {d['old']:.4g} -> {d['value']:.4g}")
        json.dump({str(r): dict(value=d['value'], why=d['why']) for r, d in out.items()},
                  open(f'{R}/cache/tmp/ag_rul151_patch_{split}.json', 'w'), indent=1)
