"""rul176: spurious meas_why='target=prior_object' after a repaired collapse.
A size row whose raw geo says target=prior_object (so the pipeline treated it as a weak vlm_2d_ratio row and hedged it,
dis_size_gm / agree_*) although in the raw's own track dir the target key and the prior key are DIFFERENT objects
(median per-frame IoU < 0.1), the prior object is a constant-speed mover (centre path > 50 px, per-segment speed CV < 0.2),
the target is static (centre path < 20 px), geo_value is finite and the video's 2d_ratio m/px (meas_mpp, own row or
same-video same-prior 2d_ratio rows) agrees within 5% with stated speed / prior track px speed: geo already scaled the
separated target by the separated prior track, so the row is an ordinary 2d_ratio size row -> pipeline fusion
(agree within 2x -> g^0.65 cs^0.35 (agree_w size:0.65:strong), else geo). Segmented copies take the plain answer."""
import json, os, sys
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + '/cache/tmp'); sys.path.insert(0, R + '/analysis/v3')
SUB = os.environ.get('C85_SUB', 'sub_xt_Nzz27')
TAG = 'rul176'
OWN = {'v1': '', 'v2': '_v2sc', 'v2sc': '_v2sc', 'v3': '_v3', 'v3nt': '_v3nt', 'v3nt32': '_v3nt32'}
W = 0.65
DIAG = bool(os.environ.get('C85_DIAG'))   # diagnostics only: ignore the source gate, write nothing


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0


def track(J, phrase, qk):
    q = [x for x in (J.get(qk) or []) if x.get('key', '').rstrip('|').strip() == phrase]
    if not q or q[0].get('obj') is None:
        return None, None
    o = q[0]['obj']; tr = J['objs'].get(str(o)) if isinstance(J['objs'], dict) else J['objs'][o]
    if tr is None:
        return o, None
    return o, [e['st']['box'] if (e and e.get('st') and e['st'].get('box')) else None for e in tr]


def centres(B, fidx):
    t, c = [], []
    for f, b in zip(fidx, B):
        if b is not None:
            t.append(f); c.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
    return np.array(t, float), np.array(c, float)


def path_len(c):
    return float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1))) if len(c) > 1 else 0.0


def seg_speed(t, c, fps, nseg=4):
    """per-segment px/s (path per segment / its time) and overall px/s (path / time)"""
    if len(t) < 2 * nseg:
        return None, None
    sp = []
    for ii in np.array_split(np.arange(len(t)), nseg):
        a, b = ii[0], ii[-1]
        if b <= a:
            return None, None
        sp.append(path_len(c[a:b + 1]) / ((t[b] - t[a]) / fps))
    return np.array(sp), path_len(c) / ((t[-1] - t[0]) / fps)


def run(split):
    det = pd.read_csv(f'{R}/submissions/{SUB}_{"valdetail" if split == "val" else "detail"}.csv', index_col=0)
    raw = pd.read_csv(f'{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv', index_col=0)
    PP = {}
    for l in open(f'{R}/geo/out/parse_{split}_v2.jsonl'):
        j = json.loads(l); PP[int(j['id'])] = j
    out, log = {}, []
    for i in raw.index[raw.meas_why.astype(str) == 'target=prior_object']:
        if i not in det.index:
            continue
        r = det.loc[i]; src = str(r.answer_source); p = PP.get(int(i))
        if (src.startswith('patch:') and not DIAG) or p is None:
            log.append((i, r.video_id, 'src', src)); continue
        if not DIAG and not (src.startswith('dis_') or src.startswith('agree') or src in ('caw', 'vlm', 'cs')):
            log.append((i, r.video_id, 'src', src)); continue
        pk = [x.get('kind') for x in p.get('priors') or []]
        if p['q'].get('kind') != 'size' or pk != ['speed']:
            log.append((i, r.video_id, 'kind', p['q'].get('kind'), pk)); continue
        g = raw.loc[i, 'geo_value']
        if not (pd.notna(g) and g > 0):
            log.append((i, r.video_id, 'nog')); continue
        tp = (p.get('target') or {}).get('phrase'); pp = (p.get('prior_obj') or {}).get('phrase')
        if not tp or not pp or tp == pp:
            log.append((i, r.video_id, 'samephrase', tp, pp)); continue
        own = OWN.get(str(raw.loc[i, 'meas_tracks']))
        f = f'{R}/geo/out/tracks_{split}{own}/{r.video_id}.json' if own is not None else None
        if f is None or not os.path.exists(f):
            log.append((i, r.video_id, 'notracks', raw.loc[i, 'meas_tracks'])); continue
        J = json.load(open(f)); fps = float(raw.loc[i, 'fps'])
        # m/px of the video's ordinary 2d_ratio rows with the same prior (own row first)
        mm = raw.loc[i, 'meas_mpp']; msrc = 'own'
        if not (pd.notna(mm) and mm > 0):
            sib = raw[(raw.video_id == r.video_id) & (raw.ground_truth_prior == raw.loc[i, 'ground_truth_prior'])
                      & (raw.meas_mode == '2d_ratio') & raw.meas_mpp.notna()].meas_mpp
            mm = float(sib.median()) if len(sib) and sib.max() / sib.min() <= 1.02 else np.nan; msrc = f'sib{len(sib)}'
        ok = None
        for qk in ('queries', 'queries_d'):
            ot, bt = track(J, tp, qk); op, bp = track(J, pp, qk)
            if bt is None or bp is None:
                log.append((i, r.video_id, qk, 'notrack', ot, op)); continue
            if ot == op:
                log.append((i, r.video_id, qk, 'sameobj', ot)); continue
            v = [iou(a, b) for a, b in zip(bt, bp) if a is not None and b is not None]
            mi = float(np.median(v)) if v else 0.0
            tt, ct = centres(bt, J['fidx']); tq, cq = centres(bp, J['fidx'])
            Lt, Lp = path_len(ct), path_len(cq)
            sp, pxs = seg_speed(tq, cq, fps)
            cv = float(sp.std() / sp.mean()) if sp is not None and sp.mean() > 0 else np.inf
            mpp_tr = p['priors'][0]['value_si'] / pxs if pxs else np.nan
            dev = abs(mpp_tr / mm - 1) if (pd.notna(mm) and mm > 0 and pd.notna(mpp_tr)) else np.inf
            st = dict(qk=qk, ot=ot, op=op, iou=round(mi, 3), Lt=round(Lt, 1), Lp=round(Lp, 1), cv=round(cv, 3),
                      mpp=mm, msrc=msrc, mpp_tr=mpp_tr, dev=round(dev, 3))
            if mi < 0.1 and Lp > 50 and cv < 0.2 and Lt < 20 and dev <= 0.05:
                ok = st; break
            log.append((i, r.video_id, 'gate', st))
        if ok is None:
            continue
        cs = float(r.cs) if pd.notna(r.cs) else np.nan
        if cs > 0 and max(g, cs) / min(g, cs) <= 2:
            val, fu = float(g ** W * cs ** (1 - W)), 'agree_wgm'
        else:
            val, fu = float(g), 'geo'
        out[str(int(i))] = {'value': val, 'why': f'{TAG}: target {tp} obj{ok["ot"]} != prior {pp} obj{ok["op"]} ({ok["qk"]}, iou {ok["iou"]}, '
                            f'path {ok["Lt"]}/{ok["Lp"]}px, cv {ok["cv"]}, mpp {ok["mpp"]:.4g} {ok["msrc"]} vs track {ok["mpp_tr"]:.4g}) -> 2d_ratio; '
                            f'g {g:.4g} cs {cs:.4g} -> {fu} (was {r.answer:.4g} {src})'}
    # segmented / recoloured copies follow the plain recording (var_ans_plain)
    try:
        from variant_keys import split_variant
        sv = det.video_id.astype(str).map(split_variant)
        key = sv.str[0] + '||' + raw.question.reindex(det.index).astype(str).str.strip().str.lower()
        for k in list(out):
            if sv.loc[int(k)][1] != 'plain':
                continue
            for j in det.index[(key == key.loc[int(k)]) & (sv.str[1] != 'plain')]:
                if not str(det.loc[j, 'answer_source']).startswith('patch:'):
                    out[str(int(j))] = {'value': out[k]['value'], 'why': f'{TAG}: copy of {k}'}
    except Exception as e:
        print('variant step skipped', e)
    if not DIAG:
        json.dump(out, open(f'{R}/cache/tmp/ag_{TAG}_patch_{split}.json', 'w'), indent=1)
    print(split, 'changed', len(out))
    for k, v in out.items():
        print(' ', k, det.loc[int(k), 'video_id'], det.loc[int(k), 'cat'], det.loc[int(k), 'question'], round(v['value'], 5), v['why'])
    for l in log:
        print('  skip', l)
    return out


if __name__ == '__main__':
    res = {sp: run(sp) for sp in ('val', 'test')}
    from lab_rows import labelled
    import analyze_caw as A
    L = labelled(SUB)
    for sp in ('val', 'test'):
        ids = [int(k) for k in res[sp]]
        sub = L[(L.split == sp) & L.index.isin(ids)]
        for i, rr in sub.iterrows():
            nv = res[sp][str(i)]['value']
            print('LAB', sp, i, rr.video_id, 'y', rr.y, 'old', rr.m, 'new', A.row_mra(np.array([nv]), np.array([rr.y])))
        print(sp, 'labelled changed', len(sub))
