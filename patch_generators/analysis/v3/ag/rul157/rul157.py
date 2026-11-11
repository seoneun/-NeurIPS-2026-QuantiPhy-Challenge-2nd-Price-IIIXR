"""rul157: coverage extension of rul125. A part-of-object size row ('<N> on the <P>', "<P>'s <N>", '<N> of the <P>',
N in a part-noun list, P non-person) whose geo came from a COLLAPSED track (in the raw's own track dir the N key's box is the
P key's box, median IoU >= 0.9, checked over queries and queries_d) re-runs the rul125 part-track search over the
alternate dirs (_v3, _v3nt, _v3nt32, _v2) and scales the part extent on the asked axis by the video's ACCEPTED m/px
(>= 2 strong same-video rows within 1.25x, as rul125). Then the pipeline fusion (agree within 2x -> gm, else geo)."""
import json, os, re, sys
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + '/analysis/v3/ag/rul125'); sys.path.insert(0, R + '/cache/tmp')
import rul125 as C
SUB = os.environ.get('C81_SUB', 'sub_xt_Nzz24')
TAG = 'rul157'
PARTS = {'lettering', 'text', 'letters', 'label', 'logo', 'word', 'cap', 'neck', 'handle'}
DIRS = ['_v3', '_v3nt', '_v3nt32', '_v2']
OWN = {'v1': '', 'v2': '_v2sc', 'v2sc': '_v2sc', 'v3': '_v3', 'v3nt': '_v3nt', 'v3nt32': '_v3nt32'}
QUOTES = ['’', '“', '”', '"']


def clean(question):
    ql = question.lower()
    for ch in QUOTES:
        ql = ql.replace(ch, "'" if ch == '’' else '')
    return ql


def parent_part(question):
    ql = clean(question)
    for N in sorted(PARTS):
        pats = (r"\b" + N + r"\s+on\s+(?:the|a|an)\s+([a-z]+)\b",
                r"\b([a-z]+)'s\s+" + N + r"\b",
                r"\b" + N + r"\s+of\s+(?:the|a|an)\s+([a-z]+)\b")
        for pat in pats:
            m = re.search(pat, ql)
            if m:
                P = m.group(1)
                if P in C.PERSON or P in PARTS:
                    continue
                return P, N
    return None, None


def axis(q, question):
    d = (q.get('dim') or '').lower(); ql = question.lower()
    if d == 'height' or 'height' in ql or 'vertical' in ql or 'tall' in ql:
        return 3
    return 2


def keyboxes(J, key, qk):
    q = [x for x in (J.get(qk) or []) if x.get('key') == key]
    if not q or q[0].get('obj') is None:
        return None, None
    o = q[0]['obj']; tr = J['objs'].get(str(o)) if isinstance(J['objs'], dict) else J['objs'][o]
    if tr is None:
        return o, None
    B = [e['st']['box'] if (e and e.get('st') and e['st'].get('box')) else None for e in tr]
    return o, B


def find_key(J, word, qk):
    ks = [x.get('key') for x in (J.get(qk) or []) if x.get('key')]
    ex = [k for k in ks if k.rstrip('|').strip() == word]
    if ex:
        return ex[0]
    cand = [k for k in ks if re.search(r'\b' + word + r'\b', k.rstrip('|'))]
    return min(cand, key=len) if cand else None


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0


def collapse(J, N, P):
    best = 0.0
    for qk in ('queries', 'queries_d'):
        kn, kp = find_key(J, N, qk), find_key(J, P, qk)
        if kn is None or kp is None:
            continue
        on, bn = keyboxes(J, kn, qk); op, bp = keyboxes(J, kp, qk)
        if on is not None and on == op:
            best = max(best, 1.0); continue
        if bn is None or bp is None:
            continue
        v = [iou(a, b) for a, b in zip(bn, bp) if a is not None and b is not None]
        if v:
            best = max(best, float(np.median(v)))
    return best


def part_search(split, vid, N, P, ax):
    log = []
    for d in DIRS:
        f = f'{R}/geo/out/tracks_{split}{d}/{vid}.json'
        if not os.path.exists(f):
            continue
        J = json.load(open(f))
        for qk in ('queries', 'queries_d'):
            kn, kp = find_key(J, N, qk), find_key(J, P, qk)
            if kn is None or kp is None:
                continue
            on, bn = keyboxes(J, kn, qk); op, bp = keyboxes(J, kp, qk)
            if bn is None or bp is None or on == op:
                log.append((d, qk, 'notrack')); continue
            nf = max(1, len(J['fidx']))
            BN = np.array([b for b in bn if b is not None], float); BP = np.array([b for b in bp if b is not None], float)
            cn, cp = len(BN) / nf, len(BP) / nf
            if cn < 0.8 or cp < 0.8:
                log.append((d, qk, 'cov', cn, cp)); continue
            mn, mp_ = np.median(BN, 0), np.median(BP, 0); ins = C.inside_frac(mn, mp_)
            extN = BN[:, ax] - BN[:, ax - 2]; extP = BP[:, ax] - BP[:, ax - 2]
            eN, eP = float(np.median(extN)), float(np.median(extP)); cv = float(extN.std() / max(1e-9, extN.mean()))
            if ins < 0.9 or eN > 0.8 * eP or cv >= 0.1 or eN <= 0:
                log.append((d, qk, 'geom', round(ins, 2), eN, eP, round(cv, 3))); continue
            return (d, qk, eN, eP, ins, cv), log
    return None, log


def run(split):
    det = pd.read_csv(f'{R}/submissions/{SUB}_{"valdetail" if split == "val" else "detail"}.csv', index_col=0)
    raw = pd.read_csv(f'{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv', index_col=0)
    if 'mode' not in det.columns:
        det['mode'] = raw['meas_mode'].reindex(det.index)
    PP = {}
    for l in open(f'{R}/geo/out/parse_{split}_v2.jsonl'):
        j = json.loads(l); PP[int(j['id'])] = j
    pj = f'{R}/cache/tmp/ag_ngfall_patch_{split}.json'
    ngf = {k: v.get('why', '') for k, v in json.load(open(pj)).items()} if os.path.exists(pj) else {}
    c69f = f'{R}/cache/tmp/ag_c69partv3_patch_{split}.json'
    c69 = json.load(open(c69f)) if os.path.exists(c69f) else {}
    out, log = {}, []
    for i in det.index:
        p = PP.get(int(i))
        if p is None or p['q'].get('kind') != 'size':
            continue
        question = raw.loc[i, 'question']; P, N = parent_part(question)
        if P is None:
            continue
        r = det.loc[i]; vid = r.video_id; ax = axis(p['q'], question)
        own = OWN.get(str(raw.loc[i, 'meas_tracks']), None)
        f = f'{R}/geo/out/tracks_{split}{own}/{vid}.json' if own is not None else None
        if f is None or not os.path.exists(f):
            log.append((i, vid, N, P, 'noown', raw.loc[i, 'meas_tracks'])); continue
        co = collapse(json.load(open(f)), N, P)
        if co < 0.9:
            log.append((i, vid, N, P, 'nocollapse', round(co, 2))); continue
        hit, sl = part_search(split, vid, N, P, ax)
        if hit is None:
            if str(i) in c69 and str(r.answer_source).startswith('agree'):
                out[str(int(i))] = {'value': c69[str(i)]['value'], 'why': f'{TAG}:nohit->c69'}
            log.append((i, vid, N, P, 'nohit', sl)); continue
        mp = C.accepted_mpp(vid, det, raw, ngf)
        vals = [v for _, v, _ in mp]
        if len(vals) >= 2 and max(vals) / min(vals) <= 1.25:
            M, ms = float(np.median(vals)), f'acc n{len(vals)}'
        else:
            pr = str(raw.loc[i, 'ground_truth_prior']).lower()
            mm = raw.loc[i, 'meas_mpp']
            if str(raw.loc[i, 'meas_mode']) == '2d_ratio' and pd.notna(mm) and mm > 0 and not any(w in pr for w in PARTS):
                M, ms = float(mm), 'own2d'
            else:
                log.append((i, vid, 'mpp', mp)); continue
        d, qk, eN, eP, ins, cv = hit; V = eN * M
        sib = []
        for j, s in det[det.video_id == vid].iterrows():
            if j == i:
                continue
            pj_ = PP.get(int(j))
            if pj_ is None or pj_['q'].get('kind') != 'size':
                continue
            ql = clean(raw.loc[j, 'question'])
            if parent_part(raw.loc[j, 'question'])[0] is None and re.search(r"\b" + P + r"\b", ql):
                sib.append(float(s.answer))
        if sib and V > max(sib) * 1.0001:
            log.append((i, vid, 'sanity', V, sib)); continue
        cs = float(r.cs) if pd.notna(r.cs) else np.nan
        if cs > 0 and max(V, cs) / min(V, cs) <= 2:
            val, fu = float(np.sqrt(V * cs)), 'agree_gm'
        else:
            val, fu = V, 'geo'
        out[str(int(i))] = {'value': val, 'why': f'{TAG}:{d} {N} {eN:.1f}px in {P} {eP:.1f}px ({qk}, ins {ins:.2f}, cv {cv:.3f}, coll {co:.2f}) x mpp {M:.3g} ({ms}) = {V:.4g}; sib {sib}; caw {cs:.3g} -> {fu} (was {r.answer:.4g} {r.answer_source})'}
    json.dump(out, open(f'{R}/cache/tmp/ag_{TAG}_patch_{split}.json', 'w'), indent=1)
    print(split, 'changed', len(out))
    for k, v in out.items():
        print(' ', k, det.loc[int(k), 'video_id'], det.loc[int(k), 'cat'], round(v['value'], 5), v['why'])
    for l in log:
        print('  skip', l)
    return out


if __name__ == '__main__':
    res = {sp: run(sp) for sp in ('val', 'test')}
    assert '495' not in res['test'], 'labelled 495 must stay unchanged'
    from lab_rows import labelled
    import analyze_caw as A
    L = labelled(SUB)
    print('labelled cols', list(L.columns)[:20], len(L))
    for sp in ('val', 'test'):
        ids = [int(k) for k in res[sp]]
        sub = L[(L.split == sp) & L.index.isin(ids)]
        for i, rr in sub.iterrows():
            nv = res[sp][str(i)]['value']
            print('LAB', sp, i, rr.video_id, 'y', rr.y, 'old', rr.m, 'new', A.row_mra(np.array([nv]), np.array([rr.y])))
        print(sp, 'labelled changed', len(sub))
