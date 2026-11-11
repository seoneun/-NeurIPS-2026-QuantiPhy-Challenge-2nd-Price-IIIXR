"""P2 (tlight): traffic-light / signal HEIGHT questions in D3 videos - geo is invalid, answer through the d3size_caw route.

Why geo is invalid: the detector class 'traffic light' boxes only the signal head (housing), while the asked object in these
simulated scenes is the whole asset (pole + head); and the rows have no target depth key, so geo also borrows the prior car's
depth (mode *_tdepth=prior). Geo is therefore not a measurement of the asked quantity and must not enter the answer
(no dis_size_gm / d3size_unstable_gm / d3size_dimgm hedge). The pole is NOT re-measured (that would be QP_GROUND).

Answer = the pipeline's d3size_caw route: full-video CaW (--d3size_full: caw9b val / caw9b_orig test)
x d3size_scale (simulation: height x1.05, other dims x1.1; other video types x1.0).
Instability (pipeline cawconf 0.1): if CaW's crop and full-video answers differ by > 0.1 decade, the pipeline does not trust
the full answer alone (it answers gm(geo, crop) = d3size_unstable_gm). Geo is invalid here, so the hedge left is the other CaW read:
  TL_UNSTABLE=gm (default): gm(crop, full) x scale   [labelled unstable D3 sizes, n=11: gm 0.491 > crop 0.418 > full 0.391]
  TL_UNSTABLE=full: full x scale (literal route, V1);  TL_UNSTABLE=keep: row unchanged (V2).
Lab (captured) copies are pooled by log-mean (lab_pool) so s/x copies stay consistent (no lab rows match today).
Rows already patched (answer_source patch*) and rows of the ag_orbit3d / ag_tedge2 / ag_orbitfix patches are skipped.
Labels are not used. Output: cache/tmp/{TL_OUT}_patch_{test,val}.json  {row_id: {value, why}}.
Env: TL_GATE (decades, default 0.1), TL_UNSTABLE (gm|full|keep), TL_OUT (default ag_tlight), TL_SUB (default sub_xt_Nze).
"""
import sys, os, re, json
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
SUB = os.environ.get('TL_SUB', 'sub_xt_Nze')
GATE = float(os.environ.get('TL_GATE', '0.1'))
UNST = os.environ.get('TL_UNSTABLE', 'gm')
OUT = os.environ.get('TL_OUT', 'ag_c89tlight3_orig')
RX_T = re.compile(r'traffic[ _-]?(?:light|signal)s?|\b(?:stop ?light|signal light|signal)\b', re.I)
RX_H = re.compile(r'\b(?:height|tall)\b', re.I)
SKIPF = ['ag_orbit3d_patch', 'ag_tedge2_patch', 'ag_orbitfix_patch']
SCALE = {'height': 1.05, 'other': 1.1}                   # --d3size_scale simulation:1.1:-height,simulation:1.05:+height


def parse(split):
    P = {}
    for ln in open(f'{R}/geo/out/parse_{split}_v2.jsonl'):
        j = json.loads(ln)
        P[int(j['id'])] = j
    return P


def run(split):
    det = pd.read_csv(f'{R}/submissions/{SUB}_{"detail" if split == "test" else "valdetail"}.csv', index_col=0)
    raw = pd.read_csv(f'{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv', index_col=0)
    ft = 'caw9b_orig' if split == 'test' else 'caw9b'
    full = pd.to_numeric(pd.read_csv(f'{R}/preds/{split}_{ft}.csv').set_index('Unnamed: 0').parsed_value, errors='coerce').abs()
    crop = det['c'] if 'c' in det.columns else det['cs']
    P = parse(split)
    skip = set()
    for f in SKIPF:
        p = f'{R}/cache/tmp/{f}_{split}.json'
        if os.path.exists(p):
            skip |= {int(k) for k in json.load(open(p))}
    rows, out = [], {}
    for i in det.index:
        j = P.get(int(i), {})
        q = j.get('q', {}) or {}
        ques = str(raw.question.get(i, j.get('question', '')))
        tgt = str(q.get('target') or '')
        if det.cat[i] != 'D3' or q.get('kind') != 'size':
            continue
        if not (RX_T.search(tgt) or RX_T.search(ques)) or not RX_H.search(ques):
            continue
        src = str(det.answer_source[i])
        vid = str(det.video_id[i])
        r = dict(id=int(i), video=vid, q=ques, src=src, ans=float(det.answer[i]), g=det.g.get(i), c=crop.get(i), full=full.get(i))
        if src.startswith('patch') or int(i) in skip:
            r['act'] = 'skip(patched)'
            rows.append(r)
            continue
        f_, c_ = full.get(i), crop.get(i)
        base = f_ if (pd.notna(f_) and f_ > 0) else det.cs.get(i)
        if not (pd.notna(base) and base > 0):
            r['act'] = 'skip(no CaW)'
            rows.append(r)
            continue
        dc = abs(np.log10(c_ / f_)) if (pd.notna(c_) and c_ > 0 and pd.notna(f_) and f_ > 0) else 0.0
        r['dc'] = dc
        unstable = GATE > 0 and dc > GATE
        if unstable and UNST == 'keep':
            r['act'] = f'keep(CaW unstable crop/full {dc:.2f} dec)'
            rows.append(r)
            continue
        typ = vid.split('_')[0]
        k = (SCALE['height'] if RX_H.search(ques) else SCALE['other']) if typ == 'simulation' else 1.0
        if unstable and UNST == 'gm':
            base = float(np.sqrt(c_ * f_))
            r['how'] = f'CaW unstable (crop {c_:g} vs full {f_:g}, {dc:.2f} dec) -> gm(crop, full) {base:.3f}'
        else:
            r['how'] = f'full-video CaW {float(base):g}'
        new = float(base) * k
        r['k'] = k
        r['new'] = new
        r['act'] = 'change' if abs(np.log10(new / r['ans'])) > 1e-6 else 'same'
        rows.append(r)
    # lab_pool: captured copies (plain / s / x) share one answer = log-mean of their route values
    chg = [r for r in rows if r.get('act') == 'change']
    labs = {}
    for r in chg:
        if r['video'].startswith('captured'):
            key = (re.sub(r'(?<=\d)[sx]$', '', r['video'].replace('_segmented', '')), r['q'])
            labs.setdefault(key, []).append(r)
    for _, grp in labs.items():
        lm = float(np.exp(np.mean([np.log(r['new']) for r in grp])))
        for r in grp:
            r['new'] = lm
    for r in chg:
        out[str(r['id'])] = {'value': r['new'],
                             'why': (f"tlight P2: traffic-light height; detector boxes the signal head only and target depth borrows "
                                     f"the prior car's (tdepth=prior) -> geo invalid; d3size_caw route: {r['how']} x{r['k']:g}; "
                                     f"was {r['ans']:.3f} {r['src']} (g {r['g']:.3f})")}
    json.dump(out, open(f'{R}/cache/tmp/{OUT}_patch_{split}.json', 'w'), indent=1)
    return pd.DataFrame(rows), out


if __name__ == '__main__':
    pd.set_option('display.width', 250)
    pd.set_option('display.max_colwidth', 70)
    for sp in ('test', 'val'):
        df, out = run(sp)
        print(f'== {sp} [gate {GATE}, unstable={UNST}]: candidates {len(df)}, changed {len(out)} -> cache/tmp/{OUT}_patch_{sp}.json')
        if len(df):
            print(df.drop(columns=['q']).to_string())
