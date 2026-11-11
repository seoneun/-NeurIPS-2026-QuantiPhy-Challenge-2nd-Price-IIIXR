# check_repair grpA (2026-10-17): copy of analysis/v3/ag/rul193/c89tlight3_rule.py; one change: the label-evaluation
# block after the patch JSONs are written (labelled(SUB), then L.split[ix] on a duplicated test/val index ->
# 'The truth value of a Series is ambiguous') is removed. run() and the patch values are unchanged.
"""rul193: coverage fix for tlight P2 (traffic-light HEIGHT, D3) on rows the shipped rul436 (TL_UNSTABLE=keep) left out.

Why 1300/1325 lacked the rul436 patch: rul436 = tlight_rule with TL_UNSTABLE=keep; both rows have CaW crop vs full-video
answers > 0.1 decade apart (1300: 6.3 vs 10.5 = 0.22 dec; 1325: 2.5 vs 12.5 = 0.70 dec), so they were 'keep(CaW unstable)'.
This rule applies tlight's own (documented, default) unstable handling to them: gm(crop, full) x k (sim height k=1.05).
Conditions (all code on inputs): D3, parse kind size, dim height, target matches 'traffic light'; answer_source in
{dis_size_gm, d3size_unstable_gm, d3size_dimgm}; g < 2.0 m (head-only detector box, the tlight docstring evidence).
Accept only if new value in [2.5, 7.0] m (traffic-light assembly height); otherwise row unchanged.
The pole is NOT re-measured (QP_GROUND rejected). No labels used. Output: cache/tmp/ag_c89tlight3_patch_{test,val}.json
"""
import os, re, json
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
TAG = 'rul193'
OUT = f'ag_{TAG}'
SUB = os.environ.get('C89_SUB', 'sub_xt_Nzz27')
GATE = 0.1
SRC_OK = {'dis_size_gm', 'd3size_unstable_gm', 'd3size_dimgm'}
RX_T = re.compile(r'traffic light', re.I)
G_MAX, LO, HI = 2.0, 2.5, 7.0
K_SIM_H = 1.05


def parse(split):
    return {int(j['id']): j for j in map(json.loads, open(f'{R}/geo/out/parse_{split}_v2.jsonl'))}


def run(split):
    det = pd.read_csv(f'{R}/submissions/{SUB}_{"detail" if split == "test" else "valdetail"}.csv', index_col=0)
    ft = 'caw9b_orig' if split == 'test' else 'caw9b'
    full = pd.to_numeric(pd.read_csv(f'{R}/preds/{split}_{ft}.csv').set_index('Unnamed: 0').parsed_value, errors='coerce').abs()
    crop = det['c'] if 'c' in det.columns else det['cs']
    P = parse(split)
    rows, out = [], {}
    for i in det.index:
        q = (P.get(int(i), {}).get('q') or {})
        if det.cat[i] != 'D3' or q.get('kind') != 'size' or q.get('dim') != 'height':
            continue
        if not RX_T.search(str(q.get('target') or '')):
            continue
        src, vid, g = str(det.answer_source[i]), str(det.video_id[i]), det.g.get(i)
        c_, f_ = crop.get(i), full.get(i)
        r = dict(id=int(i), video=vid, src=src, ans=float(det.answer[i]), g=g, crop=c_, full=f_)
        if src not in SRC_OK:
            r['act'] = 'skip(src)'; rows.append(r); continue
        if not (pd.notna(g) and g < G_MAX):
            r['act'] = 'skip(g>=2)'; rows.append(r); continue
        if not (pd.notna(f_) and f_ > 0):
            f_ = det.cs.get(i)
        if not (pd.notna(f_) and f_ > 0):
            r['act'] = 'skip(no CaW)'; rows.append(r); continue
        dc = abs(np.log10(c_ / f_)) if (pd.notna(c_) and c_ > 0) else 0.0
        if dc > GATE:
            base, how = float(np.sqrt(c_ * f_)), f'CaW unstable (crop {c_:g} vs full {f_:g}, {dc:.2f} dec) -> gm(crop, full)'
        else:
            base, how = float(f_), f'full-video CaW {f_:g}'
        k = K_SIM_H if vid.startswith('simulation') else 1.0
        new = base * k
        r.update(dc=dc, new=new)
        if not (LO <= new <= HI):
            r['act'] = f'reject(range {new:.2f})'; rows.append(r); continue
        r['act'] = 'change'; rows.append(r)
        out[str(int(i))] = {'value': new, 'why': f"rul193: traffic-light height, geo is head-only box (g {g:.3f}<2) -> d3size_caw route: {how} {base:.3f} x{k:g} = {new:.3f} (in [2.5,7]); was {r['ans']:.3f} {src}"}
    json.dump(out, open(f'{R}/cache/tmp/{OUT}_patch_{split}.json', 'w'), indent=1)
    return pd.DataFrame(rows), out


if __name__ == '__main__':
    pd.set_option('display.width', 250)
    for sp in ('test', 'val'):
        df, out = run(sp)
        print(f'== {sp}: candidates {len(df)}, changed {len(out)}')
        if len(df): print(df.to_string())
        print(json.dumps(out, indent=1))
