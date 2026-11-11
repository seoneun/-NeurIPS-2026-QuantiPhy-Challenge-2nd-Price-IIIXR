"""rul051: lower/upper size bound for adult-person heights in simulation videos.
Synthetic renders use adult human assets (every sim prior text that states a person height says 1.75-1.87 m).
If the final answer for 'height of the (walking|standing|running)? person|man|woman|pedestrian|adult|player' in m
is outside [1.5, 2.05] while the raw CaW answer (geo raw vlm_value) is inside [1.55, 2.0], the answer is physically
impossible for the asset and the geometric scale is broken (e.g. keyframed 'gravity' ball -> grav_frameH) -> raw CaW.
Inputs only: parse (kind/dim/unit/target), video_id, final answer of the build, raw CaW. Labels are used only to evaluate."""
import json, sys, re
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + '/cache/tmp'); sys.path.insert(0, R + '/caw'); sys.path.insert(0, R + '/analysis/v3')
BUILD = sys.argv[1] if len(sys.argv) > 1 else 'sub_xt_Nzw'
RAW = 'geo_v4ntrj5sQ2srngsr_d_vw'
RX = re.compile(r'^(walking |standing |running )?(person|man|woman|pedestrian|adult|player|basketball player)$')
LO, HI = 1.5, 2.05          # answer band for a rendered adult
CLO, CHI = 1.55, 2.0        # CaW must already be an adult height


def patch(split):
    D = pd.read_csv(f'{R}/submissions/{BUILD}_{"detail" if split == "test" else "valdetail"}.csv')
    raw = pd.read_csv(f'{R}/preds/{split}_{RAW}_raw.csv')
    P = [json.loads(l) for l in open(f'{R}/geo/out/parse_{split}_v2.jsonl')]
    assert len(D) == len(raw) == len(P)
    out = {}
    for i, p in enumerate(P):
        q = p['q']
        vid = str(p['video_id'])
        if not (p.get('video_source') == 'simulation' or vid.startswith('simulation_')):
            continue
        if q.get('kind') != 'size' or q.get('dim') != 'height' or q.get('unit') != 'm':
            continue
        if not RX.match((q.get('target') or '').strip().lower()):
            continue
        assert D.video_id.iloc[i] == vid == raw.video_id.iloc[i]
        ans = float(D.answer.iloc[i]); c = pd.to_numeric(raw.vlm_value.iloc[i], errors='coerce')
        if LO <= ans <= HI or not (CLO <= c <= CHI):
            continue
        out[str(i)] = {'value': float(c), 'why': f'rul051: sim adult height {ans:.3f} outside [{LO},{HI}] -> raw CaW {c:.3f} '
                                                  f'(geo {D.g.iloc[i]:.3f}, {raw.meas_why.iloc[i]})'}
    return out, D


if __name__ == '__main__':
    import analyze_caw as A
    for split in ('test', 'val'):
        out, D = patch(split)
        json.dump(out, open(f'{R}/cache/tmp/ag_c36hadult_patch_{split}.json', 'w'), indent=1)
        print(f'[{split}] {len(out)} rows changed; by cat', D.cat.iloc[[int(k) for k in out]].value_counts().to_dict())
        for k, v in out.items():
            r = D.iloc[int(k)]
            print(' ', k, r.video_id, r['cat'], r.question[:60], '| old', round(r.answer, 4), r.answer_source, '-> new', round(v['value'], 4), '|', v['why'])
        if split == 'val' and 'y' in D.columns and out:
            ii = [int(k) for k in out]; y = D.y.values[ii]
            print('  val MRA before', A.row_mra(D.answer.values[ii], y), 'after', A.row_mra(np.array([out[str(i)]['value'] for i in ii]), y))
    from lab_rows import labelled
    L = labelled(BUILD)
    pt = json.load(open(f'{R}/cache/tmp/ag_c36hadult_patch_test.json')); pv = json.load(open(f'{R}/cache/tmp/ag_c36hadult_patch_val.json'))
    hit = [ix for ix, s in zip(L.index, L.split) if str(ix) in (pt if s == 'test' else pv)]
    print('labelled rows changed (val + all-237):', len(hit), hit)
    # stress test: the same rule with looser CaW band / any source -> what would fire on labelled rows
    for split in ('test', 'val'):
        D = pd.read_csv(f'{R}/submissions/{BUILD}_{"detail" if split == "test" else "valdetail"}.csv')
        raw = pd.read_csv(f'{R}/preds/{split}_{RAW}_raw.csv')
        P = [json.loads(l) for l in open(f'{R}/geo/out/parse_{split}_v2.jsonl')]
        n = 0
        for i, p in enumerate(P):
            q = p['q']
            if q.get('kind') == 'size' and q.get('dim') == 'height' and RX.match((q.get('target') or '').strip().lower()):
                f = 100.0 if q.get('unit') == 'cm' else 1.0
                a = D.answer.iloc[i] / f
                if not (LO <= a <= HI): n += 1; print('  [any-source, out-of-band]', split, i, p['video_id'], q.get('unit'), round(a, 3), 'caw', raw.vlm_value.iloc[i])
        print(f'  [{split}] any-source strict-target person heights outside band: {n}')
