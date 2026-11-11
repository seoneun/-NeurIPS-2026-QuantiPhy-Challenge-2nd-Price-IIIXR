"""Rebuild the round-4 patch JSONs of best25v from code (2026-10-20, main agent).

Runs analysis/v3/ag/<key>/reproduce.py for every round-4 key with adopted rows (each script recomputes its rows from the
pipeline outputs, the question data, validation labels and best24v, and writes reproduce_out.csv), collects the values,
checks that the union is exactly the 115 adopted rows of best25v (analysis/v3/calib/r4/collect_rows.csv, sel) and that
a row written by two keys gets the same float, then writes one patch JSON per category and a build_probe.py spec:
  cache/tmp/r4_fromcode/ag_r4_code_<CAT>_patch_test.json, cache/tmp/r4_fromcode/best25r.tsv (base best24v; must equal
  best25v). With --computed, the eight rows that output a documented constant (1376, 1394, 1418, 1680, 1690, 2305, 2324,
  2343) output their measurement instead (env R4_COMPUTED=1 in each reproduce.py), and the specs are best25c (all four
  categories) and best26c (S2, D2, D3; S3 stays best24v as in best26v).
With --compare it also reports, per row, whether the value is bit-identical to the frozen JSON used by
analysis/v3/calib/rebuild/best25v.tsv. Run it with the base python (numpy 2.3.5, pandas 2.3.3, OpenCV 4.13.0): several
rows reproduce the original pandas / rounding steps and are only bit-identical in that environment.
Usage: python analysis/v3/ag/rul351/build_r4.py [--compare] [--skip-run] [--computed]
"""
import argparse, json, os, subprocess, sys
import pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
KEYS = ['rul368', 'rul371', 'rul355', 'rul364', 'rul366',      # S2 (+ S3 1541, 1631)
        'rul353', 'rul361', 'rul363', 'rul374',                            # D2
        'rul354', 'rul362', 'rul372', 'rul373',                              # D2 / D3
        'rul352', 'rul356', 'rul365', 'rul367', 'rul370',              # D3 (+ S3 1690)
        'rul369', 'r4_main']                                                      # S3, S2
FROZEN = {'S2': 'patches/ag_r4_best26v_S2_patch_test.json', 'D2': 'patches/ag_r4_best26v_D2_patch_test.json',
          'S3': 'patches/ag_r4_best25v_S3_patch_test.json', 'D3': 'patches/ag_r4_best26v_D3_patch_test.json'}
OUT = os.path.join(R, 'cache/tmp/r4_fromcode')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--compare', action='store_true')
    ap.add_argument('--skip-run', action='store_true', help='read the existing reproduce_out.csv files')
    ap.add_argument('--computed', action='store_true', help='measurements instead of the eight documented constants')
    a = ap.parse_args()
    env = dict(os.environ, R4_COMPUTED='1' if a.computed else '0')
    tag = 'comp' if a.computed else 'code'
    os.makedirs(OUT, exist_ok=True)
    vals = {}
    for k in KEYS:
        d = os.path.join(R, 'analysis/v3/ag', k)
        if not a.skip_run:
            r = subprocess.run([sys.executable, os.path.join(d, 'reproduce.py')], cwd=R, capture_output=True, text=True,
                               env=env)
            if r.returncode != 0:
                sys.exit(f'{k}: reproduce.py failed\n{r.stderr[-2000:]}')
        o = pd.read_csv(os.path.join(d, 'reproduce_out.csv'), float_precision='round_trip')
        for i, v in zip(o.id0.astype(int), o.value.astype(float)):
            if i in vals and repr(vals[i][1]) != repr(v):
                sys.exit(f'row {i}: {vals[i][0]} gives {vals[i][1]!r}, {k} gives {v!r}')
            vals.setdefault(i, (k, v))
    X = pd.read_csv(os.path.join(R, 'analysis/v3/calib/r4/collect_rows.csv'))
    sel = X[X.sel]
    want = dict(zip(sel.id0.astype(int), sel['cat']))
    missing, extra = sorted(set(want) - set(vals)), sorted(set(vals) - set(want))
    if missing or extra:
        sys.exit(f'row sets differ: missing {missing}, extra {extra}')
    name = 'best25c' if a.computed else 'best25r'
    spec = [f'# {name}: best24v + the round-4 rows recomputed by the reproduce.py scripts (build_r4.py'
            + (' --computed: measurements instead of documented constants)' if a.computed else '); must equal best25v'),
            f'name\t{name}', 'base\tbest24v']
    spec26 = ['# best26c: best24v + the S2 / D2 / D3 round-4 rows of build_r4.py --computed (S3 = best24v, as best26v)',
              'name\tbest26c', 'base\tbest24v']
    same = 0
    for c in ['S2', 'D2', 'S3', 'D3']:
        rows = {str(i): {'value': vals[i][1], 'why': vals[i][0]} for i in sorted(want) if want[i] == c}
        p = os.path.join(OUT, f'ag_r4_{tag}_{c}_patch_test.json')
        json.dump(rows, open(p, 'w'), indent=0)
        spec.append(f'block\tr4_{c}\t{c}\tjson:{os.path.relpath(p, R)}')
        if c != 'S3':
            spec26.append(f'block\tr4_{c}\t{c}\tjson:{os.path.relpath(p, R)}')
        if a.compare:
            fz = json.load(open(os.path.join(R, FROZEN[c])))
            for i, r in rows.items():
                ok = repr(float(fz[i]['value'])) == repr(float(r['value']))
                same += ok
                if not ok:
                    print(f'DIFF {c} {i} {r["why"]}: code {r["value"]!r} frozen {fz[i]["value"]!r}')
    open(os.path.join(OUT, f'{name}.tsv'), 'w').write('\n'.join(spec) + '\n')
    if a.computed:
        open(os.path.join(OUT, 'best26c.tsv'), 'w').write('\n'.join(spec26) + '\n')
    print(f'{len(vals)} rows from {len(KEYS)} keys; spec {os.path.relpath(OUT, R)}/{name}.tsv'
          + (f'; bit-identical to the frozen JSONs: {same}/{len(vals)}' if a.compare else ''))


if __name__ == '__main__':
    main()
