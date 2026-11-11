"""rul226: narrow refresh of c63plainkey that REPLACES it in the LAST patch slot.
Link set = exactly c63plainkey's own decisions, re-derived by running c63plainkey's code (dry) on its base build
(C63_SUB, default sub_xt_Nzz13), so the ratio/dim guards see the same pre-c63 inputs. Value:
 (a) segmented copy -> the linked plain row's CURRENT final answer in D03_SUB (default sub_xt_Nzz36);
 (b) scene_prior avg-speed rows -> c63's stated value unchanged.
Never keys on an id. Re-run with D03_SUB=<newest build> whenever a linked plain row is re-patched (two-pass)."""
import sys, os, json, re, importlib.util
import pandas as pd, numpy as np
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + '/analysis/v3'); sys.path.insert(0, R + '/cache/tmp')
TAG = 'rul226'
os.environ['C63_DRY'] = '1'
os.environ.setdefault('C63_SUB', 'sub_xt_Nzz13')
CUR = os.environ.get('D03_SUB', 'sub_xt_Nzz36')
spec = importlib.util.spec_from_file_location('c63', f'{R}/analysis/v3/ag/c63plainkey/c63plainkey.py')
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
for split in ['test', 'val']:
    _, p63 = C.run(split)
    fn = f'{R}/submissions/{CUR}_detail.csv' if split == 'test' else f'{R}/submissions/{CUR}_valdetail.csv'
    d = pd.read_csv(fn).set_index('Unnamed: 0')
    src = d.answer_source.astype(str).str.replace('mo_guard:', '', regex=False)
    out = {}
    for i, e in p63.items():
        m = re.search(r'plain row (\d+) final answer', e['why'])
        if e['why'].startswith('c63plainkey(a)') and m:
            p = int(m.group(1))      # link from c63's own code decision (not a literal id)
            if src[p] == 'patch:c63plainkey' or not np.isfinite(d.answer[p]):
                out[str(i)] = dict(e); continue
            out[str(i)] = {'value': float(d.answer[p]), 'why': f'{TAG}: c63plainkey(a) link, plain row {p} current final answer in {CUR} (src {src[p]})'}
        else:
            out[str(i)] = dict(e)
        old = e['value']; new = out[str(i)]['value']
        print(f'[{split}] {i} {d.video_id[i]} {d["cat"][i]} {str(d.question[i]).strip()[:60]} | c63 {old:.4f} -> {new:.4f} | build {d.answer[i]:.4f} {src[i]}')
    json.dump(out, open(f'{R}/cache/tmp/ag_{TAG}_patch_{split}.json', 'w'), indent=1)
    print(f'[{split}] wrote {len(out)} rows (c63plainkey had {len(p63)})')
from lab_rows import labelled
L = labelled(CUR)
for split in ['test', 'val']:
    P = json.load(open(f'{R}/cache/tmp/ag_{TAG}_patch_{split}.json'))
    print(split, 'labelled rows touched:', [k for k in P if int(k) in L[L.split == split].index])
