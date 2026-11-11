"""e2xab: the S2 simulation premise that won on the portal (probe Nzz47a, S2 +.006) extended to the other categories,
one premise per category, for probe Nzz48a (base Nzz48 = Nzz47k + that S2 premise).
Premise: on SIMULATION videos, when geo and CaW agree within 2x (final answer source agree_wgm / agree_gm = the pipeline's
own 2x agreement hedge), geo alone is the better answer for sizes and for the motion of RIGID targets.
  size half   : probe flag CAT:agree_wgm,agree_gm:size:geo:simulation (same mechanism as the S2 premise; no patch file)
  motion half : this file. Rows: category CAT, video simulation_*, final source agree_wgm / agree_gm (mo_guard rows skipped),
                kind speed / accel / disp, geo > 0, question without a NONRIGID word (people, animals, hands: the regex of
                analysis/v3/ag/rul272/gen_e1x.py) -> answer = geo.
  rul276, rul278, rul277 : the motion halves for D2 / S3 / D3
  e2xs2rigidmo_inet : regenerated internet half of S2 (must equal rul274 of probe Nzz47b; check only, not used)
  e2xd3hedge_sim    : D3 simulation SIZE rows answered by a geo/CaW size hedge (d3size_dimgm / d3size_unstable_gm / dis_size_gm)
                      with geo and CaW within 2x and geo measured at the target's own depth (mode without tdepth=prior, so it is
                      disjoint from e1xd3tdepth of probe Nzz47b) -> geo. D3 sizes never carry agree_* sources; candidate only.
Selection by code conditions only (category, video type, final answer source, kind, question words, geo/CaW values, mode);
never by row id, never by label. Reads AB_SUB (default sub_xt_Nzz48) detail + parse; writes cache/tmp/ag_<tag>_patch_{test,val}.json.
"""
import sys, os, json, re
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
SUB = os.environ.get('AB_SUB', 'sub_xt_Nzz48')
NONRIGID = re.compile(r"\b(?:person|persons|man|men|woman|women|boy|girl|people|player|astronaut|pedestrian|walker|walking|swimmer|rider|runner|human|child|children|bird|eagle|dragon|dog|cat|raccoon|horse|animal|fish|shark|duck|hand|fingertip)s?\b")
def load(split):
    fn = f'{R}/submissions/{SUB}_detail.csv' if split == 'test' else f'{R}/submissions/{SUB}_valdetail.csv'
    d = pd.read_csv(fn).set_index('Unnamed: 0')
    if split == 'val':            # the val detail carries no question / kind / mode / c columns: join them from the inputs
        sys.path.insert(0, f'{R}/caw'); import analyze_caw as A
        v = A.load_val(); d['question'] = v.question.reindex(d.index)
        raw = pd.read_csv(f'{R}/preds/val_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv').set_index('Unnamed: 0')
        d['mode'] = raw.meas_mode.reindex(d.index); d['c'] = d.cs
    d['s0'] = d.answer_source.astype(str).str.replace('mo_guard:', '', regex=False)
    d['mog'] = d.answer_source.astype(str).str.startswith('mo_guard:')
    P = {}
    for ln in open(f'{R}/geo/out/parse_{split}_v2.jsonl', encoding='utf-8'):
        x = json.loads(ln); P[x['id']] = x
    if 'kind' not in d.columns:
        d['kind'] = [((P.get(int(i)) or {}).get('q') or {}).get('kind') for i in d.index]
    return d, P
def ok(x):
    return x is not None and np.isfinite(x) and x > 0
TAGS = {'rul276': ('D2', 'simulation'), 'rul278': ('S3', 'simulation'), 'rul277': ('D3', 'simulation'),
        'e2xs2rigidmo_inet': ('S2', 'internet')}
for split in ('test', 'val'):
    d, P = load(split)
    out = {k: {} for k in list(TAGS) + ['e2xd3hedge_sim']}; skip = {k: 0 for k in out}
    for i, r in d.iterrows():
        q = str(r['question']).lower(); vid = str(r['video_id'])
        for tag, (cat, vt) in TAGS.items():
            if r['cat'] == cat and vid.startswith(vt) and r.s0 in ('agree_wgm', 'agree_gm') and r['kind'] in ('speed', 'accel', 'disp') \
                    and ok(r.g) and not NONRIGID.search(q):
                if r.mog: skip[tag] += 1
                else: out[tag][str(i)] = {'value': float(r.g), 'why': f'{tag}: {cat} {vt} rigid-target {r["kind"]} agreement row ({r.s0}) -> geo {r.g:.4g} (was {r.answer:.4g})'}
        if r['cat'] == 'D3' and vid.startswith('simulation') and r['kind'] == 'size' and r.s0 in ('d3size_dimgm', 'd3size_unstable_gm', 'dis_size_gm') \
                and ok(r.g) and ok(r.cs) and 0.5 <= r.g / r.cs <= 2 and 'tdepth=prior' not in str(r['mode']):
            if r.mog: skip['e2xd3hedge_sim'] += 1
            else: out['e2xd3hedge_sim'][str(i)] = {'value': float(r.g), 'why': f'e2xd3hedge_sim: D3 simulation size hedge ({r.s0}) with geo {r.g:.4g} and CaW {r.cs:.4g} within 2x, own-depth geo -> geo (was {r.answer:.4g})'}
    for k, o in out.items():
        json.dump(o, open(f'{R}/cache/tmp/ag_{k}_patch_{split}.json', 'w'), indent=1)
        cats = d.loc[[int(z) for z in o], 'cat'].value_counts().to_dict() if o else {}
        print(f'[{split}] {k}: {len(o)} rows {cats} (mo_guard rows skipped {skip[k]})')
        if split == 'val' or os.environ.get('AB_SHOW'):
            for z in list(o)[:60]:
                rr = d.loc[int(z)]; print(f'   {z} {rr["video_id"]} | {str(rr["question"])[:70]} | {rr["answer"]:.4g} -> {o[z]["value"]:.4g}')
# the regenerated S2 internet half must equal the probe Nzz47b tag
for sp in ('test', 'val'):
    a = json.load(open(f'{R}/cache/tmp/ag_e1xs2rigidmo_inet_patch_{sp}.json')); b = json.load(open(f'{R}/cache/tmp/ag_e2xs2rigidmo_inet_patch_{sp}.json'))
    same = set(a) == set(b) and all(a[k]['value'] == b[k]['value'] for k in a)
    print(f'[{sp}] e2xs2rigidmo_inet == rul274: {same} ({len(b)} vs {len(a)} rows)')
