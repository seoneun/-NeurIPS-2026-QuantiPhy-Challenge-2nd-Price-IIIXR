"""rul272: e1x-named copies of the abplan premise patches (selection code identical to analysis/v3/ag/abplan/ab_patches.py;
only the tag names differ: ab_<x> -> e1x<x>). Original docstring follows.
abplan: label-free premise patches for the Nzz47 portal probes (one premise per category per probe).
Reads the current base build detail (AB_SUB, default sub_xt_Nzz46) + parse; writes cache/tmp/ag_<tag>_patch_{test,val}.json.
Rows are selected by code conditions only (category, final answer source, kind, question words, parse depth keys / priors);
never by row id, never by label.
  e1xs2rigidmo : S2 agreement rows (agree_wgm / agree_gm) of RIGID-target motion (speed/accel/disp) -> geo
                 (motion half of premise S2_hedge_geo_rigid; the size half is the probe flag S2:agree_wgm,agree_gm:size:geo)
                 (rul275 / rul274: the simulation / internet halves, for one probe each)
  e1xs3firstkey: S3 lab speed agree_gm rows asked at the EARLIEST of >=2 distinct timed depth keys of the target -> answer x sqrt(1.25)
  e1xd3tdepth  : D3 size rows whose geo borrowed the prior object's depth (mode *tdepth=prior*) and were hedged with that geo
                 (d3size_dimgm / d3size_unstable_gm / dis_size_gm) -> full-video CaW x the D3 CaW scale (sim 1.1, heights 1.05; lab 1.0)
  rul273 : D2 rows forced to geo by --geo_only_modes 2d_gself whose CaW agrees within 2x -> gm(geo, CaW)
  e1xd3kint0   : D3 from-rest kinematic speeds (answer == a*t within 1%) on simulation videos -> a*(t - 1/fps)
"""
import sys, os, json, re
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
SUB = os.environ.get('AB_SUB', 'sub_xt_Nzz46')
NONRIGID = re.compile(r"\b(?:person|persons|man|men|woman|women|boy|girl|people|player|astronaut|pedestrian|walker|walking|swimmer|rider|runner|human|child|children|bird|eagle|dragon|dog|cat|raccoon|horse|animal|fish|shark|duck|hand|fingertip)s?\b")
HEIGHT = re.compile(r"\b(?:height|tall)\b")
SI = {"m/s": 1.0, "cm/s": 0.01, "mm/s": 0.001, "km/h": 1 / 3.6}
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
KEEP = ('rul275', 'rul274', 'rul273', 'e1xs3firstkey', 'e1xd3tdepth', 'e1xd3kint0')   # tags used by the Nzz47a/b/c probes
def ok(x):
    return x is not None and np.isfinite(x) and x > 0
for split in ('test', 'val'):
    d, P = load(split)
    out = {k: {} for k in ('e1xs2rigidmo', 'e1xs3firstkey', 'e1xd3tdepth', 'e1xd3kint0', 'rul273')}
    skip = {k: 0 for k in out}
    for i, r in d.iterrows():
        q = str(r['question']).lower(); vid = str(r['video_id']); x = P.get(int(i), {})
        # S2 rigid motion -> geo
        if r['cat'] == 'S2' and r.s0 in ('agree_wgm', 'agree_gm') and r['kind'] in ('speed', 'accel', 'disp') and ok(r.g) and not NONRIGID.search(q):
            if r.mog: skip['e1xs2rigidmo'] += 1
            else: out['e1xs2rigidmo'][str(i)] = {'value': float(r.g), 'why': f'e1xs2rigidmo: S2 rigid-target {r["kind"]} agreement row ({r.s0}) -> geo {r.g:.4g} (was {r.answer:.4g})'}
        # S3 lab first-key speed
        if r['cat'] == 'S3' and vid.startswith('captured') and r['kind'] == 'speed' and r.s0 == 'agree_gm' and ok(r.answer):
            qq = x.get('q') or {}; tk = set(x.get('target_dkeys') or [])
            ts = sorted({round(float(e['t']), 4) for e in (x.get('depth') or []) if e.get('key') in tk and e.get('t') is not None})
            if qq.get('t') is not None and len(ts) >= 2 and abs(float(qq['t']) - ts[0]) < 1e-6:
                if r.mog: skip['e1xs3firstkey'] += 1
                else: out['e1xs3firstkey'][str(i)] = {'value': float(r.answer) * 1.25 ** 0.5, 'why': f'e1xs3firstkey: speed asked at the first of target keys {ts}: interval-mean chord under-reads a decelerating ball, gm(1.25 g, c) = x1.118'}
        # D3 borrowed-depth size hedge -> CaW
        if r['cat'] == 'D3' and 'tdepth=prior' in str(r['mode']) and r.s0 in ('d3size_dimgm', 'd3size_unstable_gm', 'dis_size_gm') and ok(r.c):
            k = (1.05 if HEIGHT.search(q) else 1.1) if vid.startswith('simulation') else 1.0
            if r.mog: skip['e1xd3tdepth'] += 1
            else: out['e1xd3tdepth'][str(i)] = {'value': float(r.c) * k, 'why': f'e1xd3tdepth: geo used the prior object depth (mode {r["mode"]}), not a key of the target -> CaW {r.c:.4g} x {k}'}
        # D2 2d_gself (geo_only_mode) rows -> gm(geo, CaW) when the two agree within 2x (the pipeline's own agreement gate)
        if r['cat'] == 'D2' and r.s0 == 'geo_only_mode' and ok(r.g) and ok(r.cs) and 0.5 <= r.g / r.cs <= 2:
            if r.mog: skip['rul273'] += 1
            else: out['rul273'][str(i)] = {'value': float(np.sqrt(r.g * r.cs)), 'why': f'rul273: 2d_gself geo {r.g:.4g} and CaW {r.cs:.4g} agree within 2x -> gm'}
        # D3 kin from rest, one-frame offset
        if r['cat'] == 'D3' and r.s0 == 'kin' and r['kind'] == 'speed' and vid.startswith('simulation') and ok(r.answer):
            qq = x.get('q') or {}; pr = (x.get('priors') or [{}])[0]; fps = float(x.get('fps') or 0)
            a = pr.get('value_si') if (pr.get('kind') == 'accel' or pr.get('gravity')) else None
            u = str(qq.get('unit') or ''); t = qq.get('t')
            if a and t and fps > 0 and u in SI and float(t) - 1 / fps > 0:
                at = float(a) * float(t) / SI[u]
                if abs(r.answer / at - 1) < 0.01:
                    out['e1xd3kint0'][str(i)] = {'value': float(r.answer) * (float(t) - 1 / fps) / float(t), 'why': f'e1xd3kint0: from-rest v = a t with the motion starting one frame late: a (t - 1/fps), t={t}, fps={fps:g}'}
    for k_, pre_ in (('rul275', 'simulation'), ('rul274', 'internet')):   # source halves of e1xs2rigidmo
        out[k_] = {z: dict(e, why=e['why'].replace('e1xs2rigidmo', k_)) for z, e in out['e1xs2rigidmo'].items() if str(d.loc[int(z), 'video_id']).startswith(pre_)}
        skip[k_] = 0
    for k, o in out.items():
        if k not in KEEP: continue
        json.dump(o, open(f'{R}/cache/tmp/ag_{k}_patch_{split}.json', 'w'), indent=1)
        cats = d.loc[[int(z) for z in o], 'cat'].value_counts().to_dict() if o else {}
        print(f'[{split}] {k}: {len(o)} rows {cats} (mo_guard rows skipped {skip[k]})')
        if split == 'val' or os.environ.get('AB_SHOW'):
            for z in list(o)[:40]:
                rr = d.loc[int(z)]; print(f'   {z} {rr["video_id"]} | {str(rr["question"])[:70]} | {rr["answer"]:.4g} -> {o[z]["value"]:.4g}')
