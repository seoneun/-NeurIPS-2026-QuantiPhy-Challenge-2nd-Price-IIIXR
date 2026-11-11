"""c98tailgeo: S3/D3 calibrated length-including-tail rows: answer geo when geo > CaW (<=1.6x) and the
video's ruler is sane. Physical argument: CaW's body-length prior for animals excludes the tail (tail adds
~40-60% of body length for a cat), so geo's box span (which spans tail) is the better measurement."""
import os, re, sys, json
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../..")); os.chdir(R); sys.path.insert(0, R + '/cache/tmp')
TAG = 'rul227'; SUB = 'sub_xt_Nzz36'
WHY = 'rul227: CaW body-length prior excludes the tail'
TAILQ = re.compile(r'including (its|the) tail', re.I)
SRC_OK = {'d3size_dimgm', 'd3size_unstable_gm'}

def veto_ids(split):
    f = f'cache/tmp/ag_c98factveto3d_patch_{split}.json'
    return set(json.load(open(f))) if os.path.exists(f) else set()

def ruler_sane(sib):
    ok = False
    for _, s in sib.iterrows():
        src = str(s.answer_source); g = s.g; a = s.answer
        if s['kind'] in ('size',): continue
        if (src.startswith('depth_interp') or src.startswith('kin') or src.startswith('agree')) and np.isfinite(g) and a > 0:
            if abs(g / a - 1) <= 0.15: ok = True
    if ok: return True, 'ruler_wit'
    sp = sib[sib.answer_source.astype(str) == 'scene_prior']
    for _, s in sp.iterrows():
        if np.isfinite(s.g) and s.answer > 0:
            r = s.g / s.answer
            if r < 0.67 or r > 1.5: return False, 'scene_prior_bad'
    return True, 'no_bad_scene_prior'

def apply(d, split):
    veto = veto_ids(split); out = {}; diag = []
    for rid, r in d.iterrows():
        if r['cat'] not in ('S3', 'D3') or r['kind'] != 'size': continue
        q = str(r.question)
        if not (re.search(r'\blength\b', q, re.I) and TAILQ.search(q)): continue
        if str(r.answer_source) not in SRC_OK: continue
        if not str(r['mode']).startswith('3d_calib') or 'tdepth' in str(r['mode']) or 'fov' in str(r['mode']): continue
        g, cs = r.g, r.cs
        if not (np.isfinite(g) and np.isfinite(cs) and cs > 0 and g > cs and g / cs <= 1.6): continue
        if str(rid) in veto: continue
        sane, how = ruler_sane(d[(d.video_id == r.video_id) & (d.index != rid)])
        diag.append((split, rid, r.video_id, q, r.answer, g, cs, sane, how))
        if sane: out[str(rid)] = {'value': float(g), 'why': WHY}
    return out, diag

if __name__ == '__main__':
    from lab_rows import labelled
    import analyze_caw as A
    d = pd.read_csv(f'submissions/{SUB}_detail.csv', index_col=0)
    pt, dg = apply(d, 'test')
    L = labelled(SUB)
    v = L[L.split == 'val']
    pv, dg2 = apply(v, 'val')
    for x in dg + dg2: print(x)
    json.dump(pt, open(f'cache/tmp/ag_{TAG}_patch_test.json', 'w'), indent=1)
    json.dump(pv, open(f'cache/tmp/ag_{TAG}_patch_val.json', 'w'), indent=1)
    print('test', len(pt), 'val', len(pv))
    # all-237 labelled (test copies too)
    pl, dg3 = apply(L, 'lab')
    print('labelled changed', len(pl), dg3); print(L.index[:5].tolist())
    # broader: labelled rows with tail-length questions at all
    m = L.question.astype(str).str.contains('tail', case=False)
    print(L[m][['split','video_id','cat','question','answer','answer_source','g','cs','mode','y','m','mg','mc']].to_string())
