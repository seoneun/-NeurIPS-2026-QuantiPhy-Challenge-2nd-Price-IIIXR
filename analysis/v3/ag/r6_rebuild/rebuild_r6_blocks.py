#!/usr/bin/env python
import os, sys, re, json, pickle, subprocess
sys.dont_write_bytecode = True
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
FZ = f'{HERE}/frozen'
CATS_ALL = ['S2', 'D2', 'S3', 'D3']
BLOCKS = [
    ('heldS2', 'S2', 'ag_r6_heldS2_patch_test.json', 'held:S2'),
    ('heldD2', 'D2', 'ag_r6_heldD2_patch_test.json', 'held:D2'),
    ('d2rabbit', 'D2', 'ag_rul376_d2rabbit_patch_test.json', 'rul376:d2rabbit'),
    ('d2chairv', 'D2', 'ag_rul376_d2chairv_patch_test.json', 'rul376:d2chairv'),
    ('heldS3', 'S3', 'ag_r6_heldS3_patch_test.json', 'held:S3'),
    ('s3ball40', 'S3', 'ag_rul376_s3ball40_patch_test.json', 'rul376:s3ball40'),
    ('s3hit29', 'S3', 'ag_r5_s3conv_s3hit29_patch_test.json', 'r5_s3conv:s3hit29'),
    ('s3dist32', 'S3', 'sub_dist32.json', 'r6_s3latF:dist32'),
    ('s3dist31', 'S3', 'sub_dist31.json', 'r6_s3latF:dist31'),
    ('heldD3', 'D3', 'ag_r6_heldD3_patch_test.json', 'held:D3'),
    ('d3labF', 'D3', 'ag_rul402_patch_test.json', 'rul402'),
    ('d3labF2E1', 'D3', 'ag_rul403_E1_patch_test.json', 'rul403:E1'),
    ('d3labF2E2', 'D3', 'ag_rul403_E2_patch_test.json', 'rul403:E2'),
    ('d3labF2E3', 'D3', 'ag_rul403_E3_patch_test.json', 'rul403:E3a'),
]




def held(od):
    H = pd.read_csv(f'{R}/analysis/v3/calib/r5/held_bundle_rows.csv', dtype={'new_json': str})
    out = {}
    for c in CATS_ALL:
        h = H[H.cat == c]
        txt = json.dumps({str(int(i)): {'value': float(v), 'why': f'r5 held bundle ({k})'} for i, v, k in zip(h.id0, h.new_json, h.key)},
                         indent=0)
        open(f'{od}/ag_r6_held{c}_patch_test.json', 'w').write(txt)
        out[f'held:{c}'] = {int(i): float(v) for i, v in zip(h.id0, h.new_json)}
    return out


def r5conv(od):
    out = {}
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'}
    for b in ('d2rabbit', 'd2chairv', 's3ball40'):
        p = subprocess.run([sys.executable, f'{R}/analysis/v3/ag/rul376/reproduce.py', b], cwd=R, env=env, capture_output=True, text=True)
        open(f'{od}/rul376_{b}.out', 'w').write(p.stdout); open(f'{od}/rul376_{b}.err', 'w').write(p.stderr)
        if p.returncode != 0:
            sys.exit('')
        v = {}
        for line in p.stdout.splitlines():
            if line.strip():
                i, x = line.split(); v[int(i)] = float(x)
        out[f'rul376:{b}'] = v
    return out


W_, H_ = 854, 480


def r4tab(path):
    L = open(path).read().splitlines()
    f = float(re.search(r'f= *([0-9.]+)', L[1]).group(1)); rows = []
    for s in L[3:]:
        a = s.split('|')[0].split()
        try: rows.append([float(x) for x in a[:7]]) if len(a) >= 7 else None
        except ValueError: continue
    return pd.DataFrame(rows, columns=['t', 'u', 'v', 'dpx', 'bd', 'Dlin', 'Dsz']), f


def pos(T, f, i, dcol, lat):
    D = T[dcol].iloc[i]
    return np.array([lat * (T.u.iloc[i] - W_ / 2) * D / f, lat * (T.v.iloc[i] - H_ / 2) * D / f, D])


def s3hit29():
    T, f = r4tab(f'{FZ}/b29.out'); fps = 24.0
    i0 = int(np.argmin(np.abs(T.t.values - 1.24)))
    def fd(i, k, dcol, lat): return float(np.linalg.norm(pos(T, f, i + k, dcol, lat) - pos(T, f, i, dcol, lat))) * 100 * fps / k
    return {'r5_s3conv:s3hit29': {2494: fd(i0, 5, 'Dlin', 1.3)}}


def d3labF(b27):
    W = f'{R}/analysis/v3/ag/rul402'
    MOT = {"speed", "avgspeed", "disp", "path", "dist2"}
    BASE = {"": "geo", "rul301": "geo re-measure (rul301)", "rul161": "copies' majority of rul301 values", "rul298": "geo copies log-mean",
            "rul285": 't18', "rul082": "gravity physics v >= g|t - t_apex|", "rul265": "gravity physics v_y = g(t_apex - t)"}
    X = pd.read_csv(f"{W}/free_rows_model.csv")
    X["w1"] = X.writers.fillna("").str.split(r"[,|]").str[0]
    m = X["mode"].astype(str)
    X["pfs"] = np.where(m.str.contains("fovdefault"), "spec", np.where(m.str.contains("knownsize"), "known", np.where(m.str.contains("calib"),
               np.where(X.grav, "grav", np.where(X.prior_kind == "size", "size", "annot")), "none")))
    STATIC = {"captured_0052"}
    ok = ~X.rec.isin(STATIC) & X.kind.isin(MOT) & (X.F >= 1.15) & (X.scored == 1) & (X.taken == 0) & X.held.isna() & ~X.prw & X.b27.notna()
    selA = ok & X.pfs.isin(["spec", "grav"]) & X.w1.isin(BASE.keys())
    selB = ok & (X.pfs == "annot") & (X.Fp <= 1.05) & ((X.b27 / X.g - 1).abs() <= 0.03)
    X["sub"] = np.where(selA, "A", np.where(selB, "B", ""))
    sel = selA | selB
    B = X[sel].copy()
    B["qn"] = B.question.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    B["Fq"] = B.groupby(["rec", "qn"]).F.transform("median")
    B["new"] = B.b27 * B.Fq
    ids = B.id0.astype(int).values
    return {'rul402': dict(zip(ids.tolist(), B.new.astype(float).tolist()))}


def qn(s): return s.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def table():
    W = f"{R}/analysis/v3/ag/rul402"; W2 = f"{R}/analysis/v3/ag/rul403"
    K = pd.read_csv(f"{W2}/cand_all.csv")
    M = pd.read_csv(f"{W}/free_rows_model.csv").set_index("id0")
    FR = pd.read_csv(f"{FZ}/free_rows.csv").set_index("id0")
    for c in ("video", "cat", "question", "fW_spec", "source"): K[c] = M.loc[K.id0, c].values
    K["r4"] = FR.loc[K.id0, "r4"].values; K["b27"] = FR.loc[K.id0, "b27"].values
    w = K.writers.fillna("")
    EXCL = [(w.str.contains("rul438"), 't19'),
            (w.str.contains("rul141"), "distance between static people (rul402 exclusion)"),
            (K.rec == "captured_0052", "distance between static props, possibly tape-measured (rul402 exclusion)"),
            (w.str.contains("rul156"), "distance between static props (white cup, basketball, book; no time in the question), as 0052"),
            (w.str.contains("rt1|rul203|rul207"), 't20')]
    K["excl"] = ""
    for m, why in EXCL: K.loc[m & (K.excl == ""), "excl"] = why
    fused = (K.portal.isna() | (K.portal.fillna("") == "")) & (w == "") & ((K.b27 / K.g - 1).abs() > 0.3)
    K["sub"] = np.select([K.excl != "", K.grp == "phys", K.portal.fillna("") != "", fused], ["X", "E1", "E2", "E3b"], "E3a")
    B = pd.read_csv(f"{W}/rows.csv"); B["qn"] = qn(B.question)
    bF = B.groupby(["rec", "qn"]).Fq.first().to_dict()
    K["qn"] = qn(K.question)
    keep = K["sub"].isin(["E1", "E2", "E3a", "E3b"])
    med = K[keep].groupby(["rec", "qn"]).F.median().to_dict()
    K["Fq"] = [bF.get((r, q), med.get((r, q), np.nan)) for r, q in zip(K.rec, K.qn)]
    K["new"] = K.b27 * K.Fq
    return K, B


def d3labF2(b27):
    K, _ = table(); out = {}
    for sb in ('E1', 'E2', 'E3a'):
        k = K[K['sub'] == sb]; ids = k.id0.astype(int).values
        out[f'rul403:{sb}'] = dict(zip(ids.tolist(), k.new.astype(float).tolist()))
    return out


GROUPS = {"dist32": [2541, 2547, 2553], "dist31": [2524, 2530, 2535]}


def s3latF(b27):
    sys.path.insert(0, f"{R}/analysis/v3/ag/rul307"); import remeasure as RM
    recs = {v: pickle.load(open(f"{FZ}/rul307_traj_test_{v}_prodP_q3_sub.pkl", "rb")) for v in "vm"}
    def recF(i):
        for v in "vm":
            r = recs[v].get(i)
            try:
                a, b = RM.value(r, 1.0), RM.value(r, 1.3)
                if a is not None and b is not None and np.isfinite(a) and np.isfinite(b) and a > 0: return float(b / a)
            except Exception: pass
        return np.nan
    out = {}
    for g, ids in GROUPS.items():
        Fm = float(np.median([recF(i) for i in ids]))
        out[f'r6_s3latF:{g}'] = {i: float(b27[i] * Fm) for i in ids}
    return out


def main():
    out = os.path.abspath(sys.argv[1]); od = f'{out}/r6'; os.makedirs(od, exist_ok=True)
    p27 = f'{out}/pred_t011_id1.csv'
    b27 = pd.read_csv(p27).parsed_value.values
    V = {}
    V.update(held(od)); V.update(r5conv(od)); V.update(s3hit29()); V.update(d3labF(b27)); V.update(d3labF2(b27)); V.update(s3latF(b27))
    for name, cat, fn, src in BLOCKS:
        got = V[src]
        if not src.startswith('held:'):
            json.dump({str(i): {'value': got[i], 'why': f'{name}: {src}'} for i in got}, open(f'{od}/{fn}', 'w'), indent=1)
    lines = ['name\tpred_t003', 'base\tpred_t011']
    lines += [f'block\t{n}\t{c}\tjson:{os.path.relpath(f"{od}/{fn}", R)}' for n, c, fn, _ in BLOCKS]
    open(f'{od}/pred_t003.tsv', 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
