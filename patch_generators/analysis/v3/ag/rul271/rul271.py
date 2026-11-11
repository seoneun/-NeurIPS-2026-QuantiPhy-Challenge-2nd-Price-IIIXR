"""rul271: un-flag consistency.c1_mask dup_value pairs whose two targets are distinct SAM2 tracks.

c1_mask flags two rows of one video (same kind, different target phrases) whose geo answers are equal to 1e-6: the
assumption is that both phrases grounded to one object. In 2D videos geo = px extent x metres-per-px, and px extents are
integers, so two different objects of equal integer extent give exactly equal answers. If the two targets resolve to
different SAM2 objects whose boxes barely overlap (median per-frame IoU < 0.5 over >= 10 shared frames), the grounding is
not shared and the flag is a false positive. Such rows (caw_c1 in the build) are re-fused with the build's agreement
fusion when geo and CaW agree within 1.5x; otherwise they stay caw_c1.

usage: tools/run_limited.sh --io python analysis/v3/ag/rul271/rul271.py [build]
writes cache/tmp/ag_dupiou_patch_{test,val}.json (final answers, question unit)
"""
import sys, os, json
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, f"{R}/caw"); sys.path.insert(0, f"{R}/analysis/v3"); sys.path.insert(0, f"{R}/geo"); sys.path.insert(0, f"{R}/cache/tmp")
import analyze_caw as A
import consistency as C
from rules import UNIT_SI

BUILD = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nzl"
GEO = os.environ.get("DUPIOU_GEO", "geo_v4ntrj5sQ2srngsr_d_vw")
TRK = os.environ.get("DUPIOU_TRK", "v2sc")
IOU_MAX, NMIN, RATIO = 0.5, 10, 1.5
STRONG = ("2d_ratio", "3d_calib")


def key(ent):                                   # = measure_v3.Video.key
    if not ent or not ent.get("phrase"):
        return None
    rel = ent.get("rel")
    return ent["phrase"] + "|" + ",".join(ent.get("where") or []) + ("|" + rel[0] + ":" + rel[1] if rel else "")


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def box(f):
    st = f.get("st") if isinstance(f, dict) else None
    return st.get("box") if st else None


def pair_iou(vj, oa, ob):
    fa, fb = vj["objs"].get(str(oa)), vj["objs"].get(str(ob))
    if fa is None or fb is None:
        return np.nan, 0
    v = [iou(box(x), box(y)) for x, y in zip(fa, fb) if box(x) is not None and box(y) is not None]
    return (float(np.median(v)) if v else np.nan), len(v)


def run(split):
    d = A.add_keys(A.load_val() if split == "val" else A.load_test())
    raw = pd.read_csv(f"{R}/preds/{split}_{GEO}_raw.csv").set_index("Unnamed: 0")
    g = pd.to_numeric(raw.geo_value, errors="coerce").reindex(d.index)
    mode_raw = raw.meas_mode.reindex(d.index).astype(str)
    info = C.parse_info(split, d.index)
    P = {}
    for ln in open(f"{R}/geo/out/parse_{split}_v2.jsonl", encoding="utf-8"):
        x = json.loads(ln); P[x["id"]] = x
    dname = "valdetail" if split == "val" else "detail"
    det = pd.read_csv(f"{R}/submissions/{BUILD}_{dname}.csv")
    det = det.set_index("Unnamed: 0") if "Unnamed: 0" in det.columns else det
    # ---- reproduce c1_mask, keeping the pair list
    gs = g * info.unit.map(lambda u: UNIT_SI.get(u, np.nan))
    partners, other_flag = {}, {}
    for v, idx in d.groupby("vid").groups.items():
        ix = [i for i in idx if np.isfinite(gs[i])]
        for a_ in range(len(ix)):
            for b_ in range(a_ + 1, len(ix)):
                i, j = ix[a_], ix[b_]
                if info.kind[i] != info.kind[j] or info.tphrase[i] == info.tphrase[j] or not info.tphrase[i] or not info.tphrase[j]:
                    continue
                if abs(gs[i] - gs[j]) <= 1e-6 * max(abs(gs[i]), 1e-12):
                    partners.setdefault(i, []).append(j); partners.setdefault(j, []).append(i)
        for i in ix:
            if info.kind[i] == "size" and info.pkind[i] == "size" and info.pval_si[i] and info.tphrase[i] != info.pphrase[i] \
                    and abs(gs[i] - info.pval_si[i]) <= 1e-6 * info.pval_si[i]:
                other_flag[i] = "equals_prior"
        sizes = [gs[i] for i in ix if info.kind[i] == "size"]
        if sizes:
            for i in ix:
                if info.kind[i] == "dist2" and gs[i] < 0.05 * max(sizes):
                    other_flag[i] = "dist_tiny"
    bad, why = C.c1_mask(d, g, info)
    mine = set(partners) | set(other_flag)
    print(f"[{split}] c1 flags: library {int(bad.sum())}, reproduced {len(mine)}, equal sets: {set(bad[bad].index) == mine}; "
          f"dup rows {len(partners)}, other {len(other_flag)}; build caw_c1 rows {int((det.answer_source == 'caw_c1').sum())}")
    # ---- per-pair track test
    vjs, rows = {}, []

    def obj_of(i):
        vid = d.vid[i]
        if vid not in vjs:
            p = f"{R}/geo/out/tracks_{split}_{TRK}/{vid}.json"
            vjs[vid] = json.load(open(p)) if os.path.exists(p) else None
        vj = vjs[vid]
        k = key((P.get(int(i)) or {}).get("target"))
        if vj is None or k is None:
            return None, None, k
        qe = {q["key"]: q.get("obj") for q in vj.get("queries") or []}
        qd = {q["key"]: q.get("obj") for q in vj.get("queries_d") or []}
        return qe.get(k), qd.get(k), k

    ok_row = {}
    for i, js in partners.items():
        oi, oid, ki = obj_of(i)
        res = []
        for j in js:
            oj, ojd, kj = obj_of(j)
            if oi is None or oj is None:
                m, n, passed = np.nan, 0, False
            elif oi == oj:
                m, n, passed = 1.0, -1, False
            else:
                m, n = pair_iou(vjs[d.vid[i]], oi, oj)
                passed = n >= NMIN and np.isfinite(m) and m < IOU_MAX
            res.append((j, oi, oj, m, n, passed))
            rows.append(dict(i=i, j=j, vid=d.vid[i], ki=ki, kj=kj, oi=oi, oj=oj, oi_d=oid, oj_d=ojd, iou=m, n=n, passed=passed,
                             g=g[i], kind=info.kind[i], src_i=det.answer_source.get(i), src_j=det.answer_source.get(j)))
        ok_row[i] = (all(r[5] for r in res) and i not in other_flag, res)
    PR = pd.DataFrame(rows)
    pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 34)
    if len(PR):
        print(PR[PR.i < PR.j].round(3).to_string())
    # ---- re-fuse the unflagged rows
    out, rep = {}, []
    for i, (ok, res) in sorted(ok_row.items()):
        if not ok or i not in det.index:
            continue
        src = str(det.answer_source[i]); c = det.cs[i]; gi = g[i]
        mode = str(det["mode"][i]) if "mode" in det.columns else mode_raw[i]
        r_ = max(gi, c) / min(gi, c) if (np.isfinite(gi) and np.isfinite(c) and gi > 0 and c > 0) else np.nan
        new, how = np.nan, ""
        if src == "caw_c1" and np.isfinite(r_) and r_ <= RATIO:
            strong = mode in STRONG
            if info.kind[i] == "size" and strong and d.cat[i] == "D2":
                new, how = gi ** 0.7 * c ** 0.3, "w0.7"
            elif info.kind[i] == "size" and strong:
                new, how = gi ** 0.65 * c ** 0.35, "w0.65"
            else:
                new, how = np.sqrt(gi * c), "gm"
            j0 = res[0]
            out[str(int(i))] = {"value": float(new),
                                "why": f"rul271: distinct tracks o{j0[1]}/o{j0[2]} IoU {j0[3]:.2f}, equal px extents; agree {how} g={gi:.4g} c={c:.4g}"}
        rep.append(dict(id=i, vid=d.vid[i], cat=d.cat[i], kind=info.kind[i], mode=mode, src=src, old=det.answer[i], g=gi, cs=c, ratio=r_,
                        new=new, how=how, q=str(d.question[i])[:60]))
    RP = pd.DataFrame(rep)
    if len(RP):
        print(RP.round(4).to_string())
    json.dump(out, open(f"{R}/cache/tmp/ag_dupiou_patch_{split}.json", "w"), indent=1)
    print(f"[{split}] patch rows: {len(out)} -> cache/tmp/ag_dupiou_patch_{split}.json")
    return out, PR, RP


if __name__ == "__main__":
    for sp in ("test", "val"):
        run(sp)
