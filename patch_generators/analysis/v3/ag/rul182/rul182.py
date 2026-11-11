"""rul182 (copy of rul146, inverted verifier: xg_v rows whose geo the free corner reproduces).
c79limbedge: free-corner speed for a target pinned on TWO ADJACENT image borders (coverage gap of rul432/rul433/rul066).
tedge* re-read a border-cut axis from the free edge but accept only when an uncut run of the same track confirms the speed
(run_px). A limb (hand/arm/...) that reaches in from a frame corner is cut on two adjacent borders in every frame, so no uncut
run exists and tedge cannot run. Its box is then the rectangle spanned by the pinned frame corner and the limb's extreme
point: the box CENTRE moves at exactly half the speed of the free (opposite) corner, and the free corner is the hand/tip.
Gates: 2D ratio mode (2d_ratio / vlm_2d_ratio), static prior; speed/avgspeed with a time window; >=95% of window frames touch
exactly one adjacent border pair (LT/RT/LB/RB), the same pair; no uncut run (rul433 run_px NaN); the free corner never touches
a border in the window; target noun is a limb (hand|arm|foot|leg|finger|paw) or the pin holds over >=95% of the whole clip;
free-corner path >= 1.5x box-centre path; not already patched. Value: free-corner px speed (rul433 own-track diag new_px when
present, the same measure_v3 fit otherwise) rescales the row's geo (g x new_px / old_px), fused with CaW by rul433's fuse().
Labels only evaluate.
writes cache/tmp/ag_c79limbedge_patch_{test,val}.json
usage: tools/run_limited.sh --io python analysis/v3/ag/c79limbedge/c79limbedge.py [SUB=sub_xt_Nzz24]"""
import os, sys, json, re, math
os.environ.setdefault("QP_PRIOR_ACCGUARD", "3"); os.environ.setdefault("QP_PRIOR_DRIFT", "1.5")
os.environ.setdefault("QP_FF_SUB", "1"); os.environ.setdefault("QP_FF_SUB_SRC", "simulation")
os.environ.setdefault("QP_FF_SHORT", "30:1.0:0.06"); os.environ.setdefault("QP_FF_SIZEJIT", "0.02")
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
SUB = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nzz27"
TAG = "rul182"
sys.path.insert(0, R + "/geo")
import measure_v3 as MV
MV.FIX = set("abcdef"); MV.PARSE_TAG = ""
src = open(f"{R}/analysis/v3/ag/rul433/rule.py").read().split("\nL = labelled(SUB)")[0]
_argv = sys.argv; sys.argv = ["x", SUB, "0.03", "0.01", TAG, "own"]
G = {"__name__": "tedge3_head"}; exec(compile(src, "tedge3_rule_head", "exec"), G); sys.argv = _argv
fuse, A, labelled = G["fuse"], G["A"], G["labelled"]
D_ = {"__name__": "diag_own_head"}
dsrc = open(f"{R}/analysis/v3/ag/rul433/diag_own.py").read().split('\nif __name__ == "__main__":')[0].replace("SPLIT = sys.argv[1]", "SPLIT = None")
exec(compile(dsrc, "diag_own_head", "exec"), D_)
used_frames, sides, px_on = D_["used_frames"], D_["sides"], D_["px_on"]
MV.FIX = set("abcdef"); MV.PARSE_TAG = ""
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 70); pd.set_option("display.max_rows", 300)
TOL = 2.0
PAIRS = {"LT", "RT", "LB", "RB"}
LIMB = re.compile(r"\b(hand|arm|foot|leg|finger|paw)s?\b")
F_PIN, F_CLIP, R_PATH = 0.95, 0.95, 1.5
MODES = {"2d_ratio", "vlm_2d_ratio"}


def pin_pair(S):
    """per frame: the adjacent pair if the box touches EXACTLY that pair of borders, else ''"""
    out = []
    for s in S:
        on = "".join(c for c, f in zip("LRTB", s) if f)
        out.append(on if on in PAIRS else "")
    return np.array(out, object)


def corner_P(tr, pair):
    box = np.array([tr.st[j]["box"] for j in tr.okc], float)
    fx = box[:, 2] if pair[0] == "L" else box[:, 0]          # pinned left -> free right edge
    fy = box[:, 3] if pair[1] == "T" else box[:, 1]          # pinned top -> free bottom edge
    return np.stack([fx, fy], 1), box


def path(P, idx):
    Q = P[idx]
    return float(np.linalg.norm(np.diff(Q, axis=0), axis=1).sum()) if len(Q) > 1 else 0.0


def run(split, det, raw, diag):
    MV.TRACK_VER = "v2sc"
    byv, vjs = MV.load_split_tracks(split, "e")
    rows = []
    for v, recs in byv.items():
        cand = [r for r in recs if r["q"]["kind"] in ("speed", "avgspeed") and not (r["q"]["kind"] == "speed" and r["q"].get("t") is None)]
        if not cand or v not in vjs:
            continue
        MV.CUR_VID.clear(); MV.CUR_VID.update(src=str(recs[0].get("video_source", "")), is2d=str(recs[0]["video_type"])[1] == "2")
        vj = vjs[v]; V = MV.Video(vj, "e"); W, H = float(vj["W"]), float(vj["H"])
        for r in cand:
            i = r["id"]
            if i not in raw.index or i not in det.index:
                continue
            rr, dd = raw.loc[i], det.loc[i]
            mode = str(rr.meas_mode)
            if mode not in MODES or not str(dd["cat"]).startswith("S"):
                continue
            q = r["q"]
            tr = V.get(r["target"])
            if tr is None or tr.n < 3:
                continue
            w, win = used_frames(tr, q["kind"], q)
            if w is None or len(w) < 3:
                continue
            S = sides(tr, W, H)
            pp = pin_pair(S)
            pw = pp[w]
            if not (pw != "").any():
                continue
            vals, cnt = np.unique(pw[pw != ""].astype(str), return_counts=True)
            pair = str(vals[int(np.argmax(cnt))]); fpin = float((pw == pair).mean())
            if fpin < 0.5:
                continue
            Pc, box = corner_P(tr, pair)
            fx_touch = (Pc[w, 0] >= W - 1 - TOL) if pair[0] == "L" else (Pc[w, 0] <= TOL)
            fy_touch = (Pc[w, 1] >= H - 1 - TOL) if pair[1] == "T" else (Pc[w, 1] <= TOL)
            free_touch = int((fx_touch | fy_touch).sum())
            fclip = float((pp == pair).mean())
            tgt = ((r.get("target") or {}).get("phrase") or "").lower()
            limb = bool(LIMB.search(tgt))
            Pb = np.stack([(box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2], 1)
            pr = path(Pc, w) / max(path(Pb, w), 1e-9)
            corner_px = px_on(tr, q["kind"], q, Pc)
            boxc_px = px_on(tr, q["kind"], q, Pb)
            dg = diag.loc[i] if i in diag.index else None
            dnew = float(dg.new_px) if (dg is not None and pd.notna(dg.new_px)) else np.nan
            new_px = dnew if (np.isfinite(dnew) and str(dg.sidesw) == pair) else corner_px
            old_px = MV.px_quantity(tr, q["kind"], q.get("t"), q.get("t1"), q.get("t2"), q.get("when"))
            run_px = float(dg.run_px) if (dg is not None and pd.notna(dg.run_px)) else np.nan
            mpp = float(rr.meas_mpp)
            g = float(dd.g) if pd.notna(dd.g) else np.nan
            pk = ((r.get("priors") or [{}])[0] or {}).get("kind")
            gates = dict(g_static=pk not in ("speed", "avgspeed", "accel"), g_pin=fpin >= F_PIN, g_norun=not np.isfinite(run_px),
                         g_free=free_touch == 0, g_limb=limb or fclip >= F_CLIP, g_path=pr >= R_PATH,
                         g_unpatched=not str(dd.answer_source).startswith("patch"))
            rows.append(dict(id=i, vid=v, cat=dd["cat"], qk=q["kind"], t=q.get("t"), t1=q.get("t1"), t2=q.get("t2"), tgt=tgt, tkey=tr.q.get("key"),
                             mode=mode, pair=pair, nw=len(w), fpin=round(fpin, 3), fclip=round(fclip, 3), free_touch=free_touch, limb=limb,
                             path_ratio=round(pr, 3), old_px=old_px, boxc_px=boxc_px, corner_px=corner_px, diag_new_px=dnew,
                             new_px=new_px, run_px=run_px, mpp=mpp, meas=float(rr.meas_value), g=g, gnew_mpp=new_px * mpp, cs=dd.cs, answer=dd.answer,
                             src=dd.answer_source, merge_src=str(rr.merge_src), **gates, all=all(gates.values())))
    return pd.DataFrame(rows).set_index("id") if rows else pd.DataFrame()


L = labelled(SUB)
for split in ("test", "val"):
    det = pd.read_csv(f"{R}/submissions/{SUB}_{'detail' if split == 'test' else 'valdetail'}.csv", index_col=0)
    raw = pd.read_csv(f"{R}/preds/{split}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv", index_col=0)
    diag = pd.read_csv(f"{R}/cache/tmp/ag_tedge3_diagown_{split}.csv", index_col=0)
    D = run(split, det, raw, diag)
    P = {}
    print(f"\n===== {split}: candidate rows (2D ratio, S cat, a pinned adjacent pair in >=50% of window frames): {len(D)}")
    if len(D):
        print(D.drop(columns=["tgt"]).round(4).to_string())
        for i, x in D[D["all"]].iterrows():
            # inverted verifier: only xg_v-merged rows whose geo is NOT the own-track measurement (the rows rul146 skips),
            # and the free-corner speed (new_px*mpp) must independently reproduce that xg_v geo within 5%
            if not (x.merge_src == "xg_v" and np.isfinite(x.g) and x.meas and abs(x.g / x.meas - 1) >= 0.05):
                continue
            agree = abs(x.gnew_mpp / x.g - 1)
            print(f"  cand {i} {x.vid} g {x.g:.4g} meas {x.meas:.4g} gnew {x.gnew_mpp:.4g} |gnew/g-1| {agree:.3f}")
            if not (agree < 0.05):
                continue
            gn = math.sqrt(x.g * x.gnew_mpp)
            s_ = "dis_geo" if str(x.src) in ("dis_caw_routed", "caw_nogeo", "caw_c1") else x.src
            r_ = pd.Series(dict(answer_source=s_, answer=x.answer, g=x.g, cs=x.cs))
            v, how = fuse(r_, gn, x.qk, True)   # strong=True per rule spec (vlm_2d_ratio included)
            print(f"  FIRE {i} {x.vid} {x.qk} {x.tgt!r} pair {x.pair} fpin {x.fpin} path x{x.path_ratio}: px {x.old_px:.1f}->{x.new_px:.1f}; "
                  f"g {x.g:.4g}->{gn:.4g} (mpp*px {x.gnew_mpp:.4g}); cs {x.cs}; {x.src} -> {v} ({how})")
            if v is None or abs(v / x.answer - 1) < 1e-6:
                continue
            P[str(int(i))] = dict(value=float(v), why=f"{TAG}: target '{x.tgt}' box pinned on two adjacent borders ({x.pair}) in {x.fpin:.0%} of the "
                                                      f"{x.nw} window frames (no uncut run, tedge cannot match), free corner never on a border; free-corner "
                                                      f"path x{x.path_ratio:.2f} the box centre; px speed {x.old_px:.1f}->{x.new_px:.1f}; xg_v geo {x.g:.4g} reproduced by free corner {x.gnew_mpp:.4g} -> {gn:.4g}; src {x.src}; {how}")
    json.dump(P, open(f"{R}/cache/tmp/ag_{TAG}_patch_{split}.json", "w"), indent=1)
    print(f"[{split}] patched {len(P)}:", {k: round(v['value'], 4) for k, v in P.items()})
    LL = L[L.split == split]
    hit = [k for k in P if int(k) in LL.index]
    print(f"[{split}] labelled rows changed: {len(hit)}")
    for k in hit:
        y = LL.y[int(k)]
        print("   ", k, "y", y, "old", LL.answer[int(k)], A.row_mra(np.array([LL.answer[int(k)]]), np.array([y]))[0], "new", P[k]["value"],
              A.row_mra(np.array([P[k]["value"]]), np.array([y]))[0])
    if len(D):
        lab = D.index.intersection(LL.index)
        if len(lab):
            X = D.loc[lab].join(LL[["y"]])
            X["mra_g"] = A.row_mra(X.g.values, X.y.values); X["mra_gnew"] = A.row_mra(X.gnew_mpp.values, X.y.values)
            print(f"[{split}] labelled candidates (any gates):")
            print(X[["vid", "qk", "merge_src", "meas", "gnew_mpp", "pair", "fpin", "free_touch", "limb", "path_ratio", "g_norun", "all", "y", "g", "mra_g", "mra_gnew"]].round(3).to_string())
