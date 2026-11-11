#!/usr/bin/env python
import os, sys, re, json, math, importlib.util
os.environ.setdefault("QP_PRIOR_ACCGUARD", "3"); os.environ.setdefault("QP_PRIOR_DRIFT", "1.5")
os.environ.setdefault("QP_FF_SUB", "1"); os.environ.setdefault("QP_FF_SUB_SRC", "simulation")
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
BASE = sys.argv[1] if len(sys.argv) > 1 else "pred_t062"
GT = sys.argv[2] if len(sys.argv) > 2 else "ffmv"
TAG = "disc3dS2"
sys.path.insert(0, R + "/geo")
import measure_v3 as MV
MV.FIX = set("abcdef"); MV.PARSE_TAG = ""
_av = sys.argv; sys.argv = [_av[0], BASE]
_sp = importlib.util.spec_from_file_location("rul268_rule", f"{R}/analysis/v3/ag/rul268/rule.py")
DR = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(DR); sys.argv = _av
DIA = ("diameter", "radius")


def p75(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.percentile(x, 75)) if len(x) else np.nan


def summ_t(tr, dim, phrase, W_, H_):
    if tr is None or tr.n == 0:
        return dict(t_note="no_track")
    e, _ = tr.ext_px(dim, phrase)
    d = dict(t_old=MV.agg_ext(e, dim) if len(e) else np.nan)
    js = tr.oke
    if len(js) == 0:
        return d
    rr = np.array([tr.st[j]["rr"] for j in js], float)
    f = 0.5 if dim == "radius" else 1.0
    bx = np.array([tr.st[j]["box"] for j in js], float)
    if W_ and H_:
        d["t_bord"] = float(np.mean((bx[:, 0] <= 2) | (bx[:, 1] <= 2) | (bx[:, 2] >= W_ - 3) | (bx[:, 3] >= H_ - 3)))
    d.update(t_nfr=len(js), t_maj=f * p75(rr[:, 0]), t_clip=bool(getattr(tr, "clipped_extent", False)))
    return d


def scan(split, G, P):
    want = {}
    for i, p in P.items():
        q = p.get("q") or {}
        if q.get("kind") == "size" and q.get("dim") in DIA and i in G.index and "3d" in str(G.meas_mode[i]) \
                and DR.dclass((p.get("target") or {}).get("phrase")) in ("disc", "flat"):
            want.setdefault(p["video_id"], []).append(i)
    rows = []
    if not want:
        return pd.DataFrame(rows)
    VJ = {}
    for tv, tag in (("v2sc", "v2"), ("v3ntmv", "v1")):
        MV.TRACK_VER = tv
        _, vjs = MV.load_split_tracks(split, "e")
        VJ[tag] = {v: vjs[v] for v in want if v in vjs}
    for v, ids in want.items():
        MV.CUR_VID.clear(); MV.CUR_VID.update(src=str(P[ids[0]].get("video_source", "")).replace("segmentaion", "segmentation"),
                                              is2d=str(P[ids[0]]["video_type"])[1] == "2")
        VV = {}
        for i in ids:
            p = P[i]; q = p["q"]
            mt = G.meas_tracks[i] if G.meas_tracks[i] in ("v1", "v2") else "v2"
            d = dict(id=i, vid=v, cat=p["cat"], tdia=True, tgt=p["target"]["phrase"], mode=str(G.meas_mode[i]), msrc=G.merge_src[i], mtr=mt)
            if v not in VJ[mt]:
                d["t_note"] = "no_track_file"; rows.append(d); continue
            if mt not in VV:
                VV[mt] = MV.Video(VJ[mt][v], "e")
            vj = VJ[mt][v]
            d.update(summ_t(VV[mt].get(p["target"]), q["dim"], p["target"]["phrase"], vj.get("W"), vj.get("H")))
            d["W"] = float(vj.get("W") or np.nan)
            rows.append(d)
    return pd.DataFrame(rows).set_index("id")


pd.set_option("display.width", 400); pd.set_option("display.max_rows", 200); pd.set_option("display.max_colwidth", 200)
for split in ("test",):
    det = pd.read_csv(f"{R}/submissions/{BASE}_{'detail' if split == 'test' else 'valdetail'}.csv", index_col=0)
    G = pd.read_csv(f"{R}/preds/{split}_geo_v4nt{GT}srngsr_d_vw_raw.csv", index_col=0,
                    usecols=["Unnamed: 0", "merge_src", "geo_value", "meas_value", "meas_f", "meas_tracks", "meas_mode"])
    P = {}
    for ln in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        p_ = json.loads(ln); P[int(p_["id"])] = p_
    S = scan(split, G, P)
    S.to_csv(f"{R}/cache/tmp/ag_{TAG}_scan_{split}.csv")
    out, logs = {}, []
    for i, r in S.iterrows():
        cls = DR.dclass(r.tgt)
        L = dict(id=i, vid=r.vid, cat=r["cat"], cls=cls, mode=r["mode"], msrc=r.msrc, old=r.get("t_old", np.nan), new=r.get("t_maj", np.nan))
        old, maj = L["old"], L["new"]
        gd = None
        if not (np.isfinite(old) and np.isfinite(maj) and old > 0):
            gd = "nan"
        elif r.get("t_nfr", 0) < 3:
            gd = "few_frames"
        elif bool(r.get("t_clip", False)) or r.get("t_bord", 0) > 0.3:
            gd = "border"
        elif maj / old < DR.MINF:
            gd = "small"
        elif maj / old > DR.MAXF:
            gd = "too_big"
        elif maj < DR.MINPX:
            gd = "tiny"
        elif r["cat"] == "D3":
            gd = "d3_size_route"
        elif i not in det.index:
            gd = "no_row"
        elif str(det.answer_source[i]).startswith("patch"):
            gd = "already_patched"
        if gd is None:
            rw = G.loc[i]; row = det.loc[i].copy()
            if "mode" not in det.columns:
                row["mode"] = str(rw.meas_mode)
            if "kind" not in det.columns:
                row["kind"] = P[i]["q"]["kind"]
            geo, mv = float(rw.geo_value), float(rw.meas_value)
            xg = str(rw.merge_src).startswith("xg")
            if not (np.isfinite(row.g) and row.g > 0 and np.isfinite(geo) and abs(np.log(row.g / geo)) < 1e-3):
                gd = "g_not_raw_geo"
            elif not (np.isfinite(mv) and mv > 0):
                gd = "no_meas"
            elif xg and abs(np.log(geo / mv)) > DR.XG_TOL:
                gd = f"xg_far({geo / mv:.2f})"
            elif not xg and abs(np.log(geo / mv)) > 1e-3:
                gd = "geo_not_meas"
            if gd is None:
                p = P[i]; tk = set(p.get("target_dkeys") or [])
                dd = [e["d"] for e in (p.get("depth") or []) if e.get("key") in tk and e.get("t") is None]
                u = DR.UNIT.get(str((p.get("q") or {}).get("unit", "")).lower(), np.nan)
                if len(dd) != 1 or not np.isfinite(r.W) or not np.isfinite(u) or not np.isfinite(rw.meas_f):
                    gd = "no_repro_check"
                else:
                    rep = old * dd[0] / (float(rw.meas_f) * r.W) * u
                    L.update(repro=rep / mv, meas_f=float(rw.meas_f))
                    if abs(np.log(rep / mv)) > DR.REPRO_TOL:
                        gd = f"repro({rep / mv:.3f})"
        if gd is None:
            fac = maj / old; gnew = float(row.g * fac)
            L.update(src=row.answer_source, g=row.g, gnew=gnew, cs=row.cs, ans=row.answer, fac=fac)
            if np.isfinite(row.cs) and row.cs > 0 and abs(np.log10(row.g / row.cs)) > DR.LG2 and abs(np.log10(gnew / row.cs)) > abs(np.log10(row.g / row.cs)):
                gd = "pushes_dis"
            else:
                val, how = DR.fuse(row, gnew)
                L.update(new_ans=val, route=how)
                if val is None:
                    gd = how
                elif abs(np.log10(val / float(row.answer))) < 0.002:
                    gd = "no_change"
                else:
                    out[str(int(i))] = {"value": float(val), "why": f"{TAG}: own target '{r.tgt}' diameter {old:.0f}->{maj:.0f}px (flat {cls}, ellipse long axis, "
                                        f"{r['mode']}{', xg geo/meas ' + format(geo / mv, '.2f') if xg else ''}); own geo {row.g:.4g}->{gnew:.4g}; "
                                        f"{row.answer_source}->{how} (own CaW {row.cs:.4g})"}
        L["guard"] = gd
        logs.append(L)
    json.dump(out, open(f"{R}/cache/tmp/ag_{TAG}_patch_{split}.json", "w"), indent=1)
    print('')
    if logs:
        print(pd.DataFrame(logs).round(4).to_string(index=False))
