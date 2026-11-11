#!/usr/bin/env python
import json, sys
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
sys.path.insert(0, f"{R}/geo"); sys.path.insert(0, f"{R}/tools")
import measure_v3 as MV
import track_common as tc
from qp_common import OUT_COLS
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70); pd.set_option("display.max_rows", 300)
split = sys.argv[1] if len(sys.argv) > 1 else "val"
import os
XG_TRK, XG_RD, XG_RV = os.environ.get("XG_TRK", "v2"), os.environ.get("XG_RD", "geo_v3m_r"), os.environ.get("XG_RV", "geo_v3v_r")
XG_BASE, XG_OUT = os.environ.get("XG_BASE", "geo_v4x_d_vw"), os.environ.get("XG_OUT", "geo_v4xg")
MV.FIX = set("abcdef"); MV.TRACK_VER = XG_TRK
byv, vjs = MV.load_split_tracks(split, "e")
VG = {}
for ln in open(f"{R}/geo/out/vlmground_{split}{os.environ.get('XG_VG', '')}.jsonl", encoding="utf-8"):
    x = json.loads(ln)
    VG.setdefault((x["video_id"], x["key"]), []).append(x)
VG2 = {}
if os.environ.get("XG_VG2"):
    for ln in open(f"{R}/geo/out/vlmground_{split}{os.environ['XG_VG2']}.jsonl", encoding="utf-8"):
        x = json.loads(ln)
        VG2.setdefault((x["video_id"], x["key"]), []).append(x)


def coincide(G):
    out = {}
    for (v, k), xs in G.items():
        vj = vjs.get(v)
        if vj is None:
            continue
        V = MV.Video(vj, "e")
        trk = V.tr.get(k)
        if trk is None or not trk.n:
            continue
        hit = n = 0
        for x in xs:
            j = x["j"]
            bl = [b for b in x["boxes_rel1000"] if b and len(b) == 4]
            if not bl or j >= len(trk.st) or not trk.st[j]:
                continue
            tb = trk.st[j]["box"]
            n += 1
            hit += max(tc.iou(tb, b) for b in bl) >= 0.5
        out[(v, k)] = (hit, n)
    return out


agree2 = coincide(VG2) if VG2 else {}
agree = {}
for (v, k), xs in VG.items():
    vj = vjs.get(v)
    if vj is None:
        continue
    V = MV.Video(vj, "e")
    trk = V.tr.get(k)
    if trk is None or not trk.n:
        continue
    hit = n = 0
    for x in xs:
        j = x["j"]
        bl = [b for b in x["boxes_rel1000"] if b and len(b) == 4]
        if not bl or j >= len(trk.st) or not trk.st[j]:
            continue
        tb = trk.st[j]["box"]
        n += 1
        hit += max(tc.iou(tb, b) for b in bl) >= 0.5
    agree[(v, k)] = (hit, n)
flag_rows, rows = set(), []
for v, recs in byv.items():
    for r in recs:
        bad = []
        for role in ("target", "target2", "prior_obj"):
            k = MV.Video.key(r.get(role))
            if not k or (v, k) not in agree:
                continue
            h, n = agree[(v, k)]
            if n >= 2 and h == 0:
                h2, n2 = agree2.get((v, k), (0, 0))
                if agree2 and n2 >= 2 and h2 > 0:
                    continue
                bad.append(role)
        if bad:
            flag_rows.add(r["id"]); rows.append((r["id"], v, ",".join(bad)))
print('')
rd = pd.read_csv(f"{R}/preds/{split}_{XG_RD}_raw.csv").set_index("Unnamed: 0")
rv = pd.read_csv(f"{R}/preds/{split}_{XG_RV}_raw.csv").set_index("Unnamed: 0").reindex(rd.index)
base = pd.read_csv(f"{R}/preds/{split}_{XG_BASE}_raw.csv").set_index("Unnamed: 0")
fl = pd.Series(base.index.isin(sorted(flag_rows)), index=base.index)
gv = pd.to_numeric(rv.geo_value, errors="coerce").where(rv.answer_source == "meas")
for tag, drop in ((f"{XG_OUT}_d_vw", False), (f"{XG_OUT}d_d_vw", True)):
    o = base.copy()
    take = fl & gv.notna()
    o.loc[take, "geo_value"] = gv[take]
    o.loc[take, "meas_mode"] = ("" if os.environ.get("XG_STRONG") == "1" else "vlm_") + rv.meas_mode[take].fillna("").astype(str)
    o.loc[take, "answer_source"] = "meas"; o.loc[take, "meas_valid"] = True
    o.loc[take, "merge_src"] = "xg_v"
    if drop:
        dr = fl & gv.isna() & (base.answer_source == "meas")
        o.loc[dr, "geo_value"] = np.nan; o.loc[dr, "meas_valid"] = False; o.loc[dr, "meas_mode"] = np.nan
        vlm = pd.to_numeric(o.vlm_value, errors="coerce"); h1 = pd.to_numeric(o.h1_value, errors="coerce").abs()
        with np.errstate(divide="ignore", invalid="ignore"):
            vok = np.isfinite(vlm) & (vlm > 0) & (np.abs(np.log10(vlm / h1)) <= 2)
        o.loc[dr, "answer_source"] = np.where(vok[dr], "vlm", "h1")
        o.loc[dr, "merge_src"] = "xg_drop"
    o["parsed_value"] = np.where(o.answer_source == "meas", o.geo_value, np.where(o.answer_source == "vlm", o.vlm_value, o.h1_value))
    o.to_csv(f"{R}/preds/{split}_{tag}_raw.csv")
    o.reset_index()[OUT_COLS].to_csv(f"{R}/preds/{split}_{tag}.csv", index=False)
    print(f"  {tag}: flagged {int(fl.sum())}, take VLM-track value {int(take.sum())}" + (f", dropped {int(dr.sum())}" if drop else ""))
if split == "val":
    T = pd.read_csv(f"{R}/analysis/v3/v7_val_rows.csv", index_col=0)
    F = pd.DataFrame(rows, columns=["id", "vid", "roles"]).set_index("id")
    F["gd"] = pd.to_numeric(rd.geo_value, errors="coerce").where(rd.answer_source == "meas").reindex(F.index)
    F["gv"] = gv.reindex(F.index); F["y"] = T.y.reindex(F.index); F["cs"] = T.cs.reindex(F.index); F["v7"] = T.f.reindex(F.index)
    F["mra7"] = T.mra.reindex(F.index); F["src7"] = T.src.reindex(F.index); F["q"] = T.q.reindex(F.index)
    print(F.round(3).to_string())
