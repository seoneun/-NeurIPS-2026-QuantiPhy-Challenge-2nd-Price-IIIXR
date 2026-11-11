#!/usr/bin/env python
"""analysis/v3/ag/rul320/rul320.py - caw_nogeo rows whose geo was computed but rejected by the sanity checks
("2-decade sanity rule" / "implausible(...)"). Two code-detectable failures are repaired; everything else is left alone.

A  cross-dimension 2-decade rejection of a gravity-ruled 2D size (sim droplet/bottle, internet ping-pong ball):
   the arbitration compares the measurement with the PRIOR's number even when the prior is an acceleration
   (7.7 mm droplet vs "9.7 m/s^2" = 3.1 decades -> rejected). The measurement itself is a 2d_ratio size from a free-fall
   scale. Restored when: kind size, prior kind accel, mode 2d_ratio, the reconstructed px extent matches the stored value,
   the extent used is >= 8 px, and it is not the short side of a flat box (bw >= 3 bh: a flat object seen at a grazing
   angle, the vertical extent is foreshortened - the sim phones, 5 px "width"). A question that says "horizontal"
   (parse q.horizontal) takes the horizontal box extent instead of the rotated-rect long side (a falling teardrop's
   long side is vertical).
C  "implausible" 3D size from a corrupt depth_info entry: the target's camera distance is >20x every other distance of
   the row (captured_0040*: "distance_white_cup_camera = 8420m" next to 0.6900m / 0.9100m / 0.9730m). An integer with
   the other entries' 4-digit mantissa is read as 0.<digits> m (dropped "0."), kept only if that lies inside the
   scene's depth range (x1.5); 3d_calib size is linear in the target depth, so geo x Z_new / Z_old.
Final answer = pipeline fusion: CaW (x1.12 caw_prescale for simulation D2 sizes) within 2x -> g^w c^(1-w)
(w 0.7 D2, 0.65 otherwise: --agree_w size:0.65:strong + probe D2:agree_wgm:size:w0.7), else geo (strong modes).
usage: python analysis/v3/ag/rul320/rul320.py [sub=sub_xt_Nzi]
writes cache/tmp/ag_ngsanity_patch_{test,val}.json"""
import json, os, re, sys
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, f"{R}/caw"); sys.path.insert(0, f"{R}/analysis/v3"); sys.path.insert(0, f"{R}/geo"); sys.path.insert(0, f"{R}/cache/tmp")
import consistency as C_  # noqa: E402
from rules import UNIT_SI  # noqa: E402

SUB = sys.argv[1] if len(sys.argv) > 1 else "sub_xt_Nzi"
GEO = "geo_v4ntrj5sQ2srngsr_d_vw"
MEAS = "v3m_rrj5sQ"           # the measurement file the Q2 raw was merged from (mode of rejected rows)
MEAS_V = "v3v_rntrj5sQ"       # the VLM-track measurement the merge takes for xground-flagged rows (merge_src xg_v)
TRK = "_v2sc"
MIN_PX = 8.0
FLAT = 3.0
PL_SIZE = (1e-4, 500.0)


def grp_of(w):
    return str(w).split(":")[0].split("(")[0]


def ext_stats(split, vid, key):
    p = f"{R}/geo/out/tracks_{split}{TRK}/{vid}.json"
    if not os.path.exists(p):
        return None
    T = json.load(open(p))
    if not isinstance(T.get("queries"), list) or "objs" not in T:
        return None
    qs = [q for q in T["queries"] if q.get("key") == key]
    if not qs:
        return None
    rows = []
    for x in T["objs"][str(qs[0]["obj"])]:
        s = x.get("st") if isinstance(x, dict) and "st" in x else x
        if not s or s.get("border", 0) != 0:
            continue
        b = s["box"]
        rows.append((b[2] - b[0], b[3] - b[1], s["rr"][0], s["rr"][1]))
    if len(rows) < 3:
        return None
    a = np.array(rows, float)
    return dict(bw=float(np.median(a[:, 0])), bh=float(np.median(a[:, 1])), rr0=float(np.median(a[:, 2])),
                rr1=float(np.median(a[:, 3])), rr1p75=float(np.percentile(a[:, 3], 75)), n=len(a))


def depth_entries(di):
    """[(key, raw number string, value m)] of the static and timed distance entries"""
    out = []
    for m in re.finditer(r"distance_(\w+?)_camera\s*=\s*([\d.]+)\s*m", str(di)):
        out.append((m.group(1), m.group(2), float(m.group(2))))
    return out


def run(split):
    det = f"{R}/submissions/{SUB}_detail.csv" if split == "test" else f"{R}/submissions/{SUB}_valdetail.csv"
    D = pd.read_csv(det).set_index("Unnamed: 0")
    G = pd.read_csv(f"{R}/preds/{split}_{GEO}_raw.csv").set_index("Unnamed: 0").reindex(D.index)
    assert (G.video_id.astype(str).values == D.video_id.astype(str).values).all(), "raw rows misaligned"
    mp = f"{R}/geo/out/meas_{split}_v2_{MEAS}.csv"
    Mm = pd.read_csv(mp).set_index("id") if os.path.exists(mp) else pd.DataFrame(columns=["mode", "value"])
    mpv = f"{R}/geo/out/meas_{split}_v2_{MEAS_V}.csv"
    Mv = pd.read_csv(mpv).set_index("id") if os.path.exists(mpv) else pd.DataFrame(columns=["mode", "value"])
    info = C_.parse_info(split, D.index)
    P = {}
    for ln in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
        j = json.loads(ln); P[int(j["id"])] = j
    src = D.answer_source.astype(str).str.replace("mo_guard:", "", regex=False)
    cand = (src == "caw_nogeo") & G.meas_why.map(grp_of).isin(["2-decade sanity rule", "implausible"])
    out, log = {}, []
    for i in D.index[cand]:
        p = P.get(int(i)) or {}
        q = p.get("q") or {}
        g = G.loc[i]
        mv = float(g.meas_value) if pd.notna(g.meas_value) else np.nan
        mode = str(Mm["mode"].get(i, "")) if i in Mm.index else ""
        mval = float(Mm["value"].get(i, np.nan)) if i in Mm.index else np.nan
        unit = q.get("unit") or info.unit.get(i)
        usi = UNIT_SI.get(unit, np.nan)
        rec = dict(id=i, vid=g.video_id, cat=D.loc[i, "cat"], q=str(g.get("question", ""))[:70], grp=grp_of(g.meas_why), mode=mode,
                   kind=q.get("kind"), pk=info.pkind.get(i), mv=mv, cs=D.loc[i, "cs"], old=D.loc[i, "answer"], new=np.nan, why="")
        if not (np.isfinite(mv) and np.isfinite(mval) and abs(np.log(abs(mv) / abs(mval))) < 1e-6):
            rec["why"] = "skip: meas file mismatch"; log.append(rec); continue
        gnew, why = np.nan, ""
        # ---- A: cross-dimension 2-decade rejection of a gravity-ruled 2D size
        if rec["grp"] == "2-decade sanity rule":
            if q.get("kind") != "size" or info.pkind.get(i) != "accel" or mode != "2d_ratio":
                rec["why"] = "skip A: not (size, accel prior, 2d_ratio)"; log.append(rec); continue
            mpp = float(g.meas_mpp)
            t = p.get("target") or {}
            key = (t.get("phrase") or "") + "|" + ",".join(t.get("where") or []) + ("|" + t["rel"][0] + ":" + t["rel"][1] if t.get("rel") else "")
            e = ext_stats(split, g.video_id, key)
            dim = q.get("dim") or "length"
            if e is None or not (mpp > 0):
                rec["why"] = "skip A: no track/mpp"; log.append(rec); continue
            used = {"height": e["bh"], "length": e["rr0"], "size": e["rr0"], "wingspan": e["rr0"], "width": e["rr1"],
                    "thickness": e["rr1"], "side": e["rr1"], "diameter": e["rr1p75"], "radius": e["rr1p75"] / 2}.get(dim, e["rr0"])
            rec_si = used * mpp
            if not abs(np.log(rec_si / (mv * usi))) < 0.1:
                rec["why"] = f"skip A: extent reconstruction {rec_si:.4g} vs {mv * usi:.4g}"; log.append(rec); continue
            ext, why = used, f"A: cross-dim 2-decade rejection of gravity-ruled 2d_ratio {dim} ({used:.0f}px)"
            if q.get("horizontal") and dim in ("length", "width", "size"):
                ext, why = e["bw"], f"A: cross-dim 2-decade rejection; 'horizontal' {dim} = box width ({e['bw']:.0f}px, was {used:.0f}px)"
            if ext < MIN_PX:
                rec["why"] = f"skip A: extent {ext:.0f}px < {MIN_PX:.0f}px"; log.append(rec); continue
            if dim in ("width", "diameter", "thickness", "side") and e["bw"] >= FLAT * e["bh"] and not q.get("horizontal"):
                rec["why"] = f"skip A: short side of a flat box ({e['bw']:.0f}x{e['bh']:.0f}px, foreshortened)"; log.append(rec); continue
            gsi = ext * mpp
            if not (PL_SIZE[0] <= gsi <= PL_SIZE[1]):
                rec["why"] = "skip A: implausible"; log.append(rec); continue
            gnew = gsi / usi
        # ---- C: corrupt target depth entry (dropped "0.")
        elif rec["grp"] == "implausible":
            dk = p.get("target_dkeys") or []
            if q.get("kind") != "size" or mode != "3d_calib" or len(dk) != 1:
                rec["why"] = "skip C: not a 3d_calib size with one target depth key"; log.append(rec); continue
            ents = depth_entries(g.depth_info)
            tgt = [x for x in ents if x[0] == dk[0]]
            oth = np.array([x[2] for x in ents if x[0] != dk[0]])
            if len(tgt) != 1 or len(oth) < 2:
                rec["why"] = "skip C: depth entries"; log.append(rec); continue
            k_, s_, z_ = tgt[0]
            # the merge took the VLM-track measurement for this video's other rows with the same prior (xground flag:
            # the detector's prior track never coincides with the VLM box - captured_0040s/x bind "credit card" to the
            # poker tower, f/W 0.53 instead of 0.62): the same measurement source for this row
            sib = (G.video_id == g.video_id) & (G.ground_truth_prior.astype(str) == str(g.ground_truth_prior)) & (G.index != i)
            msrc = "v3m"
            if (G.merge_src[sib].astype(str) == "xg_v").any() and i in Mv.index and str(Mv["mode"].get(i)) == "3d_calib" and np.isfinite(float(Mv["value"].get(i))):
                mv = float(Mv["value"].get(i)); msrc = "v3v(xg_v)"
            if not (z_ > 20 * oth.max() and oth.max() < 10 and re.fullmatch(r"\d{2,6}", s_)):
                rec["why"] = "skip C: target depth not an outlier integer"; log.append(rec); continue
            z2 = float("0." + s_)
            if not (oth.min() / 1.5 <= z2 <= oth.max() * 1.5):
                rec["why"] = f"skip C: repaired depth {z2} outside the scene range"; log.append(rec); continue
            gnew = mv * z2 / z_
            if not (PL_SIZE[0] <= gnew * usi <= PL_SIZE[1]):
                rec["why"] = "skip C: implausible after repair"; log.append(rec); continue
            why = f"C: depth_info '{s_}m' for {k_} is {z_ / oth.max():.0f}x the scene's other distances -> 0.{s_} m [{msrc}]"
        else:
            continue
        # ---- pipeline fusion with CaW
        cs = float(D.loc[i, "cs"]) if pd.notna(D.loc[i, "cs"]) else np.nan
        if np.isfinite(cs) and cs > 0 and str(g.video_id).startswith("simulation") and rec["cat"] == "D2" and q.get("kind") == "size":
            cs = cs * 1.12
        if np.isfinite(cs) and cs > 0 and max(gnew, cs) / min(gnew, cs) <= 2.0:
            w = 0.7 if rec["cat"] == "D2" else 0.65
            val = gnew ** w * cs ** (1 - w); fz = f"agree_wgm(w{w}, geo {gnew:.4g}, caw {cs:.4g})"
        else:
            val = gnew; fz = f"geo (strong {mode}; caw {cs:.4g} >2x)"
        rec.update(new=val, g_new=gnew, why=why + "; " + fz)
        out[str(int(i))] = dict(value=float(val), why=why + "; " + fz)
        log.append(rec)
    json.dump(out, open(f"{R}/cache/tmp/ag_ngsanity_patch_{split}.json", "w"), indent=1)
    return out, pd.DataFrame(log)


if __name__ == "__main__":
    pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 120); pd.set_option("display.max_rows", 200)
    for split in ("test", "val"):
        out, lg = run(split)
        print(f"== {split}: candidates {len(lg)}, patched {len(out)}")
        if len(lg):
            print(lg[["id", "vid", "cat", "grp", "mode", "mv", "cs", "old", "new", "why"]].to_string(index=False))
