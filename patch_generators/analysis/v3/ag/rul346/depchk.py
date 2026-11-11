"""latgeo2: depth_info vs shown-camera geometry. For each depth entry of an object with a known size (prior text or rul308
value), implied f/W = size_px * Z / (D * W) (size_px = median of box w,h over +-2 frames; spec 0.596). For objects with two
timed entries: apparent-size ratio vs inverse depth ratio."""
import re, json, os, numpy as np, pandas as pd
R = "/data/02_Work/QuantiPhy-Challenge-2026"
t = pd.read_parquet(f"{R}/data/test/test_dataset.parquet")
lp = json.load(open(f"{R}/cache/tmp/ag_labprops_patch_test.json"))
recs = ["captured_0021", "captured_0022", "captured_0029", "captured_0030", "captured_0032", "captured_0033", "captured_0034a", "captured_0034b",
        "captured_0035a", "captured_0035b", "captured_0036", "captured_0038", "captured_0039", "captured_0040", "captured_0054a", "captured_0054b"]
def sizes_known(vid):
    S = {}
    d = t[t.video_id == vid]
    m = re.search(r"diameter of (?:the )?([a-z ]+?) = ([\d.]+) ?(cm|m)", str(d.prior.iat[0]).lower())
    if m: S[m.group(1).strip()] = float(m.group(2)) / (100 if m.group(3) == "cm" else 1)
    for i, r in d.iterrows():
        q = r.question.lower(); mm = re.search(r"diameter of the ([a-z ]+?) in (cm|m)", q)
        if mm and str(i) in lp: S.setdefault(mm.group(1).strip(), lp[str(i)]["value"] / (100 if mm.group(2) == "cm" else 1))
    return S
for vid in recs:
    d = t[t.video_id == vid]
    if not len(d): print("==", vid, "not in test"); continue
    dep = str(d.depth_info.iat[0]); S = sizes_known(vid)
    p = f"{R}/geo/out/tracks_test_v3ntmv/{vid}.json"; p = p if os.path.exists(p) else p.replace("v3ntmv", "v2sc")
    tr = json.load(open(p)); fps, st, W = tr["fps"], tr["stride"], tr["W"]
    ph = {}
    for q in tr["queries"]:
        if q.get("obj") is not None: ph.setdefault(q["phrase"].lower(), q["obj"])
    print(f"== {vid} prior: {d.prior.iat[0][:60]} | known sizes {S}")
    ents = re.findall(r"(?:t=([\d.]+)\s*s,\s*)?distance_([a-z_]+?)_camera\s*=\s*([\d.]+)\s*m", dep.lower())
    byobj = {}
    for tt, name, z in ents:
        obj = name.replace("_", " "); z = float(z); tq = float(tt) if tt else None
        oid = ph.get(obj); D = S.get(obj)
        if oid is None:
            cand = [k for k in ph if obj.split()[-1] in k]; oid = ph[cand[0]] if cand else None
        sz = np.nan
        if oid is not None:
            seq = tr["objs"][str(oid)]; j = int(round((tq or 0) * fps / st)) if tq is not None else None
            js = range(max(0, j - 2), min(len(seq), j + 3)) if j is not None else range(len(seq))
            vals = [max(s["st"]["box"][2] - s["st"]["box"][0], s["st"]["box"][3] - s["st"]["box"][1]) for s in (seq[k] for k in js) if s and s.get("st")]
            sz = float(np.median(vals)) if vals else np.nan
        fW = sz * z / (D * W) if D else np.nan
        byobj.setdefault(obj, []).append((tq, z, sz))
        print(f"   t={tq} {obj:16s} Z={z:.4f} obj={oid} size_px={sz:.1f} D={D} implied f/W={fW:.3f} ({fW/0.5959:.2f}x spec)")
    for obj, L in byobj.items():
        L = [x for x in L if x[0] is not None]
        if len(L) >= 2 and L[0][0] != L[1][0]:
            (t1, z1, s1), (t2, z2, s2) = L[0], L[1]
            print(f"   evolution {obj}: Z2/Z1={z2/z1:.3f}  size1/size2={s1/s2:.3f}  (equal if depth is the shown camera's)")
