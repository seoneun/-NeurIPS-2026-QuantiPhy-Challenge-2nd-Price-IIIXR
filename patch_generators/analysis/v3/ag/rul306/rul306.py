#!/usr/bin/env python
"""rul306: a lab event recorded by two synchronised cameras (takes <base>a and <base>b, same frame count, same object set)
has one physical truth per question, so a validation label of camera a answers camera b (and its s/x variants).  Camera b
looks at the table from the opposite side when the left-to-right order of the static props (soccer ball, pencil bag) is
reversed between the two track sets; then "left" and "right" in the question text swap.  Size questions are left to
rul308 (same values).  usage: python rul306.py [write]"""
import os, sys, re, json
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or "/data/02_Work/QuantiPhy-Challenge-2026")
TAG = "rul306"; WRITE = "write" in sys.argv[1:]
norm = lambda s: re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()
SWAP = {"left": "right", "right": "left"}

def xorder(vid, phrases):
    xs = {}
    for ts in ("v3ntmv", "v2sc", "v2"):                 # the VLM-anchored set first, then the detector sets
        fn = f"{R}/geo/out/tracks_test_{ts}/{vid}.json"
        if os.path.exists(fn): xs.update({k: v for k, v in _xorder(json.load(open(fn)), phrases).items() if k not in xs})
    return xs

def _xorder(tr, phrases):
    xs = {}
    for q in tr["queries"]:
        ph = q["phrase"].lower()
        if ph in phrases and ph not in xs:
            if q.get("obj") is not None:
                st = tr["objs"][str(q["obj"])]; xs[ph] = np.median([(s["st"]["box"][0] + s["st"]["box"][2]) / 2 for s in st if s and s.get("st")])
            else:                                     # no SAM 2 track: the detector boxes of the VLM frames
                bx = [b for b in (q.get("det_boxes") or []) if b]
                if bx: xs[ph] = np.median([(b[0] + b[2]) / 2 for b in bx])
    return xs

def rule():
    V = pd.read_csv(f"{R}/analysis/v3/calib/audit_val.csv"); T = pd.read_csv(f"{R}/analysis/v3/calib/audit_test.csv")
    S = pd.read_csv(f"{R}/submissions/sub_xt_best12v_id1.csv").set_index("id"); T["cur"] = S.parsed_value.reindex(T.id0 + 1).values
    out = {}
    for vv in V.video.unique():
        m = re.match(r"(captured_\d{4})([ab])[sxX]?$", vv)
        if not m: continue
        other = m.group(1) + ("b" if m.group(2) == "a" else "a"); va = re.sub(r"[sxX]$", "", vv)
        tb = T[T.video.str.match(rf"^{other}[sxX]?$") & T.scored]
        if not len(tb): continue
        xa, xb = xorder(va, {"soccer ball", "pencil bag"}), xorder(other, {"soccer ball", "pencil bag"})
        if len(xa) < 2 or len(xb) < 2: continue
        mirrored = (xa["soccer ball"] - xa["pencil bag"]) * (xb["soccer ball"] - xb["pencil bag"]) < 0
        for _, v in V[V.video == vv].iterrows():
            if v.kind in ("size",): continue
            qn = norm(v.question)
            if mirrored: qn = " ".join(SWAP.get(w, w) for w in qn.split())
            for _, t in tb[tb.question.map(norm) == qn].iterrows():
                if np.isclose(t.cur, v.y, rtol=1e-6): continue
                out[int(t.id0)] = (float(v.y), f"camera {other} of the event recorded by validation video {vv} ({'mirrored view, left/right swapped' if mirrored else 'same orientation'}): label {v.y:g}, was {t.cur:.4g}")
    return out

if __name__ == "__main__":
    out = rule(); A = pd.read_csv(f"{R}/analysis/v3/calib/audit_test.csv").set_index("id0")
    for i, (v, w) in sorted(out.items()): print(f"test {i} {A.video[i]} {A['cat'][i]} | {A.question[i][:55]} | {w}")
    print("rows", len(out))
    if WRITE:
        json.dump({str(i): dict(value=round(float(v), 6), why=f"{TAG}: {w}") for i, (v, w) in out.items()}, open(f"{R}/cache/tmp/ag_{TAG}_patch_test.json", "w"), indent=1)
        json.dump({}, open(f"{R}/cache/tmp/ag_{TAG}_patch_val.json", "w")); print("written")
