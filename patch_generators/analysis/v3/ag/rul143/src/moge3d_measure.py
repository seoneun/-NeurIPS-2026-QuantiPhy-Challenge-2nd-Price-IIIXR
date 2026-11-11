"""Prototype: object sizes in 3D from MoGe-2 point maps (scripts/ours/moge3d_dump.py), scaled per video by the stated
camera distances, measured in a ground-plane frame (RANSAC plane on the scene points, normal = up):
  height   = extent along up;   length / width = major / minor extent of the footprint (PCA in the plane);
  diameter = max(height, major extent).
The target is the object of the row's track set (merge_src xg_v -> v3ntmv, else v2sc), keyed like geo/measure_v3.Video.key.

    envs/quantiphy/bin/python scripts/ours/moge3d_measure.py val [GEO_TABLE]
-> geo/out/moge3d_<split>.csv (one row per 3D size question; labels are read only for the printed evaluation)
"""
import json
import re
import sys

import cv2
import numpy as np
import pandas as pd

S = sys.argv[1] if len(sys.argv) > 1 else "val"
GEO = sys.argv[2] if len(sys.argv) > 2 else f"preds/{S}_geo_v4ntrj5sQ2csrngsr_d_vw_raw.csv"
DTAG = sys.argv[3] if len(sys.argv) > 3 else ""          # dump directory suffix (moge3d_dump.py --out_tag)
UNIT = {"m": 1.0, "cm": 0.01, "mm": 0.001, "km": 1000.0}
rng = np.random.default_rng(0)


def norm(s):
    return re.sub(r"[^a-z]", "", (s or "").lower())


def tkey(ent):
    if not ent or not ent.get("phrase"):
        return None
    rel = ent.get("rel")
    return ent["phrase"] + "|" + ",".join(ent.get("where") or []) + ("|" + rel[0] + ":" + rel[1] if rel else "")


class Dump:
    def __init__(self, vid):
        z = np.load(f"geo/out/moge3d_{S}{DTAG}/{vid}.npz")
        self.pos = z["pos"]
        self.K = z["K"]
        self.depth = z["depth"].astype(np.float32)
        self.W, self.H, self.fps, self.fidx = int(z["W"]), int(z["H"]), float(z["fps"]), z["fidx"]
        self.h, self.w = self.depth.shape[1:]
        self.masks = {k[5:]: z[k] for k in z.files if k.startswith("mask_")}
        self.tsec = self.fidx[self.pos] / self.fps
        self._planes = {}

    def mask(self, key, f):
        m = self.masks.get(key)
        if m is None:
            return None
        return np.unpackbits(m[f], axis=1)[:, : self.w].astype(bool)

    def points(self, f, sel, scale):
        v, u = np.nonzero(sel)
        z = self.depth[f][v, u] * scale
        ok = np.isfinite(z) & (z > 0)
        u, v, z = u[ok] + 0.5, v[ok] + 0.5, z[ok]
        K = self.K[f]
        fx, fy, cx, cy = K[0, 0] * self.w, K[1, 1] * self.h, K[0, 2] * self.w, K[1, 2] * self.h
        return np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], 1)

    def up(self, f, scale):
        """RANSAC plane on the lower 60 % of the frame outside every object mask; normal towards the camera = up."""
        if f in self._planes:
            return self._planes[f]
        sel = np.isfinite(self.depth[f])
        sel[: int(0.4 * self.h)] = False
        for k in self.masks:
            m = self.mask(k, f)
            if m is not None:
                sel &= ~m
        P = self.points(f, sel, scale)
        res = None
        if len(P) >= 500:
            P = P[rng.choice(len(P), min(len(P), 20000), replace=False)]
            thr = 0.015 * float(np.median(P[:, 2]))
            best = 0
            for _ in range(300):
                a, b, c = P[rng.choice(len(P), 3, replace=False)]
                n = np.cross(b - a, c - a)
                if np.linalg.norm(n) < 1e-9:
                    continue
                n /= np.linalg.norm(n)
                inl = np.abs((P - a) @ n) < thr
                if inl.sum() > best:
                    best, bn, ba = int(inl.sum()), n, a
            if best >= 0.2 * len(P):
                if np.dot(bn, -ba) < 0:      # camera (origin) side of the plane = up
                    bn = -bn
                res = (bn, best / len(P))
        self._planes[f] = res
        return res


def erode(m):
    return cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def obj_key(tr, ent):
    """mask key '<set>_<obj>' of a parsed entity (target / prior / target2) in the first track set that has it"""
    k = tkey(ent)
    for s in ("v2sc", "v3ntmv"):
        if k and s in tr:
            q = next((x for x in tr[s]["queries"] if x.get("key") == k and x.get("obj") is not None), None)
            if q is not None:
                return f"{s}_{q['obj']}"
    return None


def calibrate(vid, D, recs_v, tr):
    """scale = median over the video's stated distances of stated / MoGe range on the object's mask; the object of a
    depth key is the one the teammates' parse binds it to (target_dkeys / prior_dkeys / target2_dkeys of a row)"""
    pairs = {}
    for r in recs_v:
        for ent, keys in ((r.get("target"), r.get("target_dkeys")), (r.get("prior_obj"), r.get("prior_dkeys")),
                          (r.get("target2"), r.get("target2_dkeys"))):
            mk = obj_key(tr, ent)
            if mk is None or not keys:
                continue
            for e in r.get("depth") or []:
                if e["key"] in keys and e.get("d") is not None:
                    pairs[(mk, e["key"], e.get("t"))] = e
    seen, rs = set(), []
    for (mk, _, _), e in pairs.items():
        if True:
            t = float(e["t"]) if e.get("t") is not None else 0.0
            f = int(np.argmin(np.abs(D.tsec - t)))
            if abs(D.tsec[f] - t) > 0.1:
                continue
            m = D.mask(mk, f)
            if m is None:
                continue
            m = erode(m)
            if m.sum() < 10:
                continue
            P = D.points(f, m, 1.0)
            rng_ = np.linalg.norm(P, axis=1)
            rs.append(float(e["d"]) / float(np.median(rng_)))
    if not rs:
        return None, None, 0
    return float(np.median(rs)), float(np.exp(np.std(np.log(rs)))), len(rs)


def measure(D, mkey, scale, border):
    out = []
    for f in range(len(D.pos)):
        if border.get(int(D.pos[f]), 1) != 0:
            continue
        m = D.mask(mkey, f)
        if m is None:
            continue
        m = erode(m)
        if m.sum() < 15:
            continue
        pl = D.up(f, scale)
        if pl is None:
            continue
        up = pl[0]
        P = D.points(f, m, scale)
        if len(P) < 15:
            continue
        zlo, zhi = np.percentile(P[:, 2], [2, 98])
        P = P[(P[:, 2] >= zlo) & (P[:, 2] <= zhi)]
        hgt = P @ up
        height = float(np.percentile(hgt, 98) - np.percentile(hgt, 2))
        Q = P - np.outer(hgt, up)
        e1 = np.cross(up, [0.0, 0.0, 1.0])
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(up, [1.0, 0.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(up, e1)
        C = np.stack([Q @ e1, Q @ e2], 1)
        C -= C.mean(0)
        w_, V_ = np.linalg.eigh(np.cov(C.T))
        a1, a2 = C @ V_[:, 1], C @ V_[:, 0]
        major = float(np.percentile(a1, 98) - np.percentile(a1, 2))
        minor = float(np.percentile(a2, 98) - np.percentile(a2, 2))
        out.append((height, major, minor, pl[1]))
    return np.array(out) if out else None


def main():
    recs = [json.loads(x) for x in open(f"geo/out/parse_{S}_v2.jsonl")]
    geo = pd.read_csv(GEO).set_index("Unnamed: 0")
    rows = []
    by_vid = {}
    for r in recs:
        by_vid.setdefault(r["video_id"], []).append(r)
    for vid, rv in sorted(by_vid.items()):
        try:
            D = Dump(vid)
        except FileNotFoundError:
            continue
        tr = {}
        for s in ("v2sc", "v3ntmv"):
            try:
                t_ = json.load(open(f"geo/out/tracks_{S}_{s}/{vid}.json"))
                if t_.get("objs"):
                    tr[s] = t_
            except FileNotFoundError:
                pass
        scale, spread, nk = calibrate(vid, D, rv, tr) if tr else (None, None, 0)
        for r in rv:
            q = r["q"]
            if q.get("kind") != "size" or not r.get("depth"):
                continue
            src = geo.merge_src.get(r["id"]) if r["id"] in geo.index else None
            sets = ["v3ntmv", "v2sc"] if src == "xg_v" else ["v2sc", "v3ntmv"]
            k = tkey(r.get("target"))
            val, info = None, {}
            for s in sets:
                if s not in tr or k is None:
                    continue
                qq = next((x for x in tr[s]["queries"] if x.get("key") == k and x.get("obj") is not None), None)
                if qq is None:
                    continue
                fl = tr[s]["objs"][str(qq["obj"])]
                border = {i: (fl[i]["st"]["border"] if fl[i] and fl[i].get("st") else 1) for i in range(len(fl))}
                res = measure(D, f"{s}_{qq['obj']}", scale or 1.0, border)
                if res is None or len(res) == 0:
                    continue
                hgt, maj, mnr = np.median(res[:, 0]), np.median(res[:, 1]), np.median(res[:, 2])
                dim = q.get("dim") or ""
                si = {"height": hgt, "length": maj, "width": mnr, "diameter": max(hgt, maj), "radius": max(hgt, maj) / 2}.get(dim)
                info = dict(set=s, nf=len(res), height=hgt, major=maj, minor=mnr, plane_inl=float(np.median(res[:, 3])))
                if si is not None:
                    val = si / UNIT.get(q.get("unit"), 1.0)
                break
            rows.append(dict(id=r["id"], video_id=vid, cat=r["cat"], dim=q.get("dim"), question=r["question"][:70], m3d=val,
                             scale=scale, spread=spread, nkeys=nk, **info))
    out = pd.DataFrame(rows).set_index("id")
    out.to_csv(f"geo/out/moge3d_{S}{DTAG}.csv")
    if S == "val":
        lab = pd.read_csv("data/val/validation_dataset.csv").set_index("Unnamed: 0").ground_truth_posterior
        out["gt"] = lab.reindex(out.index)
        out["geo"] = geo.geo_value.reindex(out.index)
        out["caw"] = pd.to_numeric(pd.read_csv("preds/val_caw9b_orig_crop.csv").set_index("Unnamed: 0").parsed_value, errors="coerce").reindex(out.index)
        TH = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])

        def mra(p, y):
            p = np.asarray(p, float)
            r_ = np.abs(p - y) / np.abs(y)
            return np.where(np.isfinite(r_), (r_[:, None] < (1 - TH)[None, :]).mean(1), 0.0)
        for c in ("m3d", "geo", "caw"):
            out[f"mra_{c}"] = mra(out[c].values, out["gt"].values)
        pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
        cols = ["video_id", "cat", "dim", "gt", "m3d", "geo", "caw", "mra_m3d", "mra_geo", "mra_caw", "scale", "spread", "nkeys", "nf", "plane_inl", "question"]
        print(out[cols].round(3).to_string())
        both = out.m3d.notna() & out.geo.notna()
        print(f"\n3D size rows {len(out)}; with m3d {int(out.m3d.notna().sum())}; both {int(both.sum())}")
        print("MRA on rows with both: m3d", round(out.mra_m3d[both].mean(), 3), "geo", round(out.mra_geo[both].mean(), 3),
              "caw", round(out.mra_caw[both].mean(), 3))
        print(out[both].groupby("dim")[["mra_m3d", "mra_geo", "mra_caw"]].mean().round(3).assign(n=out[both].groupby("dim").size()))


if __name__ == "__main__":
    main()
