import json
import re
import sys
import numpy as np
import pandas as pd
import cv2

OUT = sys.argv[1]


def hl(e, th, ml, mg):
    L = cv2.HoughLinesP(e, 1, np.pi / 720, th, minLineLength=ml, maxLineGap=mg)
    return [] if L is None else [tuple(int(v) for v in l) for l in L.reshape(-1, 4)]


def oblique(e, y0, y1):
    m = e.copy(); m[:y0] = 0; m[y1:] = 0; out = []
    for x1, y1_, x2, y2 in hl(m, 30, 30, 4):
        a = abs(np.degrees(np.arctan2(y2 - y1_, x2 - x1))); a = min(a, 180 - a)
        if 12 < a < 85:
            out.append((x1, y1_, x2, y2))
    return out


def vp(lines):
    P = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            x1, y1, x2, y2 = lines[i]; x3, y3, x4, y4 = lines[j]
            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(den) < 1e-6:
                continue
            P.append((((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den,
                      ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den))
    P = np.array(P)
    if len(P) < 5:
        return None
    return float(np.median(P[:, 0])), float(np.median(P[:, 1]))


def through(l, v, tol=12):
    x1, y1, x2, y2 = l; d = np.array([x2 - x1, y2 - y1], float); d /= np.linalg.norm(d)
    return abs((v[0] - x1) * d[1] - (v[1] - y1) * d[0]) < tol


def refine(lab, y0, x0, x1):
    P = lab[:, x0:x1].mean(1); d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    lo, hi = max(1, int(y0) - 6), min(len(d) - 1, int(y0) + 6)
    return int(lo + np.argmax(d[lo:hi + 1])) + 0.5


def track(T, key):
    for q in T["queries"]:
        if q["key"].split("|")[0] == key and q.get("obj") is not None:
            st = T["objs"][str(q["obj"])]
            return np.array([s["st"]["box"] if s and s.get("st") else [np.nan] * 4 for s in st], float)
    return None


def run(i, Q):
    r = Q.loc[i]; vid = str(r.video_id).strip(); pr = str(r.prior).strip()
    T = json.load(open(f"geo/out/tracks_test_v3/{vid}.json")); W, H = T["W"], T["H"]; fps = float(T["fps"])
    keys = [q["key"].split("|")[0] for q in T["queries"]]
    cap = cv2.VideoCapture(f"data/test_480p/{vid}.mp4"); fr = T["fidx"][len(T["fidx"]) // 2]; cap.set(1, fr); ok, im = cap.read(); im = cv2.resize(im, (W, H))
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(float)
    e = cv2.Canny(g, 20, 60)
    cl = oblique(e, 0, int(0.35 * H)); fl = oblique(e, int(0.72 * H), H)
    v = vp(cl + fl)
    if v is None or abs(v[0] - W / 2) > 0.1 * W or abs(v[1] - H / 2) > 0.1 * H:
        return None
    yh = v[1]
    cc = [l for l in cl if through(l, v, 20)]
    j = T["fidx"].index(fr); free = np.ones((H, W), bool)
    for o_ in T["objs"].values():
        s_ = o_[j]
        if s_ and s_.get("st"):
            x1, y1, x2, y2 = [int(q) for q in s_["st"]["box"]]
            free[max(0, y1 - 3):y2 + 3, max(0, x1 - 3):x2 + 3] = False
    dy = np.zeros((H, W)); dy[:-1] = np.linalg.norm(np.diff(lab, axis=0), axis=2)
    fr2 = free[:-1] & free[1:]; fr2 = np.vstack([fr2, np.zeros((1, W), bool)])

    def band_row(y0, y1):
        rows = []
        for x in range(0, W - 39, 40):
            D = np.where(fr2[y0:y1, x:x + 40], dy[y0:y1, x:x + 40], np.nan)
            cov = np.isfinite(D).mean(1); m_ = np.nanmean(np.where(cov[:, None] >= 0.5, D, np.nan), axis=1) if np.isfinite(D).any() else None
            if m_ is None or not np.isfinite(m_).any():
                continue
            k = int(np.nanargmax(m_)); rows.append((y0 + k, float(m_[k])))
        if not rows:
            return None
        R_ = np.array(rows); best = None
        for y in np.unique(R_[:, 0]):
            sel = np.abs(R_[:, 0] - y) <= 2; sc = R_[sel, 1].sum()
            if best is None or sc > best[1]:
                best = (float(np.median(R_[sel, 0])) + 0.5, sc)
        return best[0]

    def tracked_edge(names, side):
        for q in T["queries"]:
            k = q["key"].split("|")[0]
            if q.get("obj") is not None and any(k == n or k.startswith(n + " ") for n in names):
                s_ = T["objs"][str(q["obj"])][j]
                if s_ and s_.get("st"):
                    bx = s_["st"]["box"]
                    if bx[2] - bx[0] > 0.6 * W:
                        return refine(lab, bx[3] if side == "bottom" else bx[1], 0, W)
        return None
    yc = tracked_edge(["ceiling"], "bottom")
    if yc is None:
        if len(cc) >= 3:
            ye = max(max(l[1], l[3]) for l in cc); yc = band_row(max(1, ye - 3), min(int(yh) - 30, ye + 20))
        else:
            yc = band_row(3, int(yh) - 40)
    yf = tracked_edge(["floor", "ground"], "top"); amb = False
    if yf is None:
        wcol = np.median(lab[int(yh) - 20:int(yh) + 20][free[int(yh) - 20:int(yh) + 20]], axis=0)
        yb = int(yh + 0.6 * (H - yh)); Lb = lab[yb:H][free[yb:H]]
        Lb = Lb[Lb[:, 0] > 0.6 * np.median(Lb[:, 0])] if len(Lb) else Lb
        fcol = np.median(Lb, axis=0) if len(Lb) > 50 else None
        cand = []
        if fcol is not None:
            for jj in np.linspace(0.1, 0.9, 5):
                fk = T["fidx"][int(jj * (len(T["fidx"]) - 1))]; cap.set(1, fk); ok2, im2 = cap.read()
                if not ok2:
                    continue
                lab2 = cv2.cvtColor(cv2.resize(im2, (W, H)), cv2.COLOR_BGR2LAB).astype(float); jk = T["fidx"].index(fk); free2 = np.ones((H, W), bool)
                for o_ in T["objs"].values():
                    s_ = o_[jk]
                    if s_ and s_.get("st"):
                        x1, y1, x2, y2 = [int(q) for q in s_["st"]["box"]]; free2[max(0, y1 - 3):y2 + 3, max(0, x1 - 3):x2 + 3] = False
                dw = np.linalg.norm(lab2 - wcol, axis=2); df = np.linalg.norm(lab2 - fcol, axis=2)
                isw = free2 & (dw < df) & (dw < 15); isf = free2 & (df < dw) & (df < 15)
                for x in range(0, W, 2):
                    for y in range(int(yh) + 20, H - 4):
                        if isf[y + 1:y + 4, x].all() and isw[y - 11:y + 1, x].mean() >= 0.8:
                            cand.append((y, x))
        if len(cand) >= 10:
            C_ = np.array(cand); ca = C_[:, 0]
            cnt = {int(y): len(set((C_[np.abs(ca - y) <= 2, 1] // 40).tolist())) for y in np.unique(ca)}
            top = max(cnt.values()); best = min(y for y, c in cnt.items() if c == top)
            yf0 = float(np.median(ca[np.abs(ca - best) <= 2]))
            rival = max([c for y, c in cnt.items() if abs(y - best) > 6] or [0])
            amb = cnt[best] - rival < 2
            yf = refine(lab, yf0, 0, W)
        else:
            yf = band_row(int(yh) + 40, H - 4)
    if yc is None or yf is None:
        return None
    m = re.match(r"\s*(length|width|height|speed) of (?:the )?(.+?)\s*=\s*([\d.]+)", pr, re.I)
    if not m:
        m2 = re.match(r"\s*walking speed\s*=\s*([\d.]+)", pr, re.I)
        if m2:
            m = ("speed", next((k for k in keys if k in ("man", "person", "woman", "human", "pedestrian")), None), m2.group(1))
    else:
        m = (m.group(1).lower(), m.group(2).strip().lower(), m.group(3))
    if not m or m[1] is None:
        return None
    kind, obj, val = m[0], m[1], float(m[2])
    key = next((k for k in keys if k == obj or obj.endswith(k) or k.endswith(obj)), None)
    B = track(T, key) if key else None
    if B is None or not np.isfinite(B[:, 0]).any():
        return None
    ok_ = np.isfinite(B[:, 0]); t = np.array(T["fidx"], float) / fps
    if kind in ("length", "width"):
        s = float(np.nanmedian(B[:, 2] - B[:, 0])) / val
    elif kind == "height":
        s = float(np.nanmedian(B[:, 3] - B[:, 1])) / val
    else:
        cx = (B[:, 0] + B[:, 2]) / 2; pxs = abs(np.polyfit(t[ok_], cx[ok_], 1)[0]); s_obj = pxs / val
        yx = float(np.nanmedian(B[:, 3])); sup = None
        if yx < yf - 3:
            for k in keys:
                if k == key:
                    continue
                S = track(T, k)
                if S is None or not np.isfinite(S[:, 0]).any():
                    continue
                sb = np.nanmedian(S, axis=0)
                if sb[0] - 20 <= np.nanmin(B[:, 0]) and np.nanmax(B[:, 2]) <= sb[2] + 20 and abs(sb[1] - yx) < 15 and sb[3] > yf:
                    sup = k; yx = float(sb[3]); break
            if sup is None:
                return None
        s = s_obj * (yf - yh) / (yx - yh)
    if amb:
        return None
    return round((yf - yc) / s, 3)


Q = pd.read_parquet("data/test/test_dataset.parquet")
rows = [int(i) for i in Q.index[Q.question.str.contains(r"\bceiling\b", case=False) & Q.question.str.contains(r"\bheight\b", case=False)]]
R = pd.DataFrame([(i, run(i, Q)) for i in rows], columns=["id0", "value"]).dropna()
R.to_csv(OUT, index=False)
