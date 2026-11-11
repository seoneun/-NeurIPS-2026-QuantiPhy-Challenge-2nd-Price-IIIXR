#!/usr/bin/env python
"""Basketball-key scene-ruler consistency check (label-free, frames only).

geo/scene_ruler.py:basketball_key picks the key (paint) sides as the inner line pair at 0.30-0.345 of an outer line pair.
Its line profile (scene_ruler.cross_profile, offsets kept within +-diagonal/2) drops every near-vertical line right of
x ~ 1100 in a 1920x1080 landscape frame, so on internet_0011 the key's right side is never seen and the 0.32 pattern
locks onto 3-point line / outer-rectangle / frame-edge lines (230 px; the key is 540 px). Lines here are profiled over
the whole frame (court_lines_full).

Regulation geometry ties the key width k to two circles centred on the key axis:
  free-throw circle radius 1.80 m (FIBA, key 4.90 m) / 1.83 m (NBA, key 4.88 m): r/k = 0.367-0.375
  no-charge (restricted-area) arc radius 1.25 m (FIBA) / 1.22 m (NBA):           r/k = 0.250-0.255
A pair of long lines is accepted as the key only if one of these circles (at least a solid half: the half inside the key
is often unpainted) sits centred on the pair's axis. Per frame: every long-line pair -> best half-circle support over the
centre position along the axis and over r/k in the regulation band; the best pair with support >= SUP is that frame's key.
A pair must also be >= 6% of the frame's long side (the rim / net is a small ring) and both lines >= 10% of the strongest
line's weight. Video key = median over frames (>= 2 frames). If the scene_ruler px passes the check in at least half the
frames it is kept; otherwise the key is re-derived. Person width > 1.2 m is printed as a diagnostic only (patch.py).

usage: tools/run_limited.sh --io python analysis/v3/ag/rul305/key_check.py [--split test]
writes cache/tmp/ag_keyruler_keys_<split>.json"""
import os, sys, json, math, argparse
import numpy as np, cv2

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test")
ap.add_argument("--nframes", type=int, default=8)
ap.add_argument("--sup", type=float, default=0.6)
ap.add_argument("--debug", type=int, default=1)
a = ap.parse_args()
sys.argv = [sys.argv[0], "--debug", "0"]
sys.path.insert(0, f"{R}/geo")
import scene_ruler as SR  # noqa: E402  (read-only import: frames / marking_mask / court_lines)

FT = np.arange(0.330, 0.4001, 0.005)  # free-throw circle r / key width (FIBA 0.367, NBA 0.375; amateur courts vary)
NC = np.arange(0.225, 0.2751, 0.005)  # no-charge arc r / key width (NBA 0.250, FIBA 0.255)
NPHI = 96
SUP, CON = 0.6, 0.35                  # half-circle support on the ring, and its contrast to rings at r -/+ delta
MINK = 0.06                           # key width >= 6% of the frame's long side (rules out the rim / net)


def to_img(u, v, th):
    """inverse of the scene_ruler offsets: u = -x sin th + y cos th (long-line offset), v = -x cos th - y sin th"""
    s, c = math.sin(th), math.cos(th)
    return -s * u - c * v, c * u - s * v


def ring(m, uc, r, th, vgrid):
    """centres (uc, v0), v0 in vgrid -> per half circle (the two halves on either side of the FT-line direction through
    the centre): share of in-frame samples on marking pixels, nan where < 70% of the half is in frame. shape (n_v, 2)"""
    H, W = m.shape
    phi = np.linspace(0, 2 * np.pi, NPHI, endpoint=False)
    du, dv = r * np.cos(phi), r * np.sin(phi)
    U = uc + du[None, :] + 0 * vgrid[:, None]
    V = vgrid[:, None] + dv[None, :]
    X, Y = to_img(U, V, th)
    xi, yi = np.round(X).astype(int), np.round(Y).astype(int)
    inn = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    hit = np.zeros_like(inn, dtype=float)
    hit[inn] = m[yi[inn], xi[inn]]
    out = np.full((len(vgrid), 2), np.nan)
    for h, half in enumerate((np.sin(phi) > 0.02, np.sin(phi) < -0.02)):
        n_in = inn[:, half].sum(1)
        out[:, h] = np.where(n_in >= 0.7 * half.sum(), hit[:, half].sum(1) / np.maximum(n_in, 1), np.nan)
    return out


def circle_fit(m, uc, k, th, band, vgrid):
    """best (score, r/k, v0, support, contrast) over r in band and v0 in vgrid; a painted circle is a thin ring: support
    on the ring >= SUP and support minus the larger of the supports on rings at r -/+ delta >= CON"""
    best = (0.0, None, None, 0.0, 0.0)
    for rr in band:
        r = rr * k
        dl = max(8.0, 0.12 * r)
        on, lo, hi = (ring(m, uc, r_, th, vgrid) for r_ in (r, r - dl, r + dl))
        con = on - np.fmax(lo, hi)
        good = (on >= SUP) & (con >= CON)
        if not np.any(good):
            continue
        sc = np.where(good, on + con, -1.0)
        q = np.unravel_index(int(np.nanargmax(sc)), sc.shape)
        if sc[q] > best[0]:
            best = (float(sc[q]), float(rr), float(vgrid[q[0]]), float(on[q]), float(con[q]))
    return best


def court_lines_full(f, tol=2.0):
    """SR.court_lines with a profile that covers the whole frame: SR.cross_profile keeps offsets in [-size/2, size/2]
    with size = the frame diagonal, so for near-vertical lines (offset ~ -x) in a landscape frame every line right of
    x ~ diagonal/2 is dropped (internet_0011: the key's right side at x ~ 1254 of 1920 is never seen)"""
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d
    segs = [SR.segments(SR.marking_mask(f)[0])]
    th = SR.dominant_theta(segs[0])
    if th is None:
        return None
    H, W = f.shape[:2]
    size = 2 * (int(math.hypot(W, H)) + 4)
    out = {}
    for ax, t_ in (("long", th), ("short", (th + 90) % 180)):
        prof, off = SR.cross_profile(segs, t_, size, tol=tol)
        p = gaussian_filter1d(prof, 1.0)
        pk, _ = find_peaks(p, prominence=0.04 * max(p.max(), 1e-9), distance=3)
        out[ax] = (t_, pk.astype(float) - off, p[pk])
    return out


def frame_key(f):
    L = court_lines_full(f)
    if L is None:
        return None
    th_deg, x, w = L["long"]
    _, ys, wy = L["short"]
    th = math.radians(th_deg)
    m = cv2.dilate(SR.marking_mask(f)[0], np.ones((3, 3), np.uint8)).astype(float)
    H, W = m.shape
    corners = np.array([[0, 0], [W, 0], [0, H], [W, H]], float)
    vv = -corners[:, 0] * math.cos(th) - corners[:, 1] * math.sin(th)
    wmax = float(np.max(w)) if len(w) else 1.0
    cands = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            k = x[j] - x[i]
            if k < MINK * max(W, H) or k > 0.6 * max(W, H) or min(w[i], w[j]) < 0.1 * wmax:
                continue
            uc = (x[i] + x[j]) / 2
            tol = max(4.0, 0.03 * k)                      # FT circle: centred on a short line (the free-throw line)
            vft = np.unique(np.concatenate([np.arange(y_ - tol, y_ + tol + 1, 2.0) for y_ in ys])) if len(ys) else np.zeros(0)
            ft = circle_fit(m, uc, k, th, FT, vft) if len(vft) else (0.0, None, None, 0.0, 0.0)
            nc = circle_fit(m, uc, k, th, NC, np.arange(vv.min(), vv.max(), max(2.0, 0.25 * k / 20)))
            sc = (ft[0] > 0) + (nc[0] > 0) + 0.01 * (ft[0] + nc[0])
            cands.append(dict(k=float(k), lines=[float(x[i]), float(x[j])], w=[round(float(w[i]) / wmax, 3), round(float(w[j]) / wmax, 3)],
                              score=float(sc), ok=bool(ft[0] > 0 or nc[0] > 0), ft=ft, nc=nc))
    cands.sort(key=lambda c: -c["score"])
    bk = SR.basketball_key(L)                            # the old selection on the full-coverage lines (diagnostic)
    return dict(theta=float(th_deg), cands=cands, lines=[float(v) for v in x], mask_frac=float(m.mean()),
                old_sel_full=(round(bk["px"], 1), round(bk["check"], 3)) if bk else None)


def pair_support(fk, px, tol=0.03):
    """support of the best candidate whose width is within tol of px (the scene_ruler pair) in one frame"""
    s = [float(c["ok"]) for c in fk["cands"] if abs(math.log(c["k"] / px)) <= math.log(1 + tol)]
    return max(s) if s else 0.0


if __name__ == "__main__":
    sr = json.load(open(f"{R}/geo/out/scene_ruler_{a.split}.json"))
    vids = sorted(v for v, r in sr.items() if r and r.get("kind") == "basketball_key")
    out = {}
    for vid in vids:
        old = sr[vid]
        path = f"{R}/data/{a.split}/{vid}.mp4"
        if not os.path.exists(path):
            path = f"{R}/data/{a.split}/validation_videos/{vid}.mp4"
        fr = SR.frames(path, a.nframes, 1920)              # same frames / resolution as scene_ruler.court_ruler
        Wf = fr[0].shape[1]
        per, old_sup = [], []
        for n, f in enumerate(fr):
            fk = frame_key(f)
            if fk is None or not fk["cands"]:
                continue
            old_sup.append(pair_support(fk, old["px"]))
            b = fk["cands"][0]
            top = [(round(c["k"], 1), c["w"], round(c["score"], 2), [round(z, 3) if z else z for z in c["ft"]], [round(z, 3) if z else z for z in c["nc"]]) for c in fk["cands"][:4]]
            print(vid, "frame", n, "mask", round(fk["mask_frac"], 4), "old-selection-on-full-lines", fk["old_sel_full"], "| old-px ok", old_sup[-1], "| top4", top, flush=True)
            if b["ok"]:
                per.append(b)
        old_pass = len(old_sup) > 0 and float(np.mean(old_sup)) >= 0.5
        rec = dict(old_px=old["px"], old_ok=bool(old.get("ok")), old_support=float(np.mean(old_sup)) if old_sup else None,
                   old_pass=bool(old_pass), W=Wf, n_frames=len(per))
        if len(per) >= 2:
            px = float(np.median([p["k"] for p in per]))
            rec.update(new_px=px, px_spread=float(np.std([p["k"] for p in per]) / px),
                       n_ft=sum(p["ft"][0] > 0 for p in per), n_nc=sum(p["nc"][0] > 0 for p in per),
                       ft_ratio=float(np.median([p["ft"][1] for p in per if p["ft"][0] > 0] or [np.nan])),
                       nc_ratio=float(np.median([p["nc"][1] for p in per if p["nc"][0] > 0] or [np.nan])))
        rec["use_px"] = old["px"] if old_pass else rec.get("new_px")
        rec["changed"] = bool(rec["use_px"] is not None and abs(math.log(rec["use_px"] / old["px"])) > math.log(1.05))
        out[vid] = rec
        print(vid, json.dumps(rec), flush=True)
        if a.debug and rec.get("new_px"):
            f = fr[len(fr) // 2].copy()
            fk = frame_key(f)
            b = min(fk["cands"], key=lambda c: abs(c["k"] - rec["use_px"]))
            th = math.radians(fk["theta"])
            for c_ in b["lines"]:
                p0 = to_img(c_, -3000, th); p1 = to_img(c_, 3000, th)
                cv2.line(f, tuple(int(round(z)) for z in p0), tuple(int(round(z)) for z in p1), (0, 0, 255), 3)
            for nm in ("ft", "nc"):
                if b[nm][0] > 0:
                    cx, cy = to_img((b["lines"][0] + b["lines"][1]) / 2, b[nm][2], th)
                    cv2.circle(f, (int(cx), int(cy)), int(b[nm][1] * b["k"]), (0, 255, 0) if nm == "ft" else (255, 0, 255), 3)
            sc = 900 / max(f.shape[:2])
            cv2.imwrite(f"{R}/cache/tmp/ag_keyruler_{vid}.jpg", cv2.resize(f, None, fx=sc, fy=sc), [cv2.IMWRITE_JPEG_QUALITY, 80])
    json.dump(out, open(f"{R}/cache/tmp/ag_keyruler_keys_{a.split}.json", "w"), indent=1)
