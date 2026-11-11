import re, math
import numpy as np, cv2

COLOR_WORDS = ["white", "black", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "grey", "gray", "silver"]


def sing(w):
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if (w.endswith("ses") or w.endswith("xes") or w.endswith("ches") or w.endswith("shes")) and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and len(w) > 3 and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def load_video(path):
    cap = cv2.VideoCapture(path)
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        s = min(h, w)
        if s > 480:
            f = cv2.resize(f, (int(round(w * 480 / s)), int(round(h * 480 / s))), interpolation=cv2.INTER_AREA)
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return fr


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms(b, sc, thr):
    order = np.argsort(-np.asarray(sc))
    keep = []
    for i in order:
        if all(iou(b[i], b[j]) < thr for j in keep):
            keep.append(int(i))
    return keep


def _hsv_masks(hsv):
    H, S, V = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    return {"white": (S < 50) & (V > 170), "black": V < 65, "grey": (S < 45) & (V >= 65) & (V <= 190),
            "gray": (S < 45) & (V >= 65) & (V <= 190), "silver": (S < 45) & (V >= 90),
            "red": ((H < 9) | (H > 170)) & (S > 90) & (V > 50), "orange": (H >= 8) & (H < 22) & (S > 100) & (V > 80),
            "yellow": (H >= 20) & (H < 36) & (S > 80) & (V > 80), "green": (H >= 36) & (H < 88) & (S > 50) & (V > 40),
            "blue": (H >= 88) & (H < 132) & (S > 60) & (V > 40), "purple": (H >= 125) & (H < 160) & (S > 40) & (V > 30),
            "pink": (((H >= 150) & (H <= 179)) | (H < 8)) & (S > 30) & (S < 160) & (V > 120),
            "brown": (H >= 5) & (H < 25) & (S > 60) & (V < 160) & (V > 30)}


def color_frac(img, box, cw):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    w, h = x1 - x0, y1 - y0
    if w < 2 or h < 2:
        return 0.0
    cx0, cy0, cx1, cy1 = x0 + int(0.2 * w), y0 + int(0.2 * h), x1 - int(0.2 * w), y1 - int(0.2 * h)
    patch = img[max(cy0, 0):max(cy1, 1), max(cx0, 0):max(cx1, 1)]
    if patch.size == 0:
        return 0.0
    hsv = cv2.cvtColor(np.ascontiguousarray(patch), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(int)
    return float(_hsv_masks(hsv)[cw].mean())


def mask_colors(img, m):
    px = img[m]
    if len(px) < 10:
        return None
    hsv = cv2.cvtColor(np.ascontiguousarray(px.reshape(-1, 1, 3)), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(int)
    mk = _hsv_masks(hsv)
    return {c: round(float(mk[c].mean()), 3) for c in COLOR_WORDS}


def mask_poly(m, eps=1.0, min_area=4.0):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        if len(c) < 3 or cv2.contourArea(c) < min_area:
            continue
        a = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        out.append(a.tolist())
    return out if out else None


def mask_stats(m):
    ys, xs = np.nonzero(m)
    if len(xs) < 4:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    pts = np.stack([xs, ys], 1).astype(np.float32)
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts)
    H, W = m.shape
    border = int(x0 <= 1) + int(y0 <= 1) + int(x1 >= W - 1) + int(y1 >= H - 1)
    c = np.cov(pts.T) if len(xs) > 5 else np.eye(2)
    ev = np.sort(np.linalg.eigvalsh(c))[::-1]
    return dict(area=int(len(xs)), box=[int(x0), int(y0), int(x1), int(y1)], c=[float(xs.mean()), float(ys.mean())],
                rr=[float(max(rw, rh)), float(min(rw, rh)), float(ang)], pca=[float(4 * math.sqrt(max(ev[0], 0))), float(4 * math.sqrt(max(ev[1], 0)))],
                border=border, bottom=float(y1), top=float(y0))


def camera_motion(frames):
    g = [cv2.cvtColor(cv2.resize(f, (f.shape[1] // 2, f.shape[0] // 2)), cv2.COLOR_RGB2GRAY).astype(np.float32) for f in frames]
    win = cv2.createHanningWindow(g[0].shape[::-1], cv2.CV_32F)
    out = [[0.0, 0.0, 1.0]]
    for a, b in zip(g[:-1], g[1:]):
        (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
        out.append([2 * dx, 2 * dy, float(resp)])
    return out


SYN_CLASS = {"human": "person", "people": "person", "pedestrian": "person", "man": "person", "woman": "person",
             "boy": "person", "girl": "person", "player": "person", "walker": "person", "lady": "person", "kid": "person",
             "child": "person", "astronaut": "person", "swimmer": "person", "runner": "person", "passer": "person",
             "receiver": "person", "bike": "bicycle", "cyclist": "bicycle", "automobile": "car", "vehicle": "car",
             "aeroplane": "plane", "airplane": "plane", "jet": "plane", "seagull": "bird", "ship": "boat",
             "football": "soccer", "mug": "cup", "glass": "cup", "trashbin": "bin", "trashcan": "bin"}
QUAL_GROUPS = [{"white", "black", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "grey", "gray", "silver", "golden", "gold"},
               {"big", "large", "biggest", "largest", "small", "little", "smallest", "tall", "short"},
               {"left", "right", "middle", "center", "centre"}, {"upper", "top", "lower", "bottom"},
               {"front", "near", "back", "far", "closest", "farthest"}, {"first", "second", "third", "last"}]
_CUTW = r"\b(?:riding|walking on|holding|wearing|with|in|on|at|of|from|near|next|behind|under|above)\b"


def head_noun(p):
    s = (p or "").lower()
    toks = re.findall(r"[a-z]+", s)
    if not toks:
        return ""
    m = re.search(_CUTW, s)
    if m and m.start() > 0:
        pre = re.findall(r"[a-z]+", s[:m.start()])
        if pre:
            toks = pre
    return sing(toks[-1])


def heads_compatible(pa, pb):
    ha, hb = head_noun(pa), head_noun(pb)
    if not ha or not hb:
        return False
    if ha == hb or SYN_CLASS.get(ha, ha) == SYN_CLASS.get(hb, hb):
        return True
    return (len(hb) >= 3 and ha.endswith(hb)) or (len(ha) >= 3 and hb.endswith(ha))


def qualifier_conflict(wa, wb):
    for g in QUAL_GROUPS:
        a, b = wa & g, wb & g
        if a and b and a != b:
            return True
    return False


def contrastive_v2(pa, where_a, pb, where_b):
    where_a = [w for w in (where_a or [])]; where_b = [w for w in (where_b or [])]
    if "#second" in where_a or "#second" in where_b:
        return True
    if not heads_compatible(pa, pb):
        return True
    wa = set(re.findall(r"[a-z]+", (pa or "") + " " + " ".join(w for w in where_a if not w.startswith("#"))))
    wb = set(re.findall(r"[a-z]+", (pb or "") + " " + " ".join(w for w in where_b if not w.startswith("#"))))
    return qualifier_conflict(wa, wb)
