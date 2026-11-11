import math, re
import numpy as np

EXCL = re.compile(r"\b(toy|model|miniature|lego|paper airplane|paper plane|robot|sculpture|statue|drawing|picture|poster|sign|lettering|text|logo|icon|"
                  r"shadow|reflection|image|cartoon|mario|plush|dragon|phoenix|spacecraft|rocket|astronaut|virus|star|planet)\b")
PREP = re.compile(r"\s+(of|on|in|with|at|from|to|over|when|while|that|formed|riding)\s+")

SIZE = {
    "person": (0.15, 2.30), "man": (0.15, 2.30), "woman": (0.15, 2.30), "human": (0.15, 2.30), "pedestrian": (0.15, 2.30),
    "player": (0.15, 2.30), "swimmer": (0.15, 2.30), "shooter": (0.15, 2.30), "baby": (0.20, 1.00),
    "car": (1.00, 6.50), "police car": (1.00, 6.50), "truck": (1.40, 25.0), "fire truck": (1.40, 25.0), "bus": (2.0, 25.0),
    "motorcycle": (0.40, 2.60), "motor cycle": (0.40, 2.60), "bicycle": (0.30, 2.30), "bike": (0.30, 2.30),
    "bicycle wheel": (0.25, 0.80), "forklift": (0.80, 6.00), "scooter": (0.25, 2.00),
    "plane": (2.0, 90.0), "airplane": (2.0, 90.0), "aircraft": (2.0, 90.0), "helicopter": (1.5, 40.0), "canoe": (0.30, 7.50),
    "house": (2.0, 60.0), "door": (0.50, 4.00), "table": (0.30, 4.00), "desk": (0.40, 2.50), "chair": (0.30, 1.30),
    "bench": (0.25, 5.00), "sofa": (0.40, 3.50), "book": (0.003, 0.60), "notebook": (0.003, 0.40), "note book": (0.003, 0.40),
    "pen": (0.005, 0.25), "pencil": (0.005, 0.25), "cup": (0.03, 0.25), "mug": (0.03, 0.25), "teacup": (0.03, 0.25),
    "tea cup": (0.03, 0.25), "wine glass": (0.05, 0.30), "bottle": (0.02, 0.60),
    "soccer ball": (0.17, 0.24), "football": (0.15, 0.30), "basketball": (0.15, 0.26), "tennis ball": (0.055, 0.075),
    "ping pong ball": (0.036, 0.042), "yoga ball": (0.40, 0.95), "swimming ring": (0.30, 1.40), "swim ring": (0.30, 1.40),
    "lane": (1.00, 5.50), "streetlight": (0.15, 20.0), "traffic light": (0.20, 10.0), "fire hydrant": (0.15, 1.20),
    "telephone booth": (0.60, 2.80), "trash bin": (0.15, 2.50), "trash can": (0.15, 2.50), "whiteboard": (0.30, 6.00),
    "chalkboard": (0.30, 6.00), "tv": (0.03, 3.00), "keyboard": (0.01, 0.55), "phone": (0.005, 0.20), "iphone": (0.005, 0.20),
    "credit card": (0.0005, 0.090), "us one-dollar bill": (0.05, 0.16), "macbook": (0.008, 0.45), "laptop": (0.008, 0.45),
    "wallet": (0.005, 0.25), "apple": (0.04, 0.13), "avocado": (0.04, 0.16), "pizza": (0.15, 0.90), "baguette": (0.03, 1.00),
    "ice cube": (0.005, 0.08), "acorn": (0.008, 0.06), "cookie": (0.003, 0.30), "bowling pin": (0.04, 0.40),
    "shoe": (0.04, 0.36), "trolley": (0.30, 1.60), "floor mirror": (0.02, 2.50), "toilet paper roll": (0.03, 0.15),
    "long side of a marked parking space": (4.0, 8.0), "pickle ball court": (6.0, 14.0), "football pitch": (40.0, 125.0),
    "billiard table": (0.70, 3.80), "chessboard": (0.20, 0.70), "chinese chess board": (0.20, 0.70), "sword": (0.40, 1.60),
    "dog": (0.10, 1.60), "cat": (0.10, 1.20), "rabbit": (0.08, 0.90), "raccoon": (0.15, 1.10), "horse": (0.50, 3.00),
    "deer": (0.40, 3.00), "antelope": (0.40, 3.00), "elephant": (0.70, 8.00), "whale": (1.0, 33.0), "shark": (0.15, 20.0),
    "turtle": (0.03, 2.50), "eagle": (0.20, 2.80), "bird": (0.03, 3.60), "butterfly": (0.01, 0.30),
    "red blood cell": (3e-6, 1.5e-5), "droplet": (0.0001, 0.010),
}
SPEED = {
    "walking pedestrian": 3.5, "walking person": 3.5, "person": 12.5, "man": 12.5, "woman": 12.5, "human": 12.5, "player": 12.5,
    "pedestrian": 12.5, "swimmer": 3.0, "swimming man": 3.0, "swimming woman": 3.0, "bicycle": 25, "bike": 25, "car": 70,
    "police car": 70, "truck": 45, "forklift": 8, "train": 100, "motorcycle": 90, "boat": 60, "speedboat": 60, "speed boat": 60,
    "canoe": 6, "sailboat": 30, "plane": 300, "airplane": 300, "ball": 75, "soccer ball": 60, "football": 60, "basketball": 20,
    "tennis ball": 75, "ping pong ball": 35, "puck": 50, "dog": 20, "cat": 14, "horse": 25, "rabbit": 22, "raccoon": 7,
    "elephant": 7, "deer": 30, "antelope": 30, "eagle": 90, "bird": 110, "shark": 20, "whale": 15, "turtle": 10, "duck": 25,
    "droplet": 10, "bubble": 5, "book": 20, "cup": 20, "mug": 20, "box": 20, "typewriter": 20, "magnifying glass": 20,
    "wallet": 20, "credit card": 20, "water bottle": 20, "apple": 20, "pen": 20, "plate": 20, "toilet paper": 20,
    "ice cube": 20, "vacuum cleaner": 3,
}
G = 9.80665
ACCEL = {
    "ball": 2 * G, "soccer ball": 2 * G, "basketball": 2 * G, "tennis ball": 2 * G, "football": 2 * G,
    "box": 1.5 * G, "book": 1.5 * G, "cup": 1.5 * G, "chess": 1.5 * G, "general chess": 1.5 * G, "acorn": 1.5 * G,
    "bread slice": 1.5 * G, "credit card": 1.5 * G, "car": 1.5 * G, "forklift": 5.0, "train": 3.0, "boat": G,
    "person": 1.5 * G, "man": 1.5 * G, "woman": 1.5 * G, "swimming man": 1.5 * G, "swimming woman": 1.5 * G, "swimmer": 1.5 * G,
}
UNIT = {"m": 1.0, "cm": 0.01, "mm": 0.001, "m/s": 1.0, "cm/s": 0.01, "m/s^2": 1.0, "cm/s^2": 0.01}
KINDS = {"size": "size", "speed": "speed", "avgspeed": "speed", "hvel": "speed", "accel": "accel"}


def head(target):
    t = re.sub(r"\s+", " ", str(target).lower().strip())
    t = re.sub(r"^(the|a|an)\s+", "", t)
    return PREP.split(t)[0].strip(), t


def klass(table, target):
    h, full = head(target)
    if EXCL.search(full): return None
    if full in table: return full
    for k in table:
        if " of " in k and full.startswith(k): return k
    w = h.split()
    for k in (h, " ".join(w[-3:]), " ".join(w[-2:]), w[-1] if w else ""):
        if k in table: return k
    return None


def rng_for(row):
    p = row.get("parse") or {}; q = p.get("q") or {}
    kind = KINDS.get(q.get("kind")); u = UNIT.get(q.get("unit"))
    if kind is None or u is None or q.get("target2"): return None
    tg = q.get("target")
    if not tg or str(tg).lower() == "none": return None
    if kind == "size":
        k = klass(SIZE, tg)
        if k is None: return None
        lo, hi = SIZE[k]
    elif kind == "speed":
        k = klass(SPEED, tg)
        if k is None: return None
        lo, hi = 0.0, SPEED[k]
    else:
        k = klass(ACCEL, tg)
        if k is None: return None
        lo, hi = 0.0, ACCEL[k]
    return lo / u, hi / u, k, kind




def make(kind, variant):
    def rule(row):
        r = rng_for(row)
        if r is None or r[3] != kind: return None
        lo, hi, k, _ = r; c = float(row["cur"])
        out = (lo > 0 and c < lo) or c > hi
        if not out: return None
        if variant == "V3" and not ((lo > 0 and c < lo / 2) or c > 2 * hi): return None
        inside = [(nm, v) for nm, v in witnesses(row) if (lo <= v <= hi)]
        if inside:
            nm, v = min(inside, key=lambda x: abs(math.log(x[1] / c)))
            return v, f"{k}: {c:.4g} outside [{lo:.4g},{hi:.4g}] -> own {nm} {v:.4g}"
        if variant == "V1": return None
        e = lo if (lo > 0 and c < lo) else hi
        return e, f"{k}: {c:.4g} outside [{lo:.4g},{hi:.4g}], no own witness inside -> edge {e:.4g}"
    rule.__name__ = f"PL_{kind}_{variant}"
    return rule
