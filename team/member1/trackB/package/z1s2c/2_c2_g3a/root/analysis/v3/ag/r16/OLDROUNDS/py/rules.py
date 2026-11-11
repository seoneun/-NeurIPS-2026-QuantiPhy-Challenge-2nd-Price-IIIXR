import os, sys, json, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from spec import SPEC
USI = {"m": 1.0, "cm": 0.01, "mm": 0.001, "km": 1000.0}


def _cache(name, r):
    cd = os.environ.get("R16_CACHE", os.path.join(os.path.dirname(HERE), "cache"))
    p = f"{cd}/{name}_{r['split']}_{int(r['id'])}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def make(name, mode, gate):
    def rule(r):
        c = _cache(name, r)
        if not c:
            return None
        cur = float(r["cur"])
        if not (cur > 0 and math.isfinite(cur)):
            return None
        v = float(c["value"])
        if mode == "mult":
            v = v * cur / float(c["base"])
        elif mode == "floor" and not v > cur * (1 + 1e-9):
            return None
        raw = c.get("raw")
        if gate:
            g = gate[0]
            if g == "within" and not max(raw / cur, cur / raw) <= gate[1]:
                return None
            if g == "skip" and (raw is None or gate[1] <= raw / cur <= gate[2]):
                return None
            if g == "lt" and not v < gate[1] * cur:
                return None
            if g == "outside_m":
                u = USI.get((((r.get("parse") or {}).get("q") or {}).get("unit")) or "m")
                if u is None or gate[1] <= cur * u <= gate[2]:
                    return None
        return v, f"{name} [{c['src']}] " + str(c.get("why", ""))[:300]
    rule.__name__ = name
    return rule


ACCEPTED = [(n, make(n, mode, gate)) for n, tier, src, u, var, mode, gate, rx in SPEC]
TIER = {n: tier for n, tier, *_ in SPEC}
