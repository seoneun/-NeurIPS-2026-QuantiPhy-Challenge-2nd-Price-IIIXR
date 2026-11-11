#!/usr/bin/env python
import sys, os, re, json, importlib.util
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
TAG = "p3qns"
_sp = importlib.util.spec_from_file_location("pxq", f"{R}/analysis/v3/compliance/rules/pxq.py")
PX = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(PX)
PQ = PX.PQ
MARGIN = PQ.MARGIN
SZ = "height width length diameter size"
MIRROR = r"(?:floor|standing|full[- ]length|cheval|leaning|dressing) mirror"
P3 = [
    ("cup / mug", r"(?:tea ?cup|coffee cup|espresso cup|cup|mug)", "cp", r"trophy|world cup|stanley|suction|cupcake",
     ("size", SZ), 0.04, 0.20, "cups and mugs (manufacturer specifications): espresso cups about 5-6 cm high; teacups about 6-8 cm high with 8-10 cm rims; 11 oz coffee mugs about 9.5 cm high and 8.2 cm wide (about 12 cm with the handle); 20 oz mugs about 12-14 cm high; saucers 12-17 cm across"),
    ("floor mirror", MIRROR, "cp", r"wall|vanity|bathroom",
     ("size", "width"), 0.25, 1.3, "full-length (floor / standing / leaning) mirrors, retail sizes: about 30-80 cm wide (IKEA HOVET 78 x 196 cm, IKEA NISSEDAL 65 x 150 cm); oversized leaning mirrors up to about 122 cm wide (48 x 72 in); narrow over-door models about 30-36 cm wide"),
    ("floor mirror", MIRROR, "cp", r"wall|vanity|bathroom",
     ("size", "height"), 1.0, 2.2, "full-length mirrors about 1.2-2.0 m high (IKEA HOVET 196 cm, IKEA NISSEDAL 150 cm, 48 x 72 in models 1.83 m); cheval mirrors about 1.5-1.9 m"),
    ("motorcycle", r"(?:motor ?cycle|motor ?bike|dirt ?bike)", "cp", r"toy|model|tricycle|sidecar",
     ("size", "length"), 1.2, 2.7, "motorcycles, overall length (manufacturer specifications): mini bikes about 1.3 m (Honda Z50 Monkey), Honda Monkey 125 1.71 m, 125-600 cc road bikes about 2.0-2.1 m, touring bikes up to about 2.6 m (Honda Gold Wing 2.575 m; Harley-Davidson Road Glide about 2.5 m)"),
    ("football (game ball)", r"(?:american football|football|rugby ball|soccer ball)", "cp", r"foosball|table football",
     ("size", "length"), 0.18, 0.30, "game balls, long axis: NFL ball 11-11.25 in (27.9-28.6 cm; NFL rules); rugby ball 280-300 mm (World Rugby Regulation 12); a round football (soccer ball) is 18.5-22.3 cm across (sizes 3-5; IFAB Law 2: size 5 circumference 68-70 cm)"),
]


def lookup(question, x):
    e1, cp = PQ.lookup(question, x)
    if e1 is not None:
        return None, cp
    e2, _ = PX.lookup(question, x)
    if e2 is not None:
        return None, cp
    q = x.get("q") or {}
    kind, dim = q.get("kind"), PQ.DIMSYN.get(q.get("dim") or "", q.get("dim") or "")
    ql = str(question).lower().replace("’", "'")
    if not cp or re.search(PQ.NOT_ALL, cp) or re.search(PX.QEXCL, ql):
        return None, cp
    for e in P3:
        cls, rx, where, nx, kd, lo, hi, src = e
        if kind != "size" or dim not in kd[1].split():
            continue
        if not re.search(r"(?:^|\s)(?:" + rx + r")s?$", cp):
            continue
        if nx and re.search(nx, ql):
            continue
        return e, cp
    return None, cp


def prior_scaled(x):
    if PX.prior_scaled(x):
        return True
    for p in x.get("priors") or []:
        if p.get("kind") != "size" or not p.get("value_si"):
            continue
        obj = p.get("obj") or ((x.get("prior_obj") or {}).get("phrase")) or ""
        e, _ = lookup(str(p.get("raw") or ""), {"q": {"kind": "size", "dim": p.get("dim") or "", "target": obj}})
        if e is not None and not (e[5] <= float(p["value_si"]) <= e[6]):
            return True
    return False


def main():
    BASE = sys.argv[1] if len(sys.argv) > 1 else "pred_t064"
    CUR = sys.argv[2] if len(sys.argv) > 2 else "pred_t055"
    GT = sys.argv[3] if len(sys.argv) > 3 else "qdir3"
    for split in ("test", "val"):
        det, cur, cf, cc, gu, P = PX.load(split, BASE, CUR, GT)
        out, n_flag = {}, 0
        for i in range(len(det)):
            x = P[i]
            e, cp = lookup(x["question"], x)
            if e is None:
                continue
            f = PQ.si_factor(x)
            a = float(cur[i]) * f
            cls, rx, where, nx, kd, lo, hi, src = e
            if not PQ.flag(a, lo, hi) or prior_scaled(x):
                continue
            n_flag += 1
            fus, g, cs = float(det.answer.iloc[i]), float(det.g.iloc[i]), float(det.cs.iloc[i])
            other = cf[i] if np.isfinite(cs) and np.isclose(cs, cc[i]) and not np.isclose(cc[i], cf[i]) else cc[i]
            cands = [("fusion", fus * f), ("g", g * f), ("cs", cs * f), ("caw_other", float(other) * f), ("g_u", float(gu[i]) * f)]
            ch = PQ.choose(cands, lo, hi)
            if ch is None:
                continue
            nm, c, how = ch
            q = x.get("q") or {}
            out[str(int(det.index[i]))] = dict(value=c / f, why=f"{TAG}: {cls} {q.get('dim') or q.get('kind')} answer {a:.4g} SI outside "
                                                 f"[{lo / MARGIN:.4g}, {hi * MARGIN:.4g}] (range {lo}-{hi}: {src}); own {nm} {c:.4g} SI ({how} range); "
                                                 f"phrase '{cp}'")
        json.dump(out, open(f"{R}/cache/tmp/ag_{TAG}_patch_{split}.json", "w"), indent=0)
        print(split, "flagged", n_flag, "written", len(out), sorted(int(k) for k in out))


if __name__ == "__main__":
    main()
