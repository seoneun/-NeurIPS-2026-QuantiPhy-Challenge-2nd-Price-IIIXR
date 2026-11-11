#!/usr/bin/env python
import sys, os, re, json, importlib.util
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
TAG = "pxq"
_sp = importlib.util.spec_from_file_location("rul336", f"{R}/analysis/v3/compliance/rules/rul336.py")
PQ = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(PQ)
MARGIN = PQ.MARGIN
SPD = "speed avgspeed hvel"
BIRD = r"eagle|hawk|falcon|vulture|condor|bird|seagull|gull|pigeon|dove|crow|raven|sparrow|swallow|parrot|owl|heron|stork|crane bird|pelican|albatross"
PX = [
    ("flying bird", r"\b(?:" + BIRD + r")s?\b", "tgt", r"div(?:e|ing)|stoop|swoop|fall|drop|plung|walk|swim|float|paddl|perch|toy|paper|kite|plane|drone|robot|phoenix|dragon|rubber|wooden",
     (SPD,), 0.0, 31.0, "level flight: cruising airspeeds of 138 bird species 8-23 m/s (Alerstam et al. 2007, PLoS Biol 5:e197); fastest measured level flight 31.1 m/s (common swift; Henningsson, Johansson & Hedenstrom 2010, J Avian Biol 41:94); stoops (dives) are excluded by the question words"),
    ("train", r"\b(?:train|tram|streetcar|locomotive|subway|metro)s?\b", "tgt", r"toy|model|lego|training|trained|wooden|cable car",
     ("accel",), 0.0, 3.0, "rail vehicles: service acceleration / braking about 0.5-1.5 m/s^2 (metro EMUs about 1.0-1.3 m/s^2); tram emergency braking with magnetic track brakes about 2.5-3 m/s^2"),
    ("train", r"\b(?:train|tram|streetcar|locomotive|subway|metro)s?\b", "tgt", r"toy|model|lego|training|trained|wooden|cable car",
     (SPD,), 0.0, 160.0, "rail speed record 574.8 km/h = 159.7 m/s (TGV V150, 3 April 2007); service speeds up to about 97 m/s (350 km/h)"),
    ("forklift", r"\bfork ?lift(?: truck)?s?\b", "tgt", r"toy|model",
     (SPD,), 0.0, 7.0, "counterbalance forklifts: maximum travel speed about 16-20 km/h (4.4-5.6 m/s) for 1.5-3.5 t trucks, up to about 25 km/h (6.9 m/s) for large trucks (manufacturer spec sheets)"),
    ("shark", r"\bsharks?\b", "tgt", r"toy|model|robot|whale shark",
     (SPD,), 0.0, 20.0, "shortfin mako, the fastest shark: burst speeds up to about 70 km/h (19.4 m/s); cruising sharks about 0.3-2 m/s"),
    ("turtle", r"\b(?:sea )?(?:turtle|tortoise|terrapin)s?\b", "tgt", r"toy|model|robot",
     (SPD,), 0.0, 10.0, "fastest reptile in water: leatherback sea turtle 35.28 km/h (9.8 m/s; Guinness World Records); land tortoises below 0.5 m/s"),
    ("raccoon", r"\brac+oons?\b", "tgt", r"toy|model|robot",
     (SPD,), 0.0, 6.7, "raccoons run at up to about 16-24 km/h (4.4-6.7 m/s; Zeveloff 2002, Raccoons: A Natural History)"),
    ("truck", r"(?<!toy )(?<!fire )\b(?:truck|lorry|pickup)s?\b", "tgt", r"toy|model|robot|fork|hand truck|monster",
     (SPD,), 0.0, 50.0, "trucks: heavy goods vehicles speed-limited to 90 km/h (25 m/s, EU Directive 92/6/EEC); pickup trucks up to about 50 m/s (180 km/h)"),
    ("motorcycle", r"\b(?:motorcycle|motorbike|motor bike|moped)s?\b", "tgt", r"toy|model|robot",
     (SPD,), 0.0, 90.0, "production motorcycles: top speeds about 299-300 km/h (83 m/s, the manufacturers' voluntary limit since 2001); 90 m/s allows for unrestricted machines"),
    ("airplane", r"\b(?:air|aero)?planes?\b|\b(?:aircraft|airliner|jet)s?\b", "tgt", r"paper|toy|model|\brc\b|drone|glider|inclined|ramp|slope|jet ski|water jet|jet of",
     (SPD,), 0.0, 340.0, "subsonic aircraft: airliners cruise at about 240-260 m/s; speed of sound at sea level 340 m/s"),
    ("telephone booth", r"\b(?:tele)?phone (?:booth|box|kiosk)s?\b", "tgt", r"toy|model|miniature",
     ("size", "width length"), 0.75, 1.40, "telephone kiosks / booths: GPO K6 0.91 m square, K2 about 1.37 m wide; Western Electric booths about 0.8-1.0 m wide"),
    ("telephone booth", r"\b(?:tele)?phone (?:booth|box|kiosk)s?\b", "tgt", r"toy|model|miniature",
     ("size", "height"), 2.1, 2.9, "telephone kiosks / booths: GPO K6 about 2.5 m high, K2 about 2.7-2.8 m high; Western Electric booths about 2.1-2.3 m high"),
    ("bicycle height", r"height of the (?:moving |parked |\w+ )?(?:bicycle|bike)\b(?! ?(?:wheel|tire|tyre|seat|saddle|rack|lane|path))", "q", r"\b(?:motor\w*|toy|model|rider|riding|person|man|woman|child|boy|girl)\b",
     ("size", "height"), 0.6, 1.3, "bicycle height to handlebar / saddle about 0.9-1.2 m; children's bicycles from about 0.6 m (rul336 bicycle height entry)"),
    ("bicycle wheel", r"(?:(?:wheel|tire|tyre)s? of the (?:moving |parked |\w+ )?(?:bicycle|bike)\b|\b(?:bicycle|bike)(?:'s|’s)? (?:wheel|tire|tyre)s?\b)", "q", r"motor|toy|model|e-?bike|electric",
     ("size", "diameter size"), 0.30, 0.76, "ISO 5775 bead seat diameters 203-635 mm (12 in to 29 in); outer diameter about 0.30-0.75 m (rul336 bicycle wheel entry)"),
    ("car tyre", r"(?:\bcar(?:'s|’s) (?:wheel|tire|tyre)s?\b|(?:wheel|tire|tyre)s? of the (?:\w+ )?car\b)", "q", r"toy|model|robot|race|racing|formula|kart|golf|cable|train|\brc\b",
     ("size", "diameter size"), 0.50, 0.85, "passenger car / SUV tyres: outer diameter 0.53 m (155/65R13) to 0.83 m (285/70R17)"),
    ("truck tyre", r"(?:\btruck(?:'s|’s) (?:wheel|tire|tyre)s?\b|(?:wheel|tire|tyre)s? of the (?:\w+ )?truck\b)", "q", r"toy|model|robot|monster|mining|haul|dump|fork",
     ("size", "diameter size"), 0.55, 1.15, "truck tyres: kei trucks 145R12 about 0.55 m; light trucks LT245/75R16 about 0.77 m; heavy trucks 295/80R22.5 about 1.04 m, 445/65R22.5 about 1.15 m"),
    ("aircraft wheel", r"(?:landing gear|\b(?:aircraft|airplane|plane|airliner|jet)(?:'s|’s)? (?:wheel|tire|tyre)s?\b|(?:wheel|tire|tyre)s? of the (?:\w+ )?(?:aircraft|airplane|plane|airliner|jet)\b)", "q", r"toy|model|paper|\brc\b|drone",
     ("size", "diameter size"), 0.30, 1.42, "aircraft tyres (Goodyear / Michelin aircraft tire data): light aircraft 5.00-5 / 6.00-6 about 0.36-0.44 m; Boeing 777 main 52x21.0R22 1.32 m; Airbus A380 main 1400x530R23 1.40 m"),
    ("bench seat", r"(?:height of the (?:\w+ )?bench seat|seat height of the (?:\w+ )?bench)\b", "q", r"toy|model|back|top of the back",
     ("size", "height"), 0.35, 0.55, "bench seat height: ADA 2010 Standards 903.5 430-485 mm; public benches typically 0.40-0.50 m; children's benches from about 0.35 m"),
]
QEXCL = r"\b(?:toy|model|miniature|lego|figurine|statue|sculpture|cartoon|drawing|picture|poster|painting|replica|robot|rc car|remote[- ]controlled)\b"


def lookup(question, x):
    e0, cp = PQ.lookup(question, x)
    if e0 is not None:
        return None, cp
    q = x.get("q") or {}
    kind, dim = q.get("kind"), PQ.DIMSYN.get(q.get("dim") or "", q.get("dim") or "")
    ql = str(question).lower().replace("’", "'")
    tg = re.sub(r"\s+", " ", str(q.get("target") or "").lower()).strip()
    if (cp and re.search(PQ.NOT_ALL, cp)) or re.search(QEXCL, ql):
        return None, cp
    for e in PX:
        cls, rx, where, nx, kd, lo, hi, src = e
        if kd[0] == "size":
            if kind != "size" or dim not in kd[1].split():
                continue
        elif kind not in kd[0].split():
            continue
        txt = tg if where == "tgt" else ql
        if not txt or not re.search(rx, txt):
            continue
        if nx and re.search(nx, ql):
            continue
        return e, (tg if where == "tgt" else cp)
    return None, cp


def prior_scaled(x):
    if PQ.prior_scaled(x):
        return True
    for p in x.get("priors") or []:
        if p.get("kind") != "size" or not p.get("value_si"):
            continue
        obj = p.get("obj") or ((x.get("prior_obj") or {}).get("phrase")) or ""
        raw = str(p.get("raw") or "")
        e, _ = lookup(raw, {"q": {"kind": "size", "dim": p.get("dim") or "", "target": obj}})
        if e is not None and not (e[5] <= float(p["value_si"]) <= e[6]):
            return True
    return False


def load(split, BASE, CUR, GT):
    det = pd.read_csv(f"{R}/submissions/{BASE}_{'detail' if split == 'test' else 'valdetail'}.csv", index_col=0)
    cur = pd.read_csv(f"{R}/submissions/{CUR}_id1.csv").parsed_value.values if split == "test" else det.answer.values
    cf = pd.read_csv(f"{R}/preds/{split}_caw9b_orig.csv").parsed_value.values
    cc = pd.read_csv(f"{R}/preds/{split}_caw9b_orig_crop.csv").parsed_value.values
    gu = pd.read_csv(f"{R}/preds/{split}_geo_v4nt{GT}usrngsr_d_vw_raw.csv").geo_value.values
    P = [json.loads(l) for l in open(f"{R}/geo/out/parse_{split}_v2.jsonl")]
    assert len(P) == len(det) == len(cf) == len(cc) == len(gu) == len(cur)
    assert [int(x["id"]) for x in P] == [int(k) for k in det.index]
    return det, cur, cf, cc, gu, P


def main():
    BASE = sys.argv[1] if len(sys.argv) > 1 else "pred_t064"
    CUR = sys.argv[2] if len(sys.argv) > 2 else "pred_t054"
    GT = sys.argv[3] if len(sys.argv) > 3 else "qdir3"
    for split in ("test", "val"):
        det, cur, cf, cc, gu, P = load(split, BASE, CUR, GT)
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
