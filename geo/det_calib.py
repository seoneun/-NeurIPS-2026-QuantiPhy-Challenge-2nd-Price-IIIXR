#!/usr/bin/env python
import argparse
import ast
import json
import math
import os
import re
import sys

import numpy as np

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, f"{R}/geo")
import track_common as tc

EPS = 1e-3
FEATS = ["one", "ls", "has_o", "lo"]
FEATS_RICH = FEATS + ["s", "o", "ls_lo"]


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def phi(s, o, feats=None):
    s, o = np.asarray(s, float), np.asarray(o, float)
    has = (o > 0).astype(float)
    lo = has * logit(np.where(o > 0, o, 0.5))
    cols = dict(one=np.ones_like(s), ls=logit(s), has_o=has, lo=lo, s=s, o=o, ls_lo=logit(s) * lo)
    return np.stack([cols[k] for k in (feats or FEATS)], -1)


def load_vote(path):
    js = json.load(open(path))
    feats = list(js["coef"])
    b = np.array([js["coef"][k] for k in feats])
    qm = js.get("qmap")

    def vote(s, o):
        p = float(1.0 / (1.0 + np.exp(-(phi(s, o, feats) @ b))))
        if qm:
            p = float(np.interp(p, qm["p"], qm["sum"]))
        return p
    return vote


def owl_support(box, owl_dets):
    return max([s for b, s in owl_dets if tc.iou(b, box) >= 0.5] + [0.0])


def load_vlm(split):
    out = {}
    p = f"{R}/geo/out/vlmground_{split}_notext.jsonl"
    for ln in open(p, encoding="utf-8"):
        x = json.loads(ln)
        out[(x["video_id"], x["key"], int(x["j"]))] = x
    return out


def groups(split):
    vlm = load_vlm(split)
    d = f"{R}/geo/out/detdump_{split}"
    out = []
    for fn in sorted(os.listdir(d)):
        j = json.load(open(f"{d}/{fn}"))
        W, H, di = j["W"], j["H"], j["det_frames"]
        for q in j["queries"]:
            for k, fi in enumerate(di):
                x = vlm.get((j["video_id"], q["key"], fi))
                if x is None or x.get("multi"):
                    continue
                vb = x.get("boxes_rel1000") or []
                if not vb:
                    continue
                sx, sy = W / x["W"], H / x["H"]
                vb = [vb[0][0] * sx, vb[0][1] * sy, vb[0][2] * sx, vb[0][3] * sy]
                ph = q["phrase"]
                cands = [(b, s, cf) for b, s, cf in j["gd"][ph][k] if (b[2] - b[0]) * (b[3] - b[1]) < 0.85 * W * H and s >= 0.15]
                if not cands:
                    continue
                owl_c = j["owl"][ph][k]
                o = [owl_support(b, owl_c) for b, s, cf in cands]
                col = [(0.35 + 1.3 * min(max(cf), 0.5)) if cf else 1.0 for b, s, cf in cands]
                cw = any(cf for b, s, cf in cands) or None
                out.append(dict(vid=j["video_id"], split=split, key=q["key"], k=k, plain=not [w for w in q["where"] if w != "#second"] and not q.get("rel"),
                                s=np.array([c[1] for c in cands]), o=np.array(o), col=np.array(col), color=cw is not None,
                                y=np.array([tc.iou(b, vb) >= 0.5 for b, s, cf in cands], float)))
    return out


def fit_logreg(X, y, lam=1e-2):
    from scipy.optimize import minimize

    def f(b):
        z = X @ b
        ll = np.logaddexp(0, z) - y * z
        g = X.T @ (1 / (1 + np.exp(-z)) - y)
        return ll.sum() + 0.5 * lam * (b[1:] ** 2).sum(), g + lam * np.r_[0, b[1:]]

    return minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x


def fit_clogit(G, lam=1e-2, feats=None):
    from scipy.optimize import minimize
    feats = feats or FEATS
    Gs = [(phi(g["s"], g["o"], feats)[:, 1:], g["y"]) for g in G if g["y"].any() and len(g["y"]) > 1]

    def f(b):
        v, gr = 0.0, np.zeros_like(b)
        for X, y in Gs:
            z = X @ b
            lse = np.logaddexp.reduce(z)
            lpos = np.logaddexp.reduce(z[y > 0])
            v -= lpos - lse
            p, q = np.exp(z - lse), np.where(y > 0, np.exp(z - lpos), 0.0)
            gr -= X.T @ (q - p)
        return v + 0.5 * lam * (b ** 2).sum(), gr + lam * b

    return np.r_[0.0, minimize(f, np.zeros(len(feats) - 1), jac=True, method="L-BFGS-B").x]


def stack(G):
    X = np.concatenate([phi(g["s"], g["o"]) for g in G])
    y = np.concatenate([g["y"] for g in G])
    return X, y


def scores(g, how, b=None, feats=None):
    if how == "gd":
        v = g["s"]
    elif how == "sum":
        v = g["s"] + g["o"]
    else:
        v = 1 / (1 + np.exp(-(phi(g["s"], g["o"], feats) @ b)))
    return v * g["col"]


def ece(p, y, nb=10):
    bins = np.minimum((p * nb).astype(int), nb - 1)
    e = 0.0
    for i in range(nb):
        m = bins == i
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def cmd_eval(a):
    G = [g for sp in a.splits.split(",") for g in groups(sp)]
    vids = sorted({(g["split"], g["vid"]) for g in G})
    rng = np.random.default_rng(0)
    fold = dict(zip(vids, rng.permutation(len(vids)) % a.folds))
    res = {h: [] for h in ("gd", "sum", "calib", "clogit", "clogit_rich")}
    P = {"calib": [], "sum_platt": []}
    Y = []
    for f in range(a.folds):
        tr = [g for g in G if fold[(g["split"], g["vid"])] != f]
        te = [g for g in G if fold[(g["split"], g["vid"])] == f]
        b = fit_logreg(*stack(tr))
        bc = fit_clogit(tr)
        bcr = fit_clogit(tr, feats=FEATS_RICH)
        Xs = np.concatenate([np.stack([np.ones_like(g["s"]), g["s"] + g["o"]], 1) for g in tr])
        bs = fit_logreg(Xs, stack(tr)[1])
        for g in te:
            pos = g["y"].any()
            for h in res:
                sc = scores(g, h, {"clogit": bc, "clogit_rich": bcr}.get(h, b), FEATS_RICH if h == "clogit_rich" else None)
                res[h].append((pos, g["plain"], len(g["s"]) > 1, bool(g["y"][int(np.argmax(sc))]), g["split"] + "/" + g["vid"]))
            P["calib"].append(1 / (1 + np.exp(-(phi(g["s"], g["o"]) @ b))))
            P["sum_platt"].append(1 / (1 + np.exp(-(np.stack([np.ones_like(g["s"]), g["s"] + g["o"]], 1) @ bs))))
            Y.append(g["y"])
    y = np.concatenate(Y)
    print('')
    for sub, sel in [("all groups with a positive", lambda r: r[0]), ("... and > 1 candidate", lambda r: r[0] and r[2]),
                     ("... plain queries, > 1 candidate", lambda r: r[0] and r[1] and r[2])]:
        line = []
        for h in res:
            rr = [r for r in res[h] if sel(r)]
            line.append(f"{h} {np.mean([r[3] for r in rr]):.4f}")
        print(f"top-1 accuracy, {sub:34s} (n={len([r for r in res['sum'] if sel(r)])}): " + "  ".join(line))
    rng = np.random.default_rng(1)
    for h in ("clogit", "clogit_rich", "calib"):
        rows = [(r[4], float(r[3]) - float(q[3])) for r, q in zip(res[h], res["sum"]) if r[0] and r[2]]
        vv = sorted({v for v, _ in rows})
        per = {v: [0.0, 0] for v in vv}
        for v, x in rows:
            per[v][0] += x; per[v][1] += 1
        S, N = np.array([per[v][0] for v in vv]), np.array([per[v][1] for v in vv])
        bs = []
        for _ in range(2000):
            k = rng.integers(0, len(vv), len(vv))
            bs.append(S[k].sum() / N[k].sum())
        print('')
    for h, pp in P.items():
        p = np.clip(np.concatenate(pp), 1e-6, 1 - 1e-6)
        ll = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
        print('')


def cmd_fit(a):
    G = [g for sp in a.splits.split(",") for g in groups(sp)]
    feats = FEATS_RICH if a.method == "clogit_rich" else FEATS
    b = fit_logreg(*stack(G)) if a.method == "platt" else fit_clogit(G, feats=feats)
    X = np.concatenate([phi(g["s"], g["o"], feats) for g in G])
    p = 1 / (1 + np.exp(-(X @ b)))
    sm = np.concatenate([g["s"] + g["o"] for g in G])
    qs = np.linspace(0, 1, 201)
    qp, qsum = np.quantile(p, qs), np.quantile(sm, qs)
    qp = np.maximum.accumulate(qp + np.arange(len(qp)) * 1e-9)
    js = dict(coef=dict(zip(feats, map(float, b))), method=a.method, n_groups=len(G), splits=a.splits,
              qmap=dict(p=qp.tolist(), sum=qsum.tolist()) if a.qmap else None,
              rule="vote = sigmoid(coef . features(s, o)) [mapped onto the sum's distribution with qmap]; labels = IoU >= 0.5 with the Qwen3-VL-8B referring box")
    json.dump(js, open(a.out, "w"), indent=1)
    print(json.dumps(js["coef"], indent=1), "->", a.out)


class Img:
    def __init__(self, cf):
        self.cf = cf


def track_v2_funcs(vote):
    src = open(f"{R}/geo/track_v2.py", encoding="utf-8").read()
    tree = ast.parse(src)
    keep = {"owl_support", "heads", "contrastive", "rel_pick", "choose", "queries_for", "select_anchors"}
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in keep]
    body += [n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", "") in ("COLOR_WORDS", "CONTRAST") for t in n.targets)]
    mod = ast.Module(body=body, type_ignores=[])
    tc2 = type(sys)("tc2")
    tc2.__dict__.update({k: getattr(tc, k) for k in dir(tc) if not k.startswith("__")})
    tc2.color_frac = lambda img, b, c: img.cf.get((tuple(b), c), 0.0)
    ns = dict(np=np, re=re, math=math, tc=tc2, vote=vote)
    exec(compile(mod, "track_v2_select", "exec"), ns)
    return ns


def replay_video(j, ns):
    di, W, H = j["det_frames"], j["W"], j["H"]
    dets = {ph: [[(b, s) for b, s, cf in j["gd"][ph][k]] for k in range(len(di))] for ph in j["phrases"]}
    owl = [{ph: [(b, s) for b, s in j["owl"][ph][k]] for ph in j["phrases"]} for k in range(len(di))]
    imgs = []
    for k in range(len(di)):
        cf = {}
        for ph in j["phrases"]:
            cws = [c for c in tc.COLOR_WORDS if f" {c} " in f" {ph} "]
            for b, s, fr in j["gd"][ph][k]:
                for c, x in zip(cws, fr):
                    cf[(tuple(b), c)] = x
        imgs.append(Img(cf))
    outs, anchors, resolved = ns["select_anchors"](j["queries"], di, imgs, dets, owl, W, H, True)
    return outs


def cmd_replay(a):
    calib = load_vote(a.calib)
    ns_sum, ns_cal = track_v2_funcs(lambda s, o: s + o), track_v2_funcs(calib)
    d = f"{R}/geo/out/detdump_{a.split}"
    n_q = n_anchor_sum = n_anchor_cal = n_changed = n_same = 0
    ch = []
    frozen_ok = frozen_n = 0
    for fn in sorted(os.listdir(d)):
        j = json.load(open(f"{d}/{fn}"))
        o_sum, o_cal = replay_video(j, ns_sum), replay_video(j, ns_cal)
        fz = f"{R}/geo/out/tracks_{a.split}_v2/{fn}"
        if os.path.exists(fz):
            jf = json.load(open(fz))
            for qa, qf in zip(o_sum, jf["queries"]):
                if qa.get("anchor") is not None and qf.get("anchor") is not None:
                    frozen_n += 1
                    bq = qa["det_boxes"][j["det_frames"].index(qf["anchor"][0])]
                    frozen_ok += int(bq is not None and tc.iou(bq, qf["anchor"][1]) > 0.7)
        for q, a1, a2 in zip(j["queries"], o_sum, o_cal):
            n_q += 1
            n_anchor_sum += a1["anchor"] is not None
            n_anchor_cal += a2["anchor"] is not None
            if a1["anchor"] is None or a2["anchor"] is None:
                if (a1["anchor"] is None) != (a2["anchor"] is None):
                    n_changed += 1
                    ch.append((j["video_id"], q["key"], a1["anchor"], a2["anchor"]))
                continue
            same = a1["anchor"][0] == a2["anchor"][0] and tc.iou(a1["anchor"][1], a2["anchor"][1]) > 0.9
            k1 = j["det_frames"].index(a1["anchor"][0])
            b2 = a2["det_boxes"][k1]
            same = same or (b2 is not None and tc.iou(a1["anchor"][1], b2) > 0.7)
            n_same += same
            if not same:
                n_changed += 1
                ch.append((j["video_id"], q["key"], a1["anchor"], a2["anchor"]))
    print('')
    if frozen_n:
        print('')
    json.dump(ch, open(a.out or f"{R}/geo/out/det_calib_changes_{a.split}.json", "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--splits", default="val,test")
    e.add_argument("--folds", type=int, default=5)
    f = sub.add_parser("fit")
    f.add_argument("--splits", default="val,test")
    f.add_argument("--out", default=f"{R}/geo/out/det_calib.json")
    f.add_argument("--method", default="clogit", choices=["clogit", "clogit_rich", "platt"])
    f.add_argument("--qmap", action="store_true")
    r = sub.add_parser("replay")
    r.add_argument("--split", default="test")
    r.add_argument("--calib", default=f"{R}/geo/out/det_calib.json")
    r.add_argument("--out", default="")
    a = ap.parse_args()
    {"eval": cmd_eval, "fit": cmd_fit, "replay": cmd_replay}[a.cmd](a)
