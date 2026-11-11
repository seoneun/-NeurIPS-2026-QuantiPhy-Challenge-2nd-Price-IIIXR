#!/usr/bin/env python
import argparse
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, ndtr

LN2PI = np.log(2 * np.pi)
CS = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
LO, HI = np.log(1 - CS), np.log(1 + CS)
STRONG2D, STRONG3D = "2d_ratio", "3d_calib"
PMIN = 1e-3
BLOCKS = ["mu_g", "ls_g", "lp_g", "mu_c", "ls_c", "lp_c"]
INIT = {"mu_g": 0.0, "ls_g": np.log(0.2), "lp_g": -2.5, "mu_c": 0.0, "ls_c": np.log(0.35), "lp_c": -2.0}
SPECS = {
    "small": {"mu_g": ["one"], "ls_g": ["one", "g3d", "gweak"], "lp_g": ["one", "gweak"],
              "mu_c": ["one", "motion"], "ls_c": ["one", "motion"], "lp_c": ["one", "motion"]},
    "mid": {"mu_g": ["one", "gweak"], "ls_g": ["one", "g3d", "gweak", "dyn"], "lp_g": ["one", "gweak", "gvlm"],
            "mu_c": ["one", "motion", "lab"], "ls_c": ["one", "motion", "is3d"], "lp_c": ["one", "motion", "dyn"]},
    "full": {"mu_g": ["one", "g3d", "gweak", "gvlm", "dyn"], "ls_g": ["one", "g3d", "gweak", "gvlm", "dyn"],
             "lp_g": ["one", "g3d", "gweak", "gvlm", "dyn"],
             "mu_c": ["one", "motion", "is3d", "dyn", "lab", "inet"], "ls_c": ["one", "motion", "is3d", "dyn", "lab", "inet"],
             "lp_c": ["one", "motion", "is3d", "dyn", "lab", "inet"]},
}
SPECS.update({k + "_t2": dict(v, tau="split") for k, v in list(SPECS.items())})


def base_mode(m):
    m = m if isinstance(m, str) else ""
    return m.replace("vlm_", "").replace("_clipped", "").replace("_scene", "")


def feat_frame(df):
    mode = df["mode"].map(base_mode)
    cat, src = df["cat"].astype(str), df["src"].astype(str)
    f = pd.DataFrame(index=df.index)
    f["one"] = 1.0
    f["g3d"] = (mode == STRONG3D).astype(float)
    f["gweak"] = (~mode.isin([STRONG2D, STRONG3D])).astype(float)
    f["gvlm"] = df["mode"].astype(str).str.startswith("vlm_").astype(float)
    f["dyn"] = cat.str.startswith("D").astype(float)
    f["is3d"] = cat.str.endswith("3").astype(float)
    f["motion"] = (df["kind"].astype(str) != "size").astype(float)
    f["lab"] = (src == "lab").astype(float)
    f["inet"] = (src == "internet").astype(float)
    return f


def design(df, spec):
    f = feat_frame(df)
    return {b: f[spec[b]].values for b in BLOCKS}


def sizes(spec):
    return [len(spec[b]) for b in BLOCKS]


def unpack(theta, spec):
    out, k = {}, 0
    for b, n in zip(BLOCKS, sizes(spec)):
        out[b] = theta[k:k + n]
        k += n
    out["rho"] = np.tanh(theta[k])
    out["ltau_g"] = theta[k + 1]
    out["ltau_c"] = theta[k + 2] if spec.get("tau") == "split" else theta[k + 1]
    return out


def theta0(spec):
    th = []
    for b, n in zip(BLOCKS, sizes(spec)):
        th += [INIT[b]] + [0.0] * (n - 1)
    return np.array(th + [np.arctanh(0.3), np.log(1.5)] + ([np.log(1.5)] if spec.get("tau") == "split" else []))


def n_tail(spec):
    return 3 if spec.get("tau") == "split" else 2


def row_params(P, Z):
    clip = lambda p: np.clip(p, PMIN, 1 - PMIN)
    return dict(mu_g=Z["mu_g"] @ P["mu_g"], sg=np.exp(Z["ls_g"] @ P["ls_g"]), pg=clip(expit(Z["lp_g"] @ P["lp_g"])),
                mu_c=Z["mu_c"] @ P["mu_c"], sc=np.exp(Z["ls_c"] @ P["ls_c"]), pc=clip(expit(Z["lp_c"] @ P["lp_c"])),
                rho=P["rho"], tg=np.exp(P["ltau_g"]), tc=np.exp(P["ltau_c"]))


def _lnorm(x, s):
    return -0.5 * (x / s) ** 2 - np.log(s) - 0.5 * LN2PI


def _lbinorm(a, b, sa, sb, r):
    q = 1 - r ** 2
    z = (a / sa) ** 2 - 2 * r * (a / sa) * (b / sb) + (b / sb) ** 2
    return -0.5 * z / q - np.log(sa) - np.log(sb) - 0.5 * np.log(q) - LN2PI


def loglik(rp, eg, ec):
    hg, hc = np.isfinite(eg), np.isfinite(ec)
    a, b = np.nan_to_num(eg - rp["mu_g"]), np.nan_to_num(ec - rp["mu_c"])
    sg, sc, pg, pc, r, tg, tc = rp["sg"], rp["sc"], rp["pg"], rp["pc"], rp["rho"], rp["tg"], rp["tc"]
    lg = np.logaddexp(np.log1p(-pg) + _lnorm(a, sg), np.log(pg) + _lnorm(a, tg))
    lc = np.logaddexp(np.log1p(-pc) + _lnorm(b, sc), np.log(pc) + _lnorm(b, tc))
    both = np.logaddexp.reduce([np.log1p(-pg) + np.log1p(-pc) + _lbinorm(a, b, sg, sc, r),
                                np.log1p(-pg) + np.log(pc) + _lnorm(a, sg) + _lnorm(b, tc),
                                np.log(pg) + np.log1p(-pc) + _lnorm(a, tg) + _lnorm(b, sc),
                                np.log(pg) + np.log(pc) + _lnorm(a, tg) + _lnorm(b, tc)])
    return np.where(hg & hc, both, np.where(hg, lg, np.where(hc, lc, 0.0)))


def fit(df, lam=1.0, spec="small"):
    spec = SPECS[spec] if isinstance(spec, str) else spec
    y = np.log(df["y"].astype(float).values)
    eg = np.log(df["g"].astype(float).values) - y
    ec = np.log(df["cs"].astype(float).values) - y
    keep = np.isfinite(eg) | np.isfinite(ec)
    Z = {b: m[keep] for b, m in design(df, spec).items()}
    eg, ec = eg[keep], ec[keep]
    nt = n_tail(spec)
    pen = np.concatenate([[0.0] + [1.0] * (n - 1) for n in sizes(spec)] + [[0.0] * nt])

    def obj(th):
        rp = row_params(unpack(th, spec), Z)
        return (-loglik(rp, eg, ec).sum() + 0.5 * lam * (pen * th ** 2).sum()
                + 0.5 * (th[-nt] / 0.5) ** 2 + 0.5 * ((th[-nt + 1:]) ** 2).sum() / 0.25)

    res = minimize(obj, theta0(spec), method="L-BFGS-B", bounds=[(-8, 8)] * len(theta0(spec)))
    return dict(spec=spec, theta=res.x)


def to_json(m):
    P = unpack(m["theta"], m["spec"])
    out = {b: dict(zip(m["spec"][b], map(float, P[b]))) for b in BLOCKS}
    out["rho"], out["tau_g"], out["tau_c"] = float(P["rho"]), float(np.exp(P["ltau_g"])), float(np.exp(P["ltau_c"]))
    out["tau_mode"] = m["spec"].get("tau", "shared")
    return out


def from_json(js):
    spec = {b: list(js[b]) for b in BLOCKS}
    spec["tau"] = js.get("tau_mode", "shared")
    th = []
    for b in BLOCKS:
        th += [js[b][k] for k in spec[b]]
    tail = [np.arctanh(js["rho"]), np.log(js["tau_g"])] + ([np.log(js["tau_c"])] if spec["tau"] == "split" else [])
    return dict(spec=spec, theta=np.array(th + tail))


def posterior(rp, xg, xc):
    hg, hc = np.isfinite(xg), np.isfinite(xc)
    a, b = np.nan_to_num(xg - rp["mu_g"]), np.nan_to_num(xc - rp["mu_c"])
    sg, sc, pg, pc, r = rp["sg"], rp["sc"], rp["pg"], rp["pc"], rp["rho"]
    n = len(a)
    tg, tc = np.full(n, rp["tg"]), np.full(n, rp["tc"])
    LW, Mu, Sd = [], [], []
    for s1, s2, rr, lw in [(sg, sc, r, np.log1p(-pg) + np.log1p(-pc)), (sg, tc, 0.0, np.log1p(-pg) + np.log(pc)),
                           (tg, sc, 0.0, np.log(pg) + np.log1p(-pc)), (tg, tc, 0.0, np.log(pg) + np.log(pc))]:
        c12 = rr * s1 * s2
        V = s1 ** 2 + s2 ** 2 - 2 * c12
        w = (s2 ** 2 - c12) / V
        LW.append(lw + _lnorm(a - b, np.sqrt(V)))
        Mu.append(w * a + (1 - w) * b)
        Sd.append(np.sqrt((s1 ** 2 * s2 ** 2 - c12 ** 2) / V))
    LW, Mu, Sd = np.stack(LW, 1), np.stack(Mu, 1), np.stack(Sd, 1)
    ninf = np.full(n, -np.inf)
    for has, x, s, p, t in [(hg & ~hc, a, sg, pg, tg), (hc & ~hg, b, sc, pc, tc)]:
        LW[has] = np.stack([np.log1p(-p), np.log(p), ninf, ninf], 1)[has]
        Mu[has] = np.stack([x] * 4, 1)[has]
        Sd[has] = np.stack([s, t, s, s], 1)[has]
    W = np.exp(LW - LW.max(1, keepdims=True))
    return W / W.sum(1, keepdims=True), Mu, Sd


def expected_mra(W, Mu, Sd, lp):
    e = np.zeros(lp.shape)
    z = lp[:, :, None]
    for k in range(W.shape[1]):
        m, s = Mu[:, k][:, None, None], Sd[:, k][:, None, None]
        e += W[:, k][:, None] * (ndtr((z - LO - m) / s) - ndtr((z - HI - m) / s)).mean(2)
    return e


def decide(m, df, grid=801, chunk=512, both_only=False, rule="emra"):
    rp = row_params(unpack(m["theta"], m["spec"]), design(df, m["spec"]))
    xg = np.log(df["g"].astype(float).values)
    xc = np.log(df["cs"].astype(float).values)
    W, Mu, Sd = posterior(rp, xg, xc)
    out = np.full(len(xg), np.nan)
    ok = (np.isfinite(xg) & np.isfinite(xc)) if both_only else (np.isfinite(xg) | np.isfinite(xc))
    idx = np.where(ok)[0]
    if rule == "map":
        k = W[idx].argmax(1)
        out[idx] = np.exp(Mu[idx][np.arange(len(idx)), k])
        return out
    u = np.linspace(-1.0, 1.0, grid)
    for i0 in range(0, len(idx), chunk):
        ii = idx[i0:i0 + chunk]
        w, mu, sd = W[ii], Mu[ii], Sd[ii]
        live = w > 1e-6
        lo = np.where(live, mu - 1.5 * np.minimum(sd, 1.0) - 0.7, np.inf).min(1)
        hi = np.where(live, mu + 1.5 * np.minimum(sd, 1.0) + 0.7, -np.inf).max(1)
        lp = (lo + hi)[:, None] / 2 + (hi - lo)[:, None] / 2 * u[None, :]
        out[ii] = np.exp(lp[np.arange(len(ii)), expected_mra(w, mu, sd, lp).argmax(1)])
    return out


def describe(m):
    P = to_json(m)
    lines = [f"rho {P['rho']:+.3f}  tau_g {P['tau_g']:.3f}  tau_c {P['tau_c']:.3f}"]
    for b in BLOCKS:
        lines.append(f"{b:5s} " + "  ".join(f"{k}={v:+.3f}" for k, v in P[b].items()))
    return "\n".join(lines)


def row_mra(p, y):
    p, y = np.abs(np.asarray(p, float)), np.asarray(y, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.abs(p - y) / y
        s = sum((r < c).astype(float) for c in CS) / 10
    return np.where(np.isfinite(p) & (p != 0), s, 0.0)


def macro(s, cat):
    m = pd.Series(np.asarray(s, float), index=cat.index).groupby(cat).mean().reindex(["S2", "D2", "S3", "D3"])
    return float(m.mean()), m


def lovo_fits(df, lam, spec):
    return {v: fit(df[df.vid != v], lam, spec) for v in sorted(df.vid.unique())}


def cmd_fit(a):
    df = pd.read_csv(a.dump, index_col=0)
    m = fit(df, a.lam, a.spec)
    print(describe(m))
    js = dict(lam=a.lam, spec_name=a.spec, n_rows=int((df.g.notna() | df.cs.notna()).sum()), default=to_json(m),
              lovo={v: to_json(t) for v, t in lovo_fits(df, a.lam, a.spec).items()})
    json.dump(js, open(a.out, "w"), indent=1)
    print("wrote", a.out)


def cmd_eval(a):
    df = pd.read_csv(a.dump, index_col=0)
    both = df.g.notna() & df.cs.notna()
    b = pd.Series(row_mra(df.base_v, df.y), index=df.index)
    print('')
    for lam in a.lams:
        held = pd.Series(np.nan, index=df.index)
        for v, m in lovo_fits(df, lam, a.spec).items():
            te = (df.vid == v).values
            held[te] = decide(m, df[te], both_only=True)
        s = pd.Series(row_mra(held, df.y), index=df.index)
        mc = macro(s[both], df.cat[both])
        print('')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("dump")
    f.add_argument("out")
    f.add_argument("--lam", type=float, default=1.0)
    f.add_argument("--spec", default="small", choices=list(SPECS))
    e = sub.add_parser("eval")
    e.add_argument("dump")
    e.add_argument("--lams", type=float, nargs="+", default=[0.5, 1.0, 4.0])
    e.add_argument("--spec", default="small", choices=list(SPECS))
    a = ap.parse_args()
    {"fit": cmd_fit, "eval": cmd_eval}[a.cmd](a)
