#!/usr/bin/env python
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






def row_params(P, Z):
    clip = lambda p: np.clip(p, PMIN, 1 - PMIN)
    return dict(mu_g=Z["mu_g"] @ P["mu_g"], sg=np.exp(Z["ls_g"] @ P["ls_g"]), pg=clip(expit(Z["lp_g"] @ P["lp_g"])),
                mu_c=Z["mu_c"] @ P["mu_c"], sc=np.exp(Z["ls_c"] @ P["ls_c"]), pc=clip(expit(Z["lp_c"] @ P["lp_c"])),
                rho=P["rho"], tg=np.exp(P["ltau_g"]), tc=np.exp(P["ltau_c"]))


def _lnorm(x, s):
    return -0.5 * (x / s) ** 2 - np.log(s) - 0.5 * LN2PI










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














