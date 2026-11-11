import os, re, math, importlib.util
AG = os.environ["QP_R"] + "/analysis/v3/ag"


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


SR = _load("r23_scales_b", f"{AG}/r23/SCALES/py/rules.py")
WR = _load("r23_weights_b", f"{AG}/r23/WEIGHTS/py/rules.py")
PL = WR.PL
SIM_GEO_UNITS = ["probe_S2_geo_simulation", "probe_D2_geo_simulation", "probe_S3_geo_simulation", "probe_D3_geo_simulation"]
P0 = dict(w_agree=0.65, sim_geo=True, up=1.1, up_h=1.05, up_scope="old", k_dyn=0.9, k_dyn_size=0.81, k_st=1.0, k_st_nonsize=1.0,
          win_w=2.6, win_l=1.6, bound="plaus", up_d2=None, units=("agree_w", "sim_geo", "sim_up", "mra"))


def stage_fusion(p):
    out = []
    if "agree_w" in p["units"]:
        WR.WT["agree_w065"] = p["w_agree"]; out.append(WR.make("agree_w065"))
    if "sim_geo" in p["units"] and p["sim_geo"]:
        out += [WR.make(u) for u in SIM_GEO_UNITS]
    return out


def stage_caw(p):
    on = set(p["units"])

    def rule(r):
        cl, x = SR.classify(r)
        if cl == "other": return None
        v = float(r["cur"]); cat = r["cat"]; kd = SR.kind(r); sim = str(r["video_id"]).startswith("simulation")
        q = str(r["question"]).lower(); ish = re.search(SR.RX["height"], q) is not None
        up = p["up_h"] if ish else p["up"]
        if p.get("up_d2") and cat == "D2": up = p["up_d2"]
        gv = SR._fin((r.get("geo") or {}).get("geo_value")); why = []
        if p["up_scope"] == "old":
            up_ok = sim and kd == "size" and ((cl in ("fusion", "caw") and cat == "D2" and gv) or (cl == "d3caw" and cat == "D3"))
        else:
            up_ok = sim and kd == "size"
        up_ok = up_ok and "sim_up" in on
        if cl == "fusion":
            if not (up_ok and x > 0): return None
            f = up ** x; return v * f, f"r23b U3 sim_up x{up}^{x:.3f} (CaW weight of the fused answer): {v:.4g} -> {v * f:.4g}"
        if cl == "caw":
            base = x; f = 1.0
            if up_ok: f *= up; why.append(f"U3 sim_up x{up}")
            if "mra" in on and not SR.kin_exempt(r, base):
                k = (p["k_dyn_size"] if kd == "size" else p["k_dyn"]) if cat[0] == "D" else (p["k_st"] if kd == "size" else p["k_st_nonsize"])
                if k != 1.0: f *= k; why.append(f"U4 mra x{k}")
            nv = base * f
            if abs(nv / v - 1) < 1e-12: return None
            return nv, f"r23b CaW-alone (own read {base:.4g}): " + "; ".join(why or ["none"]) + f": {v:.4g} -> {nv:.4g}"
        nv = v
        if up_ok: nv *= up; why.append(f"U3 sim_up x{up}")
        g = SR._fin(r.get("s1_g"))
        if "dimgm" in on and sim and cat == "D3" and g and not ish:
            for dm, win in (("width", p["win_w"]), ("length", p["win_l"])):
                if win and re.search(SR.RX[dm], q) and abs(math.log10(g / nv)) <= math.log10(win):
                    if dm == "width" and p["bound"] == "plaus":
                        rg = PL.rng_for(r)
                        if rg is not None and not (rg[0] <= g <= rg[1]): continue
                    if dm == "width" and p["bound"] == "old":
                        km = SR._kin(); b = km.class_bound(str(r["question"]), str(r["prior"]), "size")
                        if b is not None and not (b[0] <= g * km.q_si(str(r["question"])) <= b[1]): continue
                    nv = math.sqrt(nv * g); why.append(f"U5 dimgm {dm} gm with own geo {g:.4g} (within {win}x)")
        if abs(nv / v - 1) < 1e-12: return None
        return nv, "r23b CaW-route D3 size: " + "; ".join(why) + f": {v:.4g} -> {nv:.4g}"
    return rule


def plaus_wrap(fn):
    def rule(r):
        o = fn(r)
        if o is None: return None
        return o if SR.plaus_ok(r, float(r["cur"]), float(o[0])) else None
    return rule


def stages(p=None):
    p = dict(P0, **(p or {}))
    return stage_fusion(p) + [plaus_wrap(stage_caw(p))]


ACCEPTED = stages()
