#!/usr/bin/env python
import argparse, os, sys
import numpy as np
import pandas as pd

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
sys.path.insert(0, f"{R}/caw")
sys.path.insert(0, f"{R}/analysis/v3")
import analyze_caw as A
import analyze_v2caw as V
import fuse_v3 as F
import consistency as C

ap = argparse.ArgumentParser()
ap.add_argument("geo")
ap.add_argument("out")
ap.add_argument("--c2", default="")
ap.add_argument("--caw", default="caw9b")
ap.add_argument("--no_c1", action="store_true")
ap.add_argument("--rul263", default="")
ap.add_argument("--d2size", default="")
ap.add_argument("--caw_class_scale", default="")
ap.add_argument("--caw_prescale", default="")
ap.add_argument("--cawalone_size_type", default="")
ap.add_argument("--phys_bounds_final", action="store_true")
ap.add_argument("--rul263_dimgm", default="")
ap.add_argument("--rul263_scale", default="")
ap.add_argument("--caw_override", default="")
ap.add_argument("--caw_tgm", default="")
ap.add_argument("--caw_full", default="")
ap.add_argument("--rul263_full", action="store_true")
ap.add_argument("--caw_speed32", default="")
ap.add_argument("--caw_speed_spread", default="")
ap.add_argument("--d3lab_focal", default="")
ap.add_argument("--cawconf", type=float, default=0.0)
ap.add_argument("--cawfull", default="")
ap.add_argument("--cawconf_rules", default="d3,dis")
ap.add_argument("--var_caw", default="")
ap.add_argument("--sib_caw", default="")
ap.add_argument("--pool_caw", default="")
ap.add_argument("--dis_motion", default="")
ap.add_argument("--kin_toff", action="store_true")
ap.add_argument("--rul263_skip_modes", default="")
ap.add_argument("--patch_json", default="")
ap.add_argument("--depth_bound", action="store_true")
ap.add_argument("--geo_only_modes", default="")
ap.add_argument("--clipped_caw", action="store_true")
ap.add_argument("--dis_size_gm", default="")
ap.add_argument("--caw_kinds", default="")
ap.add_argument("--sib_geo_fill", action="store_true")
ap.add_argument("--event_geo", default="", nargs="?", const="all")
ap.add_argument("--agree_w", default="")
ap.add_argument("--kin", default="")
ap.add_argument("--caw_shrink", default="")
ap.add_argument("--xrow_prior", action="store_true")
ap.add_argument("--c2_anchor", default="")
ap.add_argument("--var_geo_outlier", action="store_true")
ap.add_argument("--probe_kin_exempt", action="store_true")
ap.add_argument("--probe", default="")
ap.add_argument("--inet_geo", default="")
ap.add_argument("--depth_copy", action="store_true")
ap.add_argument("--sruler_geo", default="")
ap.add_argument("--scene_prior", default="")
ap.add_argument("--sib_caw_var", default="")
ap.add_argument("--std_court", action="store_true")
ap.add_argument("--speed_series", default="")
ap.add_argument("--disp_bound", default="")
ap.add_argument("--var_norm", action="store_true")
ap.add_argument("--scene_prior_ext", default="")
ap.add_argument("--family_pool", default="")
ap.add_argument("--lab_major", type=float, default=0.0)
ap.add_argument("--lab_pool", default="")
ap.add_argument("--var_ans_plain", default="")
ap.add_argument("--var_geo_plain", default="")
ap.add_argument("--phys_bounds", default="", nargs="?", const="base")
ap.add_argument("--event_caw", default="")
ap.add_argument("--dis_size_low", default="")
ap.add_argument("--cqc", action="store_true")
ap.add_argument("--cqc_mode", default="")
ap.add_argument("--group_geo", default="")
ap.add_argument("--pool_size", default="")
ap.add_argument("--calib_fuse", default="")
ap.add_argument("--geo_kinds", default="")
a = ap.parse_args()


def answers(split):
    d = A.add_keys(A.load_val() if split == "val" else A.load_test())
    caw = pd.to_numeric(pd.read_csv(f"{R}/preds/{split}_{a.caw}.csv").set_index("Unnamed: 0").parsed_value, errors="coerce")
    if a.caw_full:
        import consistency as C_
        ftk_ = (a.cawfull or "caw9b,caw9b_orig").split(",")[0 if split == "val" else 1]
        full_ = pd.to_numeric(pd.read_csv(f"{R}/preds/{split}_{ftk_}.csv").set_index("Unnamed: 0").parsed_value, errors="coerce").reindex(d.index)
        kd_ = C_.parse_info(split, d.index).kind.astype(str)
        selC = pd.Series(False, index=d.index)
        for ck_ in a.caw_full.split(","):
            c__, k__ = ck_.split(":")
            selC = selC | ((d.cat == c__) & ((kd_ == k__) if k__ != "motion" else kd_.isin(["speed", "avgspeed", "accel", "disp", "path", "hvel"])))
        selC = selC & full_.notna()
        caw = caw.copy(); caw.loc[selC[selC].index] = full_[selC]
        print('', file=sys.stderr)
    if a.caw_tgm:
        import json as _json
        import consistency as C_
        cats_, kinds_, mode_ = a.caw_tgm.split("/")
        p_ = f"{R}/caw/sc/out/{'val' if split == 'val' else 'test_orig_t1'}/t1n16.jsonl"
        tg_ = {}
        for ln_ in open(p_, encoding="utf-8"):
            x_ = _json.loads(ln_)
            v_ = np.sort(np.log([abs(float(q)) for q in x_["parsed"] if q is not None and np.isfinite(float(q)) and float(q) != 0]))
            if len(v_) >= 4:
                k_ = len(v_) // 4
                tg_[int(x_["original_id"])] = float(np.exp(v_[k_:len(v_) - k_].mean()))
        tg_ = pd.Series(tg_, dtype=float).reindex(d.index)
        kd_ = C_.parse_info(split, d.index).kind.astype(str)
        sel_ = d.cat.isin(cats_.split("+")) & kd_.isin(kinds_.split("+")) & tg_.notna() & caw.reindex(d.index).notna()
        cv_ = caw.reindex(d.index).abs()
        nw_ = tg_ if mode_ == "rep" else np.sqrt(cv_ * tg_)
        caw = caw.copy(); caw.loc[sel_[sel_].index] = nw_[sel_]
        print('', file=sys.stderr)
    if a.caw_speed32:
        import json as _json
        import consistency as C_
        p_ = f"{R}/caw/sc/out/{a.caw_speed32.split(',')[0 if split == 'val' else 1]}/g32.jsonl"
        g32_ = {}
        if os.path.exists(p_):
            for ln_ in open(p_, encoding="utf-8"):
                x_ = _json.loads(ln_)
                q_ = x_["parsed"][0]
                if q_ is not None and np.isfinite(float(q_)) and float(q_) != 0:
                    g32_[int(x_["original_id"])] = abs(float(q_))
        kd_ = C_.parse_info(split, d.index).kind
        sp_ = pd.Series(g32_, dtype=float).reindex(d.index)
        sel_ = kd_.isin(["speed", "avgspeed", "hvel"]) & sp_.notna()
        caw = caw.copy(); caw.loc[sel_[sel_].index] = sp_[sel_]
        print('', file=sys.stderr)
    if a.var_caw:
        from variant_keys import split_variant
        sv = d.vid.astype(str).map(split_variant)
        base, var = sv.str[0], sv.str[1]
        qn = d.question.astype(str).str.strip().str.lower()
        cv = caw.reindex(d.index).abs()
        lc = np.log10(cv.where(cv > 0))
        key = base + "||" + qn
        gm = 10 ** lc.groupby(key).transform("mean")
        nv = d.vid.astype(str).groupby(key).transform("nunique")
        sel = var.isin(a.var_caw.split(",")) & (nv > 1) & cv.notna() & gm.notna()
        caw = caw.copy()
        caw.loc[sel[sel].index] = gm[sel]
        print('', file=sys.stderr)
    if a.sib_caw and split == "test":
        import json as _json
        cv = caw.reindex(d.index).abs()
        ls_ = np.log10(cv.where(cv > 0)); ln_ = ls_.notna().astype(float); ls_ = ls_.fillna(0.0)
        for dn in a.sib_caw.split(","):
            sib = {}
            for ln in open(f"{R}/caw/sc/out/{dn}/g16.jsonl", encoding="utf-8"):
                x_ = _json.loads(ln)
                p_ = x_["parsed"][0]
                if p_ is not None and float(p_) != 0:
                    sib[int(x_["original_id"])] = abs(float(p_))
            sb = pd.Series(sib, dtype=float).reindex(d.index)
            ok_ = sb.notna() & cv.notna() & (cv > 0)
            ls_ = ls_ + np.log10(sb.where(ok_)).fillna(0.0); ln_ = ln_ + ok_.astype(float)
        sel = (ln_ > 1) & cv.notna()
        if a.sib_caw_var:
            from variant_keys import split_variant
            vv6_ = d.vid.astype(str).map(lambda z: split_variant(z)[1])
            sel = sel & vv6_.isin(a.sib_caw_var.split(","))
        caw = caw.copy()
        caw.loc[sel[sel].index] = (10 ** (ls_ / ln_))[sel]
        print('', file=sys.stderr)
    if a.pool_caw:
        from variant_keys import split_variant
        var = d.vid.astype(str).map(lambda x: split_variant(x)[1])
        pk = d.ground_truth_prior.astype(str).str.replace(r"\s+", "", regex=True).str.lower()
        qn = d.question.astype(str).str.strip().str.lower()
        key = pk + "||" + qn
        cv = caw.reindex(d.index).abs()
        lc = np.log10(cv.where(cv > 0))
        member = lc.where(var == "plain") if a.pool_caw == "plain" else lc
        ssum = member.groupby(key).transform("sum"); scnt = member.groupby(key).transform("count")
        own_in = member.notna()
        tot_s = ssum - member.fillna(0) + lc.fillna(0)
        tot_n = scnt - own_in.astype(int) + lc.notna().astype(int)
        gm = 10 ** (tot_s / tot_n)
        nv = d.vid.astype(str).groupby(key).transform("nunique")
        sel = (nv > 1) & cv.notna() & (tot_n > 1)
        if os.environ.get("POOL_KINDS"):
            import consistency as C_
            sel = sel & C_.parse_info(split, d.index).kind.isin(os.environ["POOL_KINDS"].split(","))
        caw = caw.copy()
        caw.loc[sel[sel].index] = gm[sel]
        print('', file=sys.stderr)
    if a.event_caw:
        from variant_keys import split_event
        ev_ = d.vid.astype(str).map(split_event)
        isev_ = ev_.str[1] != ""
        key_ = ev_.str[0] + "||" + d.question.astype(str).str.strip().str.lower()
        cv_ = caw.reindex(d.index).abs()
        lc_ = np.log10(cv_.where(cv_ > 0))
        gm_ = 10 ** lc_.groupby(key_).transform("mean")
        nv_ = d.vid.astype(str).groupby(key_).transform("nunique")
        sel_ = isev_ & (nv_ > 1) & cv_.notna() & gm_.notna()
        if a.event_caw == "size":
            import consistency as C_
            sel_ = sel_ & (C_.parse_info(split, d.index).kind == "size")
        caw = caw.copy(); caw.loc[sel_[sel_].index] = gm_[sel_]
        print('', file=sys.stderr)
    if a.caw_prescale:
        import consistency as C_
        typ_ = d.vid.astype(str).str.split("_").str[0]
        kd_ = C_.parse_info(split, d.index).kind.astype(str)
        gv_ = pd.to_numeric(pd.read_csv(f"{R}/preds/{split}_{a.geo}_raw.csv").set_index("Unnamed: 0").geo_value, errors="coerce").reindex(d.index)
        for spec_ in a.caw_prescale.split(","):
            t_, dm_, k_, f_ = spec_.split(":")
            sel_ = typ_.str.startswith(t_) & ((d.cat == dm_) if len(dm_) == 2 else (d.cat.str[1] == dm_)) & (kd_ == k_) & (gv_ > 0) & caw.reindex(d.index).notna()
            caw = caw.copy(); caw.loc[sel_[sel_].index] = caw.reindex(d.index)[sel_] * float(f_)
            print('', file=sys.stderr)
    if a.caw_class_scale:
        import re as re_
        typ_ = d.vid.astype(str).str.split("_").str[0]
        for spec_ in a.caw_class_scale.split(";"):
            t_, rx_, k_ = spec_.split(":")
            sel_ = typ_.str.startswith(t_) & d.question.astype(str).str.lower().str.contains(rx_, regex=True) & caw.reindex(d.index).notna()
            caw = caw.copy(); caw.loc[sel_[sel_].index] = caw.reindex(d.index)[sel_] * float(k_)
            print('', file=sys.stderr)
    if a.caw_override and split == "test":
        import json as _json
        ov_ = {}
        for ln_ in open(f"{R}/caw/sc/out/{a.caw_override}/g16.jsonl", encoding="utf-8"):
            x_ = _json.loads(ln_); q_ = x_["parsed"][0]
            if q_ is not None and np.isfinite(float(q_)) and float(q_) != 0:
                ov_[int(x_["original_id"])] = abs(float(q_))
        ov_ = pd.Series(ov_, dtype=float).reindex(d.index)
        sel_ = ov_.notna()
        caw = caw.copy(); caw.loc[sel_[sel_].index] = ov_[sel_]
        print('', file=sys.stderr)
    V.V2 = a.geo
    M = F.methods2(d, split, caw)
    info = C.parse_info(split, d.index)
    if a.var_geo_outlier:
        from variant_keys import split_variant
        sv_ = d.vid.astype(str).map(split_variant)
        key_ = sv_.str[0] + "||" + d.question.astype(str).str.strip().str.lower()
        lg_ = np.log10(M.g.where(M.g > 0))
        n_out = 0
        newg = M.g.copy(); newm = M["mode"].copy()
        for k_, z_ in lg_.groupby(key_):
            z_ = z_.dropna()
            if len(z_) < 3:
                continue
            for i in z_.index:
                oth = z_.drop(i)
                if len(oth) >= 2 and (oth.max() - oth.min()) <= np.log10(1.25) and abs(z_[i] - oth.median()) > np.log10(1.5):
                    newg[i] = 10 ** oth.median(); newm[i] = M["mode"][oth.index[0]]; n_out += 1
        M = M.copy(); M["g"] = newg; M["mode"] = newm
        print('', file=sys.stderr)
    if a.var_geo_plain:
        from variant_keys import split_variant
        thr_, *opt_ = a.var_geo_plain.split(":")
        thr_ = float(thr_)
        sv_ = d.vid.astype(str).map(split_variant)
        key_ = sv_.str[0] + "||" + d.question.astype(str).str.strip().str.lower()
        isp_ = (sv_.str[1] == "plain") & M.g.notna() & (M.g > 0)
        if "strong" in opt_:
            isp_ = isp_ & M["mode"].isin(["2d_ratio", "3d_calib"])
        if "2d" in opt_:
            isp_ = isp_ & M["mode"].astype(str).str.contains("2d")
        pidx_ = pd.Series(d.index[isp_], index=key_[isp_].values)
        pidx_ = pidx_[~pidx_.index.duplicated()]
        n_vp = 0
        M = M.copy()
        deg_ = d.index[(sv_.str[1] != "plain") & M.g.notna() & (M.g > 0)]
        if "video" in opt_:
            g0_ = M.g.copy()
            n_vs = 0
            for vid_, ii_ in pd.Series(deg_, index=d.vid[deg_].values).groupby(level=0):
                ii_ = list(ii_.values)
                lr_ = []
                for i in ii_:
                    j = pidx_.get(key_[i])
                    if j is not None and "2d" in str(M["mode"][i]) and "2d" in str(M["mode"][j]):
                        lr_.append(np.log10(g0_[j] / g0_[i]))
                if len(lr_) >= 2 and abs(np.median(lr_)) > np.log10(thr_) and (np.all(np.array(lr_) > 0) or np.all(np.array(lr_) < 0)):
                    f_ = 10 ** float(np.median(lr_))
                    for i in ii_:
                        if pidx_.get(key_[i]) is None and "2d" in str(M["mode"][i]):
                            M.loc[i, "g"] = g0_[i] * f_; n_vs += 1
            print('', file=sys.stderr)
        for i in deg_:
            j = pidx_.get(key_[i])
            if j is None:
                continue
            if abs(np.log10(M.g[i] / M.g[j])) > np.log10(thr_):
                for c_ in ["g", "mode", "w1", "w2", "w3", "lab3d"]:
                    M.loc[i, c_] = M.loc[j, c_]
                n_vp += 1
        print('', file=sys.stderr)
    if a.event_geo:
        from variant_keys import split_event
        ev_ = d.vid.astype(str).map(split_event)
        isev_ = ev_.str[1] != ""
        key_ = ev_.str[0] + "||" + d.question.astype(str).str.strip().str.lower()
        strong_ = M["mode"].isin(["2d_ratio", "3d_calib"]) & M.g.notna()
        lgs_ = np.log10(M.g.where(strong_))
        evg_ = 10 ** lgs_.groupby(key_).transform("mean")
        sel_ = isev_ & evg_.notna()
        if a.event_geo == "weak":
            sel_ = sel_ & ~strong_
        M = M.copy()
        M.loc[sel_, "g"] = evg_[sel_]
        M.loc[sel_, "mode"] = "3d_calib"
        for w in ["w1", "w2", "w3", "lab3d"]:
            M.loc[sel_, w] = False
        print('', file=sys.stderr)
    if a.sib_geo_fill:
        from variant_keys import split_variant
        sv_ = d.vid.astype(str).map(split_variant)
        key_ = sv_.str[0] + "||" + d.question.astype(str).str.strip().str.lower() + "||" + d.ground_truth_prior.astype(str).str.replace(r"\s+", "", regex=True).str.lower()
        lg_ = np.log10(M.g)
        sib_ = 10 ** ((lg_.groupby(key_).transform("sum") - lg_.fillna(0)) / (lg_.groupby(key_).transform("count") - lg_.notna().astype(int)).replace(0, np.nan))
        fill_ = M.g.isna() & sib_.notna()
        M = M.copy()
        M.loc[fill_, "g"] = sib_[fill_]
        M.loc[fill_, "mode"] = "vlm_sibfill"
        for w in ["w1", "w2", "w3"]:
            M.loc[fill_, w] = True
        print('', file=sys.stderr)
    bad, why = C.c1_mask(d, M.g, info)
    if a.no_c1:
        bad = bad & False
    M1 = M.copy()
    M1["g"] = M.g.where(~bad)
    for w in ["w1", "w2", "w3", "lab3d"]:
        M1[w] = M[w] & ~bad
    v, s = V.recipe2(M1, os.environ.get("QP_RECIPE", "d_r2_gm_c1w3"))
    if os.environ.get("QP_DUMP_FUSION"):
        dump_ = M1[["g", "c", "cs", "q", "h", "mode", "w3", "lab3d"]].copy()
        dump_["vid"], dump_["cat"], dump_["src"] = d.vid.astype(str), d.cat, d.src
        dump_["video_type"], dump_["video_source"] = d.video_type.astype(str), d.video_source.astype(str)
        dump_["kind"], dump_["unit"] = info.kind.astype(str), info.unit.astype(str)
        dump_["base_v"], dump_["base_s"] = v, s
        ftk_ = (a.cawfull or "caw9b,caw9b_orig").split(",")[0 if split == "val" else 1]
        dump_["cfull"] = A.pos(pd.read_csv(f"{R}/preds/{split}_{ftk_}.csv").set_index("Unnamed: 0").parsed_value.reindex(d.index))
        graw_ = pd.read_csv(f"{R}/preds/{split}_{a.geo}_raw.csv").set_index("Unnamed: 0").reindex(d.index)
        dump_["merge_src"], dump_["owlvote"] = graw_.merge_src.astype(str), graw_.meas_owlvote
        if split == "val":
            dump_["y"] = d.y
        os.makedirs(os.environ["QP_DUMP_FUSION"], exist_ok=True)
        dump_.to_csv(f"{os.environ['QP_DUMP_FUSION']}/{a.out}_{split}.csv")
    s = s.where(~(bad & M.g.notna()) | (s != "caw_nogeo"), "caw_c1")
    if a.calib_fuse:
        import json as _json
        sys.path.insert(0, f"{R}/analysis/v3/calib")
        import fusion_calib as FC
        path_, *opt_ = a.calib_fuse.split(":")
        js_ = _json.load(open(path_ if os.path.isabs(path_) else f"{R}/{path_}"))
        X_ = M1[["g", "cs", "mode"]].copy()
        X_["cat"], X_["src"], X_["kind"] = d.cat, d.src, info.kind.astype(str)
        sel_ = s.isin(["agree_gm", "dis_geo", "dis_caw_routed"]) & M1.g.notna() & M1.cs.notna()
        cv_ = pd.Series(np.nan, index=d.index)
        if split == "val" and "lovo" in opt_:
            for vid_ in d.vid[sel_].unique():
                ii_ = sel_ & (d.vid == vid_)
                cv_[ii_] = FC.decide(FC.from_json(js_["lovo"][str(vid_)]), X_[ii_])
        else:
            cv_[sel_] = FC.decide(FC.from_json(js_["default"]), X_[sel_])
        v = v.copy(); v[sel_] = cv_[sel_]
        if "relabel" in opt_:
            s = s.where(~sel_, np.where(s == "agree_gm", "agree_calib", "dis_calib"))
        print('', file=sys.stderr)
    if a.geo_only_modes:
        go_ = M1["mode"].astype(str).isin(a.geo_only_modes.split(",")) & M1.g.notna() & (M1.g > 0)
        v = v.copy(); v[go_] = M1.g[go_]; s = s.where(~go_, "geo_only_mode")
        print('', file=sys.stderr)
    if a.rul263 in ("caw", "caw_strong"):
        sel = (d.cat == "D3") & (info.kind == "size") & M1.cs.notna() & s.isin(["agree_gm", "dis_geo"])
        if a.rul263 == "caw_strong":
            sel = sel & M1["mode"].isin(["2d_ratio", "3d_calib"])
        if a.rul263_skip_modes:
            sel = sel & ~M1["mode"].astype(str).isin(a.rul263_skip_modes.split(","))
        v = v.copy(); v[sel] = M1.cs[sel]; s = s.where(~sel, "rul263_caw")
    if a.rul263_full:
        ftk_ = (a.cawfull or "caw9b,caw9b_orig").split(",")[0 if split == "val" else 1]
        full_ = pd.to_numeric(pd.read_csv(f"{R}/preds/{split}_{ftk_}.csv").set_index("Unnamed: 0").parsed_value, errors="coerce").abs().reindex(d.index)
        selF = (s == "rul263_caw") & full_.notna() & (full_ > 0)
        v = v.copy(); v[selF] = full_[selF]
        print('', file=sys.stderr)
    if a.d3lab_focal:
        import json as _json
        spec_ = a.d3lab_focal.split(":")
        dev_ = float(spec_[1]) if len(spec_) > 1 else 1.0
        act_ = spec_[2] if len(spec_) > 2 else "gm"
        rawf_ = pd.to_numeric(pd.read_csv(f"{R}/preds/{split}_{a.geo}_raw.csv").set_index("Unnamed: 0").meas_f, errors="coerce").reindex(d.index)

        def vres_(sp_, vids_):
            out_ = {}
            for v_ in vids_:
                for sub_ in ("_v2", "_v3nt", ""):
                    p_ = f"{R}/geo/out/tracks_{sp_}{sub_}/{v_}.json"
                    if os.path.exists(p_):
                        vj_ = _json.load(open(p_)); out_[v_] = f"{vj_['W']}x{vj_['H']}"; break
            return out_
        if spec_[0] == "auto":
            rt_ = pd.read_csv(f"{R}/preds/test_{a.geo}_raw.csv").set_index("Unnamed: 0")
            rt_ = rt_[rt_.video_id.astype(str).str.startswith("captured") & (rt_.inference_type.astype(str).str[0] == "S")
                      & (rt_.video_type.astype(str).str[1] == "3") & (rt_.meas_mode.astype(str) == "3d_calib") & rt_.meas_f.notna()]
            fv_ = rt_.groupby("video_id").meas_f.median()
            rv_ = pd.Series(vres_("test", fv_.index)).reindex(fv_.index)
            pool_ = fv_.groupby(rv_).median().to_dict()
        else:
            pool_ = {k: float(x) for k, x in (p.split("=") for p in spec_[0].split(","))}
        res_ = vres_(split, [v_ for v_ in d.vid.astype(str).unique() if v_.startswith("captured")])
        fp_ = d.vid.astype(str).map(res_).map(pool_)
        sel = ((d.cat == "D3") & (info.kind == "size") & d.vid.astype(str).str.startswith("captured") & M1["mode"].isin(["3d_calib", "vlm_3d_calib"])
               & M1.g.notna() & M1.cs.notna() & rawf_.notna() & fp_.notna() & s.isin(["rul263_caw", "agree_gm", "dis_geo"]))
        sel = sel & (np.abs(np.log10(rawf_ / fp_)) > np.log10(dev_))
        gnew_ = M1.g * rawf_ / fp_
        v = v.copy()
        if act_ == "geo":
            v[sel] = gnew_[sel]
        else:
            v[sel] = np.sqrt(gnew_[sel] * M1.cs[sel])
        s = s.where(~sel, "d3lab_" + act_)
        inb_ = spec_[3] if len(spec_) > 3 else ""
        if inb_:
            selb = ((d.cat == "D3") & (info.kind == "size") & d.vid.astype(str).str.startswith("captured")
                    & M1["mode"].isin(["3d_calib", "vlm_3d_calib"]) & M1.g.notna() & M1.cs.notna() & rawf_.notna()
                    & fp_.notna() & (s == "rul263_caw") & (np.abs(np.log10(rawf_ / fp_)) <= np.log10(dev_)))
            v = v.copy(); v[selb] = M1.g[selb] if inb_ == "geo" else np.sqrt(M1.g[selb] * M1.cs[selb])
            s = s.where(~selb, "d3lab_inband_" + inb_)
            print('', file=sys.stderr)
        print('', file=sys.stderr)
    if a.d2size == "caw":
        sel = (d.cat == "D2") & (info.kind == "size") & M1.cs.notna() & s.isin(["agree_gm", "dis_geo"])
        v = v.copy(); v[sel] = M1.cs[sel]; s = s.where(~sel, "d2size_caw")
    if a.cawconf > 0:
        ft = (a.cawfull or "caw9b,caw9b_orig").split(",")[0 if split == "val" else 1]
        full = pd.to_numeric(pd.read_csv(f"{R}/preds/{split}_{ft}.csv").set_index("Unnamed: 0").parsed_value, errors="coerce").abs().reindex(d.index)
        with np.errstate(divide="ignore", invalid="ignore"):
            un = (np.abs(np.log10(M1.c / full)) > a.cawconf) & M1.c.notna() & full.notna()
            agree = np.abs(np.log10(M1.g / M1.cs)) <= np.log10(2)
        rl = a.cawconf_rules.split(",")
        m3 = (s == "rul263_caw") & un & M1.g.notna() & agree & ("d3" in rl)
        md = (s == "dis_caw_routed") & un & M1.g.notna() & ("dis" in rl)
        v = v.copy(); v[m3] = np.sqrt(M1.g[m3] * M1.cs[m3]); v[md] = M1.g[md]
        s = s.where(~m3, "rul263_unstable_gm").where(~md, "dis_unstable_geo")
    if a.dis_motion:
        kinds_ = [k for k in a.dis_motion.split(",") if k not in ("3d", "norej", "nofov")]
        dm = (s == "dis_caw_routed") & info.kind.isin(kinds_) & M1.g.notna()
        if "3d" in a.dis_motion.split(","):
            dm = dm & (d.video_type.str[1] == "3")
        if "norej" in a.dis_motion.split(","):
            dm = dm & ~M1["mode"].astype(str).str.contains("calibrejected")
        if "nofov" in a.dis_motion.split(","):
            dm = dm & ~M1["mode"].astype(str).str.contains("fovdefault")
        v = v.copy(); v[dm] = M1.g[dm]; s = s.where(~dm, "dis_motion_geo")
    if a.dis_size_gm:
        dz = (s == "dis_caw_routed") & (info.kind == "size") & M1.g.notna() & M1.cs.notna()
        if a.clipped_caw:
            dz = dz & ~M1["mode"].astype(str).str.contains("_clipped")
        if a.dis_size_gm.startswith("3d"):
            dz = dz & (d.video_type.str[1] == "3")
        if "_sim" in a.dis_size_gm:
            dz = dz & d.video_source.astype(str).str.lower().str.startswith(("simulation", "segment"))
        if a.dis_size_gm.endswith("_unstable"):
            ft_ = (a.cawfull or "caw9b,caw9b_orig").split(",")[0 if split == "val" else 1]
            full_ = pd.to_numeric(pd.read_csv(f"{R}/preds/{split}_{ft_}.csv").set_index("Unnamed: 0").parsed_value, errors="coerce").abs().reindex(d.index)
            with np.errstate(divide="ignore", invalid="ignore"):
                un_ = (np.abs(np.log10(M1.c / full_)) > 0.1) & M1.c.notna() & full_.notna()
            dz = dz & un_
        v = v.copy(); v[dz] = np.sqrt(M1.g[dz] * M1.cs[dz]); s = s.where(~dz, "dis_size_gm")
    if a.caw_kinds:
        ck = (s == "agree_gm") & info.kind.isin(a.caw_kinds.split(",")) & M1.cs.notna()
        v = v.copy(); v[ck] = M1.cs[ck]; s = s.where(~ck, "agree_caw_kind")
    if a.geo_kinds:
        gk = (s == "agree_gm") & info.kind.isin(a.geo_kinds.split(",")) & M1.g.notna()
        v = v.copy(); v[gk] = M1.g[gk]; s = s.where(~gk, "agree_geo_kind")
    if a.agree_w:
        sp_ = a.agree_w.split(":")
        kinds_, w_ = sp_[0].split(","), float(sp_[1])
        aw = (s == "agree_gm") & info.kind.isin(kinds_) & M1.g.notna() & M1.cs.notna()
        if "strong" in sp_[2:]:
            aw = aw & M1["mode"].isin(["2d_ratio", "3d_calib"])
        for dm_ in ("2d", "3d"):
            if dm_ in sp_[2:]:
                aw = aw & (d.video_type.str[1] == dm_[0])
        v = v.copy(); v[aw] = (M1.g[aw] ** w_) * (M1.cs[aw] ** (1 - w_)); s = s.where(~aw, "agree_wgm")
        if os.environ.get("QP_TIGHT_CAW"):
            tr_ = float(os.environ["QP_TIGHT_CAW"])
            tight = aw & (np.abs(np.log10(M1.g / M1.cs)) < np.log10(tr_))
            v[tight] = M1.cs[tight]; s = s.where(~tight, "agree_tight_caw")
            print('', file=sys.stderr)
        print('', file=sys.stderr)
    if a.kin:
        import kinematic as K_
        sp_ = a.kin.split(":")
        srcs_ = ("simulation", "segmentation") if sp_[0] == "sim" else None
        use_gm = "gm" in sp_[1:]
        K_.BEFORE_OK = "before" in sp_[1:]
        types_ = next((x.split(",") for x in sp_[1:] if x not in ("gm", "full", "grav", "before", "vT")), None)
        kv = pd.Series(np.nan, index=d.index); kt = pd.Series("", index=d.index)
        for i in d.index:
            if srcs_ and not str(d.video_source[i]).lower().startswith(srcs_):
                continue
            K_.TOFF = (1.0 / float(d.fps[i])) if (a.kin_toff and str(d.video_source[i]).lower().startswith(("simulation", "segment")) and float(d.fps[i]) > 0) else 0.0
            x_, h_ = K_.propagate(str(d.ground_truth_prior[i]), str(d.question[i]))
            if not np.isfinite(x_) and "grav" in sp_[1:]:
                x_, h_ = K_.propagate_gravity(str(d.ground_truth_prior[i]), str(d.question[i]))
                gi_ = M1.g.get(i, np.nan)
                if not (np.isfinite(x_) and np.isfinite(gi_) and gi_ > 0 and abs(np.log10(gi_ / x_)) <= np.log10(1.25)):
                    x_ = np.nan
            if not np.isfinite(x_) and "vT" in sp_[1:]:
                pq_ = str(d.ground_truth_prior[i]); qq_ = str(d.question[i])
                if (K_.pkind(pq_) == "speed" and K_.qkind(qq_) == "disp" and K_.times(qq_)[0] == "none" and K_.obj(pq_) == K_.obj(qq_)
                        and len(K_.obj(pq_)) >= 3 and not K_.prior_windowed(pq_) and str(M1["mode"].get(i, "")) in ("2d_ratio", "3d_calib")
                        and np.isfinite(M1.g.get(i, np.nan))):
                    x_, h_ = float(M1.g[i]), "v*T(geo)"
            if np.isfinite(x_) and (types_ is None or h_ in types_):
                kv[i], kt[i] = x_, h_
        sel = kv.notna()
        v = v.copy()
        old_ = v[sel].copy()
        v[sel] = np.sqrt(kv[sel] * v[sel].abs()) if use_gm else kv[sel]
        s = s.where(~sel, "kin")
        with np.errstate(divide="ignore", invalid="ignore"):
            dl = np.abs(np.log10(v[sel] / old_))
        print('', file=sys.stderr)
    if a.caw_shrink:
        for spec_ in a.caw_shrink.split(";"):
            parts_ = spec_.split(":")
            srcs_, kinds_, k_ = parts_[:3]
            sk = s.isin(srcs_.split(",")) & info.kind.isin(kinds_.split(","))
            for f_ in parts_[3:]:
                if f_.startswith("src!="):
                    sk = sk & ~d.video_source.astype(str).str.lower().str.startswith(f_[5:])
                elif f_.startswith("src="):
                    sk = sk & d.video_source.astype(str).str.lower().str.startswith(f_[4:])
            if len(parts_) > 3 and parts_[3] in ("unstable", "stable"):
                ftk_ = (a.cawfull or "caw9b,caw9b_orig").split(",")[0 if split == "val" else 1]
                full_k = pd.to_numeric(pd.read_csv(f"{R}/preds/{split}_{ftk_}.csv").set_index("Unnamed: 0").parsed_value, errors="coerce").abs().reindex(d.index)
                with np.errstate(divide="ignore", invalid="ignore"):
                    unk = (np.abs(np.log10(M1.c / full_k)) > 0.1) & M1.c.notna() & full_k.notna()
                sk = sk & (unk if parts_[3] == "unstable" else ~unk)
            v = v.copy(); v[sk] = v[sk] * float(k_)
            print('', file=sys.stderr)
    if a.dis_size_low:
        dl_ = (s == "dis_geo") & (info.kind == "size") & M1.g.notna() & M1.cs.notna() & (M1.g < M1.cs)
        for dm_ in ("2d", "3d"):
            if dm_ in a.dis_size_low.split(":"):
                dl_ = dl_ & (d.video_type.str[1] == dm_[0])
        v = v.copy(); v[dl_] = np.sqrt(M1.g[dl_] * M1.cs[dl_]); s = s.where(~dl_, "dis_size_low_gm")
        print('', file=sys.stderr)
    if a.cqc:
        import kinematic as K_
        import re as re_
        def usi_(q):
            q = q.lower()
            return 0.01 if re_.search(r"\bcm\b|cm/s", q) else 0.001 if re_.search(r"\bmm\b", q) else 1.0
        n_sw = 0
        dis_ = s.isin(["dis_geo", "dis_caw_routed", "dis_motion_geo"])
        for vid_, z_ in d.groupby("vid"):
            idx_ = list(z_.index)
            qs_ = {i: str(d.question[i]) for i in idx_}
            ob_ = {i: K_.obj(qs_[i]) for i in idx_}; kd_ = {i: K_.qkind(qs_[i]) for i in idx_}; tm_ = {i: K_.times(qs_[i]) for i in idx_}
            def trusted_(j):
                gj, cj = M1.g[j], M1.cs[j]
                if np.isfinite(gj) and np.isfinite(cj) and gj > 0 and cj > 0 and abs(np.log10(gj / cj)) <= np.log10(1.5):
                    return np.sqrt(gj * cj) * usi_(qs_[j])
                return None
            for i in idx_:
                if not dis_[i]:
                    continue
                refs_ = []
                if kd_[i] == "disp" and tm_[i][0] == "int":
                    t1_, t2_ = tm_[i][1], tm_[i][2]
                    for j in idx_:
                        if j != i and ob_[j] == ob_[i] and kd_[j] == "speed" and tm_[j][0] == "at" and t1_ - 0.3 <= tm_[j][1] <= t2_ + 0.3:
                            x_ = trusted_(j)
                            if x_ is not None:
                                refs_.append(x_ * (t2_ - t1_))
                elif "cqc2" in (a.cqc_mode or "") and kd_[i] in ("speed", "avgspeed"):
                    ti_ = tm_[i]
                    for j in idx_:
                        if j == i or ob_[j] != ob_[i] or tm_[j][0] != "int":
                            continue
                        a1_, b1_ = tm_[j][1], tm_[j][2]
                        if b1_ - a1_ < 0.2:
                            continue
                        cover_ = (ti_[0] == "at" and a1_ - 0.2 <= ti_[1] <= b1_ + 0.2) or (ti_[0] == "int" and abs(ti_[1] - a1_) < 0.3 and abs(ti_[2] - b1_) < 0.3)
                        if not cover_:
                            continue
                        x_ = trusted_(j)
                        if x_ is not None and kd_[j] == "disp":
                            refs_.append(x_ / (b1_ - a1_))
                        elif x_ is not None and kd_[j] == "avgspeed":
                            refs_.append(x_)
                if "cqc3" in (a.cqc_mode or "") and kd_[i] == "accel" and tm_[i][0] in ("at", "int"):
                    tc_ = tm_[i][1] if tm_[i][0] == "at" else 0.5 * (tm_[i][1] + tm_[i][2])
                    sp_ = sorted([(tm_[j][1], trusted_(j)) for j in idx_ if j != i and ob_[j] == ob_[i] and kd_[j] == "speed" and tm_[j][0] == "at"
                                  and trusted_(j) is not None], key=lambda z: z[0])
                    for (ta_, va_), (tb_, vb_) in zip(sp_[:-1], sp_[1:]):
                        if tb_ - ta_ >= 0.2 and ta_ - 0.6 <= tc_ <= tb_ + 0.6 and abs(vb_ - va_) > 0.2 * max(va_, vb_):
                            refs_.append(abs(vb_ - va_) / (tb_ - ta_))
                if not refs_:
                    continue
                ref_ = np.mean(refs_) / usi_(qs_[i])
                gi, ci = M1.g[i], M1.cs[i]
                if not (np.isfinite(gi) and np.isfinite(ci) and gi > 0 and ci > 0):
                    continue
                okg, okc = abs(np.log10(gi / ref_)) <= np.log10(2), abs(np.log10(ci / ref_)) <= np.log10(2)
                if okg and not okc and abs(v[i] - gi) > 1e-12 * gi:
                    v = v.copy(); v[i] = gi; s = s.copy(); s[i] = "cqc_geo"; n_sw += 1
                elif okc and not okg and abs(v[i] - ci) > 1e-12 * ci:
                    v = v.copy(); v[i] = ci; s = s.copy(); s[i] = "cqc_caw"; n_sw += 1
        print('', file=sys.stderr)
    if a.group_geo:
        from variant_keys import split_event
        sp_ = a.group_geo.split(":")
        how_ = sp_[0]; srcs_ = sp_[1].split(",") if len(sp_) > 1 and sp_[1] else ["caw_nogeo", "dis_caw_routed", "caw_c1", "rul263_caw"]
        ev_ = d.vid.astype(str).map(lambda z: split_event(z)[0])
        key_ = d.ground_truth_prior.astype(str).str.replace(r"\s+", "", regex=True).str.lower() + "||" + d.question.astype(str).str.strip().str.lower()
        strong_ = M1["mode"].isin(["2d_ratio", "3d_calib"]) & M1.g.notna() & (M1.g > 0) & (info.kind == "size")
        lg_ = np.log10(M1.g.where(strong_))
        tab_ = pd.DataFrame({"key": key_, "ev": ev_, "lg": lg_})[strong_]
        sel = s.isin(srcs_) & (info.kind == "size") & v.notna() & (v > 0)
        gv = pd.Series(np.nan, index=d.index)
        for i in sel[sel].index:
            z_ = tab_[(tab_.key == key_[i]) & (tab_.ev != ev_[i])]
            if len(z_):
                gv[i] = 10 ** z_.lg.median()
        sel = sel & gv.notna()
        v = v.copy(); v[sel] = np.sqrt(gv[sel] * v[sel]) if how_ == "gm" else gv[sel]
        s = s.where(~sel, "group_geo")
        print('', file=sys.stderr)
    if a.pool_size:
        from variant_keys import split_event
        sp_ = a.pool_size.split(":")
        how_ = sp_[0]; kinds_ = sp_[1].split(",") if len(sp_) > 1 and sp_[1] else ["size"]; nmin_ = int(sp_[2]) if len(sp_) > 2 else 2
        ev_ = d.vid.astype(str).map(lambda z: split_event(z)[0])
        key_ = d.ground_truth_prior.astype(str).str.replace(r"\s+", "", regex=True).str.lower() + "||" + d.question.astype(str).str.strip().str.lower()
        ks = info.kind.isin(kinds_) & v.notna() & (v > 0)
        lv = np.log10(v.where(ks))
        nrec = ev_.where(ks).groupby(key_).transform("nunique")
        agg_ = lv.groupby(key_).transform("median" if how_ == "med" else "mean")
        sel = ks & (nrec >= nmin_) & agg_.notna()
        old_ = v[sel].copy()
        v = v.copy(); v[sel] = 10 ** agg_[sel]
        s = s.where(~(sel & (np.abs(np.log10(v / old_.reindex(v.index))) > 1e-9)), "pool_size")
        print('', file=sys.stderr)
    if a.phys_bounds:
        import kinematic as K_
        K_.EXTRA_BOUNDS = a.phys_bounds in ("extra", "extra2", "extra3", "extra3clean")
        K_.EXTRA2 = a.phys_bounds in ("extra2", "extra3", "extra3clean")
        K_.EXTRA3 = a.phys_bounds in ("extra3", "extra3clean")
        if a.phys_bounds == "extra3clean":
            K_.SIZE3 = [x_ for x_ in K_.SIZE3 if not any(w_ in x_[0] for w_ in ("pepperoni", "rabbit"))]
        n_pb = 0
        for i in d.index:
            b_ = K_.class_bound(str(d.question[i]), str(d.ground_truth_prior[i]), info.kind.get(i))
            if b_ is None:
                continue
            f_ = K_.q_si(str(d.question[i]))
            inside = lambda x: np.isfinite(x) and x > 0 and b_[0] <= x * f_ <= b_[1]
            if inside(v[i]):
                continue
            gi, ci = M1.g.get(i, np.nan), M1.cs.get(i, np.nan)
            cand = [x for x in (gi, ci) if inside(x)]
            if len(cand) == 1:
                v = v.copy(); v[i] = cand[0]; s = s.copy(); s[i] = "phys_bound"; n_pb += 1
        print('', file=sys.stderr)
    if a.xrow_prior:
        import kinematic as K_
        import re as re_
        from variant_keys import split_event
        Bw = chr(92) + "b"
        DIMS_ = ["length", "width", "height", "diameter", "radius", "wingspan", "thickness", "depth"]
        def qty_(txt, is_prior):
            tl = str(txt).lower()
            if is_prior:
                tl = tl.split("=")[0]
            k = K_.pkind(tl) if is_prior else K_.qkind(tl)
            if k in ("speed", "avgspeed"):
                qq = "speed"
            elif k == "accel":
                qq = "accel"
            else:
                mm = re_.search(Bw + "(" + "|".join(DIMS_) + ")" + Bw, tl)
                qq = mm.group(1) if mm else None
            return K_.obj(txt if not is_prior else str(txt).split("=")[0] + " = 1"), qq
        ev_ = d.vid.astype(str).map(lambda z: split_event(z)[0])
        n_x = 0
        for b_, z_ in d.groupby(ev_):
            pri_ = z_.ground_truth_prior.astype(str).drop_duplicates()
            if len(pri_) < 2:
                continue
            ps_ = [(p_, *qty_(p_, True), K_.prior_value(p_)) for p_ in pri_]
            for i in z_.index:
                qo_, qq_ = qty_(str(d.question[i]), False)
                if qq_ is None or len(qo_) < 3:
                    continue
                for p_, po_, pq_, (pv_, pu_) in ps_:
                    if p_ == str(d.ground_truth_prior[i]) or pv_ is None:
                        continue
                    if po_ == qo_ and pq_ == qq_:
                        v = v.copy(); v[i] = abs(pv_) * K_.to_si(pu_) / K_.q_scale(str(d.question[i])); s = s.copy(); s[i] = "xrow_prior"; n_x += 1
                        break
        print('', file=sys.stderr)
    if a.c2_anchor:
        import kinematic as K_
        K_.EXTRA_BOUNDS = True; K_.EXTRA2 = True; K_.EXTRA3 = True
        viol = set()
        for i in d.index:
            b_ = K_.class_bound(str(d.question[i]), str(d.ground_truth_prior[i]), info.kind.get(i))
            gi = M1.g.get(i, np.nan)
            if b_ is not None and np.isfinite(gi) and gi > 0 and not (b_[0] <= gi * K_.q_si(str(d.question[i])) <= b_[1]):
                viol.add(d.vid[i])
        for spec in a.c2_anchor.split(";"):
            thr, nmin, act, dims = (spec.split(",") + ["2"])[:4]
            v2, rows, badv = C.c2_apply(d, M1, info, v, float(thr), int(nmin), act, dims)
            selfmo = pd.Series([K_.obj(str(d.ground_truth_prior[i])) == K_.obj(str(d.question[i])) and info.kind.get(i) in
                                ("speed", "avgspeed", "accel", "disp", "path", "hvel") for i in d.index], index=d.index)
            keep = rows & d.vid.isin(viol) & ~s.isin(["kin", "phys_bound", "xrow_prior"]) & ~selfmo
            v = v.copy(); v[keep] = v2[keep]; s = s.where(~keep, f"c2a_{act}")
            print('', file=sys.stderr)
    if a.c2:
        for spec in a.c2.split(";"):
            thr, nmin, act, dims = (spec.split(",") + ["2"])[:4]
            v2, rows, badv = C.c2_apply(d, M1, info, v, float(thr), int(nmin), act, dims)
            s = s.where(~rows, f"c2{dims}_{act}")
            v = v2
    if a.caw_speed_spread:
        import json as _json
        k_, thr_ = [float(x) for x in a.caw_speed_spread.split(":")]
        p_ = f"{R}/caw/sc/out/{'test_orig_t1' if split == 'test' else 'val'}/t1n16.jsonl"
        sdm_ = {}
        if os.path.exists(p_):
            for ln_ in open(p_, encoding="utf-8"):
                x_ = _json.loads(ln_)
                vals_ = [abs(float(z_)) for z_ in (x_.get("parsed") or []) if z_ is not None and np.isfinite(float(z_)) and float(z_) != 0]
                if len(vals_) >= 6:
                    sdm_[int(x_["original_id"])] = float(np.std(np.log10(vals_)))
        sd_ = pd.Series(sdm_, dtype=float).reindex(d.index)
        sel_ = s.isin(["caw_nogeo", "dis_caw_routed", "caw_c1"]) & info.kind.isin(["speed", "avgspeed"]) & (sd_ >= thr_)
        v = v.copy(); v[sel_] = v[sel_] * k_
        print('', file=sys.stderr)
    if a.cawalone_size_type:
        import consistency as C_
        typ_ = d.vid.astype(str).str.split("_").str[0]
        kd_ = C_.parse_info(split, d.index).kind.astype(str)
        for spec_ in a.cawalone_size_type.split(","):
            t_, k_ = spec_.split(":")
            sel_ = s.isin(["caw_nogeo", "dis_caw_routed", "caw_c1"]) & (kd_ == "size") & typ_.str.startswith(t_)
            v = v.copy(); v[sel_] = v[sel_] * float(k_)
            print('', file=sys.stderr)
    if a.rul263_scale:
        typ_ = d.vid.astype(str).str.split("_").str[0]
        for spec_ in a.rul263_scale.split(","):
            t_, k_, *xd_ = spec_.split(":")
            sel_ = (s == "rul263_caw") & typ_.str.startswith(t_)
            if xd_ and xd_[0].startswith("+"):
                RXI_ = {"height": r"\b(?:height|tall)\b", "width": r"\b(?:width|wide|wingspan)\b", "length": r"\b(?:length|long)\b", "diam": r"\b(?:diameter|radius)\b"}
                qi_ = d.question.astype(str).str.lower()
                inc_ = pd.Series(False, index=d.index)
                for dm_ in xd_[0][1:].split("+"):
                    inc_ = inc_ | qi_.str.contains(RXI_[dm_], regex=True)
                sel_ = sel_ & inc_
            if xd_ and xd_[0].startswith("-"):
                RXX_ = {"height": r"\b(?:height|tall)\b", "width": r"\b(?:width|wide|wingspan)\b", "length": r"\b(?:length|long)\b", "diam": r"\b(?:diameter|radius)\b"}
                qx_ = d.question.astype(str).str.lower()
                for dm_ in xd_[0][1:].split("+"):
                    sel_ = sel_ & ~qx_.str.contains(RXX_[dm_], regex=True)
            v = v.copy(); v[sel_] = v[sel_] * float(k_)
            print('', file=sys.stderr)
    for spec_dg_ in (a.rul263_dimgm.split(";") if a.rul263_dimgm else []):
            import re as re_
            parts_ = spec_dg_.split(":"); t_, dims_ = parts_[0], parts_[1]
            gmin_ = float(parts_[2]) if len(parts_) > 2 else 0.0
            rmax_ = float(parts_[3]) if len(parts_) > 3 else 1e9
            typ_ = d.vid.astype(str).str.split("_").str[0]
            RXD_ = {"width": r"\b(?:width|wide|wingspan)\b", "length": r"\b(?:length|long)\b", "height": r"\b(?:height|tall)\b", "diam": r"\b(?:diameter|radius)\b"}
            qd_ = d.question.astype(str).str.lower()
            dsel_ = pd.Series(False, index=d.index)
            for dm_ in dims_.split("+"):
                dsel_ = dsel_ | qd_.str.contains(RXD_[dm_], regex=True)
            if "height" not in dims_:
                dsel_ = dsel_ & ~qd_.str.contains(RXD_["height"], regex=True)
            gg_ = M1.g.reindex(d.index)
            import consistency as C_
            gsi_ = gg_ * C_.parse_info(split, d.index).unit.map(lambda u: {"cm": 0.01, "mm": 0.001}.get(str(u), 1.0))
            sel_ = (s == "rul263_caw") & typ_.str.startswith(t_) & dsel_ & (gg_ > 0) & (gsi_ >= gmin_) & (np.abs(np.log10(gg_ / v)) <= np.log10(rmax_))
            if os.environ.get("QP_DIMGM_BOUND", "1") == "1" and "width" in dims_:
                import kinematic as K_
                K_.EXTRA_BOUNDS = K_.EXTRA2 = K_.EXTRA3 = True
                for i_ in sel_[sel_].index:
                    b_ = K_.class_bound(str(d.question[i_]), str(d.ground_truth_prior[i_]), "size")
                    if b_ is not None and not (b_[0] <= gg_[i_] * K_.q_si(str(d.question[i_])) <= b_[1]):
                        sel_[i_] = False
            v = v.copy(); v[sel_] = np.sqrt(v[sel_] * gg_[sel_]); s = s.copy(); s[sel_] = "rul263_dimgm"
            print('', file=sys.stderr)
    if a.probe:
        n_pr = {}
        v = v.copy()
        for spec_ in a.probe.split(";"):
            parts4_ = spec_.split(":")
            cat_, srcs_, kinds_, act_ = parts4_[:4]
            sel_ = d.cat == cat_
            if len(parts4_) > 4 and parts4_[4]:
                sel_ = sel_ & d.vid.astype(str).str.startswith(parts4_[4])
            if srcs_ != "any":
                sel_ = sel_ & s.isin(srcs_.split(","))
            if kinds_ != "any":
                ks_ = kinds_.split(",")
                kk_ = info.kind.isin([k for k in ks_ if k not in ("size", "motion")])
                if "size" in ks_:
                    kk_ = kk_ | (info.kind == "size")
                if "motion" in ks_:
                    kk_ = kk_ | info.kind.isin(["speed", "avgspeed", "accel", "disp", "path", "dist2", "hvel"])
                if "scenesize" in ks_:
                    sys.path.insert(0, f"{R}/geo")
                    from measure import SCENE as SCENE_
                    qsz_ = d.question.astype(str).str.lower()
                    scn_ = info.tphrase.astype(str).str.lower().str.contains(SCENE_, regex=True) | qsz_.str.contains(
                        r"\b(?:lane|road|street|tile|curb|kerb|pier|deck|floor|ceiling|wall|pool|court|field|track|gap|spacing|distance between|cabin|parking|lattice|frame)\b", regex=True)
                    kk_ = kk_ | ((info.kind == "size") & scn_)
                if "objsize" in ks_:
                    sys.path.insert(0, f"{R}/geo")
                    from measure import SCENE as SCENE_
                    qsz_ = d.question.astype(str).str.lower()
                    scn_ = info.tphrase.astype(str).str.lower().str.contains(SCENE_, regex=True) | qsz_.str.contains(
                        r"\b(?:lane|road|street|tile|curb|kerb|pier|deck|floor|ceiling|wall|pool|court|field|track|gap|spacing|distance between|cabin|parking|lattice|frame)\b", regex=True)
                    kk_ = kk_ | ((info.kind == "size") & ~scn_)
                if "other" in ks_:
                    kk_ = kk_ | ~info.kind.isin(["size", "speed", "avgspeed", "accel", "disp", "path", "dist2", "hvel"])
                sel_ = sel_ & kk_
            g_, c_ = M1.g.reindex(d.index), M1.cs.reindex(d.index)
            if act_ == "geo":
                sel_ = sel_ & g_.notna() & (g_ > 0); v[sel_] = g_[sel_]
            elif act_ == "caw":
                sel_ = sel_ & c_.notna() & (c_ > 0); v[sel_] = c_[sel_]
            elif act_ == "gm":
                sel_ = sel_ & g_.notna() & (g_ > 0) & c_.notna() & (c_ > 0); v[sel_] = np.sqrt(g_[sel_] * c_[sel_])
            elif act_.startswith("w"):
                w_ = float(act_[1:])
                sel_ = sel_ & g_.notna() & (g_ > 0) & c_.notna() & (c_ > 0); v[sel_] = g_[sel_] ** w_ * c_[sel_] ** (1 - w_)
            elif act_.startswith("x"):
                if a.probe_kin_exempt:
                    if "kin_ex_" not in dir():
                        import json as js2_
                        P2_ = {}
                        for ln_ in open(f"{R}/geo/out/parse_{split}_v2.jsonl", encoding="utf-8"):
                            x_ = js2_.loads(ln_); P2_[x_["id"]] = x_
                        SI2_ = {"m": 1, "cm": 0.01, "mm": 0.001, "m/s": 1, "cm/s": 0.01, "km/h": 1 / 3.6}
                        c_raw_ = M1.c.reindex(d.index)
                        kin_ex_ = pd.Series(False, index=d.index)
                        for i_ in d.index:
                            x_ = P2_.get(int(i_))
                            if not x_ or not x_.get("priors"):
                                continue
                            pr_ = x_["priors"][0]; q_ = x_.get("q") or {}
                            gv_ = pr_.get("value_si") if (pr_.get("gravity") or (pr_.get("kind") == "accel" and 9.0 <= float(pr_.get("value_si") or 0) <= 10.0)) else None
                            u_ = str(q_.get("unit") or ""); c0_ = c_raw_.get(i_)
                            if not gv_ or u_ not in SI2_ or c0_ is None or not np.isfinite(c0_) or c0_ <= 0:
                                continue
                            kv_ = None
                            if q_.get("kind") == "speed" and q_.get("t"):
                                kv_ = gv_ * float(q_["t"])
                            elif q_.get("kind") == "disp" and q_.get("t1") is not None and q_.get("t2") is not None:
                                kv_ = 0.5 * gv_ * (float(q_["t2"]) ** 2 - float(q_["t1"]) ** 2)
                            if kv_ and kv_ > 0 and abs(c0_ * SI2_[u_] / kv_ - 1) < 0.01:
                                kin_ex_[i_] = True
                    sel_ = sel_ & ~kin_ex_
                v[sel_] = v[sel_] * float(act_[1:])
            n_pr[spec_] = int(sel_.sum())
        print(f"[{split}] probe: {n_pr}", file=sys.stderr)
    if a.inet_geo:
        g_ = M1.g.reindex(d.index)
        cats_, srcs_ = (a.inet_geo.split(":", 1) if ":" in a.inet_geo else ("S2,D2,S3,D3", a.inet_geo))
        sel_ = (d.video_id.astype(str).str.startswith("internet") & s.isin(srcs_.split(",")) & d.cat.isin(cats_.split(","))
                & g_.notna() & (g_ > 0))
        v = v.copy(); v[sel_] = g_[sel_]; s = s.copy(); s[sel_] = "inet_geo"
        print('', file=sys.stderr)
    if a.depth_copy:
        import json as js_, re as re_
        P_ = {}
        for ln_ in open(f"{R}/geo/out/parse_{split}_v2.jsonl", encoding="utf-8"):
            x_ = js_.loads(ln_); P_[x_["id"]] = x_
        n_dc = 0
        v = v.copy(); s = s.copy()
        for i_ in d.index:
            x_ = P_.get(int(i_))
            if x_ is None:
                continue
            ql_ = str(x_["question"]).lower()
            if not re_.search(r"\bcamera\b", ql_) or not re_.search(r"\bdistance\b", ql_):
                continue
            mt_ = re_.search(r"\bat (\d+(?:\.\d+)?)\s*s\b", ql_)
            keys_ = set(x_.get("target_dkeys") or [])
            ent_ = [e for e in (x_.get("depth") or []) if e.get("key") in keys_ and e.get("t") is not None]
            if not mt_ or not ent_:
                continue
            T_ = float(mt_.group(1))
            ex_ = [e["d"] for e in ent_ if abs(float(e["t"]) - T_) < 1e-6]
            if ex_:
                val_ = float(ex_[0])
            else:
                pts_ = sorted({float(e["t"]): float(e["d"]) for e in ent_}.items())
                if len(pts_) < 2 or T_ < pts_[0][0] - 1.0 or T_ > pts_[-1][0] + 1.0:
                    continue
                (t0_, d0_), (t1_, d1_) = (pts_[0], pts_[1]) if T_ <= pts_[1][0] else (pts_[-2], pts_[-1])
                val_ = d0_ + (d1_ - d0_) * (T_ - t0_) / (t1_ - t0_)
            u_ = str((x_.get("q") or {}).get("unit") or "m")
            v[i_] = val_ / {"m": 1.0, "cm": 0.01, "mm": 0.001, "km": 1000.0}.get(u_, 1.0)
            s[i_] = "depth_copy" if ex_ else "depth_interp"; n_dc += 1
        print('', file=sys.stderr)
    if a.sruler_geo:
        ms_ = pd.read_csv(f"{R}/preds/{split}_{a.geo}_raw.csv").set_index("Unnamed: 0").merge_src.reindex(d.index)
        ks_ = a.sruler_geo.split(",")
        sel_ = (ms_ == "sruler") & M1.g.reindex(d.index).notna() & (M1.g.reindex(d.index) > 0)
        if "any" not in ks_:
            kk_ = info.kind.isin([k for k in ks_ if k not in ("size", "motion")])
            if "size" in ks_:
                kk_ = kk_ | (info.kind == "size")
            if "motion" in ks_:
                kk_ = kk_ | info.kind.isin(["speed", "avgspeed", "accel", "disp", "path", "dist2", "hvel"])
            sel_ = sel_ & kk_
        v = v.copy(); v[sel_] = M1.g.reindex(d.index)[sel_]; s = s.copy(); s[sel_] = "sruler_geo"
        print('', file=sys.stderr)
    if a.family_pool:
        import re as re3_
        fp_ = a.family_pool.split(":")
        minq3_ = int(fp_[0]); dims3_ = fp_[1].split("+") if len(fp_) > 1 else ["length", "width", "height", "diameter"]
        RX3_ = {"height": r"\b(?:height|tall)\b", "width": r"\b(?:width|wide)\b", "length": r"\b(?:length|long)\b", "diameter": r"\b(?:diameter|radius)\b"}
        sb3_ = d.vid.astype(str).str.replace("_segmented", "", regex=False)
        sim3_ = sb3_.str.startswith("simulation_")
        qn3_ = d.question.astype(str).str.strip().str.lower()
        qs3_ = d[sim3_].groupby(sb3_[sim3_]).question.apply(lambda z: set(z.astype(str).str.strip().str.lower()))
        vids3_ = list(qs3_.index)
        par3_ = {x: x for x in vids3_}
        def find3_(x):
            while par3_[x] != x:
                par3_[x] = par3_[par3_[x]]; x = par3_[x]
            return x
        for i3_ in range(len(vids3_)):
            for j3_ in range(i3_ + 1, len(vids3_)):
                if len(qs3_[vids3_[i3_]] & qs3_[vids3_[j3_]]) >= minq3_:
                    par3_[find3_(vids3_[i3_])] = find3_(vids3_[j3_])
        fam3_ = sb3_.map(lambda x: find3_(x) if x in par3_ else None)
        isz3_ = pd.Series(False, index=d.index)
        for dm3_ in dims3_:
            isz3_ = isz3_ | qn3_.str.contains(RX3_[dm3_], regex=True)
        sel3_ = sim3_ & isz3_ & (info.kind == "size") & fam3_.notna() & (v > 0)
        key3_ = fam3_.astype(str) + "||" + qn3_
        n_fp = 0
        v = v.copy()
        for k3_, g3_ in v[sel3_].groupby(key3_[sel3_]):
            if sb3_[g3_.index].nunique() < 2:
                continue
            med3_ = 10 ** float(np.median(np.log10(g3_.values)))
            v[g3_.index] = med3_; n_fp += len(g3_)
        print('', file=sys.stderr)
    if a.scene_prior:
        import re as re_
        minq_ = int(a.scene_prior)
        SI_ = {"m": 1.0, "meter": 1.0, "meters": 1.0, "cm": 1e-2, "mm": 1e-3, "km": 1e3}
        DIMS2_ = r"(diameter|length|width|height|radius|thickness|wingspan|size)"
        STOP2_ = {"the", "a", "an", "of", "in", "at", "is", "what", "meters", "meter", "cm", "m", "mm", "s", "total", "each", "one"}
        def nobj_(x):
            return " ".join(w for w in re_.sub(r"[^a-z ]", " ", x.lower()).split() if w not in STOP2_)
        def pitems_(p):
            out_ = []
            for line_ in str(p).split("\n"):
                m_ = re_.match(r"\s*(.+?)\s*[=~]\s*([-+]?\d*\.?\d+)\s*([a-z/\^0-9]+)?", line_.strip().lower())
                if not m_:
                    continue
                u_ = (m_.group(3) or "").strip()
                dm_ = re_.search(DIMS2_, m_.group(1))
                if not dm_ or u_ not in SI_:
                    continue
                out_.append((dm_.group(1), nobj_(re_.sub(DIMS2_, " ", m_.group(1))), float(m_.group(2)) * SI_[u_]))
            return out_
        EXT_ = set(x for x in a.scene_prior_ext.split(",") if x)
        def qitem_(q):
            pre_ = r"(?:(?:body|total|overall|full|whole) )?" if "dimpre" in EXT_ else ""
            dm_ = re_.search(r"what is the " + pre_ + DIMS2_ + r"(?: \([^)]*\))? of (?:the |a |each )?(.+?) in (meters|meter|m|cm|mm)\b", str(q).lower())
            return (dm_.group(1), nobj_(dm_.group(2)), dm_.group(3)) if dm_ else None
        sbase_ = d.vid.astype(str).str.replace("_segmented", "", regex=False)
        issim_ = sbase_.str.startswith("simulation_")
        qsets_ = d.groupby(sbase_).question.apply(lambda z: set(z.astype(str).str.strip().str.lower()))
        pri_ = {b: pitems_(g.ground_truth_prior.iloc[0]) for b, g in d[issim_].groupby(sbase_[issim_])}
        sb_ = [b for b in qsets_.index if b.startswith("simulation_")]
        link_ = {b: [c for c in sb_ if c != b and len(qsets_[b] & qsets_[c]) >= minq_] for b in sb_}
        if "trans" in EXT_:
            comp_ = {}
            for b0_ in sb_:
                if b0_ in comp_:
                    continue
                stack_, seen_ = [b0_], {b0_}
                while stack_:
                    x_ = stack_.pop()
                    for y_ in link_.get(x_, []):
                        if y_ not in seen_:
                            seen_.add(y_); stack_.append(y_)
                for x_ in seen_:
                    comp_[x_] = seen_
            link_ = {b: sorted(comp_[b] - {b}) for b in sb_}
            print('', file=sys.stderr)
        n_sp = 0
        v = v.copy()
        for i in d.index[issim_]:
            it_ = qitem_(d.question[i])
            if not it_:
                continue
            dim_, obj_, unit_ = it_
            b = sbase_[i]
            if any(dd == dim_ and oo == obj_ for dd, oo, _ in pri_.get(b, [])):
                continue
            vals_ = [val for c in link_.get(b, []) for dd, oo, val in pri_.get(c, []) if dd == dim_ and oo == obj_]
            if not vals_ or max(vals_) / min(vals_) > 1.02:
                continue
            v[i] = float(np.median(vals_)) / SI_[unit_]; s[i] = "scene_prior"; n_sp += 1
        def sprior_(p):
            m_ = re_.match(r"\s*(?:the )?(?:average )?speed of (?:the )?(.+?)\s*=\s*([\d.]+)\s*m/s\s*$", str(p).lower().strip())
            if not m_ or re_.search(r"\d\s*s\b|at t|from|between", str(p).lower().split("=")[0]):
                return None
            return nobj_(m_.group(1)), float(m_.group(2))
        spri_ = {b: sprior_(g.ground_truth_prior.iloc[0]) for b, g in d[issim_].groupby(sbase_[issim_])}
        n_ss = 0
        for i in d.index[issim_]:
            ql_ = str(d.question[i]).lower()
            if "average speed" not in ql_ or "m/s" not in ql_ or "m/s^2" in ql_:
                continue
            mo_ = re_.search(r"average speed of (?:the )?(.+?)(?: from| in m/s| between|\?|,)", ql_)
            if not mo_:
                continue
            o_ = nobj_(mo_.group(1))
            own_ = spri_.get(sbase_[i])
            if own_ and own_[0] == o_:
                continue
            vals_ = [spri_[c][1] for c in link_.get(sbase_[i], []) if spri_.get(c) and spri_[c][0] == o_]
            if not vals_ or max(vals_) / min(vals_) > 1.02:
                continue
            v[i] = float(np.median(vals_)); s[i] = "scene_prior"; n_ss += 1
        print('', file=sys.stderr)
    if a.var_ans_plain:
        from variant_keys import split_variant
        sv_ = d.vid.astype(str).map(split_variant)
        qn_ = d.question.astype(str).str.strip().str.lower()
        if a.var_norm:
            import re as re_
            def qnorm_(x):
                x = re_.sub(r"(\d)\s+s\b", r"\1s", x)
                x = re_.sub(r"(?<![\d.])(\d+)s\b", r"\1.0s", x)
                x = re_.sub(r"what is the initial (speed|velocity) of (?:the )?(.+?) in m/s\??$", r"what is the \1 of the \2 at 0.0s in m/s?", x)
                return re_.sub(r"\s+", " ", x).strip()
            qn_ = qn_.map(qnorm_)
        key_ = sv_.str[0] + "||" + qn_
        isp_ = sv_.str[1] == "plain"
        pv_ = pd.Series(v[isp_].values, index=key_[isp_].values); pv_ = pv_[~pv_.index.duplicated()]
        sel_ = sv_.str[1].isin(a.var_ans_plain.split(",")) & key_.isin(pv_.index)
        v = v.copy(); v[sel_] = pv_.reindex(key_[sel_].values).values
        s = s.where(~sel_, "var_ans_plain")
        print('', file=sys.stderr)
    if a.lab_pool:
        from variant_keys import split_variant, split_event
        mode_, *cap_ = a.lab_pool.split(":")
        if mode_ == "event":
            base_ = d.vid.astype(str).map(lambda z: split_event(z)[0])
        else:
            base_ = d.vid.astype(str).map(lambda z: split_variant(z)[0])
        key_ = base_ + "||" + d.question.astype(str).str.strip().str.lower()
        lab_ = d.vid.astype(str).str.startswith("captured_") & (v > 0)
        lv_ = np.log10(v.where(lab_))
        pooled_ = 10 ** lv_.groupby(key_).transform("mean")
        n_ = lv_.groupby(key_).transform("count")
        sel_ = lab_ & (n_ >= 2) & pooled_.notna()
        if cap_:
            spread_ = lv_.groupby(key_).transform("max") - lv_.groupby(key_).transform("min")
            sel_ = sel_ & (spread_ <= np.log10(float(cap_[0])))
        v = v.copy(); v[sel_] = pooled_[sel_]
        print('', file=sys.stderr)
        if a.lab_major > 0:
            n_mj = 0
            lv2_ = np.log10(v.where(lab_))
            for k_, g_ in lv2_.groupby(key_):
                g_ = g_.dropna()
                if len(g_) < 3 or (g_.max() - g_.min()) <= (np.log10(float(cap_[0])) if cap_ else 0):
                    continue
                best_ = None
                for c_ in g_.values:
                    mem_ = g_[(g_ >= c_ - 1e-12) & (g_ <= c_ + np.log10(a.lab_major) + 1e-12)]
                    if best_ is None or len(mem_) > len(best_):
                        best_ = mem_
                if best_ is not None and len(best_) >= 2 and len(best_) > len(g_) / 2:
                    v.loc[g_.index] = 10 ** float(best_.mean()); n_mj += len(g_)
            print('', file=sys.stderr)
    if a.phys_bounds_final:
        import kinematic as K_
        K_.EXTRA_BOUNDS = K_.EXTRA2 = K_.EXTRA3 = True
        n_pf = 0
        for i in d.index:
            b_ = K_.class_bound(str(d.question[i]), str(d.ground_truth_prior[i]), info.kind.get(i))
            if b_ is None:
                continue
            f_ = K_.q_si(str(d.question[i]))
            inside = lambda x: np.isfinite(x) and x > 0 and b_[0] <= x * f_ <= b_[1]
            if inside(v[i]):
                continue
            cand = [x for x in (M1.g.get(i, np.nan), M1.cs.get(i, np.nan)) if inside(x)]
            if len(cand) == 1:
                v = v.copy(); v[i] = cand[0]; s = s.copy(); s[i] = "phys_bound_final"; n_pf += 1
        print('', file=sys.stderr)
    if a.disp_bound:
        import json as js3_
        P3_ = {}
        for ln_ in open(f"{R}/geo/out/parse_{split}_v2.jsonl", encoding="utf-8"):
            x_ = js3_.loads(ln_); P3_[int(x_["id"])] = x_
        SI3_ = {"m": 1.0, "cm": 0.01, "mm": 0.001, "m/s": 1.0, "cm/s": 0.01, "km/h": 1 / 3.6}
        si_ = v * info.unit.map(SI3_)
        n_db = 0
        for i in d.index[info.kind.isin(["disp", "path"])]:
            q_ = (P3_.get(int(i)) or {}).get("q") or {}
            if q_.get("t1") is None or q_.get("t2") is None or not np.isfinite(si_.get(i, np.nan)):
                continue
            dt_ = float(q_["t2"]) - float(q_["t1"])
            sp_ = si_[(d.vid == d.vid[i]) & (info.tphrase == info.tphrase[i]) & info.kind.isin(["speed", "avgspeed"])].dropna()
            if dt_ <= 0 or not len(sp_):
                continue
            cap_ = float(sp_.max()) * dt_
            if si_[i] > float(a.disp_bound) * cap_:
                v = v.copy(); v[i] = cap_ / SI3_[info.unit[i]]; s = s.copy(); s[i] = "disp_bound"; n_db += 1
        print('', file=sys.stderr)
    if a.depth_bound:
        import json as js6_
        SI6_ = {"m": 1.0, "cm": 0.01, "mm": 0.001}
        n6_ = n6c_ = 0
        for ln_ in open(f"{R}/geo/out/parse_{split}_v2.jsonl", encoding="utf-8"):
            x_ = js6_.loads(ln_); q_ = x_.get("q") or {}; i = int(x_["id"])
            if q_.get("kind") != "dist2" or not x_.get("depth") or i not in v.index:
                continue
            k1_, k2_ = x_.get("target_dkeys") or [], x_.get("target2_dkeys") or []
            if len(k1_) != 1 or len(k2_) != 1 or k1_ == k2_:
                continue
            def dep6_(k, t=q_.get("t")):
                es = [e for e in x_["depth"] if e["key"] == k[0]]
                if t is not None:
                    es = [e for e in es if e["t"] is not None and abs(e["t"] - t) < 0.051] or [e for e in es if e["t"] is None]
                return es[0]["d"] if len(es) == 1 else None
            d1_, d2_ = dep6_(k1_), dep6_(k2_)
            if d1_ is None or d2_ is None:
                continue
            si6_ = SI6_.get(q_.get("unit") or "m", 1.0)
            b_ = abs(d1_ - d2_) / si6_
            cs_, g_ = M1.cs.get(i, np.nan), M1.g.get(i, np.nan)
            caw_flat_ = np.isfinite(cs_) and (cs_ < b_ * 1.05)
            if caw_flat_ and np.isfinite(g_) and g_ >= b_ and str(s[i]).replace("mo_guard:", "") in ("dis_caw_routed", "caw_nogeo", "caw_c1"):
                v = v.copy(); v[i] = g_; s = s.copy(); s[i] = "depth_bound_geo"; n6_ += 1
            if v[i] < b_:
                v = v.copy(); v[i] = b_; s = s.copy(); s[i] = "depth_bound_clamp"; n6c_ += 1
        print('', file=sys.stderr)
    if a.patch_json:
        import json as js7_
        for tag7_ in a.patch_json.split(","):
            p7_ = os.path.join(os.environ.get("QP_PATCH_DIR") or f"{R}/patches", f"ag_{tag7_}_patch_{split}.json")
            if not os.path.exists(p7_):
                if os.environ.get("QP_PATCH_STRICT", "0") == "1":
                    sys.exit('')
                print('', file=sys.stderr); continue
            P7_ = js7_.load(open(p7_)); n7_ = 0
            for k7_, e7_ in P7_.items():
                i = int(k7_)
                if i in v.index and e7_.get("value") is not None and np.isfinite(float(e7_["value"])) and float(e7_["value"]) > 0:
                    v = v.copy(); v[i] = float(e7_["value"]); s = s.copy(); s[i] = f"patch:{tag7_}"; n7_ += 1
            print('', file=sys.stderr)
        from variant_keys import split_variant as sv7f_
        import re as re7_
        sv7_ = d.vid.astype(str).map(sv7f_)
        def qn7_(x):
            x = re7_.sub(r"(\d)\s+s", r"s", str(x).strip().lower())
            x = re7_.sub(r"(?<![\d.])(\d+)s", r".0s", x)
            return re7_.sub(r"\s+", " ", x).strip()
        k7_ = sv7_.str[0] + "||" + d.question.map(qn7_)
        pat7_ = s.astype(str).str.startswith("patch:") & (sv7_.str[1] == "plain")
        pv7_ = pd.Series(v[pat7_].values, index=k7_[pat7_].values); pv7_ = pv7_[~pv7_.index.duplicated()]
        cp7_ = s.astype(str).str.replace("mo_guard:", "").eq("var_ans_plain") & k7_.isin(pv7_.index)
        if cp7_.any():
            v = v.copy(); v[cp7_] = pv7_.reindex(k7_[cp7_].values).values; s = s.where(~cp7_, "patch_copy")
        print('', file=sys.stderr)
    if a.std_court:
        import re as re5_
        STD5_ = {"pickleball": (6.10, 13.41), "badminton": (6.10, 13.40), "volleyball": (9.0, 18.0),
                 "tennis": (10.97, 23.77), "basketball": (15.0, 28.0)}
        SI5_ = {"m": 1.0, "meter": 1.0, "meters": 1.0, "cm": 0.01, "mm": 0.001, "ft": 0.3048, "feet": 0.3048}
        n_sc = 0
        for i in d.index:
            q5_ = str(d.question[i]).lower().replace("pickle ball", "pickleball").replace("basket ball", "basketball")
            p5_ = str(d.ground_truth_prior[i]).lower().replace("pickle ball", "pickleball").replace("basket ball", "basketball")
            mq_ = re5_.search(r"\b(length|width)\b.*\b(" + "|".join(STD5_) + r")\s*(?:ball\s*)?court\b", q5_)
            mp_ = re5_.search(r"\b(length|width)\b[^=]*\b(" + "|".join(STD5_) + r")\s*(?:ball\s*)?court\b[^=]*=\s*([\d.]+)\s*([a-z]+)", p5_)
            if not mq_ or not mp_ or mq_.group(2) != mp_.group(2) or mq_.group(1) == mp_.group(1) or mp_.group(4) not in SI5_:
                continue
            w5_, l5_ = STD5_[mq_.group(2)]
            pv_ = float(mp_.group(3)) * SI5_[mp_.group(4)]
            ref_ = w5_ if mp_.group(1) == "width" else l5_
            if abs(pv_ / ref_ - 1) > 0.03:
                continue
            ans_si_ = l5_ if mq_.group(1) == "length" else w5_
            u5_ = str(info.unit.get(i) or "m")
            v = v.copy(); v[i] = ans_si_ / SI5_.get(u5_, 1.0); s = s.copy(); s[i] = "std_court"; n_sc += 1
        print('', file=sys.stderr)
    if a.speed_series:
        import re as re4_
        K4_ = float(a.speed_series)
        tq_ = d.question.astype(str).str.extract(r"at (?:time )?(\d+(?:\.\d+)?)\s*s\b")[0].astype(float)
        sel4_ = d.vid.astype(str).str.startswith("simulation") & (info.kind == "speed") & tq_.notna()
        n_ss4 = 0
        for (vv_, oo_), g_ in d[sel4_].groupby([d.vid[sel4_], info.tphrase[sel4_]]):
            if len(g_) < 3:
                continue
            ix_ = list(tq_[g_.index].sort_values().index)
            for k_ in range(1, len(ix_) - 1):
                a_, b_, c_ = v[ix_[k_ - 1]], v[ix_[k_]], v[ix_[k_ + 1]]
                cw_ = M1.cs.get(ix_[k_], np.nan)
                if not (a_ > 0 and b_ > 0 and c_ > 0) or max(a_, c_) / min(a_, c_) > 1.5:
                    continue
                if min(b_ / a_, b_ / c_) > K4_ or max(b_ / a_, b_ / c_) < 1 / K4_:
                    if np.isfinite(cw_) and cw_ > 0 and max(cw_ / a_, a_ / cw_, cw_ / c_, c_ / cw_) <= 1.5:
                        v = v.copy(); v[ix_[k_]] = cw_; s = s.copy(); s[ix_[k_]] = "speed_series"; n_ss4 += 1
        print('', file=sys.stderr)
    return d, M1, v, s


def main():
    t, M, v, s = answers("test")
    o = t[[c for c in A.OUT_COLS if c != "parsed_value"]].copy()
    o["parsed_value"] = v.values
    o = o[A.OUT_COLS]
    o1 = o.rename(columns={"Unnamed: 0": "id"})
    o1["id"] = o1["id"] + 1
    o1.to_csv(f"{R}/submissions/{a.out}_id1.csv", index=False)
    det = t[["Unnamed: 0", "video_id", "question", "cat", "src"]].copy()
    det["answer"], det["answer_source"] = v.values, s.values
    for k in ["g", "c", "cs", "h", "mode", "kind", "w3", "tri_rej"]:
        det[k] = M[k].values
    det.to_csv(f"{R}/submissions/{a.out}_detail.csv", index=False)


if __name__ == "__main__":
    main()
