"""rul204: copy of c86inetgam for simulation S2 videos with a size prior, labels live in the prior's world (gamma x pixels).
When geo/CaW in a video is one consistent world factor (>=3 rows, |median log ratio| >= log 1.4, log-IQR <= 0.35),
an agree_* hedge row whose own 2d_ratio geo matches that factor takes geo instead of the geometric-mean hedge.
Segmented var_ans_plain copies of a patched plain row follow. Labels are never read here."""
import sys, json, os
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
SUB = os.environ.get("C95_SUB", "sub_xt_Nzz29")
RAW = "geo_v4ntrj5sQ2srngsr_d_vw"
TAG = "rul204"
HEDGE = ("agree_wgm", "agree_gm")


def load(split):
    P = {}
    for ln in open(f"{R}/geo/out/parse_{split}_v2.jsonl", encoding="utf-8"):
        x = json.loads(ln); P[int(x["id"])] = x
    raw = pd.read_csv(f"{R}/preds/{split}_{RAW}_raw.csv"); raw = raw.set_index(raw.columns[0])
    fn = f"{R}/submissions/{SUB}_detail.csv" if split == "test" else f"{R}/submissions/{SUB}_valdetail.csv"
    D = pd.read_csv(fn).set_index("Unnamed: 0")
    D["question"] = [P[i]["question"] for i in D.index]
    D["qn"] = D.question.astype(str).str.strip().str.lower()
    D["qkind"] = [(P[i].get("q") or {}).get("kind") for i in D.index]
    D["mm"] = raw["meas_mode"].reindex(D.index).astype(str)

    def pk(i):
        ps = P[i].get("priors") or []
        if not ps:
            return "none"
        p = ps[0]
        if p.get("gravity") or (p.get("kind") == "accel" and 9.0 <= float(p.get("value_si") or 0) <= 10.0):
            return "gravity"
        return p.get("kind") or "other"
    D["pk"] = [pk(i) for i in D.index]
    D["src0"] = D.answer_source.astype(str)
    return D


def group_stats(z):
    ok = z.g.notna() & z.cs.notna() & (z.g > 0) & (z.cs > 0) & (z.qkind != "accel")
    r = np.log(z.g[ok] / z.cs[ok])
    r = r[(r >= np.log(1 / 3)) & (r <= np.log(3))]
    if len(r) < 3:
        return None
    med = float(np.median(r)); iqr = float(np.percentile(r, 75) - np.percentile(r, 25))
    return med, iqr, len(r)


def run(split):
    D = load(split)
    out, glog = {}, []
    inet = D[D.video_id.astype(str).str.startswith("simulation") & (D.cat == "S2")]
    for vid, z in inet.groupby("video_id"):
        if z.pk.iloc[0] != "size":
            continue
        st = group_stats(z)
        if st is None:
            continue
        med, iqr, n = st
        glog.append((vid, round(np.exp(med), 3), round(iqr, 3), n))
        if abs(med) < np.log(1.4) or iqr > 0.35:
            continue
        for i, r in z.iterrows():
            if r.src0 not in HEDGE or r.mm != "2d_ratio" or r.qkind == "accel":
                continue
            if not (np.isfinite(r.g) and np.isfinite(r.cs) and r.g > 0 and r.cs > 0):
                continue
            lr = np.log(r.g / r.cs)
            if abs(lr - med) > np.log(1.35) or np.sign(lr) != np.sign(med) or abs(lr) < np.log(1.2):
                continue
            out[int(i)] = {"value": float(r.g), "why": f"{TAG}: sim S2 prior-world factor {np.exp(med):.2f}x (n={n}, iqr={iqr:.2f}); row g/cs={np.exp(lr):.2f}; hedge {r.answer:.4g} -> geo",
                           "_old": float(r.answer), "_vid": vid, "_qn": r.qn}
    # segmented copies
    for i, e in list(out.items()):
        if e["_vid"].endswith("_segmented"):
            continue
        sv = e["_vid"] + "_segmented"
        c = D[(D.video_id == sv) & (D.src0 == "var_ans_plain") & (D.qn == e["_qn"])]
        for j, r in c.iterrows():
            if int(j) in out:
                continue
            if abs(r.answer - e["_old"]) <= 1e-6 * max(1.0, abs(e["_old"])):
                out[int(j)] = {"value": e["value"], "why": f"{TAG}: segmented copy of plain row {i}", "_old": float(r.answer), "_vid": sv, "_qn": r.qn}
    return D, out, glog


if __name__ == "__main__":
    for sp in ("test", "val"):
        D, out, glog = run(sp)
        print(f"== {sp}: groups {glog}")
        for i, e in sorted(out.items()):
            print(sp, i, D.loc[i, "video_id"], D.loc[i, "cat"], D.loc[i, "question"][:60], f"{e['_old']:.4g} -> {e['value']:.4g}", D.loc[i, "src0"])
        print(sp, "by cat:", D.loc[list(out), "cat"].value_counts().to_dict() if out else {})
        json.dump({str(k): {"value": v["value"], "why": v["why"]} for k, v in out.items()},
                  open(f"{R}/cache/tmp/ag_{TAG}_patch_{sp}.json", "w"), indent=1)
