import os, sys, json, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "tools"))
from qp_common import load_split, OUT_COLS, R
THETA = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def mra(pred, gt):
    pred = abs(float(pred)) if np.isfinite(pred) else np.nan
    if not np.isfinite(pred) or pred == 0:
        return 0.0
    e = abs(pred - float(gt)) / float(gt)
    return float(np.mean([e < 1 - t for t in THETA]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--tag", default="geo_v2")
    ap.add_argument("--meas_tag", default="")
    ap.add_argument("--maxdec", type=float, default=2.0)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    df = load_split(a.split).reset_index(drop=True)
    meas = pd.read_csv(f"{R}/geo/out/meas_{a.split}_v2{a.meas_tag}.csv").set_index("id")
    vlm = pd.read_csv(f"{R}/preds/{a.split}_qwen3vl8b_video.csv").set_index("Unnamed: 0").parsed_value
    h1 = pd.read_csv(f"{R}/preds/{a.split}_h1_prior_copy.csv").set_index("Unnamed: 0").parsed_value
    ids = df["Unnamed: 0"].astype(int)
    m_ok0 = ids.map(meas.ok).fillna(False).astype(bool).values
    m_val = ids.map(meas.value).values.astype(float)
    v_val = ids.map(vlm).values.astype(float)
    h_val = np.abs(ids.map(h1).values.astype(float))
    v_ok = np.isfinite(v_val) & (v_val > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        m_dec = np.abs(np.log10(m_val / h_val))
        v_dec = np.abs(np.log10(v_val / h_val))
    m_ok = m_ok0.copy()
    if a.maxdec > 0:
        m_rej = m_ok & ~(m_dec <= a.maxdec)
        v_rej = v_ok & ~(v_dec <= a.maxdec)
        m_ok = m_ok & ~m_rej
        v_ok = v_ok & ~v_rej
    final = np.where(m_ok, m_val, np.where(v_ok, v_val, h_val))
    src = np.where(m_ok, "meas", np.where(v_ok, "vlm", "h1"))
    out = df.copy()
    out["parsed_value"] = final
    p0 = f"{R}/preds/{a.split}_{a.tag}.csv"
    out[OUT_COLS].to_csv(p0, index=False)
    raw = out.copy()
    raw["answer_source"] = src
    raw["meas_valid"] = m_ok
    raw["meas_value"] = m_val; raw["vlm_value"] = v_val; raw["h1_value"] = h_val
    raw["geo_value"] = np.where(m_ok, m_val, np.nan)
    why = ids.map(meas.why).fillna("").values.astype(object)
    why = np.where(m_ok0 & ~m_ok, "2-decade sanity rule", why)
    raw["meas_why"] = why
    raw["meas_mode"] = ids.map(meas["mode"]).values
    raw["meas_kind"] = ids.map(meas.kind).values
    for c in ("grav", "gap", "tracks", "f", "mpp", "owlvote", "nseg", "colswap"):
        if c in meas.columns:
            raw["meas_" + c] = ids.map(meas[c]).values
    raw.to_csv(p0.replace(".csv", "_raw.csv"), index=False)
    if not a.quiet:
        print("wrote", p0, "| sources", pd.Series(src).value_counts().to_dict(), "| sanity-rejected meas", int((m_ok0 & ~m_ok).sum()))


if __name__ == "__main__":
    main()
