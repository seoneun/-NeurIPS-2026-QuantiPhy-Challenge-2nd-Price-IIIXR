"""rul269: Track-B clean derivation (main session q604.sh, 10-05 04:17).

Rule (general, no hand-chosen row/video IDs):
  The `disc` rule re-measures the prior ruler of a video (a disc-shaped prior seen as an
  ellipse: the old px diameter is replaced by the ellipse long axis), so every geo value
  measured on that ruler scales by f = geo_new / geo_old.  `rul303` rows keep the plain geo
  value, which was measured on the OLD ruler.  For every rul303 row whose video also has
  disc-patched rows, multiply the rul303 value by that video's disc factor f.

  f is taken from the lowest-id disc row of the SAME video, preferring rows the disc rule
  measured itself (not 'copy of row N' var_ans_plain copies, which carry another video's
  ruler).  All disc rows of one video share the same px-diameter ratio, so the choice only
  affects rounding in the 4-significant-figure `why` text.

The original derive_discfix.py hard-coded dz['688'] (simulation_0224) as the factor source
and applied it to rul303 rows of ANY disc-patched video.  On the current inputs the general
rule selects the same single row (test 690, simulation_0224) with an identical value
(688 is the lowest-id non-copy disc row of simulation_0224).  Both splits are processed.
"""
import json, re, os, sys
import pandas as pd

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
os.chdir(R)
RAW = "preds/{s}_geo_v4ntrj5sdGsrngsr_d_vw_raw.csv"
GEO_RE = re.compile(r"geo ([\d.eE+-]+)->([\d.eE+-]+)")
COPY_RE = re.compile(r"copy of row \d+")


def disc_factor(why):
    m = GEO_RE.search(why)
    if not m:
        return None
    a, b = float(m.group(1)), float(m.group(2))
    return b / a if a > 0 else None


def run(split):
    dz = json.load(open(f"cache/tmp/ag_disc_patch_{split}.json"))
    ia = json.load(open(f"cache/tmp/ag_impacc_patch_{split}.json"))
    vid = pd.read_csv(RAW.format(s=split), usecols=["Unnamed: 0", "video_id"]).set_index("Unnamed: 0").video_id
    # per-video disc factor: lowest-id non-copy disc row with a parsable geo ratio (fallback: any)
    fac = {}
    for prefer_own in (True, False):
        for k in sorted(dz, key=int):
            v = vid[int(k)]
            if v in fac or (prefer_own and COPY_RE.search(dz[k]["why"])):
                continue
            f = disc_factor(dz[k]["why"])
            if f is not None:
                fac[v] = (f, int(k))
    fix = {}
    for k in sorted(ia, key=int):
        v = vid[int(k)]
        if v not in fac:
            continue
        f, src = fac[v]
        fix[k] = {"value": ia[k]["value"] * f,
                  "why": f"rul269: rul303 geo on the disc ruler (x{f:.4f}, {v}; factor from disc row {src} of the same video)"}
    print(split, "disc videos/factors:", {v: (round(f, 5), s) for v, (f, s) in fac.items()})
    print(split, "rows:", {k: (round(ia[k]["value"], 4), round(x["value"], 4)) for k, x in fix.items()})
    json.dump(fix, open(f"cache/tmp/ag_discfix_patch_{split}.json", "w"), indent=0)
    return fix


if __name__ == "__main__":
    for s_ in ("test", "val"):
        run(s_)
