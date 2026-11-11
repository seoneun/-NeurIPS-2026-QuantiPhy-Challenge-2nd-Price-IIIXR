import os
import re

import numpy as np
import pandas as pd

R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
C = f"{R}/caw"
os.makedirs(f"{C}/inputs", exist_ok=True)

src_dir = f"{R}/data/val/validation_videos"
vd = f"{C}/val_videos"
os.makedirs(vd, exist_ok=True)
for f in sorted(os.listdir(src_dir)):
    if f.endswith(".mp4"):
        dst = os.path.join(vd, f.strip())
        if not os.path.lexists(dst):
            os.symlink(os.path.join(src_dir, f), dst)
val = pd.read_csv(f"{R}/data/val/validation_dataset.csv")
val = val[[c for c in val.columns if not c.startswith("Unnamed") or c == "Unnamed: 0"]]
ids = val.video_id.astype(str).str.strip()
miss = [v for v in ids.unique() if not os.path.isfile(f"{vd}/{v}.mp4")]
print('')

t = pd.read_parquet(f"{R}/data/test/test_dataset.parquet").reset_index(drop=True)
out = pd.DataFrame({"": np.arange(len(t)), "video_id": t.video_id, "video_source": t.video_source,
                    "video_type": t.video_type, "fps": t.fps, "inference_type": t.inference_type,
                    "question": t.question, "ground_truth_prior": t.prior, "depth_info": t.depth_info,
                    "ground_truth_posterior": ""})
out.to_csv(f"{C}/inputs/test_input.csv", index=False)
miss = [v for v in t.video_id.unique() if not os.path.isfile(f"{R}/data/test_480p/{v}.mp4")]
print('')
chk = pd.read_csv(f"{R}/preds/test_geo_v1.csv")
assert (chk["Unnamed: 0"].values == out[""].values).all() and (chk.video_id.values == out.video_id.values).all() \
    and (chk.question.values == out.question.values).all()
print('')


def scale_prior(p, k):
    s = str(p)
    cut = max(s.rfind("="), s.rfind("~"))
    head, tail = s[:cut + 1], s[cut + 1:]
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", tail)
    assert m
    v = float(m.group(0)) * k
    return head + tail[:m.start()] + f"{v:.6g}" + tail[m.end():]


v2 = val[(val.video_type.str[1] == "2") & ~val.ground_truth_prior.str.contains("gravit", case=False)].copy()
v2 = v2.sort_values("Unnamed: 0").groupby("video_id", sort=False).head(4)
ctl = v2.copy()
x2 = v2.copy()
x2["Unnamed: 0"] = x2["Unnamed: 0"] + 100000
x2["ground_truth_prior"] = x2.ground_truth_prior.map(lambda p: scale_prior(p, 2))
cf = pd.concat([ctl, x2])
cf["ground_truth_posterior"] = ""
cf = cf.rename(columns={"Unnamed: 0": ""})
cf.to_csv(f"{C}/inputs/val_cf_input.csv", index=False)
print('')
for a, b in sorted(set(zip(ctl.ground_truth_prior, x2.ground_truth_prior))):
    print(f"   {a!r} -> {b!r}")
