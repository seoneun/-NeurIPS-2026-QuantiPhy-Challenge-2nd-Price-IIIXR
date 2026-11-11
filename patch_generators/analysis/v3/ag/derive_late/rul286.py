"""rul286: fallblob without width rows whose width extent collapsed onto the height axis.

Verifier filter (cycle 19). A width row is dropped when its measured ext_width is
within 10% of the ext_height that the same rule measured for the same video and
target phrase. Without the box-aspect clause this is the conservative superset:
it only drops rows, and dropping keeps the current answer.
Inputs: the fallblob patch, the parse, the test questions. No labels.
"""
import json, re
import pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
T = pd.read_csv(f"{R}/submissions/sub_xt_Nzm_detail.csv").set_index("Unnamed: 0")
parse = {}
for l in open(f"{R}/geo/out/parse_test_v2.jsonl"):
    j = json.loads(l); parse[int(j.get("id", j.get("row", -1)))] = j
for split in ("test", "val"):
    src = json.load(open(f"{R}/cache/tmp/ag_fallblob_patch_{split}.json")); out = dict(src)
    if split == "test":
        def ext(k, dim):
            m = re.search(rf"ext_{dim} ([0-9.]+) px", src[k]["why"]); return float(m.group(1)) if m else None
        def tgt(k):
            p = parse.get(int(k), {}); q = p.get("q", {}) if isinstance(p, dict) else {}
            return (T.loc[int(k), "video_id"], str(q.get("target") or "").lower().strip())
        for k in list(src):
            w = ext(k, "width")
            if w is None: continue
            for k2 in src:
                h = ext(k2, "height")
                if k2 != k and h and tgt(k2) == tgt(k) and abs(w / h - 1) <= 0.10:
                    out.pop(k, None); print("drop", k, "width", w, "vs height", h, "of", k2, tgt(k))
    json.dump(out, open(f"{R}/cache/tmp/ag_fallblob2_patch_{split}.json", "w"), indent=1)
    print(split, sorted(out))
