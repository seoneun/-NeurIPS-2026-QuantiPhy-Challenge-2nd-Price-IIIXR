"""d08adultband_mvc_rep (check_repair grpD, 2026-10-17): writer of ag_d08adultband_mvc_patch_{test,val}.json (no writer exists in the repo
or its git history; checked with git grep over 0b65f41, 2d80e24 and HEAD).
The base rule d08adultband is run first in the same shadow (previous TSV line) and writes cache/tmp/ag_d08adultband_patch_<split>.json.
This script keeps the base rule's rows whose OWN inference_type (dataset column, analyze_caw load_test / load_val) does not
start with "S", and prefixes why with "camsplit_mvc(inference_type=<it>): ". The filter is inferred: camsplit/make_camsplit.py
writes _mv by the same test on video_type, and every stored _mvc row (5 tags) has inference_type DS while the stored _stc
rows (c86inetgamf) have SD (checked: patches/ag_*_mvc_patch_*.json, ag_c86inetgamf_stc_patch_test.json).
usage: python analysis/v3/compliance/check_repair/d08adultband_mvc_rep.py"""
import json, os, sys
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
sys.path.insert(0, R + "/caw")
import analyze_caw as A
BASE_TAG = "d08adultband"

for split in ("test", "val"):
    d = A.load_val() if split == "val" else A.load_test()
    it = d.inference_type.astype(str).str.strip()
    src = f"{R}/cache/tmp/ag_{BASE_TAG}_patch_{split}.json"
    P = json.load(open(src))                 # must exist: written by the base rule line of this run
    out = {}
    for k, e in P.items():
        i = int(k)
        if i in d.index and not it.loc[i].startswith("S"):
            e2 = dict(e); e2["why"] = f"camsplit_mvc(inference_type={it.loc[i]}): " + str(e.get("why", ""))
            out[k] = e2
    json.dump(out, open(f"{R}/cache/tmp/ag_{BASE_TAG}_mvc_patch_{split}.json", "w"), indent=1)
    print(f"[{split}] {BASE_TAG} -> {BASE_TAG}_mvc: {len(P)} -> {len(out)} rows; dropped "
          f"{ {k: it.loc[int(k)] for k in P if k not in out} }; kept {sorted(int(k) for k in out)}")
