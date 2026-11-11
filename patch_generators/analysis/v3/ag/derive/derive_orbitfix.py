"""rul324 (main session q610.sh, 10-05 04:47): orbit-diameter rows take the 2v/omega geometry alone instead of the fused value."""
import json, os
os.chdir(os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../..")))
for s in ("test", "val"):
    full = json.load(open(f"cache/tmp/ag_orbit_full_{s}.json"))
    pat = json.load(open(f"cache/tmp/ag_orbit_patch_{s}.json"))
    fix = {}
    for k, v in full.items():
        if v.get("kind") == "orbitdiam" and k in pat and v.get("geo"):
            fix[k] = {"value": float(v["geo"]), "why": f"rul324: orbit diameter = 2v/omega geometry alone (was fused {pat[k]['value']:.3f})"}
    json.dump(fix, open(f"cache/tmp/ag_orbitfix_patch_{s}.json", "w"), indent=0)
    print(s, {k: (round(pat[k]["value"], 3), round(v["value"], 3)) for k, v in fix.items()})
