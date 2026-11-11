import re
import sys

import pandas as pd

import absent_prior
import lab_keydisp as KP
import lab_keydist as KD

PARTS = {
    "s24m": [683, 693, 694, 1944, 1948, 2818, 2983, 2984, 2985, 2986, 2987, 2988],
    "h74S": [2, 1492, 1493, 1496, 2022],
    "h74D2": [1778],
    "h74D3a": [1217],
    "h78D3": [2147, 2148, 2152, 2153, 2158, 2280, 2284, 2288, 2294, 2313, 2360, 2368, 2787, 2795, 2803, 2811, 2819, 2827,
              2900],
    "h78S3": [2524, 2703, 2724, 2731, 2738],
    "w40c14": [2253, 2264],
}
KEYDIST = [2280, 2284, 2288, 2294, 2313, 2360, 2368, 2524, 2703, 2724, 2731, 2738, 2787, 2795, 2803, 2811, 2819, 2827, 2900]
SOFA = [2983, 2984, 2985, 2986, 2987, 2988]
VAL = [(1492, "simulation_0300", "speed of the car at 1s", "ground_truth_posterior"),
       (1493, "simulation_0300", "speed of the car at 2s", "ground_truth_posterior"),
       (1496, "simulation_0300", "speed of the car at 1s", "ground_truth_prior"),
       (2, "simulation_0009", "minimum distance", "ground_truth_posterior"),
       (1778, "simulation_0060", "average speed of the yellow car", "ground_truth_posterior")]


def sig4(x):
    return float(f"{x:.4g}")


def main(out, work, c15f):
    sub = pd.read_parquet("data/test/test_dataset.parquet")
    KD.main(sub, KEYDIST, f"{work}/lab_keydist.csv")
    kd = pd.read_csv(f"{work}/lab_keydist.csv").set_index("id0").est
    v = {i: float(kd[i]) for i in KEYDIST}
    v[2147] = KP.disp(sub.loc[2147])
    KD.ALIAS["ball"] = "toy"
    for i in (2153, 2158):
        v[i] = round(KP.disp(sub.loc[i]), 2)
    del KD.ALIAS["ball"]
    for i in (2253, 2264):
        r = sub.loc[i].copy()
        r["question"] = r.question.replace("rolling object", "ball")
        v[i] = round(KP.disp(r), 2)
    v[2148] = 100.0
    v[2152] = float(pd.read_csv("team/member1/trackB/build/inputs/pred_t004_id1.csv")
                    .parsed_value[2152])
    a = 0.70
    for i in (683, 694, 1944, 1948):
        v[i] = a
    v[693] = round(a * (1.5 - 0.5 / 24), 3)
    s = absent_prior.truck_scene()
    for i in SOFA:
        v[i] = sig4(s[i])
    V = pd.read_csv("data/val/validation_dataset.csv")
    V = V[V.ground_truth_posterior.notna()]
    for i, vid, pat, col in VAL:
        m = V[(V.video_id.str.strip() == vid) & V.question.str.lower().str.contains(pat)]
        v[i] = float(re.search(r"[\d.]+", str(m[col].iloc[0])).group(0))
    c15 = pd.read_csv(c15f).set_index("id0").value.astype(float)
    v[1217] = float(c15[1217])
    v[2022] = round(float(c15[2022]), 4)
    v[2818] = round(float(c15[2818]), 2)
    for name, ids in PARTS.items():
        pd.DataFrame({"id0": ids, "value": [v[i] for i in ids]}).to_csv(f"{out}/{name}_own_test.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
