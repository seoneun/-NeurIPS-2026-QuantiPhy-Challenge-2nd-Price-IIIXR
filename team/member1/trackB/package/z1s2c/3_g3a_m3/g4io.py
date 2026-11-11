import json
import pandas as pd



def load_patch(p):
    return {int(k): v for k, v in json.load(open(p)).items()}




def write_lines(base, P, out):
    txt = open(base).read().splitlines()
    recs, cur = [], None
    for line in txt[1:]:
        cur = line if cur is None else cur + chr(10) + line
        if cur.count('"') % 2 == 0:
            recs.append(cur); cur = None
    newl, n = [txt[0]], 0
    for line in recs:
        rid = int(line.split(",", 1)[0])
        if rid - 1 in P:
            head, val = line.rsplit(",", 1)
            line = head + "," + repr(float(P[rid - 1]["value"])); n += 1
        newl.append(line)
    open(out, "w").write("\n".join(newl) + "\n")


def read_float(p):
    return pd.read_csv(p, float_precision="round_trip")


def write_float(S0, P, out, check_before=True):
    S = S0.copy()
    for i, v in sorted(P.items()):
        m = S.id == i + 1
        if isinstance(v, dict):
            if check_before:
                pass
            v = v["value"]
        S.loc[m, "parsed_value"] = float(v)
    S.to_csv(out, index=False, lineterminator="\n")
    S2 = read_float(out)
    for c in S0.columns:
        if c != "parsed_value":
            pass
    return S2


def write_str(base, P, out):
    b = pd.read_csv(base, float_precision="round_trip", keep_default_na=False, dtype=str)
    c = b.copy()
    for k, v in P.items():
        m = c.id.astype(int) == int(k) + 1
        c.loc[m, "parsed_value"] = repr(float(v["value"]))
    c.to_csv(out, index=False, lineterminator="\n")


def cat_of(df):
    return [str(t)[0] + ("3" if isinstance(d, str) and d.strip() else "2") for t, d in zip(df.inference_type, df.depth_info)]
