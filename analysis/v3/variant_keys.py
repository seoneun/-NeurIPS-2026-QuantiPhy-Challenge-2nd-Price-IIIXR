import re

RX = re.compile(r"^(captured_\d+[a-z]?)(s|x)$", re.I)


def split_variant(v):
    v = str(v).strip()
    if v.endswith("_segmented"):
        return v[: -len("_segmented")], "_segmented"
    m = RX.match(v)
    if m:
        return m.group(1).lower(), m.group(2).lower()
    return v.lower(), "plain"


RX_TAKE = re.compile(r"^(captured_\d+)([ab])$", re.I)


def split_event(v):
    b, var = split_variant(v)
    m = RX_TAKE.match(b)
    if m:
        return m.group(1).lower(), m.group(2).lower(), var
    return b, "", var


_EVMAP = None


def event_map():
    global _EVMAP
    if _EVMAP is not None:
        return _EVMAP
    import pandas as pd
    import os
    t = pd.read_csv(os.path.join(os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")), "caw/inputs/test_input.csv"))
    t["v"] = t.video_id.astype(str).str.strip()
    t["base"] = t.v.map(lambda x: split_variant(x)[0])
    t["pk"] = t.ground_truth_prior.astype(str).str.replace(r"\s+", "", regex=True).str.lower()
    t["q"] = t.question.astype(str).str.strip().str.lower()
    pl = t[t.v == t.base]
    sig = pl.groupby("base").agg(pk=("pk", "first"), qs=("q", lambda s: "||".join(sorted(set(s)))))
    m = {}
    for b in t.base.unique():
        e = split_event(b)
        m[b] = e[0]
    gen = sig.pk.str.contains(r"gravit|acc=9\.8|acceleration=9\.8")
    for (pk, qs), x in sig[~gen].groupby(["pk", "qs"]):
        if len(x) > 1 and all(str(b).startswith("simulation") for b in x.index):
            eid = "simevent_" + sorted(x.index)[0]
            for b in x.index:
                m[b] = eid
    _EVMAP = m
    return m


def event_of(v):
    b, var = split_variant(v)
    return event_map().get(b, split_event(v)[0])
