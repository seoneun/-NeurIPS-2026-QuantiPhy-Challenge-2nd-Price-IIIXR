"""rul328: '<part> <dim> of <owner>' size questions whose parse target is the owner, not the part.
Match the lowercased question with  what is the (\w+) (length|width|height|diameter) of (the )?(.+?) in
and require: group1 in PART lexicon, parse q.target == group 4 (the owner), group1 not in q.target.
Then the geo measures the whole owner (e.g. a person for 'shoe length of the left person') -> geo is invalid
for the row; it is routed through the standard no-geo path (CaW)."""
import re, json
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
PART = {"shoe", "foot", "hand", "arm", "leg", "head", "shoulder", "finger", "tail", "wing", "wheel", "tire",
        "door", "window", "handle", "ceiling"}
RX = re.compile(r"what is the (\w+) (length|width|height|diameter) of (the )?(.+?) in")


def parse(sp):
    P = {}
    for l in open(f"{R}/geo/out/parse_{sp}_v2.jsonl"):
        x = json.loads(l)
        P[int(x["id"]) if "id" in x else len(P)] = x
    return P


def match(question, prs):
    m = RX.search(str(question).lower())
    if not m:
        return None
    part, dim, owner = m.group(1), m.group(2), m.group(4).strip()
    if part not in PART:
        return None
    tgt = str(((prs or {}).get("q") or {}).get("target") or "").strip().lower()
    if tgt != owner or part in tgt:
        return None
    return {"part": part, "dim": dim, "owner": owner, "target": tgt}
