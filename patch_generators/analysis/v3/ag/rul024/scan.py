"""rul024 scan: rows whose question has a time range 'between A[s] and B[s]' (unit optional on either end) and whose
parse missed it (q.t1 null) or kept the range inside the target phrase."""
import json, re, sys
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
RNG = re.compile(r"between\s+(\d+(?:\.\d+)?)\s*s?\s+and\s+(\d+(?:\.\d+)?)\s*s?\b")


def gate(r):
    ql = r["question"].lower()
    m = RNG.search(ql)
    if not m:
        return None, None
    q = r["q"]
    tps = [q.get("target") or "", (r.get("target") or {}).get("phrase") or "", (r.get("target") or {}).get("raw") or ""]
    bad = q.get("t1") is None or any("between" in (t or "").lower() for t in tps)
    return m, bad


if __name__ == "__main__":
    for sp in ("test", "val"):
        n = 0; hits = []
        for l in open(f"{R}/geo/out/parse_{sp}_v2.jsonl"):
            r = json.loads(l)
            m, bad = gate(r)
            if m is None:
                continue
            n += 1
            t1, t2 = float(m.group(1)), float(m.group(2))
            q = r["q"]
            agree = q.get("t1") is not None and abs(q["t1"] - t1) < 1e-9 and abs(q["t2"] - t2) < 1e-9
            if bad or not agree:
                hits.append((r["id"], r["video_id"], r["cat"], q.get("kind"), q.get("t1"), q.get("t2"), t1, t2, q.get("target"), (r.get("target") or {}).get("phrase"), r["question"]))
        print(f"[{sp}] range-regex rows {n}; gated or t1/t2 disagreeing: {len(hits)}")
        for h in hits:
            print("  ", h)
