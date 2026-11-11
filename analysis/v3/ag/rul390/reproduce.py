#!/usr/bin/env python
import os, sys, subprocess
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "../../../.."))
T = [sys.executable, f"{HERE}/regrid.py"]
JOBS = [
    (2841, "test", "captured_0054a", "pink ball", "0,12,24,30", "4", "0:0,24:0", "pink ball", "c54apink"),
]
force = "--force" in sys.argv
FZ = os.environ.get("R5_FROZEN", "1") != "0" and not force
ENV = {**os.environ, "R5_TOOL_TRACKS": f"{HERE}/frozen", "R5_TABLES": f"{R}/analysis/v3/ag/r5_common/frozen"} if FZ else None
for i, s, v, ph, fr, tk, pk, ob, tag in JOBS:
    tf = f"{R}/cache/tmp/r5_tool/tracks_{s}_{tag}/{v}.json"
    if not FZ and (force or not os.path.exists(tf)):
        subprocess.run(T + ["detect", "--split", s, "--video", v, "--phrase", ph, "--frames", fr, "--topk", tk], check=True, stdout=subprocess.DEVNULL)
        subprocess.run(T + ["track", "--split", s, "--video", v, "--phrase", ph, "--pick", pk, "--obj", ob, "--out", tag], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(T + ["measure", "--split", s, "--video", v, "--tracks", tag, "--rows", str(i)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=ENV)
    D = pd.read_csv(f"{R}/cache/tmp/r5_tool/measure_{s}_{v}_{tag}.csv").set_index("id0")
    print(f"{i}, {D.chain_new[i]:.6g}")
