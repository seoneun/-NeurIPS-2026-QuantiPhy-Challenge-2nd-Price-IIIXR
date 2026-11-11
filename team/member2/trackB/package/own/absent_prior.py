import json

import cv2
import numpy as np

CM_PER_PX = (6.5 / 28 + 19.8 / 90) / 2


def truck_scene():
    v = "internet_0026_segmented"
    d = json.load(open(f"geo/out/tracks_test_v3ntmv/{v}.json"))
    fps, k = d["fps"], 1280 / d["W"]
    obj = {x["phrase"]: x.get("obj") for x in d["queries"]}

    def boxes(name):
        o = d["objs"][str(obj[name])]
        return np.array([(d["fidx"][j] / fps, *[c * k for c in x["st"]["box"]], *[c * k for c in x["st"]["c"]])
                         for j, x in enumerate(o) if isinstance(x, dict) and x.get("st")])

    baby, man, truck = boxes("baby"), boxes("man"), boxes("toy truck")
    cap = cv2.VideoCapture(f"data/test/{v}.mp4")
    ext = []
    for f in range(0, 36, 5):
        cap.set(1, f)
        ok, im = cap.read()
        x0, y0, x1, y1 = truck[np.argmin(abs(truck[:, 0] - f / fps)), 1:5].astype(int)
        m = (255 - cv2.cvtColor(im[max(0, y0 - 5):y1 + 5, max(0, x0 - 5):x1 + 5], cv2.COLOR_BGR2GRAY)) > 25
        ys, xs = np.where(m)
        ext.append((np.ptp(xs), np.ptp(ys)))
    s = CM_PER_PX
    t, c = truck[:, 0], truck[:, 5:7]
    w = abs(t - 0.5) <= 0.2
    v05 = np.hypot(np.polyfit(t[w], c[w, 0], 1)[0], np.polyfit(t[w], c[w, 1], 1)[0]) * s
    vavg = np.linalg.norm(c[-1] - c[0]) / (t[-1] - t[0]) * s
    n = min(len(baby), len(man))
    dist = np.median(np.linalg.norm(baby[:n, 5:7] - man[:n, 5:7], axis=1)) * s / 100
    return {2983: np.median(baby[:, 4] - baby[:, 2]) * s, 2984: dist, 2985: np.median([e[0] for e in ext]) * s,
            2986: np.median([e[1] for e in ext]) * s, 2987: v05, 2988: vavg}
