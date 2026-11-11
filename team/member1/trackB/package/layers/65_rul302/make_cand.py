import sys, os, re, json, csv, math
import numpy as np, pandas as pd
DET, GEO, PARSE, GROUPS, TRK, H25, H25SRC, OUT = sys.argv[1:9]
HUM = float(os.environ.get("HUM_H", "1.70"))
PER = re.compile(r"\b(person|man|woman|pedestrian|walker|boy|girl)\b")
gs = json.load(open(GROUPS))["groups"]
LAY = {int(r): int(g["gid"].split("__q")[1]) for g in gs for r in g["rows"]}
by = {}
for g in gs: by.setdefault(g["video"], []).append((int(g["gid"].split("__q")[1]), g["gid"]))
for v in by: by[v].sort()
def track_path(vid, K):
    l = by.get(vid)
    if l is None: return None
    return os.path.join(TRK, (l[K][1] if K < len(l) else l[0][1]) + ".json")
det = pd.read_csv(DET, index_col=0)
G = pd.read_csv(GEO).set_index("Unnamed: 0")
P = {json.loads(l)["id"]: json.loads(l) for l in open(PARSE)}
out = {}
for i in det.index:
    if i not in LAY: continue
    x = P.get(i)
    if x is None: continue
    vid = str(x["video_id"])
    if not vid.startswith("internet") or "_segmented" in vid: continue
    pr = (x.get("priors") or [{}])[0]
    pph = str((x.get("prior_obj") or {}).get("phrase") or "").strip().lower()
    if pr.get("kind") != "speed" or not PER.search(pph): continue
    k = (x.get("q") or {}).get("kind")
    if k not in ("size", "dist2") or str(G.meas_mode.get(i)) != "2d_ratio": continue
    src = str(det.answer_source[i])
    if src.startswith("patch") or src.startswith("phys_bound") or src in ("caw_nogeo",): continue
    g, cs, mpp = float(det.g[i]), float(det.cs[i]), float(G.meas_mpp.get(i, np.nan))
    if not (g > 0 and mpp > 0): continue
    tp = track_path(vid, LAY[i])
    if tp is None or not os.path.exists(tp): continue
    tr = json.load(open(tp))
    if "gid" in tr:
        oid = next((str(q["obj"]) for q in tr["queries"] if q.get("obj") is not None and q["phrase"].strip().lower() == pph
                    and str(q["obj"]) in tr.get("objs", {})), None)
    else:
        oid = next((o for o in tr.get("objs", {}) if tr["queries"][int(o)]["phrase"].strip().lower() == pph), None)
    if oid is None: continue
    H = tr["H"]
    hs = [e["st"]["box"][3] - e["st"]["box"][1] for e in tr["objs"][oid] if e and e.get("st") and e["st"]["box"][1] > 2 and e["st"]["box"][3] < H - 3]
    if len(hs) < 5: continue
    h = float(np.median(hs)) * mpp
    r = h / HUM
    if not ((h > 2.3 or h < 0.5) and 0.5 <= r <= 2.0 and abs(np.log(r)) >= np.log(1.4)): continue
    g2 = g / r
    if np.isfinite(cs) and cs > 0 and abs(np.log(g2 / cs)) <= np.log(2):
        nv = g2 ** 0.7 * cs ** 0.3 if k == "size" else float(np.sqrt(g2 * cs)); how = "agree"
    else:
        nv = g2; how = "geo"
    if abs(nv / det.answer[i] - 1) > 0.02:
        out[str(i)] = dict(value=float(nv), why=f"humscaleS: own prior person {h:.2f} m tall at the row's m/px (r {r:.2f} vs {HUM} m) -> geo {g2:.4g} ({how}); was {det.answer[i]:.4g} {src}")
print("humscaleS patch rows", len(out), sorted(int(k) for k in out))
SRC = {int(r['id0']): r for r in csv.DictReader(open(H25SRC))}
cur = {int(r['id']) - 1: r['parsed_value'] for r in csv.DictReader(open(H25, newline='', encoding='utf-8'))}
BA = {int(i): a for i, a in zip(det.index, det.answer.astype(str))}
rows = []
for kk, v in out.items():
    i = int(kk); x = float(v['value'])
    if i not in SRC or not math.isfinite(x) or abs(x / float(cur[i]) - 1) < 1e-9: continue
    isp = SRC[i]['src'].startswith('base_p17'); ba = BA.get(i)
    okb = ba not in (None, '', 'nan') and abs(float(ba) / float(cur[i]) - 1) < 1e-9
    if (not isp) and okb: rows.append((i, x, 'humscaleS'))
with open(OUT, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['id0', 'value', 'rule']); [w.writerow([i, repr(float(v)), t]) for i, v, t in sorted(rows)]
print('65_rul302 cand rows', len(rows))
