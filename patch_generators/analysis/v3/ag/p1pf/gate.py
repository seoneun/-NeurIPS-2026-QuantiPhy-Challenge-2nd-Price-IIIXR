"""P1 part 2: gate for a person focal resting on ONE estimate (geo/out/person_focal_<split>.json, n == 1).

geo/person_focal.py keeps a frame only when exactly one full-body 'person' box is seen, so with n == 1 nothing checks WHICH
person that box is. A seated cyclist is also a 'person' box, but its height is not the H = 1.864 m standing height, so the
focal would be wrong. The gate re-runs the same Grounding-DINO 'person.' detection at every frame with a stated person
distance, adds a 'bicycle. motorcycle.' detection, and
  - drops person boxes that overlap a bicycle / motorcycle box (intersection >= 30% of either box): riders;
  - keeps standing boxes only: full body (as person_focal.py) and height / width >= 2.0;
  - CONFIRMS the n == 1 estimate when its own box (same frame, same pixel height within 3%) survives both tests;
  - adds an estimate from every other stated-distance frame where exactly ONE standing non-rider box remains, and requires
    all estimates to agree within 15% (a second, independent check of the same ruler);
  - f_over_W = median of the confirmed estimates with H = 1.864 m (the height person_focal_<split>.json uses).
Every n == 1 video is printed with its verdict (gate firings and refusals).
usage: CUDA_VISIBLE_DEVICES= python analysis/v3/ag/p1pf/gate.py <split>   -> cache/tmp/ag_p1pf_gate_<split>.json
Inputs only: frames, parse depth_info, the person focal file. No labels."""
import os, re, sys, json
import numpy as np, pandas as pd, torch, torch.nn.functional as F
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, f"{R}/geo"); sys.path.insert(0, f"{R}/tools")
import track_common as tc  # noqa: E402
from transformers import AutoProcessor, GroundingDinoForObjectDetection  # noqa: E402

split = sys.argv[1] if len(sys.argv) > 1 else "test"
HP, THR, ASPECT, OVL, HTOL, AGREE = 1.864, 0.3, 2.0, 0.3, 0.03, 0.15
dev = "cpu"
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
gp = AutoProcessor.from_pretrained(f"{R}/models/grounding-dino-base")
gm = GroundingDinoForObjectDetection.from_pretrained(f"{R}/models/grounding-dino-base").to(dev).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
PER = re.compile(r"person|people|\bman\b|woman|girl|boy|human|player|child|kid|lady|guy", re.I)


@torch.no_grad()
def detect(frame, text):
    H, W = frame.shape[:2]
    s = 800 / min(H, W)
    if max(H, W) * s > 1333:
        s = 1333 / max(H, W)
    h, w = int(round(H * s)), int(round(W * s))
    x = torch.from_numpy(frame[None]).permute(0, 3, 1, 2).float() / 255.0
    x = (F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False, antialias=True) - MEAN) / STD
    tok = gp.tokenizer([text], return_tensors="pt", padding=True)
    out = gm(pixel_values=x, input_ids=tok.input_ids, attention_mask=tok.attention_mask,
             token_type_ids=tok.get("token_type_ids"), pixel_mask=torch.ones((1, h, w), dtype=torch.long))
    res = gp.post_process_grounded_object_detection(out, tok.input_ids, threshold=THR, text_threshold=THR, target_sizes=[(H, W)])[0]
    b = res["boxes"].cpu().numpy().astype(float); sc = res["scores"].cpu().numpy().astype(float)
    keep = tc.nms(b, sc, 0.6) if len(b) else []
    return [b[i] for i in keep]


def full_body(b, H):
    x1, y1, x2, y2 = b
    return y1 > 2 and y2 < H - 2 and (y2 - y1) >= 1.6 * max(x2 - x1, 1) and (y2 - y1) > 20


def inter(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def area(a):
    return max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))


def rider(b, bikes):
    return any(inter(b, c) / area(c) >= OVL or inter(b, c) / area(b) >= OVL for c in bikes)


def f_from(b, Z, W, H, height):
    hpx = b[3] - b[1]
    u, v = (b[0] + b[2]) / 2 - W / 2, (b[1] + b[3]) / 2 - H / 2
    f = hpx * Z / height
    for _ in range(3):
        z = Z / np.sqrt(1 + (u / f) ** 2 + (v / f) ** 2)
        f = hpx * z / height
    return f / W


PF = json.load(open(f"{R}/geo/out/person_focal_{split}.json"))
# video files of this checkout (data/test_480p, data/val/validation_videos via tools/qp_common.load_split), not the
# video_path column of the geo table, which records where the files were when the table was made
from qp_common import load_split  # noqa: E402
_ls = load_split(split)
vpath = {}
for _v, _p in zip(_ls.video_id, _ls.video_path):
    vpath.setdefault(_v, _p); vpath.setdefault(str(_v).strip(), _p)
info = {}
for ln in open(f"{R}/geo/out/parse_{split}_v2.jsonl"):
    x = json.loads(ln)
    for e in x.get("depth") or []:
        if PER.search(e["key"]):
            info.setdefault(x["video_id"], {"fps": x.get("fps") or 30.0, "ents": set()})["ents"].add((e["key"], e["t"], e["d"]))
out = {}
todo = sorted(v for v, p in PF.items() if p.get("n") == 1)
print(f"{split}: person-focal videos {len(PF)}, n == 1: {len(todo)} {todo}", flush=True)
for v in todo:
    p = PF[v]; e0 = p["ests"][0]
    if v not in info or v not in vpath:
        print(f"  {v}: no parse / video -> REFUSE"); out[v] = dict(ok=False, why="no parse/video"); continue
    frames = tc.load_video(str(vpath[v]))
    H, W = frames[0].shape[:2]; N = len(frames); fps = float(info[v]["fps"])     # parse fps (dataset value), never the container's
    ents = sorted(info[v]["ents"], key=lambda z: (z[0], -1 if z[1] is None else z[1]))
    keys = sorted({k for k, _, _ in ents})
    rec, ests, conf = [], [], None
    for k, t, Z in ents:
        if t is None:
            continue
        j = min(N - 1, int(round(t * fps)))
        pers = [b for b in detect(frames[j], "person.") if full_body(b, H)]
        bikes = detect(frames[j], "bicycle. motorcycle.")
        stand = [b for b in pers if not rider(b, bikes) and (b[3] - b[1]) >= ASPECT * max(b[2] - b[0], 1)]
        r = dict(key=k, t=t, Z=Z, j=j, n_person=len(pers), n_bike=len(bikes), n_stand=len(stand),
                 boxes=[[round(float(c), 1) for c in b] for b in pers],
                 riders=[bool(rider(b, bikes)) for b in pers], aspect=[round(float((b[3] - b[1]) / max(b[2] - b[0], 1)), 2) for b in pers])
        if j == e0["j"] and k == e0["key"]:
            same = [b for b in pers if abs((b[3] - b[1]) / e0["h_px"] - 1) <= HTOL]
            if len(pers) == 1 and same:
                b = same[0]
                conf = dict(rider=bool(rider(b, bikes)), aspect=round(float((b[3] - b[1]) / max(b[2] - b[0], 1)), 2),
                            standing=bool((b[3] - b[1]) >= ASPECT * max(b[2] - b[0], 1)), box=[round(float(c), 1) for c in b])
                if not conf["rider"] and conf["standing"]:
                    ests.append(dict(key=k, t=t, Z=Z, j=j, h_px=float(b[3] - b[1]), f_over_W=float(f_from(b, Z, W, H, HP)), src="n1"))
            else:
                conf = dict(rider=None, standing=None, why=f"est box not re-found ({len(pers)} full-body boxes)")
        elif len(keys) == 1 and len(stand) == 1:
            b = stand[0]
            ests.append(dict(key=k, t=t, Z=Z, j=j, h_px=float(b[3] - b[1]), f_over_W=float(f_from(b, Z, W, H, HP)), src="extra"))
        rec.append(r)
    ok = conf is not None and conf.get("rider") is False and conf.get("standing") is True and len(keys) == 1
    fw = np.array([e["f_over_W"] for e in ests]) if ests else np.array([])
    spread = float(fw.max() / fw.min()) if len(fw) else np.nan
    if ok and len(fw) >= 2 and spread > 1 + AGREE:
        ok, why = False, f"estimates disagree (spread {spread:.2f})"
    elif ok:
        why = f"confirmed standing non-rider (aspect {conf['aspect']}), extra estimates {len(fw) - 1}"
    else:
        why = f"not confirmed: {conf}" + ("" if len(keys) == 1 else f" (keys {keys})")
    fo = float(np.median(fw)) if (ok and len(fw)) else None
    out[v] = dict(ok=bool(ok), why=why, f_over_W=fo, f_json=p["f_over_W"], n=len(fw) if ok else 0, spread=spread, ests=ests, frames=rec, conf=conf)
    print(f"  {v:28s} {'FIRE  ' if ok else 'REFUSE'} f/W json {p['f_over_W']:.3f} gate {fo if fo is None else round(fo, 3)} "
          f"n {len(fw)} spread {spread:.3f} | {why}", flush=True)
    for r in rec:
        print(f"      t {r['t']} j {r['j']} Z {r['Z']}: person {r['n_person']} bikes {r['n_bike']} standing-non-rider {r['n_stand']} "
              f"riders {r['riders']} aspect {r['aspect']} boxes {r['boxes']}", flush=True)
json.dump(out, open(f"{R}/cache/tmp/ag_p1pf_gate_{split}.json", "w"), indent=1, default=float)
print("fired", sum(1 for x in out.values() if x["ok"]), "of", len(out))
