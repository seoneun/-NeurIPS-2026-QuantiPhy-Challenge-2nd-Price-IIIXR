import re, math, os, json
_DC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dur_cache.json")
import os
_DATA = os.path.join(os.environ.get("QP_ROOT", ""), "data")
_dur = None
def own_duration(row):
    global _dur
    if _dur is None:
        try: _dur = json.load(open(_DC))
        except Exception: _dur = {}
    k = f"{row['split']}/{row['video_id']}"
    d = _dur.get(k)
    if d is None:
        import cv2
        p = f"{_DATA}/test/{row['video_id']}.mp4" if row['split'] == 'test' else f"{_DATA}/val/validation_videos/{row['video_id']}.mp4"
        if not os.path.exists(p): return None
        c = cv2.VideoCapture(p); d = dict(n=c.get(cv2.CAP_PROP_FRAME_COUNT), fps=c.get(cv2.CAP_PROP_FPS)); c.release(); _dur[k] = d
    if not d or not d.get('fps'): return None
    return (d['n'] - 1) / d['fps']

def _words(s):
    s = re.sub(r"[^a-z ]", " ", str(s or '').lower())
    return [x for x in s.split() if x not in ('the', 'a', 'an', 'of', 'model', 'moving')]

def same_obj(a, b):
    wa, wb = _words(a), _words(b)
    if not wa or not wb or wa[-1] != wb[-1] and wa[0] != wb[0]: return False
    sa, sb = set(wa), set(wb)
    return sa <= sb or sb <= sa

def cv_pred(row):
    p = row.get('parse') or {}; q = p.get('q') or {}; pr = p.get('priors') or []
    if len(pr) != 1 or pr[0].get('kind') != 'speed' or q.get('kind') not in ('disp', 'path'): return None
    if not same_obj(q.get('target'), pr[0].get('obj')): return None
    if q.get('target2'): return None
    v = abs(float(pr[0]['value_si'])); tp = pr[0].get('t')
    t1, t2 = q.get('t1'), q.get('t2')
    if t1 is not None and t2 is not None:
        dt = float(t2) - float(t1)
        if not (0 < dt <= 1.0): return None
        if tp is not None and abs(float(tp) - float(t1)) > 1e-6: return None
    elif q.get('t') is None and q.get('when') is None and tp is None:
        dt = own_duration(row)
        if not dt: return None
    else:
        return None
    sc = 100.0 if str(q.get('unit') or '').startswith('cm') else (1000.0 if q.get('unit') == 'mm' else 1.0)
    return v * dt * sc

def make(thr, gm):
    def rule(row):
        d = cv_pred(row)
        if not d or d <= 0: return None
        c = row['cur']
        if abs(math.log(c / d)) <= thr: return None
        val = math.sqrt(c * d) if gm else d
        return val, f"cvkin: same-object speed prior x dt = {d:.4g} vs cur {c:.4g}"
    return rule

E1_V1 = make(0.15, False); E1_V2 = make(0.15, True); E1_V3 = make(0.10, True)
