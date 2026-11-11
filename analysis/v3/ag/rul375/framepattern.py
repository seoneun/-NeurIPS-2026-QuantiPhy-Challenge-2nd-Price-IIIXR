import os, sys, json
import numpy as np, pandas as pd, cv2
R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
OUT = os.environ.get('FP_OUT', f'{R}/cache/tmp/rul375')
os.makedirs(OUT, exist_ok=True)
FF = json.load(open(f'{R}/geo/out/file_fps_test.json'))
FV = json.load(open(f'{R}/geo/out/file_fps_val.json'))


def frames(path, w=320):
    cap = cv2.VideoCapture(path); out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY); h = int(round(g.shape[0] * w / g.shape[1]))
        out.append(cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA))
    return out


def local_ratio(x, k=4):
    x = np.asarray(x, float); r = np.full(len(x), np.nan)
    for i in range(len(x)):
        nb = np.r_[x[max(0, i - k):i], x[i + 1:i + 1 + k]]
        m = np.median(nb) if len(nb) else np.nan
        if m and m > 0:
            r[i] = x[i] / m
    return r


def track_steps(split, v):
    for ts in ('v2sc', 'v3ntmv'):
        p = f'{R}/geo/out/tracks_{split}_{ts}/{v}.json'
        if os.path.exists(p):
            j = json.load(open(p)); break
    else:
        return {}
    out = {}
    for oid, lst in j.get('objs', {}).items():
        c = [(e['st']['c'] if (e and e.get('st')) else None) for e in lst]
        st = []
        for a, b in zip(c[:-1], c[1:]):
            st.append(np.hypot(b[0] - a[0], b[1] - a[1]) if (a is not None and b is not None) else np.nan)
        st = np.array(st)
        if np.nanmedian(st) > 1.5 if np.isfinite(st).sum() > 8 else False:
            out[f'trk{oid}'] = st
    return out


def series(path, split, v):
    F = frames(path); S = {}
    if len(F) < 3:
        return S, len(F)
    mad, med, p95, p99 = [], [], [], []
    for a, b in zip(F[:-1], F[1:]):
        mad.append(np.abs(a.astype(np.int16) - b).mean())
        fl = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 3, 5, 1.1, 0)
        m = np.hypot(fl[..., 0], fl[..., 1]).ravel()
        med.append(np.median(m)); p95.append(np.percentile(m, 95)); p99.append(np.percentile(m, 99))
    S.update(mad=np.array(mad), fmed=np.array(med), f95=np.array(p95), f99=np.array(p99))
    S.update(track_steps(split, v))
    return S, len(F)


def main(vids):
    rows, summ = [], []
    for spec in vids:
        split, v = spec.split(':') if ':' in spec else ('test', spec)
        path = f'{R}/data/test/{v}.mp4' if split == 'test' else f'{R}/data/val/validation_videos/{v}.mp4'
        e = (FF if split == 'test' else FV).get(v, {})
        S, n = series(path, split, v)
        csv, fil = e.get('csv'), e.get('file')
        pred_drop = max(0.0, (csv - fil) / fil) if csv and fil else np.nan
        pred_dup = max(0.0, (fil - csv) / fil) if csv and fil else np.nan
        for name, x in S.items():
            r = local_ratio(x)
            for i, (xi, ri) in enumerate(zip(x, r)):
                rows.append(dict(split=split, video=v, series=name, k=i, x=xi, ratio=ri))
            ok = np.isfinite(r)
            if ok.sum() < 6:
                continue
            dbl = ((r > 1.6) & (r < 2.6))[ok]
            dup = (r < 0.3)[ok]
            idx = np.where((r > 1.6) & (r < 2.6))[0]
            gaps = np.diff(idx)
            summ.append(dict(split=split, video=v, n=n, csv=csv, file=fil, series=name, med_x=float(np.nanmedian(x)),
                             frac_dbl=float(dbl.mean()), frac_dup=float(dup.mean()), pred_drop=pred_drop, pred_dup=pred_dup,
                             dbl_at=' '.join(map(str, idx[:16])), gap_mode=(int(pd.Series(gaps).mode().iloc[0]) if len(gaps) else -1),
                             gap_mode_n=(int((gaps == pd.Series(gaps).mode().iloc[0]).sum()) if len(gaps) else 0)))
        print(v, split, n, 'done', file=sys.stderr)
    pd.DataFrame(rows).to_csv(f'{OUT}/fp_series.csv', index=False)
    s = pd.DataFrame(summ); s.to_csv(f'{OUT}/fp_summary.csv', index=False)
    return s


if __name__ == '__main__':
    vids = sys.argv[1:]
    if not vids:
        mis = [k for k, e in FF.items() if k.startswith('internet') and abs(e['file'] - e['csv']) / e['csv'] > 0.003]
        ctrl = ['internet_0004', 'internet_0005', 'internet_0013', 'internet_0024', 'internet_0041']
        vids = sorted(mis) + ctrl + ['val:internet_0001', 'val:internet_0019', 'val:internet_0027', 'val:internet_0004',
                                     'val:internet_0005', 'val:internet_0013']
    s = main(vids)
    pd.set_option('display.width', 250)
    print(s.to_string(index=False))
