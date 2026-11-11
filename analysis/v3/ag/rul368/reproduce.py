import io, json, os, re, subprocess, sys
import numpy as np, pandas as pd, cv2

D = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(D, '..', '..', '..', '..'))
FROZEN = os.path.join(D, 'frozen', 'mirror_sam.json')
SAM_FRAMES = [0, 20, 40, 60, 79]


def v3track(v, phrase, src='tracks_test_v3ntmv'):
    d = json.load(open(f'{R}/geo/out/{src}/{v}.json'))
    for i, q in enumerate(d['queries']):
        if q['phrase'] == phrase:
            return d['objs'][str(q.get('obj', i))]


def frame(v, k):
    cap = cv2.VideoCapture(f'{R}/data/test_480p/{v}.mp4'); cap.set(cv2.CAP_PROP_POS_FRAMES, k)
    ok, f = cap.read(); cap.release()
    return f


def gpu_mirror():
    import torch
    from PIL import Image
    from transformers import Sam2Model, Sam2Processor
    sp = Sam2Processor.from_pretrained(f'{R}/models/sam2.1-hiera-large')
    sm = Sam2Model.from_pretrained(f'{R}/models/sam2.1-hiera-large').to('cuda').eval()

    def sam(img, box):
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        inp = sp(images=pil, input_boxes=[[box]], return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = sm(**inp, multimask_output=False)
        m = sp.post_process_masks(out.pred_masks.cpu(), inp['original_sizes'])[0]
        return np.asarray(m).reshape(-1, img.shape[0], img.shape[1])[0] > 0
    mir = v3track('simulation_0221', 'floor mirror')
    res = {}
    for v in ['simulation_0221', 'simulation_0222']:
        for k in SAM_FRAMES:
            b = [float(x) for x in mir[k]['st']['box']]
            m = sam(frame(v, k), b)
            ys = np.where(m.any(1))[0]
            ext = np.array([np.ptp(np.where(m[y])[0]) + 1 for y in ys])
            n = max(2, int(0.15 * len(ys)))
            res[f'{v}|{k}'] = dict(box=b, rows=int(len(ys)), bottom_w=float(np.median(ext[-n:])),
                                   top_w=float(np.median(ext[:n])), area=int(m.sum()))
            print(v, k, res[f'{v}|{k}'], flush=True)
    os.makedirs(os.path.dirname(FROZEN), exist_ok=True)
    json.dump(dict(model='models/sam2.1-hiera-large (transformers Sam2Model, multimask_output=False)', frames=res),
              open(FROZEN, 'w'), indent=1)


def collected(x):
    return float(pd.read_csv(io.StringIO(f'v\n{float(x)!r}\n')).v.iloc[0])


def main():
    t = pd.read_csv(os.path.join(R, 'analysis/v3/calib/audit_test.csv'))
    b = pd.read_csv(os.path.join(R, 'submissions/pred_t007_id1.csv'))
    t['cur'] = b.parsed_value.values
    cur = dict(zip(t.id0, t.cur)); g = dict(zip(t.id0, t.g)); A = t.set_index('id0')
    M = pd.read_csv(f'{R}/geo/out/meas_test_v2_v3v_rntrj5sQu.csv')
    gmpp = lambda i: round(float(M[M.id == i].mpp.iloc[0]), 6)
    pnum = lambda i, w: float(re.search(w + r'[^=]*=\s*([0-9.]+)\s*m\b', str(A.loc[i, 'prior'])).group(1))
    new = {}
    vac = v3track('simulation_0221', 'vacuum cleaner')
    vac_px = float(np.median([max(f['st']['rr'][:2]) for f in vac if f and f.get('st')]))
    mpp = pnum(654, 'vacuum cleaner') / vac_px
    fr = json.load(open(FROZEN))['frames']
    ws = [fr[f'{v}|{k}']['bottom_w'] * mpp for v in ['simulation_0221', 'simulation_0222'] for k in SAM_FRAMES]
    w = float(np.median(ws))
    print('')
    new[654] = new[670] = w
    same = A[(A.video == 'simulation_0229') & (A.question.str.strip() == A.loc[726, 'question'].strip())].index[0]
    new[726] = cur[same]
    print('')
    wsz, hsz = [], []
    for k in [0, 25]:
        gr = cv2.cvtColor(frame('simulation_0235', k), cv2.COLOR_BGR2GRAY)
        m = (gr < 40).astype(np.uint8); m[200:] = 0
        n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
        H, W = gr.shape
        for i in range(1, n):
            x, y, ww, hh, a = st[i]
            if a > 1500 and x > 0 and y > 0 and x + ww < W and y + hh < 200:
                wsz.append(int(ww)); hsz.append(int(hh))
    new[768] = float(np.median(wsz)) * gmpp(771)
    print('')
    from scipy.optimize import curve_fit
    d = json.load(open(f'{R}/geo/out/tracks_test/simulation_0254.json'))
    for i, q in enumerate(d['queries']):
        if 'tennis' in q['phrase']:
            frs = q['frames'] if 'frames' in q else d['objs'][str(q.get('obj', i))]; fps = d['fps']; break
    c = np.array([f['st']['c'] if f and f.get('st') else [np.nan, np.nan] for f in frs])
    k = np.arange(21, 47); tt = k / fps
    v = np.linalg.norm(c[k] - c[k - 1], axis=1) * gmpp(862) * fps
    f = lambda x, a0, cc, t0: (a0 / cc) * (1 - np.exp(-cc * (x - t0)))
    p, cov = curve_fit(f, tt, v, p0=[7, 0.3, 0.0], maxfev=20000)
    v05 = float(f(0.5, *p))
    trap = cur[863] - 0.5 * (cur[866] + cur[867]) / 2
    new[862] = (v05 + trap) / 2
    print('')
    d = json.load(open(f'{R}/geo/out/tracks_test_v3ntmv/simulation_0223.json'))
    for i, q in enumerate(d['queries']):
        if 'general chess' in q['phrase']:
            frs = q['frames'] if 'frames' in q else d['objs'][str(q.get('obj', i))]; fps = d['fps']; break
    c = np.full((len(frs), 2), np.nan)
    for j, x in enumerate(frs):
        if isinstance(x, dict) and x.get('st'):
            c[j] = x['st']['c']
    kk = np.array([i for i in range(0, 43) if not np.isnan(c[i, 1])])
    pp = np.polyfit(kk, c[kk, 1], 2)
    y = np.polyval(pp, np.arange(0, 45))
    f05 = int(round(0.5 * fps))
    new[682] = float(abs(y[f05] - y[f05 - 1]) * gmpp(682) * fps)
    print('')
    d = json.load(open(f'{R}/geo/out/tracks_test/simulation_0153.json'))
    for i, q in enumerate(d['queries']):
        if 'tire' in q['phrase']:
            frs = q['frames'] if 'frames' in q else d['objs'][str(q.get('obj', i))]; break
    tire = float(np.median([max(x['st']['box'][2] - x['st']['box'][0], x['st']['box'][3] - x['st']['box'][1])
                            for x in frs[17:42] if x and x.get('st')]))
    tire_m = pnum(347, 'tire')
    cap = cv2.VideoCapture(f'{R}/data/test_480p/simulation_0153_segmented.mp4')
    wid = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        m = (img.min(2) < 200).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        ys, xs = np.where(m)
        if len(xs):
            H, W = m.shape
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            if not (x0 == 0 or x1 == W - 1 or y0 == 0 or y1 == H - 1):
                wid.append(int(x1 - x0 + 1))
    sil = float(np.median(wid) * tire_m / tire)
    geo = (g[347] + g[1852]) / 2
    new[347] = new[1852] = (sil + geo) / 2
    print('')
    ids = [654, 670, 726, 768, 862, 682, 347, 1852]
    with open(os.path.join(D, 'reproduce_out.csv'), 'w') as fo:
        fo.write('id0,value\n')
        for i in ids:
            fo.write(f'{i},{collected(round(float(new[i]), 4))!r}\n')
    for i in ids:
        print(i, 'unrounded', repr(float(new[i])), 'written', repr(collected(round(float(new[i]), 4))))


if __name__ == '__main__':
    if '--gpu' in sys.argv:
        gpu_mirror(); sys.exit(0)
    if '--rerun' in sys.argv:
        subprocess.run(['flock', f'{R}/cache/tmp/r4_gpu.lock', os.path.expanduser('~/anaconda3/envs/baba/bin/python'),
                        os.path.abspath(__file__), '--gpu'], check=True)
    main()
