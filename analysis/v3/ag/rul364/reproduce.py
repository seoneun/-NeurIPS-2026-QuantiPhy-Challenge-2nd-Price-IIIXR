import io, json, os, subprocess, sys
import numpy as np, pandas as pd

D = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(D, '..', '..', '..', '..'))
FROZEN = os.path.join(D, 'frozen', 'owl_internet_0016.json')
OWL_SPEC = ('internet_0016', [0, 15], ['basketball hoop', 'basketball rim', 'person'])


def gpu_owl():
    import cv2, torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    op = Owlv2Processor.from_pretrained(f"{R}/models/owlv2-base-patch16-ensemble")
    om = Owlv2ForObjectDetection.from_pretrained(f"{R}/models/owlv2-base-patch16-ensemble").to(dev).eval()
    v, fs, qs = OWL_SPEC
    out = {}
    cap = cv2.VideoCapture(f'{R}/data/test_480p/{v}.mp4')
    for f in fs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read(); rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]; S = max(H, W)
        inp = op(text=[qs], images=rgb, return_tensors='pt').to(dev)
        with torch.no_grad():
            o = om(**inp)
        res = op.post_process_grounded_object_detection(o, threshold=0.05, target_sizes=torch.tensor([[S, S]]).to(dev))[0]
        dets = []
        for qi, q in enumerate(qs):
            m = res['labels'] == qi
            sc = res['scores'][m]; bx = res['boxes'][m]
            for k in sc.argsort(descending=True)[:8]:
                dets.append([q, round(float(sc[k]), 3)] + [round(float(t), 1) for t in bx[k]])
        out.setdefault(v, {})[str(f)] = dets
    os.makedirs(os.path.dirname(FROZEN), exist_ok=True)
    json.dump(out, open(FROZEN, 'w'), indent=0); print('wrote', FROZEN)


def collected(x):
    return float(pd.read_csv(io.StringIO(f'v\n{float(x)!r}\n')).v.iloc[0])


def main():
    P = pd.read_csv(R + '/preds/test_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv'); P['id0'] = P['Unnamed: 0']; P = P.set_index('id0')

    def track(v, d, phrase, objsel=None):
        T = json.load(open(f'{R}/geo/out/{d}/{v}.json')); res = []
        for q in T['queries']:
            if q['phrase'] != phrase: continue
            if objsel is not None and str(q.get('obj')) != str(objsel): continue
            if 'frames' in q: fr = q['frames']
            else:
                ob = T['objs']; o = ob[int(q['obj'])] if isinstance(ob, list) else ob[str(q['obj'])]
                fr = o['frames'] if isinstance(o, dict) and 'frames' in o else o
            st = [x.get('st') if isinstance(x, dict) and 'st' in x else (x if isinstance(x, dict) else None) for x in fr]
            res.append((list(T['fidx']), st, q.get('obj')))
        return res

    def at(tr, f):
        fidx, st, _ = tr; k = min(range(len(fidx)), key=lambda i: abs(fidx[i] - f)); s = st[k]
        return np.array(s['c'], float) if s and s.get('c') else None
    OWL = json.load(open(FROZEN))

    def dets(v, f, q):
        return [(x[1], np.array([(x[2] + x[4]) / 2, (x[3] + x[5]) / 2]), x[4] - x[2], x[5] - x[3]) for x in OWL[v][str(f)] if x[0] == q]
    new = {}
    v = 'internet_0016'; mpp = float(P.loc[2082, 'meas_mpp']); assert abs(mpp - float(P.loc[2079, 'meas_mpp'])) < 1e-12
    fps = json.load(open(f'{R}/geo/out/file_fps_test.json'))[v]['csv']
    f05 = int(round(0.5 * fps))
    H = json.load(open(f'{R}/geo/out/tracks_test_v2sc/{v}.json'))['H']
    blk = at(track(v, 'tracks_test_v2sc', 'player in black')[0], f05)
    rims = [d for d in dets(v, f05, 'basketball rim') if d[2] < 40 and d[3] < 40]
    bot = max([d for d in rims if d[1][1] > H / 2], key=lambda d: d[0])
    top = max([d for d in rims if d[1][1] < H / 2], key=lambda d: d[0])
    circ = [d for d in dets(v, f05, 'basketball rim') if 80 < d[2] < 130 and 80 < d[3] < 130]
    cen = min(circ, key=lambda d: abs(d[1][1] - (H + 1) // 2))[1]
    r1 = np.linalg.norm(blk - bot[1]) * mpp; r2 = np.linalg.norm(blk - (2 * cen - top[1])) * mpp
    new[2082] = r1
    print('')
    blk0 = at(track(v, 'tracks_test_v2sc', 'player in black')[0], 0)
    pl = [(at(t, 0), t[2]) for t in track(v, 'tracks_test_v3nt', 'players')]
    wht = min([c for c, o in pl if c is not None and np.linalg.norm(c - blk0) > 30], key=lambda c: np.linalg.norm(c - bot[1]))
    d1 = np.linalg.norm(blk0 - wht) * mpp
    per = sorted(dets(v, 0, 'person'), key=lambda d: -d[0])
    per_b = [d for d in per if d[1][1] > 600 and d[2] > 30]
    d2 = np.linalg.norm(per_b[0][1] - per_b[1][1]) * mpp if len(per_b) >= 2 else float('nan')
    new[2079] = d1
    print('')
    v = 'internet_0023'; tr = track(v, 'tracks_test_v2sc', 'swimmer')[0]
    bh = np.nanmedian([s['box'][3] - s['box'][1] for s in tr[1] if s]); rl = np.nanmedian([max(s['rr'][:2]) for s in tr[1] if s and s.get('rr')])
    k = bh / rl
    for rid in (2115, 2116):
        new[rid] = float(P.loc[rid, 'geo_value']) * k
        print('')
    ids = [2082, 2079, 2115, 2116]
    val = {i: (round(float(new[i]), 4) if i in (2082, 2079) else round(new[i], 4)) for i in ids}
    with open(os.path.join(D, 'reproduce_out.csv'), 'w') as fo:
        fo.write('id0,value\n')
        for i in ids:
            fo.write(f'{i},{collected(val[i])!r}\n')
    for i in ids:
        print(i, 'unrounded', repr(float(new[i])), 'written', repr(collected(val[i])))


if __name__ == '__main__':
    if '--gpu' in sys.argv:
        gpu_owl(); sys.exit(0)
    if '--rerun' in sys.argv:
        subprocess.run(['flock', f'{R}/cache/tmp/r4_gpu.lock', os.path.expanduser('~/anaconda3/envs/baba/bin/python'),
                        os.path.abspath(__file__), '--gpu'], check=True)
    main()
