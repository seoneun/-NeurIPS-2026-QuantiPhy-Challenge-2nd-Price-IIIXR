"""Round-4 audit dossier: one table + one contact sheet per test family (2026-10-20).

A family is one recording or scene render with its copies: simulation_0015 + simulation_0015_segmented,
internet_0001 + internet_0001_segmented, captured_0013 + 0013s + 0013x (lab cameras a/b of one event share the family
captured_0034). Usage:

    python analysis/v3/ag/rul351/dossier.py --list [--cat S3] [--src simulation]   # families with free scored rows
    python analysis/v3/ag/rul351/dossier.py FAMILY [FAMILY ..]                       # write dossiers

Outputs (cache/tmp/rul351/dossier/): <family>.txt (row table), <family>_sheet.jpg (frames of each video of the
family with the track boxes (v2sc, else v3ntmv, else tracks_test) of every query object; label = query phrase).
Columns of the table:
  id0, video, cat, sc (scored in the 3010 mask), lab (label tag holding the row's best24v value, if any), kind,
  question, prior, cur (best24v value), writers (patch tags whose test JSON value equals cur), src (fusion source of the
  Nzz48a-era answer), g (geo), cs (CaW crop/full answer), qw (Qwen3-VL-8B answer), mode (geo mode), mpp, tr (tracks).
"""
import argparse, glob, json, os, re, sys

import numpy as np
import pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
OUT = os.path.join(R, 'cache/tmp/rul351/dossier')
LABEL_TAGS = ['rul308', 'rul447', 'rul444', 'rul446', 'rul306', 'rul445', 'rul452', 'rul450',
              'rul350', 'r2_simsize_animlink']


def family(v):
    v = v.replace('_segmented', '')
    m = re.match(r'(captured_\d{4})', v)
    if m:
        return m.group(1)
    return v


def load():
    t = pd.read_csv(os.path.join(R, 'analysis/v3/calib/audit_test.csv'))
    b = pd.read_csv(os.path.join(R, 'submissions/sub_xt_best24v_id1.csv'))
    t['cur'] = b.parsed_value.values
    sm = pd.read_csv(os.path.join(R, 'analysis/v3/calib/r3/inputs/scored_mask_3010.csv'))
    t = t.merge(sm[['id0', 'scored']].rename(columns={'scored': 'sc'}), on='id0', how='left')
    t['fam'] = t.video.map(family)
    # patch writers: every patch JSON value equal to the current value
    wr = {}
    files = sorted(glob.glob(os.path.join(R, 'patches/ag_*_patch_test.json'))) + \
        sorted(glob.glob(os.path.join(R, 'cache/tmp/ag_*_patch_test.json')))
    seen = set()
    cur = dict(zip(t.id0, t.cur))
    for f in files:
        tag = os.path.basename(f)[3:-len('_patch_test.json')]
        if tag in seen:
            continue
        seen.add(tag)
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            if not isinstance(v, dict) or 'value' not in v:
                continue
            try:
                i = int(k); val = float(v['value'])
            except Exception:
                continue
            c = cur.get(i)
            if c is not None and np.isfinite(val) and abs(c - val) <= 1e-6 * max(1e-9, abs(val)):
                wr.setdefault(i, []).append(tag)
    t['writers'] = t.id0.map(lambda i: ','.join(wr.get(i, [])))
    t['lab'] = t.id0.map(lambda i: ','.join(x for x in wr.get(i, []) if x in LABEL_TAGS))
    return t


def short(x, n):
    x = '' if pd.isna(x) else str(x)
    return x if len(x) <= n else x[:n - 1] + '~'


def fmt(x):
    if pd.isna(x):
        return '-'
    try:
        x = float(x)
    except Exception:
        return str(x)
    return f'{x:.4g}'


def table(t, fam):
    d = t[t.fam == fam].sort_values(['question', 'video'])
    lines = [f'# family {fam}: {len(d)} rows, scored {int(d.sc.sum())}, label-held scored {int(((d.lab != "") & d.sc).sum())}']
    pri = d.prior.dropna().unique()
    lines.append('priors: ' + ' | '.join(map(str, pri)))
    dep = d.depth.dropna().unique()
    for x in dep[:3]:
        lines.append('depth_info: ' + str(x).replace('\n', ' ; '))
    lines.append('id0\tvideo\tcat\tsc\tlab\tkind\tquestion\tcur\twriters\tsrc\tg\tcs\tqw\tmode\tmpp')
    for _, r in d.iterrows():
        lines.append('\t'.join([str(r.id0), r.video.replace('simulation_', 'sim_').replace('captured_', 'cap_'), r['cat'],
                                'Y' if r.sc else 'n', short(r.lab, 12), str(r.kind), short(r.question, 110), fmt(r.cur),
                                short(r.writers, 40), short(r.source, 22), fmt(r.g), fmt(r.cs), fmt(r.qwen),
                                short(r['mode'], 22), '-']))
    return '\n'.join(lines)


def video_path(v):
    for p in [f'data/test_480p/{v}.mp4', f'data/test/{v}.mp4']:
        if os.path.exists(os.path.join(R, p)):
            return os.path.join(R, p)
    return None


def sheet(t, fam, cols=3):
    import cv2
    vids = sorted(t[t.fam == fam].video.unique(), key=lambda v: ('segmented' in v, v))
    panels = []
    for v in vids[:4]:
        vp = video_path(v)
        tj = next((os.path.join(R, f'geo/out/{d}/{v}.json') for d in ('tracks_test_v2sc', 'tracks_test_v3ntmv',
                   'tracks_test') if os.path.exists(os.path.join(R, f'geo/out/{d}/{v}.json'))), '')
        if vp is None:
            continue
        cap = cv2.VideoCapture(vp)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        js = json.load(open(tj)) if os.path.exists(tj) else None
        picks = [int(n * 0.1), int(n * 0.5), int(n * 0.9)] if v == vids[0] else [int(n * 0.5)]
        for fi in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, im = cap.read()
            if not ok:
                continue
            if js is not None:
                sx = im.shape[1] / js['W']; sy = im.shape[0] / js['H']
                fidx = js['fidx']
                pos = int(np.argmin([abs(f - fi) for f in fidx]))
                for qi, q in enumerate(js['queries']):
                    o = q.get('obj')
                    if o is None or str(o) not in js['objs']:
                        continue
                    fr = js['objs'][str(o)][pos]
                    st = fr.get('st') if isinstance(fr, dict) else None
                    if not st or not st.get('box'):
                        continue
                    x0, y0, x1, y1 = st['box']
                    col = [(0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 255), (0, 255, 255), (255, 255, 0),
                           (128, 0, 255), (255, 0, 0)][qi % 8]
                    cv2.rectangle(im, (int(x0 * sx), int(y0 * sy)), (int(x1 * sx), int(y1 * sy)), col, 2)
                    cv2.putText(im, q['phrase'][:22], (int(x0 * sx) + 2, max(12, int(y0 * sy) - 3)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            cv2.putText(im, f'{v} f{fi}/{n}', (5, im.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
            h = 270
            im = cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
            panels.append(im)
    if not panels:
        return None
    w = max(p.shape[1] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, 0, 0, w - p.shape[1], cv2.BORDER_CONSTANT) for p in panels]
    rows = []
    for i in range(0, len(panels), cols):
        r = panels[i:i + cols]
        while len(r) < cols:
            r.append(np.zeros_like(panels[0]))
        rows.append(np.hstack(r))
    out = os.path.join(OUT, f'{fam}_sheet.jpg')
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('fams', nargs='*')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--cat')
    ap.add_argument('--src')
    ap.add_argument('--nosheet', action='store_true')
    ap.add_argument('--blind', action='store_true',
                    help='pilot mode: label-held rows show the Nzz48a-era answer, label and writer columns are blank; '
                         'outputs go to cache/tmp/rul351/dossier_blind/')
    a = ap.parse_args()
    global OUT
    t = load()
    if a.blind:
        OUT = OUT + '_blind'
        m = t.lab != ''
        t.loc[m, 'cur'] = t.loc[m, 'answer']
        t['writers'] = ''
        t['lab'] = ''
    os.makedirs(OUT, exist_ok=True)
    if a.list:
        f = t[t.sc & (t.lab == '')]
        if a.cat:
            f = f[f.cat == a.cat]
        if a.src:
            f = f[f.src_type == a.src]
        g = f.groupby('fam').agg(free=('id0', 'size'), S2=('cat', lambda c: (c == 'S2').sum()),
                                 D2=('cat', lambda c: (c == 'D2').sum()), S3=('cat', lambda c: (c == 'S3').sum()),
                                 D3=('cat', lambda c: (c == 'D3').sum()), src=('src_type', 'first'))
        g['pts'] = g.S2 / 532 + g.D2 / 1109 + g.S3 / 488 + g.D3 / 881
        print(g.sort_values('pts', ascending=False).to_string())
        return
    for fam in a.fams:
        txt = table(t, fam)
        open(os.path.join(OUT, f'{fam}.txt'), 'w').write(txt + '\n')
        sh = None if a.nosheet else sheet(t, fam)
        print(f'{fam}: {os.path.join(OUT, fam + ".txt")} {sh or "(no sheet)"}')


if __name__ == '__main__':
    main()
