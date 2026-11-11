import os, sys, re, csv, json, subprocess, random, numpy as np, pandas as pd
R = os.getcwd(); D = os.path.dirname(os.path.abspath(__file__)) + '/'; W = D; OUT = sys.argv[1] + '/'
MO = ['speed', 'dist2', 'disp', 'path', 'accel', 'maxheight', 'avgspeed']
tf = pd.read_pickle(D + 'tab_fixed.pkl'); REC = {(r.split, r.id0): r for r in tf.itertuples()}
CLEAN = ('base_z1s2c', '01_h2', '02_aud_tag', '03b_aud_geo_z', 'opt_A_copyfix', 'opt_B_h1chain', '13_rul279', '30_rul003')


def latgeo_own(row):
    i = row['id0']
    if not str(row['video']).startswith('captured') or row['cat'] != 'S3': return None
    if not row['src'].startswith(CLEAN): return None
    r = REC.get(('test', i))
    if r is None or r.kind not in MO or not (r.F == r.F) or r.F < 1.15 or r.mode == '3d_knownsize': return None
    if r.kind == 'dist2' and not re.search(r'\d\s*s\b', row['question']): return None
    vg = row['value'] / r.g if r.g == r.g and r.g else np.nan
    if not (0.4 <= vg <= 1.05): return None
    return row['value'] * r.F, 't167'


S3BALL = [2747, 2755, 2763]
def s3ball40_own(row):
    i = row['id0']
    if i not in S3BALL: return None
    m = re.search(r'velocity of the (.+?) at ([\d.]+)s', row['question']); phrase, tq = m.group(1), float(m.group(2)); key = phrase.split()[-1]
    ent = [(float(a), float(b)) for a, b in re.findall(r't=([\d.]+)s, distance_' + key + r'_camera = ([\d.]+)m', row['depth_info'])]
    trk = json.load(open(R + '/geo/out/tracks_test_v3ntmv/' + row['video'] + '.json')); r = REC[('test', i)]; fgeo = r.fW * trk['W']
    out = subprocess.run([sys.executable, W + 'cpy/grav.py', row['video'], phrase, '%.3f' % fgeo, ','.join('%g=%g' % e for e in ent), '--at', '%g' % tq],
                         capture_output=True, text=True).stdout
    q = {}
    for nm in ('f_geo', 'f_grav'):
        mm = re.search(r't=%g \(tmid [\d.]+\) %s=\d+: vert=([-\d.]+) lat=([-\d.]+) rad=([-\d.]+) speed=([\d.]+)' % (tq, nm), out)
        vz, la, ra, sp = map(float, mm.groups()); q[nm] = float(np.hypot(r.F * np.hypot(vz, la), ra))
    return float(np.sqrt(q['f_geo'] * q['f_grav'])), 's3ball40_own: own copy FD1xF %.2f / gravity-arc xF %.2f geo-mean (f_geo %.0f = own fW x W, own F %.4f)' % (q['f_geo'], q['f_grav'], fgeo, r.F)


S3LATF = [2524, 2530, 2535, 2541, 2547, 2553, 2667, 2672, 2677]
def s3latF_own(row):
    i = row['id0']
    if i not in S3LATF: return None
    r = REC.get(('test', i))
    if r is None or not (r.F == r.F): return None
    return row['value'] * r.F, 't168'


RULES = [('latgeo_own', latgeo_own), ('s3ball40_own', s3ball40_own), ('s3latF_own', s3latF_own)]
H = D + 'pred_t069_id1.csv'
h = pd.read_csv(H); src = pd.read_csv(D + 'pred_t069_id1_src.csv').set_index('id0'); st = pd.read_csv(os.environ['H1ROWS']).set_index('id0')


def run(tbl, ids, upto=None):
    out = {}
    for i in ids:
        if src.cat[i] != 'S3' or src.src[i].startswith('base_p17'): continue
        row = dict(id0=i, video=tbl.video_id[i], question=tbl.question[i], depth_info=str(tbl.depth_info[i]), value=float(tbl.parsed_value[i]), cat=src.cat[i], src=src.src[i])
        for nm, fn in RULES[:upto]:
            o = fn(row)
            if o is not None and abs(o[0] / row['value'] - 1) > 1e-9:
                out[i] = (o[0], nm, o[1]); row['value'] = o[0]
    return out


def mra(p, y):
    e = abs(p - y) / abs(y); return np.mean([e < 1 - t for t in np.arange(.5, 1, .05)])


ALL = list(range(len(h)))
full = run(h, ALL)
alone_ok = True
for i in full:
    t = h.copy(); m = np.arange(len(t)) != i
    t.loc[m, 'parsed_value'] = np.nan; t.loc[m, 'question'] = ''; t.loc[m, 'depth_info'] = ''; t.loc[m, 'video_id'] = 'zz'
    a = run(t, [i]); alone_ok &= (i in a and a[i][0] == full[i][0])
perm = list(ALL); random.Random(7).shuffle(perm)
sh = run(h, perm)
print('LEGALITY alone==full', alone_ok, 'shuffled==full', sh == full)
rows = []
for i, (v, nm, w) in sorted(full.items()):
    rows.append(dict(id0=i, video=h.video_id[i], h22=float(h.parsed_value[i]), value=v, rule=nm, p17=st.p17[i], status=st.status[i], h22src=src.src[i][:50],
                     dmra_p17=round(mra(v, st.p17[i]) - mra(float(h.parsed_value[i]), st.p17[i]), 3), why=w))
DF = pd.DataFrame(rows); DF.to_csv(OUT + 'cand_detail.csv', index=False)
with open(OUT + 'cand.csv', 'w', newline='') as f:
    wr = csv.writer(f, lineterminator='\n'); wr.writerow(['id0', 'value', 'rule'])
    for i, (v, nm, w) in sorted(full.items()): wr.writerow([i, repr(float(v)), nm])
for r in rows: print(r['id0'], r['why'])
