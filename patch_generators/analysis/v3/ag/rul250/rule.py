"""rul250: S2/D2 size rows whose target phrase is grounded (v2sc) to two different static blobs in
'queries' vs 'queries_d' (both static: span < 1.5*size; median centres > max(size_a,size_b) and > 100 px apart),
with no v3nt object for the phrase, and answer_source dis_geo with cs/g >= 2 or g/cs >= 2 -> answer CaW (cs).
No ids, no labels in the logic (labels joined at the end for evaluation only).
usage: tools/run_limited.sh --io python analysis/v3/ag/rul250/rule.py [SUB] [write]"""
import sys, json, os
import numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0, R + '/cache/tmp'); sys.path.insert(0, R + '/caw'); sys.path.insert(0, R + '/analysis/v3/ag/rul006')
import analyze_caw as A
from lab_rows import labelled
from scan import obj_stats, load_tracks
pd.set_option('display.width', 400); pd.set_option('display.max_colwidth', 70); pd.set_option('display.max_rows', 3000)
TAG = 'rul250'
args = [a for a in sys.argv[1:] if a != 'write']
SUB = args[0] if args else 'sub_xt_Nzz40'
WRITE = 'write' in sys.argv[1:]
RAWN = 'geo_v4ntrj5sQ2srngsr_d_vw'
WHY = 'bindsplit: target grounded to two different static blobs, no v3nt witness -> CaW'


def qobj(tj, phrase, which):
    if tj is None:
        return None
    for q in tj.get(which) or []:
        if (q.get("phrase") or "").strip() == phrase:
            return q.get("obj")
    return None


def stats(tj, o):
    if tj is None or o is None or str(o) not in (tj.get("objs") or {}):
        return None
    return obj_stats(tj["objs"][str(o)])


rows = []
for sp, suf in (('test', 'detail'), ('val', 'valdetail')):
    raw = pd.read_csv(f"{R}/preds/{sp}_{RAWN}_raw.csv", index_col=0)
    det = pd.read_csv(f"{R}/submissions/{SUB}_{suf}.csv", index_col=0)
    PP = {p['id']: p for p in map(json.loads, open(f"{R}/geo/out/parse_{sp}_v2.jsonl"))}
    TC = {}
    for i in det.index:
        cat = det['cat'][i]
        if cat not in ('S2', 'D2'):
            continue
        p = PP[i]
        if (p.get('q') or {}).get('kind') != 'size':
            continue
        vid = str(det.video_id[i]); ph = ((p.get('target') or {}).get('phrase') or '').strip()
        if not ph:
            continue
        if vid not in TC:
            TC[vid] = (load_tracks(sp, '_v2sc', vid), load_tracks(sp, '_v3nt', vid))
        t2, t3 = TC[vid]
        oa, ob = qobj(t2, ph, 'queries'), qobj(t2, ph, 'queries_d')
        sa, sb = stats(t2, oa), stats(t2, ob)
        o3 = qobj(t3, ph, 'queries'); o3d = qobj(t3, ph, 'queries_d')
        s3 = stats(t3, o3) or stats(t3, o3d)
        src = str(det.answer_source[i]); g = float(det.g[i]); cs = float(det.cs[i]) if det.cs[i] == det.cs[i] else np.nan
        a_ok = False; dist = np.nan
        if sa is not None and sb is not None and str(oa) != str(ob):
            dist = float(np.linalg.norm(np.median(sa['c'], 0) - np.median(sb['c'], 0)))
            stat = sa['span'] < 1.5 * sa['size'] and sb['span'] < 1.5 * sb['size']
            a_ok = stat and dist > max(sa['size'], sb['size']) and dist > 100
        b_ok = s3 is None
        c_ok = src.replace('mo_guard:', '') == 'dis_geo' and g > 0 and cs > 0 and (cs / g >= 2 or g / cs >= 2)
        rows.append(dict(sp=sp, rid=int(i), vid=vid, q=str(raw.question[i])[:60], cat=cat, src=src, ans=float(det.answer[i]), g=g, cs=cs,
                         oa=oa, ob=ob, dist=dist, a=a_ok, b=b_ok, c=c_ok, patch=src.startswith('patch'),
                         fire=a_ok and b_ok and c_ok and not src.startswith('patch')))
d = pd.DataFrame(rows)
print('size S2/D2 rows', d.groupby('sp').size().to_dict())
print('a', d.groupby('sp').a.sum().to_dict(), 'a&b', d[d.a & d.b].groupby('sp').size().to_dict())
print(d[d.a].to_string())
F = d[d.fire]
L = labelled(SUB); L = L.reset_index() if 'rid' not in L.columns else L
print('labelled cols', list(L.columns)[:20])
idc = 'id' if 'id' in L.columns else L.columns[0]
for r in F.itertuples():
    m = L[(L.split == r.sp) & (L[idc] == r.rid)]
    if len(m):
        y = float(m.y.iloc[0])
        print('LAB', r.sp, r.rid, 'y', y, 'before', A.row_mra(np.array([r.ans]), np.array([y])), 'after', A.row_mra(np.array([r.cs]), np.array([y])))
print('FIRE'); print(F.to_string())
# labelled rows that match a&b (diagnostic: how CaW vs answer do there)
LM = d[d.a].merge(L[['split', idc, 'y']].rename(columns={'split': 'sp', idc: 'rid'}), on=['sp', 'rid'], how='inner')
if len(LM):
    LM['m_ans'] = A.row_mra(LM.ans.values, LM.y.values); LM['m_cs'] = A.row_mra(LM.cs.fillna(0).values, LM.y.values)
    print('labelled rows with (a):'); print(LM[['sp', 'rid', 'vid', 'q', 'src', 'ans', 'g', 'cs', 'y', 'b', 'c', 'm_ans', 'm_cs']].to_string())
if WRITE:
    for sp in ('test', 'val'):
        out = {str(r.rid): {'value': float(r.cs), 'why': WHY} for r in F[F.sp == sp].itertuples()}
        json.dump(out, open(f"{R}/cache/tmp/ag_{TAG}_patch_{sp}.json", 'w'), indent=1)
        print('wrote', sp, len(out))
