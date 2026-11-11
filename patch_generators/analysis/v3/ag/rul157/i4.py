import json,os,numpy as np
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
def boxes(J,o):
    tr=J['objs'].get(str(o)) if isinstance(J['objs'],dict) else J['objs'][o]
    if tr is None: return None
    return np.array([e['st']['box'] for e in tr if e and e.get('st') and e['st'].get('box')],float), len(J['fidx'])
for vid in ['simulation_0196','simulation_0197','simulation_0198','simulation_0199','simulation_0200']:
  for d in ['_v2sc','_v3','_v3nt','_v3nt32','_v2']:
    f=f'{R}/geo/out/tracks_test{d}/{vid}.json'
    if not os.path.exists(f): print(vid,d,'missing'); continue
    J=json.load(open(f))
    print(vid,d,'nfidx',len(J['fidx']))
    for qk in ('queries','queries_d'):
        for q in J.get(qk,[]) or []:
            o=q.get('obj'); b=boxes(J,o) if o is not None else None
            s='' if b is None else f'n{len(b[0])}/{b[1]} med {np.median(b[0],0).round(1).tolist()}'
            print('  ',qk,q.get('key'),o,s)
