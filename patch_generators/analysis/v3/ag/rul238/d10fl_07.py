import sys, json, os, numpy as np, pandas as pd
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
def rates(sp, v, ent, lo, hi):
    fps=24.0
    k=ent['phrase']+'|'+','.join(ent.get('where') or [])+('|'+ent['rel'][0]+':'+ent['rel'][1] if ent.get('rel') else '')
    for d in ('_v2sc','_v2',''):
        p=f"{R}/geo/out/tracks_{sp}{d}/{v}.json"
        if not os.path.exists(p): continue
        j=json.load(open(p))
        if 'objs' in j:
            q=next((q for q in (j.get('queries_d') or [])+(j.get('queries') or []) if isinstance(q,dict) and q.get('key')==k and q.get('obj') is not None),None)
            fr=j['objs'][str(q['obj'])] if q else None
        else:
            q=next((q for q in (j.get('queries') or []) if q.get('key')==k and q.get('frames') is not None),None)
            fr=q['frames'] if q else None
        if fr is None: return d,'noq'
        C={f:(np.array(e['st']['c'],float),e['st']['box'][3]-e['st']['box'][1],e['st'].get('border',0)) for f,e in zip(j['fidx'],fr) if e and e.get('st')}
        h=6; rr=[]
        for f in sorted(C):
            if not (lo<=f/fps<=hi): continue
            if f-h in C and f+h in C and C[f][2]==0 and C[f-h][2]==0 and C[f+h][2]==0 and C[f][1]>4:
                rr.append(float(np.hypot(*(C[f+h][0]-C[f-h][0])))/(2*h/fps)/C[f][1])
        return d, len(rr), (round(float(np.median(rr)),3) if rr else None), round(max(j['fidx'])/fps,2)
    return 'none'
P={}
for sp in ('test','val'):
    for l in open(f"{R}/geo/out/parse_{sp}_v2.jsonl"):
        x=json.loads(l); P.setdefault((sp,x['video_id']),x)
ent={'phrase':'person','where':[],'rel':None}
for n in ('25','26','27','31','34','35','37','38'):
    for s in ('','_segmented'):
        v='simulation_00'+n+s
        for sp in ('test','val'):
            if (sp,v) in P:
                print(sp, v, P[(sp,v)]['prior_text'], '[1,2]', rates(sp,v,ent,1,2), 'whole', rates(sp,v,ent,0,99))
