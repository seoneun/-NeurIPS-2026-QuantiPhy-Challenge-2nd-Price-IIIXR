import sys,json,cv2,numpy as np
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
sys.path.insert(0,R+'/geo'); import track_common as tc
out=R+'/cache/tmp/ag_c81partcoll_viz'; import os; os.makedirs(out,exist_ok=True)
for vid in ['simulation_0199','simulation_0200','simulation_0196']:
    fr=tc.load_video(f'{R}/data/test_480p/{vid}.mp4')
    f=fr[0] if not isinstance(fr,tuple) else fr[0][0]
    f=np.ascontiguousarray(f[...,::-1]).copy()
    cols={'_v2sc':(0,0,255),'_v3':(0,255,0),'_v2':(255,0,0)}
    for d,c in cols.items():
        J=json.load(open(f'{R}/geo/out/tracks_test{d}/{vid}.json'))
        for q in J['queries']:
            if 'letter' in q['key'] or 'text' in q['key']:
                o=q.get('obj')
                if o is None: continue
                B=np.median([e['st']['box'] for e in J['objs'][str(o)] if e],0).astype(int)
                cv2.rectangle(f,(B[0],B[1]),(B[2],B[3]),c,1)
    cv2.imwrite(f'{out}/{vid}.jpg',f)
print(type(fr), len(fr))
