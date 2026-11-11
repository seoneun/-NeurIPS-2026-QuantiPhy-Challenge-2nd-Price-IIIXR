import json, glob, numpy as np
for vid in ['simulation_0219','simulation_0220','simulation_0218']:
    for p in sorted(glob.glob(f'geo/out/tracks_test_*/{vid}.json'))[:12]:
        t=json.load(open(p)); fid=t['fidx']
        qs=t.get('queries',[])
        for q in qs:
            if 'robot' not in q['key']: continue
            o=t['objs'][str(q['obj'])] if isinstance(t['objs'],dict) else t['objs'][q['obj']]
            pts=[(f,s['st']['box']) for f,s in zip(fid,o) if s and s.get('st')]
            if len(pts)<5: continue
            f=np.array([a for a,b in pts]); b=np.array([bb for a,bb in pts])
            cx=(b[:,0]+b[:,2])/2; ok=(b[:,0]>2)&(b[:,2]<852)
            if ok.sum()>3:
                sl=np.polyfit(f[ok]/24.0,cx[ok],1)[0]
                print(p.split('/')[-2],vid,q['key'],'n',int(ok.sum()),'px/s %.1f'%sl,'bottom %.0f'%np.median(b[:,3]),'h %.0f'%np.median(b[ok,3]-b[ok,1]),'w %.0f'%np.median(b[ok,2]-b[ok,0]))
