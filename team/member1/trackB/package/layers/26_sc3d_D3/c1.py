import json, glob, sys, re, numpy as np, pandas as pd
M=sys.argv[1]; OUTF=sys.argv[2]; H10F=sys.argv[3]; HP=1.75
R='.'
df=pd.read_parquet(R+'/data/test/test_dataset.parquet'); df['id0']=df.id-1 if 'id' in df else df.index
FC=pd.read_csv(R+'/geo/out/dpro_focal_test.csv').set_index('video_id')
h10=pd.read_csv(H10F); h10['id0']=h10.id-1
rows=h10[h10.question.str.contains(r'height of the streetlight',case=False)&(h10.video_source=='simulation')]
out=[]
for _,r in rows.iterrows():
    v=r.video_id; js=[json.load(open(f)) for f in sorted(glob.glob(f'{M}/{v}_*.json'))]
    if not js: continue
    W=js[0]['W']; Hh=js[0]['H']; c=(W/2,Hh/2)
    keyd={k:d for t,k,d in js[0]['dep'] if t is None and 'streetlight' in k}
    if not keyd: continue
    zf=lambda d,m,f: d*np.cos(np.arctan(np.hypot(m['cx']-c[0],m['cy']-c[1])/f))
    hum=[]
    for j in js:
        bikes=[b['box'] for b in j['det'].get('bicycle',[]) if b['score']>0.3]
        P=[p for p in j['det'].get('person',[]) if 'top' in p and (p['bot']-p['top'])>=1.8*(p['right']-p['left'])]
        for t,k,d in j['dep']:
            if k=='human' and P: hum.append((d,P[0]))
    f=1.4*W; src='own_person'
    if hum:
        for _ in range(5):
            fs=[(m['bot']-m['top'])*zf(d,m,f)/HP for d,m in hum]; f=float(np.median(fs))
        if np.ptp(fs)/f>0.05: continue
    else:
        f=float(FC.loc[v,'fw_moge'])*W; src='own_moge_focal'
    j0=[j for j in js if j['t']==0.0][0]
    sls=sorted([s for s in j0['det'].get('street light',[]) if 'top' in s and s['score']>0.3 and (s['bot']-s['top'])>3*(s['right']-s['left'])],key=lambda s:s['cx'])
    if not sls: continue
    hs=[]
    for k,d in keyd.items():
        s={'streetlight':max(sls,key=lambda s:s['bot']-s['top']),'left_streetlight':sls[0],'right_streetlight':sls[-1]}.get(k)
        if s is None or s['top']<=1 or s['bot']>=Hh-2: continue
        hs.append((s['bot']-s['top'])*zf(d,s,f)/f)
    if not hs: continue
    val=float(np.mean(hs))
    out.append(dict(id0=int(r.id0),value=round(val,4),rule=f'sc3d_sl_owndepthkey:{src}:f/W={f/W:.3f}:n={len(hs)}',h10=r.parsed_value,video=v))
o=pd.DataFrame(out); print(o.to_string())
o[['id0','value','rule']].to_csv(OUTF,index=False)
