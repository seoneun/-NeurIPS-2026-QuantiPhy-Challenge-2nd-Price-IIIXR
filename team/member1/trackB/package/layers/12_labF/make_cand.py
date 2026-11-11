import sys, re, numpy as np, pandas as pd
TF, FRM, H5, H5SRC, OUT = sys.argv[1:6]
MO=['speed','dist2','disp','path','accel','maxheight','avgspeed']
tf=pd.read_pickle(TF); REC={(r.split,r.id0):r for r in tf.itertuples()}
FR=pd.read_csv(FRM).set_index('id0')
def rule(split,id0,cat,video,question,value):
    if not str(video).startswith('captured') or cat not in ('S3','D3'): return None,'notlab'
    r=REC.get((split,id0))
    if r is None or r.kind not in MO or not (r.F==r.F): return None,'norec/notmotion'
    if r.F<1.15: return None,'radial'
    if r.mode=='3d_knownsize': return None,'knownsize'
    if split=='test' and cat=='D3' and id0 in FR.index and FR.Fp.get(id0,np.nan)>=1.15: return None,'lateral_prior_absorbs'
    if r.kind=='dist2' and not re.search(r'\d\s*s\b',question): return None,'static_pair'
    vg=value/r.g if r.g==r.g and r.g else np.nan
    if not (0.4<=vg<=1.05): return None,'value_not_plain(vg=%.2f)'%vg
    return value*r.F,'labF: value x own F %.4f (vg %.3f, %s)'%(r.F,vg,r.mode)
h5=pd.read_csv(H5); h5['id0']=np.arange(len(h5)); s0=pd.read_csv(H5SRC); CAT=s0.set_index('id0').cat.to_dict()
cat={i:CAT.get(i) for i in range(len(h5))}
out=[]
for i,rw in h5.iterrows():
    n,why=rule('test',i,cat[i],rw.video_id,rw.question,rw.parsed_value)
    if n is not None: out.append((i,n,why))
pd.DataFrame([(i,n,'labF') for i,n,why in out],columns=['id0','value','rule']).to_csv(OUT,index=False)
