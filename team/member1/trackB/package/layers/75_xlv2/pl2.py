import sys,re,csv,random,numpy as np
import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import bird2
ROLE={'plane':'big','car':'small'}
import cv2
_F={}
def flat(v):
    if v not in _F:
        c=cv2.VideoCapture(bird2.R+'/data/test/%s.mp4'%v); ok,im=c.read()
        med=np.median(im.reshape(-1,3),0); _F[v]=float((np.abs(im.astype(int)-med).sum(2)<30).mean())
    return _F[v]
def rule(q,prior,fps,M):
    pm=re.search(r'length of the (plane|car)\s*=\s*([\d.]+)\s*m',prior)
    if not pm: return None,'prior?'
    tg=re.search(r'of the (plane|car)',q)
    if not tg: return None,'tgt?'
    P=M[ROLE[pm.group(1)]]; T=M[ROLE[tg.group(1)]]
    k=float(pm.group(2))/P['L']
    if re.search(r'speed',q):
        tm=re.search(r'at ([\d.]+)\s*s',q); n=round(fps*float(tm.group(1)))
        d=T['dx'].get(n); why='bd n=%d'%n
        if d is None: d=T['v_px']; why+=' (median)'
        return abs(d)*fps*k,'speed_'+why
    if re.search(r'length',q): return T['L']*k,'length_med'
    if re.search(r'height',q): return float(np.median(T['h'])) *k if 'h' in T else None,'height'
    return None,'dim?'
if __name__=='__main__':
    sub,outp=sys.argv[1:3]
    rows=[r for r in csv.DictReader(open(sub,encoding='utf-8')) if re.search(r'length of the (plane|car)\s*=',r['ground_truth_prior']) and re.search(r'of the (plane|car)',r['question']) and r['video_type'].startswith('S2')]
    cache={}; res=[]
    for r in rows:
        v=r['video_id']
        if v not in cache:
            try: cache[v]=bird2.measure(v)
            except Exception as e: cache[v]=None; print('fail',v,repr(e)[:80])
        if cache[v] is None: continue
        m=cache[v]
        if m['big'] is m['small'] or m['big']['n']<5 or m['small']['n']<5: print('gate_tracks',v); continue
        if flat(v)<0.9: print('gate_flat',v,flat(v)); continue
        print('#',v,{s:{x:round(m[s][x],1) for x in ('area','v_px','L','Lmin','Wmax','n')} for s in ('big','small')})
        val,why=rule(r['question'],r['ground_truth_prior'],float(r['fps']),m)
        print('ROW',int(r['id'])-1,v,r['question'][:60],val,why)
        if val is None or not np.isfinite(val): continue
        res.append((int(r['id'])-1,float(val),'flat_s2:'+why))
    w=csv.writer(open(outp,'w',newline='')); w.writerow(['id0','value','rule'])
    for x in sorted(res): w.writerow([x[0],repr(x[1]),x[2]])
