import csv,json,re,sys,math,os
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); I=os.path.join(HERE,'inputs')
FX0,FY0=508.3,503.1
WIN=0.1
STATIC_FRAC=0.01
ZMIN,ZMAX=0.25,5.46
def norm(s): s=s.lower().replace('_',' '); s=re.sub(r'\bthe\b','',s); return re.sub(r'[^a-z]','',s)
MOT=r'speed|veloc|accel|distance|travel|displac|bounce|how far|path|moved'
def parse_q(q):
    ql=q.lower().strip()
    if re.search(MOT,ql): return None
    u=100.0 if re.search(r'\bin (cm|centimet)',ql) else 1000.0 if re.search(r'\bin (mm|millimet)',ql) else 1.0 if re.search(r'\bin (m|meters?|metres?)\b',ql) else None
    if u is None: return None
    m=re.search(r'what is the (height|length|width|diameter|radius|size) of (?:the )?(?:rim of (?:the )?)?(.+?) in ',ql) or \
      re.search(r"what is the (.+?)'s (height|length|width|diameter|radius|size) in ",ql)
    if not m: return None
    ax,obj=(m.group(1),m.group(2)) if m.re.pattern.startswith('what is the (h') else (m.group(2),m.group(1))
    return ax,obj.strip(),u
def parse_depth(d):
    out=[]
    for line in re.split(r'[\n;]',d):
        m=re.search(r'(?:t\s*=\s*([\d.]+)\s*s?\s*(?:\([^)]*\))?\s*,)?\s*distance_(.+?)_?(?:and_camera|camera)\s*=\s*([\d.]+)\s*(mm|cm|m)\b',line)
        if m: out.append((float(m.group(1)) if m.group(1) else None,m.group(2).strip(' _'),float(m.group(3))*{'m':1.0,'cm':0.01,'mm':0.001}[m.group(4)]))
    return out
def match(n,cands,key):
    n=norm(n); ex=[c for c in cands if norm(key(c))==n]
    if ex: return ex
    return [c for c in cands if n and (n in norm(key(c)) or norm(key(c)) in n)]
def compute(q,depth,path):
    pq=parse_q(q)
    if not pq: return None,'noq'
    ax,obj,u=pq
    if not path: return None,'notrack'
    d=json.load(open(path)); W,H,fps=d['W'],d['H'],d['fps']
    fx=FX0*W/853; fy=FY0*H/480; fm=(fx+fy)/2
    tq=[x for x in d['queries'] if 'target' in x['roles']]
    if len(tq)!=1: return None,'ntarget%d'%len(tq)
    tq=tq[0]
    if not match(obj,[tq['key'].split('|')[0]],lambda x:x): return None,'target_mismatch:%s/%s'%(obj,tq['key'])
    dk=[k for k in match(obj,parse_depth(depth),lambda x:x[1]) if ZMIN<=k[2]<=ZMAX]
    if not dk: return None,'nodepth:'+obj
    fr=[(j/fps,f['st']) for j,f in zip(d['fidx'],tq['frames']) if f and f.get('st') and f['st'].get('box')]
    if len(fr)<3: return None,'shorttrack'
    def ext(st):
        x0,y0,x1,y1=st['box']; a,b=st['rr'][0],st['rr'][1]
        if ax=='height': return (y1-y0)/fy
        if ax in ('diameter','radius','size'): return math.sqrt(4*st['area']/math.pi)/fm*(0.5 if ax=='radius' else 1)
        if ax=='length': return max(a,b)/fm
        if ax=='width': return min(a,b)/fm
    vals=[]
    for t,_,Z in dk:
        if t is not None:
            sel=[st for tt,st in fr if abs(tt-t)<=WIN+1e-6 and not st.get('border')]
        else:
            if any(x[0] is not None for x in dk): continue
            C=np.array([st['c'] for _,st in fr]); sp=np.max(np.linalg.norm(C-np.median(C,0),axis=1))
            if sp>STATIC_FRAC*W: return None,'moving_untimed:%.0fpx'%sp
            sel=[st for _,st in fr if not st.get('border')]
        if len(sel)<2: continue
        vals.append(float(np.median([ext(st) for st in sel]))*Z*u)
    if not vals: return None,'noframes'
    return float(np.median(vals)),'ok:%s:%s:k%d'%(ax,tq['key'],len(vals))
if __name__=='__main__':
    pred_t019,h34src,out=sys.argv[1:4]
    rm={k:os.path.join(HERE,v) for k,v in json.load(open(os.path.join(I,'rowmap_test_v32_D3lab.json'))).items()}
    srcr={int(z['id0']):z for z in csv.DictReader(open(h34src,encoding='utf-8'))}
    sub={int(z['id'])-1:z for z in csv.DictReader(open(pred_t019,newline='',encoding='utf-8'))}
    cand=[]
    for i,r in sorted(sub.items()):
        if r['video_source']!='lab': continue
        s=srcr[i]
        if not s['cat'].startswith('D3'): continue
        g,why=compute(r['question'],r['depth_info'],rm.get(str(i)))
        if g is None: continue
        cur=float(r['parsed_value']); h=float(s['value'])
        src=s['src'] if abs(cur-h)<=1e-9*max(1,abs(cur)) else 'PRED_T019_CHANGED'
        if re.match(r'(64|64b|67|69|76)_',src) or src.startswith('79V_rul449'): continue
        if 0.5<=g/cur<=2: cand.append((i,math.sqrt(cur*g)))
    with open(out,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f,lineterminator=chr(10)); w.writerow(['id0','value']); w.writerows(cand)
