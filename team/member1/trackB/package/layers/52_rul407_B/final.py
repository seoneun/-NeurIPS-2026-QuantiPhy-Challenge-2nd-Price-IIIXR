import os,sys,json,re,csv,numpy as np,pandas as pd
Q,ZF,OUT=sys.argv[1:4]
os.environ['QP_ROOT']=os.getcwd()
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'modules'))
import rul320 as N
import rule_width as W
N.R=Q; W.R=Q
raw=pd.read_csv(Q+'/preds/test_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv',index_col=0)
CAW=pd.read_csv(Q+'/preds/test_caw9b_orig.csv').parsed_value.values
CROP=pd.read_csv(Q+'/preds/test_caw9b_orig_crop.csv').parsed_value.values
DET={k:pd.read_csv(Q+f'/submissions/{k}_detail.csv',index_col=0) for k in ('pred_t088','pred_t091','pred_t087')}
def cs_own(i,sub):
    c=float(DET[sub].cs[i])
    for b in (CAW[i],CROP[i]):
        for k in (1.0,1.12):
            if abs(c/(float(b)*k)-1)<1e-9: return c
    raise ValueError('')
UD=pd.read_csv(Q+'/cache/tmp/ag_rul439_diag_test.csv',index_col=0)
P={}
for l in open(Q+'/geo/out/parse_test_v2.jsonl'):
    j=json.loads(l); P[int(j['id'])]=j
CATS={int(r['id'])-1:r['inference_type'][0]+r['video_type'][1] for r in csv.DictReader(open(ZF,newline='',encoding='utf-8'))}
def dcat(i): return CATS[i]
def ngA(i):
    g=raw.loc[i]; p=P[i]; q=p['q']; t=p['target']
    if not (q.get('kind')=='size' and str(g.meas_why).startswith('2-decade sanity rule')): return None
    pr=(p.get('priors') or [{}])[0]
    if not (pr.get('kind')=='accel'): return None
    key=(t.get('phrase') or '')+'|'+','.join(t.get('where') or [])+('|'+t['rel'][0]+':'+t['rel'][1] if t.get('rel') else '')
    e=N.ext_stats('test',g.video_id,key); dim=q.get('dim') or 'length'
    used={'height':e['bh'],'length':e['rr0'],'size':e['rr0'],'wingspan':e['rr0'],'width':e['rr1'],'thickness':e['rr1'],'side':e['rr1'],
          'diameter':e['rr1p75'],'radius':e['rr1p75']/2}.get(dim,e['rr0'])
    if abs(np.log(used*float(g.meas_mpp)/(float(g.meas_value)*N.UNIT_SI[q['unit']])))>=0.1: return None
    if q.get('horizontal') and dim in ('length','width','size'): used=e['bw']
    if used<N.MIN_PX: return None
    if dim in ('width','diameter','thickness','side') and e['bw']>=N.FLAT*e['bh'] and not q.get('horizontal'): return None
    gsi=used*float(g.meas_mpp)
    if not (N.PL_SIZE[0]<=gsi<=N.PL_SIZE[1]): return None
    gn=gsi/N.UNIT_SI[q['unit']]; cs=cs_own(i,'pred_t091')*(1.12 if str(g.video_id).startswith('simulation') and dcat(i)=='D2' else 1.0)
    w=0.7 if dcat(i)=='D2' else 0.65
    return gn**w*cs**(1-w) if max(gn,cs)/min(gn,cs)<=2 else gn
def uncut(i):
    d=UD.loc[i]; g=raw.loc[i]
    if pd.notna(d.sh_id) or pd.notna(d.sh_d): return None
    b=DET['pred_t088'].loc[i]
    if abs(b.g/float(g.geo_value)-1)<1e-4 and b.answer_source in ('dis_geo','dis_motion_geo','agree_geo_kind','cqc_geo','geo_cawinvalid') and b.kind=='size':
        gn=float(g.geo_value)*d.old_px/d.chord_run
        if not np.isfinite(DET['pred_t088'].cs[i]): return gn
        cs=cs_own(i,'pred_t088')
        return gn**0.7*cs**0.3 if (b.answer_source=='dis_geo' and abs(np.log(gn/cs))<=np.log(2)) else gn
    if not (abs(b.g/float(g.geo_value)-1)<1e-4 and b.answer_source in ('agree_wgm','agree_gm') and b.kind!='dist2'): return None
    gn=float(g.geo_value)*d.old_px/d.chord_run; cs=cs_own(i,'pred_t088'); gg=float(b.g); a=float(b.answer)
    if abs(np.log(gn/cs))>np.log(2): return gn
    w0=0.7 if b.kind=='size' else 0.5
    w=np.log(a/cs)/np.log(gg/cs) if abs(np.log(gg/cs))>1e-6 else w0
    if not (0.3<=w<=0.9): w=w0
    return a*(gn/gg)**w
TAILQ=re.compile(r'including (its|the) tail',re.I)
def tail(i):
    g=raw.loc[i]; q=str(g.question); gv=float(g.geo_value); cs=float(CAW[i])
    if not (re.search(r'\blength\b',q,re.I) and TAILQ.search(q) and str(g.meas_mode)=='3d_calib'): return None
    return gv if (gv>cs and gv/cs<=1.6) else None
def wing(i):
    o=P[i]; q=o['q']; ent=o.get('target') or {}; r=raw.loc[i]
    if not (W.WINGED.search((ent.get('phrase') or '').lower()) and q.get('dim') in ('width','wingspan')): return None
    k=W.key(ent); mppu=r.meas_mpp/W.USI.get(q.get('unit') or 'm',1.0)
    m,T,tj=W.find_track('test',r.video_id,k,q.get('dim'),ent.get('phrase'),r.meas_value/mppu)
    B=np.array([T.st[j]['box'] for j in T.oke],float); bw,bh=B[:,2]-B[:,0],B[:,3]-B[:,1]
    c=np.asarray(T.uv,float); dx,dy=abs(c[-1,0]-c[0,0]),abs(c[-1,1]-c[0,1])
    size=float(np.median([T.st[j]['rr'][0] for j in T.oke]))
    if max(dx,dy)<size: return None
    gn=float(np.percentile(bh if dx>=dy else bw,90))*mppu
    try: cs=cs_own(i,'pred_t087')
    except ValueError: cs=float(CROP[i])*1.12
    return W.fuse(gn,cs)[0]
RULES=[(i,ngA,'rul320_A_own') for i in (513,514,516,517,3073)]+[(i,uncut,'rul441_own') for i in (226,227,228,229,356,416,417,419,420,421,636,637,640)]+[
       (1592,tail,'rul227_own'),(1604,tail,'rul227_own'),(1815,wing,'wing_W2_own')]
full={i:f(i) for i,f,_ in RULES}; RULES=[x for x in RULES if full[x[0]] is not None]
with open(OUT,'w',newline='') as fo:
    Wc=csv.writer(fo); Wc.writerow(['id0','value','rule']); [Wc.writerow([i,repr(float(full[i])),r]) for i,f,r in RULES]
