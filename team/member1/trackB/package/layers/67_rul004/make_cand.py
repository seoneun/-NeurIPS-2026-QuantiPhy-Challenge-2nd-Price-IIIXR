import csv,re,math,sys,os
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','64b_rul326'))
import labtab
BASE,OUT=sys.argv[1:3]
BALLS=[4.0,5.7,6.7,7.0,17.5,23.2,52.2]
SIZE=re.compile(r'\b(diameter|radius|size)\b',re.I); MOT=re.compile(r'speed|veloc|accel|distance|displace|travel|reach|far|maximum|bounce|height|time',re.I)
def unit(q):
    ql=q.lower()
    if re.search(r'\bin (mm|millimet)',ql) or '(mm)' in ql: return 10.0
    if re.search(r'\bin (m|meters|metres)\b',ql) or '(m)' in ql: return 0.01
    return 1.0
def rule(video,q,cur,gate=1.25):
    if not str(video).startswith('captured'): return None
    if not re.search(r"\bthe ball('s)?\b",q,re.I) or not SIZE.search(q) or MOT.search(q): return None
    if labtab.value(q) is not None: return None
    u=unit(q); rad='radius' in q.lower()
    x=float(cur)/u*(2 if rad else 1)
    if not x>0: return None
    d=min(BALLS,key=lambda b:abs(math.log(x/b)))
    if abs(math.log(x/d))>math.log(gate): return None
    return d*u/(2 if rad else 1), 'tabsnap:%.1f'%d
P=list(csv.DictReader(open(BASE,newline='',encoding='utf-8')))
out=[]
for r in P:
    i=int(r['id'])-1; res=rule(r['video_id'],r['question'],r['parsed_value'])
    if res and abs(res[0]-float(r['parsed_value']))>1e-9: out.append((i,res[0],res[1]))
with open(OUT,'w',newline='') as f:
    w=csv.writer(f); w.writerow(['id0','value','rule']); w.writerows(out)
