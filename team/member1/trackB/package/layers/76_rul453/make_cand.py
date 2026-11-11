import csv,re,math,sys
BASE,H1,OUT=sys.argv[1:4]
def rule(r,cur):
    q=r['question'].lower()
    if not r['video_id'].startswith('captured'): return None
    if not re.search(r'\bwhite ball', q): return None
    if not re.search(r'diameter|radius|size', q): return None
    if re.search(r'veloc|speed|displace|distance|travel|height reached|acceler|between', q): return None
    unit = 0.01 if re.search(r'in meters|\(m\)| in m\b',q) else (10.0 if re.search(r'\bmm\b|millimet',q) else 1.0)
    d = 4.0*(0.5 if 'radius' in q else 1.0)
    tgt = d/ (1.0 if unit==1.0 else (100.0 if unit==0.01 else 0.1))
    if cur<=0 or abs(math.log(cur/tgt))>math.log(1.25): return None
    return tgt
S=list(csv.DictReader(open(BASE,encoding='utf-8')))
def run(rows):
    return {r['id']:rule(r,float(r['parsed_value'])) for r in rows}
out=run(S)
H={int(r['id0']):r for r in csv.DictReader(open(H1,encoding='utf-8'))}
base={int(r['id'])-1:r['parsed_value'] for r in S}
rows=[]
for r in S:
    i=int(r['id'])-1; v=out[r['id']]
    if v is None or i not in H: continue
    if abs(v-float(base[i]))<1e-12: continue
    rows.append((i,v,'paper2_rul453'))
with open(OUT,'w',newline='') as f:
    w=csv.writer(f); w.writerow(['id0','value','rule']); [w.writerow([i,repr(float(v)),t]) for i,v,t in rows]
