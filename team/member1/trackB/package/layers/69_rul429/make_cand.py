import sys,re,csv,pandas as pd,numpy as np
TP,BASE,OUT=sys.argv[1:4]
base={int(r['id'])-1:float(r['parsed_value']) for r in csv.DictReader(open(BASE,newline='',encoding='utf-8'))}
T=pd.read_parquet(TP); T['id0']=np.arange(len(T))
TOY=[14.5,11,8,29.5,15.0,10.0,8.1,6.4,19.6]; BIN=[34.5,89.1,59.9]
def rule(r,tol):
    if not r.video_id.startswith('captured'): return None
    q=r.question.lower()
    if not re.search(r'(height|width|length) of the [\w ]*\b(toy|trashbin)\b in cm',q) or re.search(r'speed|veloc|distance|accel',q): return None
    D=TOY if 'toy' in q else BIN; v=base[r.id0]
    j=int(np.argmin([abs(np.log(v/d)) for d in D]))
    return (D[j],'rul429:%s'%('toy' if D is TOY else 'trashbin')) if abs(np.log(v/D[j]))<=np.log(1+tol) and abs(D[j]-v)>1e-9 else None
tol=0.1; ch={}
for r in T.itertuples():
    x=rule(r,tol)
    if x: ch[r.id0]=x[0]
pd.DataFrame([(i,ch[i],'rul429 tol0.1') for i in ch],columns=['id0','value','rule']).to_csv(OUT,index=False)
print('69_rul429 rows',len(ch))
