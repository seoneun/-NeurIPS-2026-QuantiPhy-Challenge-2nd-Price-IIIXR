import json,csv,re,sys,os
_N = {'c2': 'pred_t015', 'c3a': 'pred_t115', 'c3b': 'pred_t116', 'c3m': 'pred_t117', 'c4': 'pred_t118', 'c4m': 'pred_t119', 'c4s': 'pred_t120', 'c5': 'pred_t121', 'c5m': 'pred_t021', 'c6': 'pred_t122', 'c7': 'pred_t123', 'c8': 'pred_t124', 'c9': 'pred_t125', 'c9b': 'pred_t126', 'c9n': 'pred_t127', 'c9u': 'pred_t128', 'g2': 'pred_t129', 'g2a': 'pred_t130', 'g3': 'pred_t131', 'g3a': 'pred_t132', 'i1': 'pred_t133', 'j1': 'pred_t134', 'j6': 'pred_t135', 'j7': 'pred_t136', 'j7s': 'pred_t137', 'j9n': 'pred_t138', 'm3': 'pred_t139', 'p1': 'pred_t140', 'r1': 'pred_t141', 's1': 'pred_t142', 'y1': 'pred_t143', 'y1s': 'pred_t144', 'z1': 'pred_t145', 'z1s2': 'pred_t146', 'z1s2c': 'pred_t004'}
sys.stdout.reconfigure(encoding='utf-8')
G=os.path.dirname(os.path.abspath(__file__))+'/../'; OUT=sys.argv[1].rstrip('/')+'/'; B=G
src=sys.argv[2]
with open(src,encoding='utf-8',newline='') as f:
    rd=csv.DictReader(f); FN=rd.fieldnames; rows=list(rd)
P={'S2':'in/r66_P_S2_patch_test.json','D2':'in/r66_P_D2_patch_test.json','S3':'in/r66_P_S3_patch_test.json','D3':'in/r66_P_D3_patch_test.json'}
XQ={'jump','rul441','rul022','rul062','rul073','rul271','rul291'}
SUSP={305,3145}
pt=[]
for c,f in P.items():
    for k,v in json.load(open(B+f,encoding='utf-8')).items():
        g=v.get('rule') or re.sub(r'^r66 D2 (\S+?):?\s.*',r'\1',v['why'])
        g=re.sub(r'\[own\]','',g); g=re.sub(r'^P_','',g).rstrip(':'); g=re.sub(r'^patch:','',g)
        i=int(k)
        pt.append(dict(i=i,c=c,g=g,v=v['value'],b=v['before']))
VAR={'y1':lambda p:True,
     'y1s':lambda p:p['g'] not in XQ and p['i'] not in SUSP}
for name,sel in VAR.items():
    fn=OUT+f'{_N[name]}_id1.csv'
    R=[dict(r) for r in rows]
    for p in pt:
        if sel(p): R[p['i']]['parsed_value']=repr(float(p['v']))
    with open(fn,'w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FN); w.writeheader(); w.writerows(R)
