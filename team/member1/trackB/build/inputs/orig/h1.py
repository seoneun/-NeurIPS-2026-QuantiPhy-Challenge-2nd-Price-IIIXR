import csv,collections,json,sys
B='blk/analysis/v3/calib/'; S='blk/submissions/'
def sub(p):
    with open(p,newline='',encoding='utf-8') as f: return {int(r['id'])-1:float(r['parsed_value']) if r['parsed_value'] not in ('','nan') else None for r in csv.DictReader(f)}
P17=sub(S+'pred_t003_id1.csv'); B17=sub(S+'pred_t013_id1.csv'); B24=sub(S+'pred_t007_id1.csv')
B26c=sub(S+'pred_t014_id1.csv'); Z=sub('../../pred_t004_id1.csv')
st={int(r['id0']):r for r in csv.DictReader(open('rows_status.csv',encoding='utf-8'))}
CHAIN=['pred_t012','pred_t006','pred_t007','pred_t008','pred_t009','pred_t010','pred_t011','pred_t003']
ADOPT={'pred_t012':{'claimA','w1_deskcorner','w1_labradius','rul345'}}
writes=collections.defaultdict(list); known=set()
for f in CHAIN:
    for r in csv.DictReader(open(B+f+'_rows.csv',encoding='utf-8')):
        if f in ADOPT and r['block'] not in ADOPT[f]: continue
        if f=='pred_t007' and r['block'] in ('w1_deskcorner','w1_labradius'): continue
        i=int(r['id0']); writes[i].append((f,r['block'],float(r['old']),float(r['new']))); known.add(i)
w1=[i for i in st if B17.get(i)!=B24.get(i) and i not in known]
print('w1-era rows', len(w1), collections.Counter((st[i]['cat'],st[i]['status']) for i in w1))
KEEP={'claimA','claimC','claimC_S3','rul345','rul348','d3labF','d3labF2E1','d3labF2E2','d3labF2E3','s3dist31','s3dist32'}
MULT={'d3labF','d3labF2E1','d3labF2E2','d3labF2E3','s3dist31','s3dist32'}
GEN_SCRIPTS=('rul383','rul375','s2card','s2fps74')
aud={int(r['id0']):r for r in csv.DictReader(open('r4r5_rows.csv',encoding='utf-8'))}
def keep_block(i,f,b):
    if b in KEEP: return True
    if b.startswith(('r5_','r4_','held')):
        a=aud.get(i); return bool(a and any(g in a['script'] for g in GEN_SCRIPTS))
    return False
H=dict(P17); why=collections.Counter(); fall=0; nc_rows=0
for i,r in st.items():
    if r['status']!='S':
        H[i]=Z[i]; why['status_'+r['status']]+=1; continue
    ws=writes.get(i,[])
    first=next((k for k,w in enumerate(ws) if not keep_block(i,w[0],w[1])),None)
    if first is None: continue
    nc_rows+=1; v=ws[first][2]; why['nc_'+ws[first][1]]+=1
    for f,b,o,n in ws[first+1:]:
        if keep_block(i,f,b): v = v*(n/o) if (b in MULT and o) else n
    H[i]=v
print('NC block rows (status S):',nc_rows); print(why.most_common())
for i in w1:
    if st[i]['status']=='S': print('w1 S row',i,st[i]['cat'],B17[i],B24[i],P17[i])
rows=list(csv.reader(open(S+'pred_t003_id1.csv',newline='',encoding='utf-8'))); vi=rows[0].index('parsed_value'); ch=0
for r in rows[1:]:
    i=int(r[0])-1
    if H.get(i)!=P17.get(i): r[vi]=repr(H[i]); ch+=1
csv.writer(open('pred_t005_id1.csv','w',newline='',encoding='utf-8'),lineterminator='\n').writerows(rows)
print('H1 changed vs pred_t003:',ch)
json.dump({str(k):v for k,v in H.items()},open('h1_vals.json','w'))
with open('h1_rows.csv','w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['id0','cat','status','reason','p17','h1','z1s2c','b17v'])
    for i,r in sorted(st.items()):
        ws=writes.get(i,[]); first=next((w_[1] for k,w_ in enumerate(ws) if not keep_block(i,w_[0],w_[1])),'')
        reason = r['status'] if r['status']!='S' else (('nc:'+first) if first else 'S')
        w.writerow([i,r['cat'],r['status'],reason,P17[i],H[i],Z[i],B17[i]])
