import pandas as pd, json, re
pat=re.compile(r'parking (space|spot|bay)|box formed by the white lines|marked .* on the (ground|floor)|painted (box|rectangle)',re.I)
for sp,f in [('test','submissions/sub_xt_Nzz20_detail.csv'),('val','submissions/sub_xt_Nzz20_valdetail.csv')]:
    d=pd.read_csv(f)
    P={}
    for l in open(f'geo/out/parse_{sp}_v2.jsonl'):
        j=json.loads(l); P[str(j.get('id',j.get('row_id')))]=j
    print(sp, d.columns.tolist()[:20], len(P))
    idcol=d.columns[0]
    for i,r in d.iterrows():
        rid=str(r[idcol]) if 'id' in idcol else str(i)
        if pat.search(str(r['question'])):
            print(sp,i,r.get('video_id'),r['cat'],r.get('kind'),r['answer_source'],r['answer'],r.get('g'),r.get('cs'),r.get('y',''),'|',r['question'][:200])
