import json, os, csv, sys
R='@QP_ROOT@'
def frames_of(t):
    res={}
    if 'objs' in t:
        o2k={}
        for q in t.get('queries',[])+t.get('queries_d',[]):
            o2k.setdefault(str(q.get('obj')),[]).append(q['key'])
        for ok,fr in t['objs'].items(): res['obj'+ok+':'+'/'.join(sorted(set(o2k.get(ok,[]))))[:60]]=fr
    else:
        for q in t['queries']: res[q['key']]=q.get('frames') or []
    return res
for v in sys.argv[1:]:
    rows=[]
    for tag,p in [('M',f'{R}/geo/out/tracks_test_v2sc/{v}.json'),('M1',f'{R}/geo/out/tracks_test/{v}.json'),('V',f'{R}/geo/out/tracks_test_v3ntmv/{v}.json')]:
        if not os.path.exists(p): continue
        t=json.load(open(p))
        for key,frs in frames_of(t).items():
            for i,fr in enumerate(frs):
                st=fr.get('st') if isinstance(fr,dict) else None
                if not st: continue
                rows.append(dict(src=tag,key=key,fidx=t['fidx'][i],t=round(t['fidx'][i]/t['fps'],4),cx=round(st['c'][0],2),cy=round(st['c'][1],2),
                    x0=st['box'][0],y0=st['box'][1],x1=st['box'][2],y1=st['box'][3],area=st['area'],rrl=round(st['rr'][0],1),rrs=round(st['rr'][1],1),rra=round(st['rr'][2],1)))
        print(v,tag,'fps',t['fps'],'N',t['N'],'W',t['W'])
    fn=f'{R}/cache/tmp/rul388/boxtracks_{v}.csv'
    with open(fn,'w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    from collections import defaultdict
    g=defaultdict(list)
    for r in rows: g[(r['src'],r['key'])].append(r)
    for k,rs in g.items():
        a=rs[0]; b=rs[-1]
        print('  ',k,'n',len(rs),'first',a['fidx'],(a['cx'],a['cy']),(a['x1']-a['x0'],a['y1']-a['y0']),'last',b['fidx'],(b['cx'],b['cy']),(b['x1']-b['x0'],b['y1']-b['y0']))
