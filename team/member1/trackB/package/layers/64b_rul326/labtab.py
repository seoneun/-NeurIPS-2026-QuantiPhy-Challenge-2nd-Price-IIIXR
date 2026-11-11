import csv,re,collections,sys
T={'green tennis ball':('d',6.7),'pink tennis ball':('d',6.7),'tennis ball':('d',6.7),'ping pong ball':('d',4.0),'yoga ball':('d',52.2),
'soccer ball':('d',17.5),'basketball':('d',23.2),'red plastic ball':('d',7.0),'red ball':('d',7.0),'orange plastic ball':('d',5.7),'orange ball':('d',23.2),
'small whiteboard':('box',(30.8,23.2,0.5)),'slope':('box',(30.8,23.2,0.5)),'large whiteboard':('box',(81,60,1)),'whiteboard':('box',(81,60,1)),
'tape':('cyl',(10.1,4.8)),'food box':('box',(19.5,6.0,7.2)),'white box':('box',(19.5,6.0,7.2)),
'cookie':('cyl',(8.0,1.8)),'cosmetic jar':('cyl',(7.2,6.0)),'glass jar':('cyl',(6.8,8.5)),'white cup':('cyl',(8.6,14.5)),'cup':('cyl',(8.6,14.5)),
'water bottle':('cyl',(6.7,19.5)),'pink bottle':('cyl',(6.7,19.5)),'bottle':('cyl',(6.7,19.5)),'marker':('box',(13.5,0.8,0.8)),'black pen':('box',(14.5,1.2,1.2)),'pen':('box',(14.5,1.2,1.2)),
'notebook':('box',(18.6,8.8,1.5)),'white-covered book':('box',(19.5,12.5,2.2)),'white book':('box',(19.5,12.5,2.2)),'pencil case':('box',(21,7.2,6.7)),'pencil bag':('box',(21,7.2,6.7)),
'credit card':('box',(8.56,5.398,0.07)),'card':('box',(8.56,5.398,0.07)),'poker':('box',(9.8,6.3,1.8)),'deck':('box',(9.8,6.3,1.8))}
KEYS=sorted(T,key=len,reverse=True)
def value(q):
    ql=q.lower()
    if not re.search(r'\b(diameter|radius|height|length|width|thickness|long|wide|tall|size)\b',ql): return None
    if re.search(r'\b(speed|velocity|acceleration|distance|travel|displacement|time|far|rebounce|bounce|reached|reach|maximum|tower|stack)\b',ql): return None
    obj=next((k for k in KEYS if k in ql),None)
    if not obj: return None
    kind,d=T[obj]
    unit=100.0 if re.search(r'in (meters|m)\b|\(m\)|in metres',ql) else (10.0 if ' mm' in ql or 'millimet' in ql else 1.0)
    unit = 1/100.0 if unit==100.0 else (10.0 if unit==10.0 else 1.0)
    if kind=='d':
        v=d/2 if 'radius' in ql else d
        if not re.search(r'diameter|radius|size|width|height',ql): return None
    elif kind=='cyl':
        dia,h=d
        if re.search(r'width|wide',ql) and obj=='tape': v=h
        elif re.search(r'diameter|width|wide',ql): v=dia
        elif re.search(r'radius',ql): v=dia/2
        elif re.search(r'height|tall|thickness|length|long',ql): v=h
        else: return None
    else:
        a=sorted(d,reverse=True)
        if re.search(r'thickness|thick',ql): v=a[2]
        elif re.search(r'length|long',ql): v=a[0]
        elif re.search(r'width|wide',ql): v=a[1]
        elif re.search(r'height|tall',ql): v=a[2] if obj not in ('trash','food box','white box','pencil case','pencil bag') else d[2]
        else: return None
    return v*unit, obj
if __name__=='__main__':
    P=list(csv.DictReader(open(sys.argv[1],newline='',encoding='utf-8')))
    out=[]
    for r in P:
        i=int(r['id'])-1
        if not r['video_id'].startswith('captured'): continue
        res=value(r['question'])
        if not res: continue
        v,obj=res; out.append((i,v,'rul326:'+obj))
    with open(sys.argv[2],'w',newline='') as f:
        w=csv.writer(f); w.writerow(['id0','value','rule']); w.writerows(out)
