# rul082 main: writes cache/tmp/ag_c48gflb_patch_{test,val}.json (final answers; floor only, never lowers).
import sys, json, os, pandas as pd, numpy as np
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
import importlib.util
spec=importlib.util.spec_from_file_location('gf',R+'/analysis/v3/ag/rul082/c48gflb_lib.py'); gf=importlib.util.module_from_spec(spec); spec.loader.exec_module(gf)
load_parse=gf.load_parse; floor_for=gf.floor_for; UNIT=gf.UNIT
SUB=sys.argv[1] if len(sys.argv)>1 else 'sub_xt_Nzz4'
out={}
for split in ('test','val'):
    P=load_parse(split)
    det=pd.read_csv(R+'/submissions/%s_%s.csv'%(SUB,'detail' if split=='test' else 'valdetail'),index_col=0)
    patch={}
    for rid,j in P.items():
        q=j.get('q') or {}
        if q.get('kind')!='speed' or q.get('t') is None: continue
        if j.get('video_source')!='lab' or rid not in det.index: continue
        tr=R+'/geo/out/tracks_%s_v2sc/%s.json'%(split,j['video_id'])
        if not os.path.exists(tr): continue
        J=json.load(open(tr))
        f=floor_for(j,J)
        if f is None: continue
        sc=UNIT.get(q.get('unit') or 'm/s')
        if sc is None: continue
        ans=float(det.loc[rid,'answer'])
        new=f['vlb']*sc
        if new>ans*1.05:
            patch[str(rid)]={'value':float(new),'why':'rul082: v>=g|t-t_apex| (apex %.3fs, %s, win %.2f-%.2f n%d, fz %.2fW, %s dir %s); was %.4g'%(f['tap'],f['phase'],f['win'][0],f['win'][1],f['n'],f['fzW'],f['obj'],f['dsrc'],ans)}
    json.dump(patch,open(R+'/cache/tmp/ag_c48gflb_patch_%s.json'%split,'w'),indent=1)
    out[split]=patch
    print(split, len(patch))
    for k,v in patch.items(): print(' ',k,P[int(k)]['video_id'],P[int(k)]['cat'],'|',P[int(k)]['question'].strip()[:60],'|',round(v['value'],4),'|',v['why'])
