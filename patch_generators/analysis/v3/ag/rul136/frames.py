import sys, cv2, json
sys.path.insert(0,'geo'); import track_common as tc
import pandas as pd
raw=pd.read_csv('preds/test_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv')
for vid in ['simulation_0218','simulation_0219','simulation_0220','simulation_0218_segmented']:
    fr=tc.load_video(f'data/test_480p/{vid}.mp4')
    n=len(fr); print(vid,n,fr[0].shape, raw[raw.video_id==vid].fps.iloc[0])
    for k,f in enumerate([n//4,n//2,3*n//4]):
        cv2.imwrite(f'cache/tmp/ag_c73pline_{vid}_{k}.jpg', cv2.cvtColor(fr[f],cv2.COLOR_RGB2BGR))
    t=json.load(open(f'geo/out/tracks_test_v2sc/{vid}.json'))
    print(' queries',t.get('queries'),t.get('queries_d'),'fidx',t['fidx'][:3],len(t['fidx']))
