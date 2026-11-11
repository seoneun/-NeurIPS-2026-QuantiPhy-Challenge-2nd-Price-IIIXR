import sys, cv2
sys.path.insert(0,'geo'); import track_common as tc
for vid in ['simulation_0220','simulation_0218_segmented']:
    fr=tc.load_video(f'data/test_480p/{vid}.mp4'); n=len(fr)
    for k,f in enumerate([n//4,n//2,3*n//4]):
        cv2.imwrite(f'cache/tmp/ag_c73pline_{vid}_{k}.jpg', cv2.cvtColor(fr[f],cv2.COLOR_RGB2BGR))
