import pandas as pd
d=pd.read_csv('submissions/sub_xt_Nzz20_detail.csv').set_index('Unnamed: 0')
pd.set_option('display.width',250)
print(d[d.video_id.str.match(r'simulation_02(18|19|20)')][['video_id','question','answer','answer_source','g','cs','mode']].to_string())
import sys; sys.path.insert(0,'cache/tmp')
from lab_rows import labelled
L=labelled('sub_xt_Nzz20'); print('labelled rows in these videos:', L[L.video_id.str.match(r'simulation_02(18|19|20)')].shape[0])
