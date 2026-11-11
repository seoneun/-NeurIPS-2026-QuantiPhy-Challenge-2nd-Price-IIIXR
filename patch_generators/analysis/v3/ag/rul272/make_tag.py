import os
# generator for this e1x tag: runs ../rul272/gen_e1x.py, which writes all six abplan premise tags under e1x names
# (rul275, rul274, rul273, e1xs3firstkey, e1xd3tdepth, e1xd3kint0) from sub_xt_Nzz46 + parse.
import runpy
runpy.run_path(os.path.join(os.environ.get('QP_ROOT', '.'), 'analysis/v3/ag/rul272/gen_e1x.py'), run_name='__main__')
