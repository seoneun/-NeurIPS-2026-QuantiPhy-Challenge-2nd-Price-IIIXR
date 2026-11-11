import subprocess, sys, csv, os
R='@QP_ROOT@'; A=f'{R}/analysis/v3/ag/rul388'; O=f'{R}/cache/tmp/rul388'
import os
R5FZ = os.environ.get('R5_FROZEN', '1') != '0'
T5 = f'{R}/analysis/v3/ag/r5_common/frozen' if R5FZ else f'{R}/cache/tmp/r5'
os.makedirs(O,exist_ok=True)
def run(*a): return subprocess.run([sys.executable,*a],capture_output=True,text=True,check=True).stdout
run(f'{A}/boxtracks.py','simulation_0141'); run(f'{A}/walls0141.py'); meas=run(f'{A}/walls0141_measure.py'); fin=run(f'{A}/walls0141_final.py','--v')
val={int(l.split()[0]):float(l.split()[1]) for l in fin.splitlines() if l[:4] in ('1510','1511')}
free={r['id0']:r for r in csv.DictReader(open(f'{T5}/free_rows.csv'))}
val[1509]=float(free['336']['c26c'])
for k in (1509,1510,1511): print(k, round(val[k],4))
