import json, csv, sys, os
O, OUT = sys.argv[1:3]; H = os.path.dirname(os.path.abspath(__file__))
src = {int(r['id0']): r for r in csv.DictReader(open(f'{H}/inputs/h21_src.csv'))}
C = []


def add(f, rule, skip=()):
    for k, v in json.load(open(f'{O}/{f}')).items():
        i = int(k)
        if i not in skip:
            C.append((i, float(v['value']), rule))


add('c45own_test.json', 'c45own(rul076, length-sibling gate dropped)')
add('c53own_test.json', 'c53own(rul094 v2, own static P-track gate; 2099 = own rul291 g x k)')
add('c65own_test.json', 'c65own(rul113 part1, own geo/CaW + own fr)')
add('c79own_test.json', 'c79own(rul151, own ruler~rs replaces echo witness)')
add('flatw2own_test.json', 'flatw2own(rul291, length-sibling gate dropped, no copy pool)', skip=(2099,))
add('ag_rul290_patch_test.json', 'rul290(own-row rul289, own tshift)')
add('ag_rul206_patch_test.json', 'rul206(own-row, witness dropped, 2x cs gate)')
add('ag_rul023_patch_test.json', 'rul023(own-row, length-sibling dropped)')
good = [(i, v, rule) for i, v, rule in C if 'z1s2c' in src[i]['src'] and abs(v / float(src[i]['value']) - 1) > 0.005]
with open(OUT, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['id0', 'value', 'rule'])
    for i, v, rule in sorted(good):
        w.writerow([i, repr(v), rule])
