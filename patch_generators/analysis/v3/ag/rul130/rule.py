"""rul130: coverage extension of the accepted echo rule (analysis/v3/ag/echo/rule.py, patch ag_echo2).
CaW says k x (stated size-prior number), k in {2, 0.5}, on a speed / avgspeed question: a speed cannot legitimately
equal a multiple of a length, so that CaW answer carries no measurement of the asked quantity -> geo alone (g).
Gates: non-integer prior number; |said/(k n) - 1| <= 1 %; q.kind in {speed, avgspeed}; priors[0].kind == size;
not same_quantity, not kin_explained; answer_source in echo.CAW_SRC (not kin / patch); geo finite > 0, meas_valid != False;
mode strong (2d_ratio / 3d_calib), or vlm_2d_ratio with head(q.target) == head(priors[0].obj) (self-ruler).
var_ans_plain copies of a fired plain row follow it.  usage: python rule.py [sub]
"""
import sys, json, numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
SUB = sys.argv[1] if len(sys.argv) > 1 else 'sub_xt_Nzz19'
_argv = sys.argv
sys.argv = ['echo', SUB, 'main']
sys.path.insert(0, R + '/analysis/v3/ag/echo')
import rule as E
sys.argv = _argv
from variant_keys import split_variant
TAG = 'rul130'
KS = (2.0, 0.5)


def run(split):
    D, P = E.load(split, SUB)
    rec = []
    for i in D.index:
        p = P[i]
        q = p['q']
        pr = p['priors'][0] if p['priors'] else {}
        said = E.said_value(D, i)
        r = dict(id=i, said=said, k=np.nan, num=np.nan, hit=False, selfr=False)
        if np.isfinite(said) and q.get('kind') in ('speed', 'avgspeed') and pr.get('kind') == 'size':
            ns = E.prior_numbers(D.ground_truth_prior[i])
            if ns:
                n = ns[0]
                if abs(n - round(n)) > 1e-9:
                    for k in KS:
                        if abs(said / (k * n) - 1) <= 0.01:
                            if not E.same_quantity(p) and not E.kin_explained(p, said, D.dur.get(i, np.nan)):
                                r.update(hit=True, k=k, num=n)
                            break
        th, ph = E.head(q.get('target')), E.head(pr.get('obj'))
        r['selfr'] = bool(th and th == ph)
        rec.append(r)
    F = pd.DataFrame(rec).set_index('id')
    D = D.join(F)
    src = D.answer_source.astype(str)
    mode = D['mode'].astype(str)
    D['caw_src'] = src.isin(E.CAW_SRC) & ~src.str.contains('kin') & ~src.str.startswith('patch')
    D['geo_ok'] = D.g.notna() & (D.g > 0) & (D.meas_valid != False)
    D['mode_ok'] = mode.isin(['2d_ratio', '3d_calib']) | ((mode == 'vlm_2d_ratio') & D.selfr)
    D['fire'] = D.hit & D.caw_src & D.geo_ok & D.mode_ok
    D['new'] = np.where(D.fire, D.g, np.nan)
    D['why'] = np.where(D.fire, 'main', '')
    # var_ans_plain copies follow a fired plain row (same as echo / patch_copy)
    sv = D.video_id.astype(str).map(split_variant)
    kq = sv.str[0] + '||' + D.question.map(E._qn)
    pl = D.fire & (sv.str[1] == 'plain')
    pv = pd.Series(D.new[pl].values, index=kq[pl].values)
    po = pd.Series(D.answer[pl].values, index=kq[pl].values)
    pv, po = pv[~pv.index.duplicated()], po[~po.index.duplicated()]
    cp = src.str.replace('mo_guard:', '').eq('var_ans_plain') & kq.isin(pv.index) & ~D.fire
    cp = cp & np.isclose(D.answer.values.astype(float), po.reindex(kq.values).values.astype(float), rtol=1e-6)
    D.loc[cp, 'new'] = pv.reindex(kq[cp].values).values
    D.loc[cp, 'fire'] = True
    D.loc[cp, 'why'] = 'copy'
    patch = {}
    for i in D.index[D.fire]:
        w = 'rul130: CaW = k x size prior on a speed question'
        if D.why[i] == 'copy':
            w += ' (segmented copy follows its plain recording)'
        else:
            w += f" (CaW {D.said[i]:.4g} = {D.k[i]:g} x {D.num[i]:.4g}; geo {D['mode'][i]})"
        patch[str(i)] = {'value': float(D.new[i]), 'why': w}
    json.dump(patch, open(f'{R}/cache/tmp/ag_{TAG}_patch_{split}.json', 'w'), indent=0)
    D.to_pickle(f'{R}/cache/tmp/ag_{TAG}_flags_{split}.pkl')
    return D


if __name__ == '__main__':
    for split in ('val', 'test'):
        D = run(split)
        h = D[D.hit]
        print(split, 'hit', len(h), 'fire', int(D.fire.sum()), D[D.fire].cat.value_counts().to_dict())
        cols = ['video_id', 'cat', 'qkind', 'answer', 'answer_source', 'g', 'cs', 'said', 'k', 'num', 'mode', 'selfr', 'meas_valid', 'fire', 'new', 'why']
        with pd.option_context('display.width', 250, 'display.max_columns', 30, 'display.max_colwidth', 30):
            print(D.loc[D.hit | D.fire, cols].to_string())
