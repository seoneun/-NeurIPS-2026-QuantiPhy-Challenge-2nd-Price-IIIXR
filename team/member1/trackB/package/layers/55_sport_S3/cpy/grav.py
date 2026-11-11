import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from labtrk import load
v, phrase, fgeo = sys.argv[1], sys.argv[2], float(sys.argv[3])
ent = [(float(a), float(b)) for a, b in (x.split('=') for x in sys.argv[4].split(','))]
ats = [float(x) for x in sys.argv[sys.argv.index('--at') + 1].split(',')] if '--at' in sys.argv else []
d = load(v); fps = d['fps']
obj = str([q for q in d['queries'] if phrase in q['phrase']][0]['obj'])
T, U, V = [], [], []
for i, e in enumerate(d['objs'][obj]):
    if e and e.get('st'):
        T.append(d['fidx'][i] / fps); U.append(e['st']['c'][0]); V.append(e['st']['c'][1])
T, U, V = map(np.array, (T, U, V))
ts = np.array([e[0] for e in ent]); ds = np.array([e[1] for e in ent])
Dfn = (lambda t: float(np.interp(t, ts, ds))) if len(ent) > 1 else (lambda t: ds[0])
vr = (ds[-1] - ds[0]) / (ts[-1] - ts[0]) * 100 if len(ent) > 1 else 0.0
b = [i for i in range(2, len(V) - 2) if V[i] >= V[max(0, i - 6):i + 7].max() and V[i] - V[max(0, i - 6):i].min() > 12 and V[i] - V[i + 1:i + 7].min() > 12]
print(v, 'obj', obj, 'bounces at', [round(T[i], 3) for i in b], 'radial(cm/s, entries)=', round(vr, 1))
arcs = []
for a0, a1 in zip([0] + b, b + [len(T) - 1]):
    idx = np.arange(a0 + 1, a1)
    if len(idx) < 5:
        continue
    pv = np.polyfit(T[idx], V[idx], 2); pu = np.polyfit(T[idx], U[idx], 2)
    av = 2 * pv[0]; res = np.std(V[idx] - np.polyval(pv, T[idx]))
    tm = T[idx].mean(); s = 980.0 / av if av > 0 else np.nan
    arcs.append((T[idx[0]], T[idx[-1]], pv, pu, s))
    print(f'  arc {T[idx[0]]:.3f}-{T[idx[-1]]:.3f} n={len(idx)} a_v={av:.0f}px/s2 resid={res:.1f}px s={s:.4f}cm/px '
          f'f_grav={Dfn(tm)*100/s if s==s else np.nan:.0f} (f_geo {fgeo:.0f}) apex t={-pv[1]/(2*pv[0]):.3f}')
for tq in ats:
    tmid = (round(tq * fps) + 0.5) / fps
    arc = [a for a in arcs if a[0] - 0.03 <= tmid <= a[1] + 0.03]
    if not arc:
        print(f'  t={tq}: no arc'); continue
    t0, t1, pv, pu, s = arc[0]
    dv = np.polyval(np.polyder(pv), tmid); du = np.polyval(np.polyder(pu), tmid); D = Dfn(tmid) * 100
    vz = s * dv
    for nm, f in [('f_geo', fgeo), ('f_grav', D / s)]:
        lat = du * D / f
        print(f'  t={tq} (tmid {tmid:.3f}) {nm}={f:.0f}: vert={vz:.1f} lat={lat:.1f} rad={vr:.1f} speed={np.sqrt(vz**2+lat**2+vr**2):.1f}')
