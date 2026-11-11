"""rul096: a size answer read from a border-cut box is only a LOWER bound; when CaW is >= 1.5x above it, CaW is the
only estimate consistent with the bound. Extends rul005 (frame-filling boxes) to boxes cut by one border.
Scan: analysis/v3/ag/c54s0/bcut.py (code over raw geo_value/meas_mpp, the build's target track, frame W/H) -> ag_c54s0_bcut.csv.
Gates: parse kind size; detail source dis_geo (unpatched); cs >= 1.5 g; geo px (geo_value/meas_mpp) matches the target box
extent on the measured axis within 5%; that box touches the frame border on that axis in >= 80% of frames; the box spans
< 95% of the frame on the other axis (rul005's domain excluded). Value: CaW. env C54_SUB (default sub_xt_Nzz8)."""
import os, sys, json, subprocess, numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))  # repo root (QP_ROOT overrides)
SUB = os.environ.get('C54_SUB', 'sub_xt_Nzz8')
env = dict(os.environ, SUB=SUB)
subprocess.run([sys.executable, R + '/analysis/v3/ag/rul096/bcut_scan.py'], env=env, stdout=subprocess.DEVNULL, check=True)
D = pd.read_csv(R + '/cache/tmp/ag_c54cutlb_bcut.csv')
D['ext'] = np.where(D.ax == 'x', D.w, D.h); D['oth'] = np.where(D.ax == 'x', D.h / D.H, D.w / D.W)
D['match'] = (D.px - D.ext).abs() / D.ext
F = D[(D.src == 'dis_geo') & (D.cs >= 1.5 * D.g) & (D.match <= 0.05) & (D.cut >= 0.8) & (D.oth < 0.95)]
print(F[['split', 'id', 'vid', 'cat', 'q', 'g', 'cs', 'ans', 'px', 'ext', 'ax', 'cut', 'oth']].round(3).to_string())
for split in ('test', 'val'):
    P = {str(int(r.id)): dict(value=float(r.cs), why=f"rul096: geo {r.g:.3g} reads a border-cut box ({r.ax} extent {r.ext:.0f}px, cut in {r.cut:.0%} of frames) "
                                                   f"-> lower bound; CaW {r.cs:.3g} >= 1.5x geo is the only bound-consistent value") for r in F[F.split == split].itertuples()}
    json.dump(P, open(f'{R}/cache/tmp/ag_c54cutlb_patch_{split}.json', 'w'), indent=1)
    print(split, {k: round(v['value'], 3) for k, v in P.items()})
