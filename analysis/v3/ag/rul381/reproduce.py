import sys
sys.path.insert(0, '@QP_ROOT@/analysis/v3/ag/rul381')
import slope_end
for k, v in sorted(slope_end.compute().items()):
    print(f"{k},{v:.4f}")
