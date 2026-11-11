#!/bin/bash
# abbuild: e1x tags (renamed abplan premises) -> probes Nzz47a/b/c via mk_sr2-generated sr129/sr130/sr131 (exact Nzz46 recipe + one premise per category)
cd ${QP_ROOT:-.}
RL() { for i in $(seq 1 360); do env CUDA_VISIBLE_DEVICES= "$@" </dev/null; rc=$?; [ $rc -ne 75 ] && return $rc; sleep 10; done; }
RLIO() { for i in $(seq 1 360); do env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 "$@" </dev/null; rc=$?; [ $rc -ne 75 ] && return $rc; sleep 10; done; }
set -o pipefail
D=analysis/v3/ag/rul272
echo "== start $(date '+%F %T')"
RLIO python $D/gen_e1x.py 2>&1 | grep -v run_limited | grep "^\[test\]\|^\[val\]"
RLIO python $D/cmp_e1x.py 2>&1 | grep -v run_limited | tee cache/tmp/ag_abbuild_cmp.txt
grep -q ALL_IDENTICAL cache/tmp/ag_abbuild_cmp.txt || { echo "e1x patches differ from ab_ patches: stop"; exit 1; }
for r in 129:Nzz47a 130:Nzz47b 131:Nzz47c; do
  RD=${r%%:*}; NM=${r#*:}
  echo "== sr$RD $NM $(date '+%T')"
  bash $D/sr$RD.sh > cache/tmp/ag_abbuild_sr$RD.out 2>&1
  grep -h "^xt_$NM\|SR${RD}_DONE\|all 237\|siblings\|  pl_\|Error" cache/tmp/ag_abbuild_sr$RD.out | head -12
  grep -A1 "^all 237\|^siblings" cache/tmp/ag_abbuild_sr$RD.out | head -6
  RLIO python analysis/v3/ag/abplan/ab_diff.py sub_xt_Nzz46 sub_xt_$NM > cache/tmp/ag_abbuild_diff_$NM.txt 2>&1
  grep "changed test rows\|^-- \|labelled \|macro\|by cat x source" cache/tmp/ag_abbuild_diff_$NM.txt | cut -c1-260
  md5sum submissions/sub_xt_${NM}_id1.csv cache/tmp/abbuild_bak/sub_xt_${NM}_id1.csv
done
RLIO python analysis/v3/ag/abplan/ab_bounds.py Nzz47a Nzz47b Nzz47c 2>&1 | grep -v run_limited
echo "ABBUILD_DONE $(date '+%F %T')"
