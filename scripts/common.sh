QP_ROOT="$(cd "${QP_ROOT:-$(dirname "${BASH_SOURCE[0]}")/..}" && pwd)"
export QP_ROOT
cd "$QP_ROOT"
if [ -n "${PYTHON:-}" ]; then PY=$PYTHON; elif command -v python >/dev/null 2>&1; then PY=python; else PY=python3; fi
export PY

export QP_PATCH_STRICT=${QP_PATCH_STRICT:-1}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-4} TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
mkdir -p logs

gpu() {
  "$@"
}

need_tracks() {
  local d miss=()
  for d in "$@"; do [ -n "$(ls -A "geo/out/$d" 2>/dev/null)" ] || miss+=("geo/out/$d"); done
  if [ ${#miss[@]} -gt 0 ]; then echo "missing track outputs of 02_track.sh: ${miss[*]}" >&2; exit 2; fi
}

stage_upstream() {
  tar -xJf intermediates/intermediates.tar.xz --skip-old-files
}

logged() {
  local lf=logs/$1; shift
  echo "+ $* > $lf"
  "$@" > "$lf" 2>&1 && return 0
  local rc=$?
  echo "FAILED (rc=$rc): $*" >&2; tail -20 "$lf" >&2; return $rc
}
