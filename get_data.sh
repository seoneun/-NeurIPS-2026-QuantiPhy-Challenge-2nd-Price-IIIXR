#!/bin/bash
set -euo pipefail
R=$(cd "$(dirname "$0")" && pwd); cd "$R"
QP_ROOT="$R" python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
R = os.environ["QP_ROOT"]
L = [l.strip() for l in open(f"{R}/scripts/dataset_files.txt") if l.strip()]
test = sorted({l.split("/")[-1] for l in L if l.startswith(("data/test/", "data/test_480p/"))})
val = sorted({l[len("data/val/"):] for l in L if l.startswith("data/val/")})
snapshot_download("PaulineLi/QuantiPhy", repo_type="dataset", revision="8b437e728577af37eaa660b35c933b128b4d19c3",
                  local_dir=f"{R}/data/test", allow_patterns=["test_dataset.parquet"] + test, max_workers=2)
snapshot_download("PaulineLi/QuantiPhy-validation", repo_type="dataset", revision="74aec82473912ccab6a589645a39fcd0e78ee23c",
                  local_dir=f"{R}/data/val", allow_patterns=val, max_workers=2)
PY
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
for f in $(grep '^data/test_480p/' scripts/dataset_files.txt); do ln -s "$R/data/test/${f##*/}" "$T/"; done
SRC=$T DST=$R/data/test_480p bash tools/make_480p.sh | grep -E "^(END|FAIL)"
