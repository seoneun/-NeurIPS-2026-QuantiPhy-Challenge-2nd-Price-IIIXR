#!/bin/bash
R="$(cd "${QP_ROOT:-$(dirname "${BASH_SOURCE[0]}")/..}" && pwd)"; export QP_ROOT="$R"
PY="${PY:-${PYTHON:-python}}"
"$PY" - <<'PY'
import os
from huggingface_hub import snapshot_download
R = os.environ["QP_ROOT"]
snapshot_download(repo_id="MirroS-Lab/Code-as-World-VL-9B", repo_type="model", revision="46b111c53f5680eb12c63a7391e5c690d1e14ab1",
                  local_dir=f"{R}/models/Code-as-World-VL-9B", max_workers=2)
PY
