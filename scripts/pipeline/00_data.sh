#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
set -euo pipefail

bash tools/download.sh
bash caw/dl_caw9b.sh
$PY - <<'PY'
import os
from huggingface_hub import hf_hub_download, snapshot_download
R = os.environ["QP_ROOT"]
snapshot_download("allenai/Molmo2-8B", revision="e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b", local_dir=f"{R}/models/Molmo2-8B", max_workers=2)
snapshot_download("ByteDance/Sa2VA-Qwen3-VL-4B-SAM3", revision="3f2bec126b22761a4607eec674781c138f9f9cca",
                  local_dir=f"{R}/models/Sa2VA-Qwen3-VL-4B-SAM3", max_workers=2)
hf_hub_download("Ruicheng/moge-2-vitl-normal", "model.pt", revision="cb0e8bbd6b1e243589717c78e750b1ba4c093acf",
                local_dir=f"{R}/models/moge-2-vitl-normal")
PY
bash tools/make_480p.sh
$PY caw/prep_inputs.py
echo "00_data done"
