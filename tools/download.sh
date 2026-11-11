#!/bin/bash
R="$(cd "${QP_ROOT:-$(dirname "${BASH_SOURCE[0]}")/..}" && pwd)"; export QP_ROOT="$R"
[ -f "$R/tools/env.sh" ] && source "$R/tools/env.sh"
export CUDA_VISIBLE_DEVICES=""
PY="${PY:-${PYTHON:-python}}"
echo "START $(date)"; df -h "$R" | tail -1
"$PY" - <<'PY' || { echo "DOWNLOAD FAILED"; exit 1; }
import time
from huggingface_hub import snapshot_download
R = __import__("os").environ["QP_ROOT"]
DET = ["*.json", "*.txt", "model.safetensors"]
jobs = [  # repo, type, revision (= the copy our runs used), local dir, allow_patterns
    ("PaulineLi/QuantiPhy-validation", "dataset", "74aec82473912ccab6a589645a39fcd0e78ee23c", f"{R}/data/val", None),
    ("PaulineLi/QuantiPhy", "dataset", "8b437e728577af37eaa660b35c933b128b4d19c3", f"{R}/data/test", None),
    ("IDEA-Research/grounding-dino-base", "model", "12bdfa3120f3e7ec7b434d90674b3396eccf88eb", f"{R}/models/grounding-dino-base", DET),
    ("facebook/sam2.1-hiera-large", "model", "665f8e2ad61cf5f53d65644ff27c8ee525124610", f"{R}/models/sam2.1-hiera-large", DET),
    ("google/owlv2-base-patch16-ensemble", "model", "cfd3195ba4ea9592eec887ded089f4c08eff231d", f"{R}/models/owlv2-base-patch16-ensemble", DET),
    ("Qwen/Qwen3-VL-8B-Instruct", "model", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b", f"{R}/models/Qwen3-VL-8B-Instruct", None),
]
for rid, rt, rev, d, pat in jobs:
    t = time.time()
    p = snapshot_download(repo_id=rid, repo_type=rt, revision=rev, local_dir=d, allow_patterns=pat, max_workers=2)
    print(f"done {rid}@{rev[:7]} -> {p} ({time.time() - t:.0f} s)", flush=True)
PY
du -sh --apparent-size "$R"/data/val "$R"/data/test "$R"/models/*; df -h "$R" | tail -1
echo "END $(date)"
