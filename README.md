# QuantiPhy Challenge 2026 — Track B, 2nd place (IIIXR Lab)

Code that produces our ranked Track B submission `sub_xt_r9mF_id1.csv` (portal score 0.857, uploaded 2026-10-26 10:06 UTC).

- Method: [METHOD.md](METHOD.md)
- Models: [MODELS.md](MODELS.md)

## Reproduce the submission (CPU, ~10 min)

```bash
conda env create -f environment/run_env.yml
bash get_data.sh   # QuantiPhy test/validation files from Hugging Face + 480p copies of 27 videos
bash run.sh        # -> output/sub_xt_r9mF_id1.csv
```

`intermediates/intermediates.tar.xz` holds the stored outputs of the model stages; `run.sh` unpacks it.

Requirements: Linux x86-64, bash, ffmpeg (6.1.1, libx264). Environment: Python 3.13.9, numpy 2.3.5, scipy 1.16.3 (MKL 2025.0.0),
pandas 2.3.3, pyarrow 21.0.0, opencv-python 4.13.0.92; `huggingface_hub` for `get_data.sh`. No GPU.
Tested on AMD Ryzen 7 3800XT (8C/16T, 31 GB RAM).

## Model stages (GPU, optional)

`scripts/pipeline/00_data.sh` … `04_geo_chain.sh` regenerate the stored model outputs.

```bash
bash environment/build_env.sh   # Python 3.10, versions in environment/env_constraints.txt
```

| job | time |
|---|---|
| Code-as-World-VL-9B, 3,289 test questions | 17 min |
| Code-as-World-VL-9B, 1,419 zoom crops | 7 min |
| Sa2VA, test videos | 46 min |
| geometric chain (4 CPU cores) | 11 min |

Hardware: NVIDIA H100 NVL (94 GB), RTX 3090 (24 GB). Re-running gives nearby, not identical, outputs (vLLM batching, SAM 2 and detector numerics).

## Layout

| path | content |
|---|---|
| `run.sh`, `get_data.sh` | entry points |
| `scripts/` | fusion, rule blocks, model-stage pipeline |
| `analysis/` | per-rule measurement and build code |
| `team/` | per-member build packages used by `run.sh` |
| `geo/`, `caw/`, `strict/` | geometric chain, Code-as-World stage, strict parsing |
| `patches/` | per-rule patch files read by `run.sh` |
| `patch_generators/` | scripts that generate `patches/`; `patch_rerun_results.csv` lists each generator and its re-run check |
| `intermediates/` | stored model-stage outputs |
| `environment/`, `tools/` | environments and helpers |
