# Models (Track B, team IIIXR Lab)

| model | public weights | revision / license | role |
|---|---|---|---|
| SAM 2.1 hiera-large | https://huggingface.co/facebook/sam2.1-hiera-large | Apache-2.0 | object tracks |
| Grounding DINO base | https://huggingface.co/IDEA-Research/grounding-dino-base | Apache-2.0 | detection of named objects |
| OWLv2 base-patch16-ensemble | https://huggingface.co/google/owlv2-base-patch16-ensemble | Apache-2.0 | detection |
| Qwen3-VL-8B-Instruct | https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct | Apache-2.0 | parsing fallback, grounding |
| Molmo2-8B | https://huggingface.co/allenai/Molmo2-8B | rev e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b, Apache-2.0 | pointing |
| Sa2VA-Qwen3-VL-4B-SAM3 | https://huggingface.co/ByteDance/Sa2VA-Qwen3-VL-4B-SAM3 | rev 3f2bec126b22761a4607eec674781c138f9f9cca, Apache-2.0 | segmentation tracks |
| MoGe-2 ViT-L normal | https://huggingface.co/Ruicheng/moge-2-vitl-normal | MIT | focal length and depth |
| Code-as-World-VL-9B | https://huggingface.co/MirroS-Lab/Code-as-World-VL-9B | rev 46b111c, Apache-2.0 | answer per question (vLLM, seed 1, temperature 0.01) |
| Kimi K3 | https://huggingface.co/moonshotai/Kimi-K3 | license "other" (kimi-k3) | answers for some simulation questions, served from these weights through OpenRouter (fp8) |
| VGGT-1B | https://huggingface.co/facebook/VGGT-1B | CC-BY-NC-4.0 | camera poses and depth |

No model weights were fine-tuned.
