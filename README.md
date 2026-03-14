# anna-containers

Container images for LLM inference servers, optimized for GPU cloud instances (Vast.ai, RunPod, etc.).

## Architecture

```
nvidia/cuda:12.4.1-devel-ubuntu22.04
    │
    ├── anna-containers/base      CUDA + Python 3.11 + SSH
    │
    ├── anna-containers/sglang    base + SGLang
    ├── anna-containers/vllm      base + vLLM
    └── anna-containers/ollama    base + Ollama
```

## Images

| Image | Description | Usage |
|-------|-------------|-------|
| `ghcr.io/sitle/anna-containers/base` | CUDA 12.4 + Python 3.11 + SSH | Base image |
| `ghcr.io/sitle/anna-containers/sglang` | SGLang inference server | `sglang-serve Qwen/Qwen3-32B-AWQ` |
| `ghcr.io/sitle/anna-containers/vllm` | vLLM inference server | `vllm-serve Qwen/Qwen3-32B-AWQ` |
| `ghcr.io/sitle/anna-containers/ollama` | Ollama inference server | `ollama-serve qwen2.5-coder:32b` |

## Quick Start

### On Vast.ai

```bash
# Create an instance with the SGLang image
vastai create instance <offer_id> \
  --image ghcr.io/sitle/anna-containers/sglang:latest \
  --env '-p 8000:8000' \
  --disk 100 --ssh --direct

# SSH in and start serving
ssh -p <port> root@<host>
sglang-serve Qwen/Qwen3-32B-AWQ
```

### Local (requires NVIDIA GPU + nvidia-container-toolkit)

```bash
docker run --gpus all -p 8000:8000 ghcr.io/sitle/anna-containers/sglang:latest \
  sglang-serve Qwen/Qwen3-32B-AWQ
```

## Helper Scripts

Each image includes a helper script:

### sglang-serve
```
sglang-serve <model> [quantization] [tool-call-parser]
sglang-serve Qwen/Qwen3-32B-AWQ                    # defaults: awq_marlin, hermes
sglang-serve Qwen/Qwen3-32B-AWQ awq_marlin hermes  # explicit
```

### vllm-serve
```
vllm-serve <model> [quantization] [tool-call-parser]
vllm-serve Qwen/Qwen3-32B-AWQ                      # defaults: awq_marlin, hermes
```

### ollama-serve
```
ollama-serve <model>
ollama-serve qwen2.5-coder:32b
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` (SGLang/vLLM) or `11434` (Ollama) | Server port |
| `HF_HUB_ENABLE_HF_TRANSFER` | `0` | HuggingFace download method |
