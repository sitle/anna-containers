# Guide des GPU pour l'inférence LLM

> Dernière mise à jour : 2026-03-14
> Contexte : choix d'un GPU sur Vast.ai pour servir des LLMs (SGLang, vLLM, Ollama)

---

## Tableau comparatif

| GPU | VRAM | Bande passante | Perf LLM (8B, tok/s) | Perf LLM (32B) | Prix Vast.ai ($/hr) | Recommandation |
|-----|------|---------------|----------------------|-----------------|---------------------|----------------|
| **RTX 5090** | 32 GB GDDR7 | 1,790 GB/s | ~213 tok/s | ~60-70 tok/s | $0.30-0.50 | **Best value pour 32B** |
| **RTX 4090** | 24 GB GDDR6X | 1,008 GB/s | ~128 tok/s | Trop juste (24 GB) | $0.25-0.45 | Best value pour 7-14B |
| **RTX 3090** | 24 GB GDDR6X | 936 GB/s | ~112 tok/s | Trop juste (24 GB) | $0.15-0.30 | Budget, modèles 7-14B |
| **RTX 4080** | 16 GB GDDR6X | 717 GB/s | ~95 tok/s | Non (16 GB) | $0.15-0.25 | Modèles 7B uniquement |
| **RTX 3080** | 10-12 GB | 760 GB/s | ~70 tok/s | Non | $0.08-0.15 | Dev/test, petits modèles |
| **RTX 4070 Ti** | 12 GB GDDR6X | 504 GB/s | ~75 tok/s | Non | $0.10-0.20 | Modèles 7B |
| **A100 (40 GB)** | 40 GB HBM2e | 1,555 GB/s | ~138 tok/s | ~50-60 tok/s | $0.70-1.20 | Multi-GPU, gros modèles |
| **A100 (80 GB)** | 80 GB HBM2e | 2,039 GB/s | ~138 tok/s | ~55-65 tok/s | $1.00-1.80 | Modèles 70B |
| **H100** | 80 GB HBM3 | 3,350 GB/s | ~144 tok/s | ~80-90 tok/s | $1.50-3.00 | Production haute perf |

---

## Règle de base : VRAM nécessaire

La VRAM détermine quel modèle peut tourner :

| Modèle | Format | VRAM minimale | GPU recommandé |
|--------|--------|---------------|----------------|
| 7-8B (Qwen3-8B, Llama 3.1-8B) | GGUF Q4 | **~5 GB** | RTX 3080+ (n'importe quel GPU moderne) |
| 7-8B | FP16 | **~16 GB** | RTX 4080, RTX 3090 |
| 14B (Qwen2.5-14B) | GGUF Q4 | **~9 GB** | RTX 3080 12GB+ |
| 14B | AWQ 4-bit | **~10 GB** | RTX 3090, RTX 4080 |
| 32B (Qwen3-32B) | GGUF Q4 | **~20 GB** | **RTX 5090** (32 GB) |
| 32B | AWQ 4-bit | **~19 GB** | **RTX 5090** (32 GB), A100 40 GB |
| 70B (Llama 3.1-70B) | AWQ 4-bit | **~38 GB** | A100 40 GB, 2x RTX 4090 |
| 70B | GGUF Q4 | **~40 GB** | A100 40 GB, A100 80 GB |
| 70B | FP16 | **~140 GB** | 2x A100 80 GB, 2x H100 |

### Formule rapide

```
VRAM ≈ (paramètres en milliards) × (bits par poids) / 8 + 2 GB overhead

Exemples :
  32B en 4-bit : 32 × 4 / 8 + 2 = 18 GB → RTX 5090
  70B en 4-bit : 70 × 4 / 8 + 2 = 37 GB → A100 40 GB
   8B en 4-bit :  8 × 4 / 8 + 2 =  6 GB → RTX 3080
```

---

## Quel GPU choisir ?

### Pour ANNA (agent LLM avec tool calling)

Notre cas d'usage : servir un modèle 32B quantifié (AWQ/GGUF) avec tool calling, single user.

**Recommandation : RTX 5090 ($0.30-0.50/hr)**
- Seul GPU consumer avec 32 GB VRAM → peut faire tourner un 32B en AWQ
- Bande passante 1.8 TB/s → ~60-70 tok/s avec SGLang
- Meilleur rapport perf/prix pour notre cas d'usage
- Alternative : A100 40 GB mais 2-3x plus cher

### Par budget

| Budget | GPU | Modèle max | Cas d'usage |
|--------|-----|-----------|-------------|
| **< $0.15/hr** | RTX 3080, RTX 3060 | 7-8B GGUF | Dev, test, prototypage |
| **$0.15-0.30/hr** | RTX 3090, RTX 4080 | 14B AWQ | Petits agents, chatbot |
| **$0.30-0.50/hr** | **RTX 5090** | **32B AWQ** | **ANNA, agents tool calling** |
| **$0.50-1.00/hr** | A100 40 GB | 70B GGUF Q4 | Gros modèles |
| **$1.00-2.00/hr** | A100 80 GB | 70B AWQ | Production |
| **$2.00+/hr** | H100 | 70B FP16 | Haute performance |

### Par cas d'usage

| Cas d'usage | GPU recommandé | Pourquoi |
|-------------|----------------|----------|
| **Développement/test** | RTX 3080/3090 | Pas cher, suffisant pour 7-8B |
| **Agent LLM (tool calling)** | RTX 5090 | 32B AWQ avec SGLang, tool calling structuré |
| **Chatbot production** | RTX 4090 ou RTX 5090 | Bon rapport perf/prix |
| **Multi-utilisateurs** | A100 40 GB | Continuous batching, plus de VRAM |
| **Gros modèles (70B+)** | A100 80 GB / H100 | Seuls GPU avec assez de VRAM |
| **Fine-tuning** | A100 80 GB / H100 | Besoin de VRAM pour gradients |
| **Multi-GPU (tensor parallel)** | 2-4x A100 / H100 | NVLink pour communication inter-GPU |

---

## Facteurs clés pour l'inférence LLM

### 1. VRAM (le plus important)
Détermine quel modèle peut tourner. Pas de VRAM = pas de modèle. C'est le critère numéro 1.

### 2. Bande passante mémoire
L'inférence LLM est **memory-bandwidth bound** (limitée par la vitesse de lecture des poids). Plus la bande passante est élevée, plus les tokens sont générés vite.

- GDDR6X (~1 TB/s) : bon
- HBM2e (~2 TB/s) : très bon
- HBM3 (~3.3 TB/s) : excellent
- GDDR7 (~1.8 TB/s) : très bon (RTX 5090)

### 3. Prix / heure
Le rapport perf/prix est souvent meilleur sur les GPU consumer (RTX) que sur les GPU datacenter (A100, H100).

### 4. Réseau de la machine (Vast.ai)
Les specs réseau affichées par Vast.ai ne sont pas toujours fiables. Tester la bande passante réelle avant de télécharger des gros modèles.

---

## Pièges à éviter

1. **RTX 4090 pour un 32B** : 24 GB de VRAM, c'est trop juste. Le modèle 32B en AWQ fait ~19 GB, il reste très peu pour le KV cache → contexte très limité.

2. **A100 40 GB pour du "simple" chat** : overkill et cher. Un RTX 4090 ou RTX 5090 fait le même travail pour 3x moins cher.

3. **GPU sans assez de VRAM** : le modèle ne se charge simplement pas, ou tourne en mode CPU/offload qui est 100x plus lent.

4. **Ignorer la bande passante** : deux GPU avec la même VRAM mais des bandes passantes différentes auront des performances très différentes. La RTX 5090 (1.8 TB/s) est bien plus rapide que la RTX 3090 (936 GB/s) malgré la même VRAM effective pour un 32B.

5. **Multi-GPU sans NVLink** : le tensor parallelism sur PCIe est beaucoup plus lent que sur NVLink. Privilégier un seul gros GPU plutôt que 2 petits.

---

## Sources

- [Best GPUs for Local LLM Inference 2025](https://localllm.in/blog/best-gpus-llm-inference-2025)
- [RTX 5090 vs RTX 4090 AI Benchmarks](https://bizon-tech.com/blog/nvidia-rtx-5090-comparison-gpu-benchmarks-for-ai)
- [RTX 5090 Ollama Benchmark](https://www.databasemart.com/blog/ollama-gpu-benchmark-rtx5090)
- [LLM Inference Speed: H100 vs A100 vs RTX 4090](http://valebyte.com/en/guides/llm-inference-speed-h100-a100-rtx-4090-cloud-benchmarks/)
- [GPU Selection ML 2026](https://markaicode.com/gpu-selection-ml-2026/)
- Tests ANNA sur Vast.ai (RTX 5090, SGLang/Ollama/vLLM)
