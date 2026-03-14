# Comparatif des serveurs d'inférence LLM

> Dernière mise à jour : 2026-03-14
> Tests réalisés sur RTX 5090 (32 GB VRAM) avec Qwen3-32B-AWQ et Qwen2.5-Coder-32B-Instruct-AWQ

---

## Vue d'ensemble

| Critère | SGLang | vLLM | Ollama |
|---------|--------|------|--------|
| **Version testée** | 0.5.9 | 0.17.1 | latest |
| **API** | OpenAI-compatible | OpenAI-compatible | Propre + OpenAI-compat (/v1) |
| **Facilité d'installation** | `pip install "sglang[all]"` | `pip install vllm` | `curl -fsSL https://ollama.com/install.sh \| sh` |
| **Facilité d'utilisation** | Moyenne | Complexe | Très simple |
| **Vitesse d'inférence** | ~58-67 tok/s (CUDA graph) / ~28-30 tok/s (sans) | ~15-25 tok/s | ~15-25 tok/s |
| **Tool calling structuré** | ✅ Oui (hermes parser, Qwen3) | ⚠️ Instable (hermes parser) | ❌ Non (texte brut) |
| **Quantization AWQ** | ✅ awq_marlin (rapide) | ✅ awq_marlin | ❌ GGUF uniquement |
| **Formats de modèles** | HuggingFace (safetensors) | HuggingFace (safetensors) | GGUF (propre format) |
| **CUDA Graph** | ✅ Oui (boost ~2x) | ✅ Oui | N/A |
| **Maturité** | Récent (2024+) | Mature (2023+) | Très populaire (2023+) |
| **Communauté** | Croissante | Grande | Très grande |
| **License** | Apache 2.0 | Apache 2.0 | MIT |

---

## SGLang

### Points forts
- **Le plus rapide** : ~67 tok/s avec CUDA graph sur RTX 5090 (2-3x plus rapide que vLLM/Ollama)
- **Tool calling structuré** fonctionne avec le hermes parser (seul serveur testé qui renvoie des `tool_calls` dans l'API avec Qwen3)
- **awq_marlin** optimisé pour les modèles quantizés AWQ
- **Chargement rapide** : ~5 secondes pour charger un modèle 32B AWQ depuis le disque
- **API OpenAI-compatible** native (port 30000 par défaut, configurable)

### Points faibles
- **Jeune** : documentation moins fournie, moins de ressources en ligne
- **CUDA graph capture** peut être lent au premier démarrage (~2-5 min sur RTX 5090)
- **Erreurs de parsing JSON** : quand le modèle génère du JSON malformé, le hermes parser échoue silencieusement (`Error in detect_and_parse`)
- **Nécessite un fix `libcuda.so`** sur certaines images Docker cloud
- **Sans CUDA graph** : la vitesse tombe à ~28-30 tok/s

### Cas d'usage recommandé
- Production avec tool calling (agents LLM)
- Modèles AWQ sur GPU unique
- Quand la vitesse d'inférence est critique

### Commande type
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-32B-AWQ \
    --port 8000 --host 0.0.0.0 \
    --quantization awq_marlin \
    --tool-call-parser hermes
```

---

## vLLM

### Points forts
- **Le plus mature** : large communauté, bien documenté, beaucoup de modèles supportés
- **Continuous batching** : optimisé pour le multi-utilisateur
- **PagedAttention** : gestion mémoire efficace
- **Nombreux formats de quantization** : AWQ, GPTQ, SqueezeLLM, etc.
- **Tensor parallelism** natif pour multi-GPU

### Points faibles
- **Configuration complexe** : beaucoup de paramètres à ajuster (`--gpu-memory-utilization`, `--max-model-len`, `--enforce-eager`, etc.)
- **Tool calling instable** avec le hermes parser (Qwen3 alterne entre structuré et texte)
- **Chargement lent** : a bloqué 15+ minutes sur Qwen2.5-Coder-32B lors de nos tests
- **Vitesse moyenne** : ~15-25 tok/s sur nos tests (possiblement lié à la config)
- **Gros en mémoire** : utilise ~31 GB VRAM vs ~19 GB pour SGLang avec le même modèle

### Cas d'usage recommandé
- Production multi-utilisateur avec continuous batching
- Déploiements multi-GPU
- Quand la compatibilité avec un large éventail de modèles est prioritaire

### Commande type
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-32B-AWQ \
    --port 8000 --host 0.0.0.0 \
    --quantization awq_marlin \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --gpu-memory-utilization 0.90
```

---

## Ollama

### Points forts
- **Le plus simple** : `ollama pull <model>` puis `ollama serve` — c'est tout
- **Aucune configuration** : pas de paramètres de quantization, VRAM, etc.
- **Catalogue de modèles** intégré (ollama.com/library)
- **Format GGUF** : compatible CPU et GPU, fonctionne sur des machines modestes
- **Auto-detection GPU** : utilise le GPU si disponible, sinon CPU
- **API simple** + endpoint OpenAI-compatible (`/v1`)

### Points faibles
- **Pas de tool calling structuré** : les tool calls sont renvoyés en texte brut dans `content`, pas dans `tool_calls`
- **Format GGUF uniquement** : pas de support AWQ/safetensors natif
- **Plus lent** : ~15-25 tok/s (pas de CUDA graph, pas d'optimisations kernel)
- **Pas de continuous batching** : une requête à la fois
- **Moins de contrôle** : pas de paramètres fins (memory utilization, tensor parallel, etc.)
- **Modèles plus gros** : GGUF Q4 prend plus de VRAM que AWQ 4-bit

### Cas d'usage recommandé
- Développement local et prototypage
- Quand la simplicité prime sur la performance
- Machines sans GPU puissant (fonctionne en CPU)
- Tests rapides de différents modèles

### Commande type
```bash
ollama serve &
ollama pull qwen2.5-coder:32b
# API disponible sur http://localhost:11434
```

---

## Tableau de décision

| Besoin | Serveur recommandé |
|--------|-------------------|
| Tool calling fiable pour agents | **SGLang** |
| Vitesse maximale (tok/s) | **SGLang** |
| Simplicité absolue | **Ollama** |
| Prototypage rapide | **Ollama** |
| Production multi-utilisateur | **vLLM** |
| Multi-GPU | **vLLM** |
| Machine sans GPU | **Ollama** |
| Modèles AWQ quantizés | **SGLang** ou **vLLM** |
| Large écosystème de modèles | **Ollama** (catalogue) ou **vLLM** (HuggingFace) |

---

## Résultats benchmark ANNA (demo5 fullstack)

Test : création d'une application fullstack (FastAPI + Next.js + Docker) avec orchestration sub-agents.

| Serveur | Modèle | Vitesse | Tool calling | Fichiers créés | Résultat |
|---------|--------|---------|-------------|----------------|----------|
| SGLang | Qwen3-32B-AWQ | ~30 tok/s | ✅ Structuré | 0 | JSON malformé dans write_files, boucle sub_agents |
| SGLang | Qwen2.5-Coder-32B-AWQ | ~67 tok/s | ❌ Texte brut | 0 | Boucle infinie chemins fichiers, context overflow |
| Ollama | Qwen2.5-Coder-32B | ~15-25 tok/s | ❌ Texte brut | 18 (partiel) | Backend manquant, crash UnicodeEncodeError |
| vLLM | Qwen3-32B-AWQ | ~15-25 tok/s | ⚠️ Instable | 0-14 | Boucle sub_agents, JSON cassé |

### Conclusion

**SGLang + Qwen3** est la meilleure combinaison pour le tool calling structuré. Le problème restant n'est pas le serveur mais le modèle qui génère du JSON invalide pour les gros payloads (`write_files` avec beaucoup de fichiers). La solution passe par :
1. Simplifier les tools (un fichier par appel au lieu de batch)
2. Ajouter de la réparation JSON côté client
3. Tester des modèles plus récents (Qwen3.5, DeepSeek-V3)
