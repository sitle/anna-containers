ARG BASE_IMAGE=ghcr.io/sitle/anna-containers/base:latest
FROM ${BASE_IMAGE}

# Install zstd (required by Ollama installer) then install Ollama via official script
RUN apt-get update && apt-get install -y --no-install-recommends zstd \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://ollama.com/install.sh | bash

# Helper script to launch Ollama and pull a model
COPY ollama-serve /usr/local/bin/ollama-serve
RUN chmod +x /usr/local/bin/ollama-serve

EXPOSE 11434 8000

ENTRYPOINT ["/bin/bash"]
