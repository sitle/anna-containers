ARG BASE_IMAGE=ghcr.io/sitle/anna-containers/base:latest
FROM ${BASE_IMAGE}

# Install Ollama via official script (zstd already in base image)
RUN curl -fsSL https://ollama.com/install.sh | bash

# Helper script to launch Ollama with proxy
COPY ollama-serve /usr/local/bin/ollama-serve
RUN chmod +x /usr/local/bin/ollama-serve

ENTRYPOINT ["/bin/bash"]
