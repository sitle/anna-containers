ARG BASE_IMAGE=ghcr.io/sitle/anna-containers/base:latest
FROM ${BASE_IMAGE}

# Install Ollama binary directly (install.sh needs systemd which is not available in containers)
RUN curl -fsSL -o /usr/local/bin/ollama https://ollama.com/download/ollama-linux-amd64 \
    && chmod +x /usr/local/bin/ollama

# Helper script to launch Ollama and pull a model
COPY ollama-serve /usr/local/bin/ollama-serve
RUN chmod +x /usr/local/bin/ollama-serve

EXPOSE 11434 8000

ENTRYPOINT ["/bin/bash"]
