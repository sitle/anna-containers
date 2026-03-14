ARG BASE_IMAGE=ghcr.io/sitle/anna-containers/base:latest
FROM ${BASE_IMAGE}

# Install vLLM (includes PyTorch)
RUN pip install --no-cache-dir vllm

# Helper script to launch vLLM easily
COPY vllm-serve /usr/local/bin/vllm-serve
RUN chmod +x /usr/local/bin/vllm-serve

EXPOSE 8000 8001

ENTRYPOINT ["/bin/bash"]
