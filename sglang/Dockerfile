ARG BASE_IMAGE=ghcr.io/sitle/anna-containers/base:latest
FROM ${BASE_IMAGE}

# Install SGLang with all dependencies (includes PyTorch, FlashInfer, etc.)
RUN pip install --no-cache-dir "sglang[all]"

# Helper script to launch SGLang with proxy
COPY sglang-serve /usr/local/bin/sglang-serve
RUN chmod +x /usr/local/bin/sglang-serve

ENTRYPOINT ["/bin/bash"]
