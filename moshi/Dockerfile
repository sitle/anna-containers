FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Moshi (pulls PyTorch with CUDA as dependency)
RUN pip install --no-cache-dir moshi

WORKDIR /workspace

# Helper script
COPY moshi-serve /usr/local/bin/moshi-serve
RUN chmod +x /usr/local/bin/moshi-serve

EXPOSE 8998

ENTRYPOINT ["/bin/bash"]
