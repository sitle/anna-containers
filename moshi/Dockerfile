FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

# System dependencies (including SSH for Vast.ai)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    openssh-server \
    tmux \
    sudo \
    rsync \
    less \
    locales \
    && locale-gen en_US.UTF-8 \
    && mkdir -p /run/sshd \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Install Moshi (pulls PyTorch with CUDA as dependency)
RUN pip install --no-cache-dir moshi

WORKDIR /workspace

# Helper script
COPY moshi-serve /usr/local/bin/moshi-serve
RUN chmod +x /usr/local/bin/moshi-serve

EXPOSE 8998

ENTRYPOINT ["/bin/bash"]
