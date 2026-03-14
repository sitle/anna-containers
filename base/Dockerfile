FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=0

# System dependencies (includes packages Vast.ai expects: tmux, sudo, rsync, locales, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    wget \
    less \
    openssh-server \
    tmux \
    sudo \
    rsync \
    locales \
    software-properties-common \
    zstd \
    debian-keyring \
    debian-archive-keyring \
    apt-transport-https \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# Install Caddy (reverse proxy with API key auth)
RUN curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg \
    && curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | tee /etc/apt/sources.list.d/caddy-stable.list \
    && apt-get update && apt-get install -y --no-install-recommends caddy \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Fix libcuda.so symlink (needed on some cloud GPU instances)
RUN ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so 2>/dev/null || true

# SSH server config
RUN mkdir -p /run/sshd \
    && echo "PermitRootLogin yes" >> /etc/ssh/sshd_config \
    && echo "PasswordAuthentication no" >> /etc/ssh/sshd_config

# Caddy reverse proxy config (API_KEY env var required at runtime)
COPY Caddyfile /etc/caddy/Caddyfile

# Helper script to start Caddy
COPY start-proxy /usr/local/bin/start-proxy
RUN chmod +x /usr/local/bin/start-proxy

# Working directory
WORKDIR /workspace

# Only expose the authenticated proxy port
EXPOSE 8080 22

CMD ["/bin/bash"]
