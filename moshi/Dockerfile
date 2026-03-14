ARG BASE_IMAGE=ghcr.io/sitle/anna-containers/base:latest
FROM ${BASE_IMAGE}

# Install Moshi and Gradio
RUN pip install --no-cache-dir moshi gradio

# Helper script to launch Moshi with proxy
COPY moshi-serve /usr/local/bin/moshi-serve
RUN chmod +x /usr/local/bin/moshi-serve

ENTRYPOINT ["/bin/bash"]
