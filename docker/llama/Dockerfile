FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl wget zstd \
    && rm -rf /var/lib/apt/lists/*

# llama.cpp via the official installer (provides llama-server + llama-cli)
# SKIP_CUDA and SKIP_ROCM force a CPU-only build inside the container
ENV SKIP_CUDA=1 SKIP_ROCM=1
RUN curl -LsSf https://llama.app/install.sh | sh

ENV PATH="/root/.local/bin:${PATH}"

# Volume mount: /models — place the GGUF model here
VOLUME ["/models"]

EXPOSE 8080

# MODEL env var can be set in docker-compose; default to LFM2.5-230M
ENV MODEL_FILE="LFM2-230M-Q4_K_M.gguf"
ENV HF_REPO="LiquidAI/LFM2.5-230M-GGUF"

# Download the model if it's not already present, then start the server
CMD ["/bin/bash", "-c", "\
    if [ ! -f /models/$MODEL_FILE ]; then \
        echo 'Downloading $MODEL_FILE from $HF_REPO...'; \
        wget -q -O /models/$MODEL_FILE \"https://huggingface.co/$HF_REPO/resolve/main/$MODEL_FILE\" || \
        { echo 'Could not find model file. Mount /models with the GGUF and set MODEL_FILE.'; exit 1; }; \
    fi; \
    echo 'Starting llama server on :8080'; \
    if command -v llama-server >/dev/null 2>&1; then \
        exec llama-server -m /models/$MODEL_FILE --host 0.0.0.0 --port 8080; \
    else \
        exec llama server -m /models/$MODEL_FILE --host 0.0.0.0 --port 8080; \
    fi \
"]