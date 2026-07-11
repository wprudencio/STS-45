FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl wget tar libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /parakeet

# Download the pre-built Linux CPU x64 parakeet-server binary
ENV PARAKEET_VERSION="v0.4.0"
ENV PARAKEET_URL="https://github.com/mudler/parakeet.cpp/releases/download/${PARAKEET_VERSION}/parakeet-${PARAKEET_VERSION}-bin-linux-cpu-x64.tar.gz"

RUN curl -L "$PARAKEET_URL" | tar xz --strip-components=1 && \
    chmod +x parakeet-server && \
    ls -la

# Volume: /models — place the STT GGUF model here
VOLUME ["/models"]

ENV STT_MODEL_FILE="tdt_ctc-110m-f16.gguf"

# Download STT model if missing, then run parakeet-server
CMD ["/bin/bash", "-c", "\
    if [ ! -f /models/$STT_MODEL_FILE ]; then \
        echo 'Downloading STT model $STT_MODEL_FILE...'; \
        wget -q -O /models/$STT_MODEL_FILE \
            'https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/tdt_ctc-110m-f16.gguf' || \
        wget -q -O /models/$STT_MODEL_FILE \
            'https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/tdt-1.1b-q5_k.gguf'; \
    fi; \
    echo 'Starting parakeet-server on :8081'; \
    exec ./parakeet-server --model /models/$STT_MODEL_FILE --host 0.0.0.0 --port 8081 \
"]