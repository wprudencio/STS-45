FROM python:3.11-slim

WORKDIR /app

# System deps for ONNX Runtime + audio + llama.cpp install
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install llama.cpp via official installer (https://llama.app)
# Skip GPU probes — add SKIP_CUDA/ROCM/VULKAN overrides if you want them.
ENV SKIP_CUDA=1 SKIP_ROCM=1 SKIP_VULKAN=1
RUN curl -LsSf https://llama.app/install.sh | sh

# Put `llama` on PATH
ENV PATH="/root/.local/bin:${PATH}"

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY chat_ui.py .

# HuggingFace cache volume — shared by chat (TTS auto-download) + llama.cpp (-hf model)
VOLUME ["/root/.cache/huggingface"]

EXPOSE 7777
EXPOSE 8080

ENV PYTHONUNBUFFERED=1

CMD ["python3", "chat_ui.py", "--host", "0.0.0.0", "--port", "7777"]
