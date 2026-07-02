FROM python:3.11-slim

WORKDIR /app

# System deps for ONNX Runtime + audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY realtime_server.py .
COPY chat_ui.py .
COPY templates/ ./templates/

# HuggingFace cache volume for model persistence
VOLUME ["/root/.cache/huggingface"]

EXPOSE 7777

ENV PYTHONUNBUFFERED=1

CMD ["python3", "realtime_server.py", "--host", "0.0.0.0", "--port", "7777"]
