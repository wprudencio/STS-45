FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# onnxruntime (used by piper-tts) needs libgomp
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY server.py realtime.py ./
COPY static/ ./static/
COPY templates/ ./templates/

# Piper voice models live here; mount as a volume to persist downloads
RUN mkdir -p /app/models/piper

# Ports are internal-only; nginx is the single external entry point (7777)
EXPOSE 7777 7778

CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "7777", "--voice", "en_US-amy-medium"]