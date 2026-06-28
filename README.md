# Supertonic — Voice Chat

Local voice assistant: mic → STT → LLM → TTS → speaker. Everything on-device.

```
Browser mic ──→ parakeet.cpp ──→ llama.cpp ──→ Supertonic TTS ──→ audio
               (STT)            (LLM)         (ONNX Runtime)
```

## Quick Start

```bash
# 1. Download STT model (one-time)
mkdir -p models
curl -L -o models/tdt_ctc-110m-q5_k.gguf \
  https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/tdt_ctc-110m-q5_k.gguf

# 2. Start llama.cpp on host
llama-server -m your-model.gguf --port 8080

# 3. Start everything else
docker compose up -d

# 4. Open http://localhost:7777
```

## Manual

```bash
pip install -r requirements.txt
python chat_ui.py --port 7777 --stt-api http://localhost:8081
```

## Stack

| Role | Tech | Port |
|------|------|------|
| STT | [parakeet.cpp](https://github.com/mudler/parakeet.cpp) (NVIDIA NeMo → ggml) | `:8081` |
| LLM | [llama.cpp](https://github.com/ggerganov/llama.cpp) server | `:8080` |
| TTS | [Supertonic 3](https://github.com/supertone-inc/supertonic) (ONNX Runtime) | — |
| UI | Flask + vanilla JS | `:7777` |

## Usage

Hold mic or Space to talk, release to transcribe, Enter to send. Edit voice, language, and system prompt in the left panel.
