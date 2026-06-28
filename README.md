# Supertonic — Voice Chat

Local voice assistant: mic → STT → LLM → TTS → speaker. Everything on-device.

```
Browser mic ──→ parakeet.cpp ──→ llama.cpp ──→ Supertonic TTS ──→ audio
               (STT)            (LLM)         (ONNX Runtime)
```

## Quick Start

```bash
# 1. Start llama.cpp on host
llama-server -m your-model.gguf --port 8080

# 2. Start everything (STT model auto-downloads on first run)
docker compose up -d

# 3. Open http://localhost:7777
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
