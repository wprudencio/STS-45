# Supertonic — Voice Chat

On-device voice assistant: speak via microphone, get spoken responses. **Everything local** — TTS, STT, and LLM all run on your machine.

## Architecture

```
Microphone ──→ [parakeet.cpp] ──→ text ──→ [llama.cpp] ──→ text ──→ [Supertonic TTS] ──→ Speaker
  (browser)     STT (NVIDIA NeMo)         LLM (local)               TTS (ONNX Runtime)
```

| Component | Technology | Runtime |
|-----------|-----------|---------|
| **STT** | parakeet.cpp (NVIDIA Parakeet → ggml) | CPU, 110M params, English |
| **LLM** | llama.cpp (OpenAI-compatible API) | CPU/GPU, any GGUF model |
| **TTS** | Supertonic 3 (flow-matching) | ONNX Runtime, 99M params, 31 langs |
| **UI** | Flask + vanilla JS (monochrome) | Browser |

Everything runs **on-device** — no cloud APIs, no audio leaves your machine.

## Quick Start — Docker Compose

```bash
# 1. Download the STT model (one-time)
mkdir -p models
curl -L -o models/tdt_ctc-110m-q5_k.gguf \
  https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/tdt_ctc-110m-q5_k.gguf

# 2. Start everything
docker compose up -d

# 3. Open http://localhost:7777
```

| Service | Port | Container |
|---------|------|-----------|
| Chat UI (TTS + web) | `7777` | `supertonic-chat` |
| STT (parakeet.cpp) | `8081` | `supertonic-stt` |
| LLM (llama.cpp) | `8080` | runs on host, not in compose |

> **Note:** llama.cpp is not included in docker-compose — start it separately on your host:
> ```bash
> llama-server -m your-model.gguf --port 8080
> ```

## Manual Setup

```bash
pip install -r requirements.txt
python chat_ui.py --port 7777 --stt-api http://localhost:8081
```

### Dependencies

- **Python 3.11+** with `supertonic`, `flask`, `numpy`, `requests`, `sounddevice`
- **parakeet-server** — [parakeet.cpp](https://github.com/mudler/parakeet.cpp) running on `:8081`
- **llama-server** — [llama.cpp](https://github.com/ggerganov/llama.cpp) on `:8080`

## Usage

Open `http://localhost:7777` in your browser:

| Action | How |
|--------|-----|
| **Talk** | Hold mic button or Space key, release to transcribe |
| **Send** | Press Enter or click Send |
| **Change voice** | Left panel → Voice dropdown |
| **Change language** | Left panel → Language dropdown |
| **Edit system prompt** | Left panel → System Prompt textarea |
| **Configure APIs** | Left panel → Configure (LLM URL, STT URL) |
| **Clear chat** | Left panel → Clear Conversation |

The assistant replies are streamed sentence-by-sentence — TTS starts playing while the LLM is still generating.

## TUI

A terminal-based version is also available:

```bash
python chat.py          # Text input (no mic)
python tui.py           # Optional TUI variant
```

## Models

| Model | Size | Purpose | Source |
|-------|------|---------|--------|
| Supertonic 3 | ~260MB | TTS (auto-downloaded) | [HuggingFace](https://huggingface.co/Supertone/supertonic-3) |
| Parakeet TDT-CTC 110M q5_k | ~137MB | STT (English) | [HuggingFace](https://huggingface.co/mudler/parakeet-cpp-gguf) |
| llama.cpp GGUF | varies | LLM | your choice |

### Multilingual STT

For non-English speech recognition, download a multilingual Parakeet model:

```bash
# 25 European languages
curl -L -o models/tdt-0.6b-v3-q5_k.gguf \
  https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/parakeet-tdt-0.6b-v3-q5_k.gguf

# Update docker-compose.yml command:
#   --model /models/tdt-0.6b-v3-q5_k.gguf
```

## File Overview

```
├── chat.py              # Terminal chat (text input + TTS)
├── chat_ui.py           # Web UI (mic + TTS + LLM streaming)
├── tui.py               # Terminal UI variant
├── docker-compose.yml   # Chat UI + Parakeet STT
├── Dockerfile           # Chat UI image
├── requirements.txt     # Python dependencies
├── models/              # Parakeet GGUF models (gitignored)
└── py/                  # Original Supertonic Python examples
```

## License

This project's sample code is released under the MIT License. The Supertonic model is under OpenRAIL-M. Parakeet models are under their respective NVIDIA licenses.
