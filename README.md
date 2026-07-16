# STS-45

Hands-free speech-to-speech voice assistant. Tap the orb, talk, it replies — everything runs on your machine, **no Docker required**.

```
Browser mic ──16kHz PCM──► Nginx :7777 ──► WS :7779 ──► VAD → parakeet STT → llama.cpp LLM → Piper TTS
                          (single URL)                                                      ↓ per clause
Browser speaker ◄──PCM──────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start — Local (no Docker, recommended)

One script sets up everything: installs nginx, downloads models, creates the Python environment, and configures the unified reverse proxy on port `7777`.

### Prerequisites

- **Ubuntu 24.04** (or Debian-based Linux). Other distros work but may need adapted package names.
- **Python 3.12+** (`python3 --version` to check). Piper TTS works on Python ≥ 3.9.
- **~1 GB free disk space** for models (~2.1 GB LLM, ~260 MB STT, ~2.1 GB parakeet binary).
- **sudo access** (for installing nginx).
- **Headphones recommended** to avoid speaker → mic echo.

### 1. Clone the repository

```bash
git clone git@github.com:wprudencio/STS-45.git
cd STS-45
git checkout local-setup   # this branch — local-first setup
```

### 2. Run the setup script

```bash
chmod +x setup.sh
./setup.sh
```

The script will:

1. Install system packages: `nginx`, `curl`, `wget`, `libgomp1`, `python3-pip`, `python3-venv`
2. Create a Python virtual environment (`.venv/`) and install `piper-tts`, `flask`, `websockets`, etc.
3. Download **llama.cpp** via the official [llama.app](https://llama.app) installer
4. Download the **LLM model** (`unsloth/gemma-4-E2B-it-qat-GGUF`, ~2.1 GB)
5. Download the **parakeet-server** binary (v0.4.0, ~2.7 MB) into `bin/`
6. Download the **STT model** (`tdt_ctc-110m-f16.gguf`, ~260 MB)
7. Configure **nginx** as a reverse proxy (port `7777` → Flask `7778` + WS `7779`)

> **Network note**: The LLM and STT models are ~400 MB total. On a slow connection the download may take several minutes.

### 3. Start the LLM server (llama.cpp)

In a **separate terminal** (or tmux pane), run:

```bash
cd ~/STS-45
export PATH="$HOME/.local/bin:$PATH"
llama-server serve -m models/gemma-4-E2B-UD-Q2.gguf --host 127.0.0.1 --port 8080 --no-kv-offload -c 2048
```

> **Why `--no-kv-offload`?** Some architectures needs this flag on CPU-only setups to avoid a segfault during model loading. If you're on a GPU machine, you can drop it (and add `-ngl 99`).

> **First load is slow**: The model takes 1–3 seconds to load on the first request. Subsequent requests are instant.

Wait for the log message `listening on http://127.0.0.1:8080` before proceeding.

### 4. Start everything else

```bash
cd ~/STS-45
./start.sh
```

This script will:

1. Start **parakeet-server** (STT) on port `8081`
2. Detect that `llama-server` is already running on `8080` and skip it
3. Verify nginx is running on port `7777`
4. Start the **Flask + WebSocket** server on ports `7778` (HTTP) and `7779` (WS)

### 5. Open the orb

Open **[http://localhost:7777](http://localhost:7777)** in your browser.

You'll see the amber CRT-style orb interface. Tap the orb, grant microphone access, and start talking. The assistant will:

1. Listen until you pause (~650 ms of silence)
2. Transcribe your speech via parakeet STT
3. Generate a reply via the LLM
4. Speak it back via Piper TTS (streaming, sentence by sentence)

### 6. Stop everything

Press `Ctrl+C` in the terminal where `start.sh` is running. It will gracefully shut down parakeet and the Flask server.

---

## One-command start (if you have tmux or a terminal multiplexer)

Skip step 3 and let `start.sh` handle the LLM too. It will automatically find the `llama-server` binary and start it:

```bash
cd ~/STS-45
./start.sh
```

The script auto-detects whether `llama-server` is already running on port `8080`. If not, it searches for the binary in common locations and starts it with the correct flags.

---

## Architecture (local, no Docker)

### Service layout

```
                        ┌──────────────────────────────────────────┐
                        │           Nginx (port 7777)              │
                        │  Unified entry point — single URL        │
                        └──────┬────────────────────┬──────────────┘
                               │                    │
                        ┌──────▼──────┐    ┌────────▼────────┐
                        │  / → Flask  │    │  /ws → WS       │
                        │  :7778      │    │  :7779          │
                        └──────┬──────┘    └────────┬────────┘
                               │                    │
                        ┌──────▼────────────────────▼──────────┐
                        │         Flask + WebSocket App         │
                        │  VAD │ STT │ LLM │ TTS pipeline      │
                        └──────┬──────────┬─────────────────────┘
                               │          │
                        ┌──────▼──┐ ┌─────▼──────┐
                        │ parakeet│ │ llama.cpp  │
                        │ STT     │ │ LLM        │
                        │ :8081   │ │ :8080      │
                        └─────────┘ └────────────┘
```

### Port map

| Port | Service | Purpose |
|------|---------|---------|
| `7777` | **Nginx** | **Single URL** — proxies `/` → Flask, `/ws` → WS |
| `7778` | Flask (HTTP) | Serves the UI, settings API, health endpoint |
| `7779` | WebSocket | Binary PCM16 stream (mic → server → TTS audio) |
| `8080` | llama.cpp | OpenAI-compatible chat completions API |
| `8081` | parakeet.cpp | STT transcription API (`/v1/audio/transcriptions`) |

### WebSocket protocol

The browser streams 16 kHz PCM16 mono audio over the WebSocket. The server runs an energy-based VAD (voice activity detection) to find utterance boundaries, transcribes each utterance, streams the LLM reply, and speaks it back sentence-by-sentence as PCM16 frames. Audio starts playing before the full reply is generated.

See `realtime.py` for the full protocol specification.

---

## Setup details (what each script does)

### `setup.sh` — One-time installation

```bash
./setup.sh
```

| Step | What it does | Downloads |
|------|-------------|-----------|
| System packages | `apt install nginx curl wget libgomp1 python3-pip python3-venv` | — |
| Python venv | Creates `.venv/`, installs `piper-tts flask numpy requests websockets` | — |
| llama.cpp | Runs `curl -LsSf https://llama.app/install.sh \| sh` | ~16 MB binary |
| LLM model | Downloads `gemma-4-E2B-UD-Q2.gguf` to `models/` | ~2.1 GB |
| parakeet binary | Downloads `parakeet-server` v0.4.0 to `bin/` | ~2.7 MB |
| STT model | Downloads `tdt_ctc-110m-f16.gguf` to `models/` | ~260 MB |
| Nginx config | Copies `docker/nginx/default.conf` → `/etc/nginx/sites-available/sts45`, enables it, restarts nginx | — |

### `start.sh` — Start all services

```bash
./start.sh
```

1. Activates the Python virtual environment
2. Kills any stale `parakeet-server` or `server.py` processes
3. Starts `parakeet-server` on port `8081`
4. Checks if `llama-server` is already running on `8080`; if not, finds the binary and starts it with `--no-kv-offload`
5. Ensures nginx is running
6. Starts the Flask+WS server (`server.py`) with `WS_CLIENT_PORT=7777`
7. Waits for all services; prints the access URL

Press `Ctrl+C` to stop everything gracefully.

---

## Manual start (for debugging or custom setups)

### Start each service individually

```bash
# Terminal 1: LLM
export PATH="$HOME/.local/bin:$PATH"
llama-server serve -m models/gemma-4-E2B-UD-Q2.gguf --host 127.0.0.1 --port 8080 --no-kv-offload

# Terminal 2: STT
./bin/parakeet-server --model models/tdt_ctc-110m-f16.gguf --port 8081

# Terminal 3: Flask + WS
source .venv/bin/activate
export WS_CLIENT_PORT=7777
python3 server.py --host 127.0.0.1 --port 7778 --ws-port 7779 \
    --stt-api http://localhost:8081 \
    --api http://127.0.0.1:8080/v1/chat/completions
```

### Start without nginx (direct Flask ports)

If you don't want to use nginx, you can run the traditional `run.sh`:

```bash
./run.sh
```

This starts the Flask server on port `7777` and the WebSocket server on port `7778` (port + 1). Open `http://localhost:7777`. Note that the JS will connect to `ws://localhost:7778/ws` directly (two ports instead of one).

---

## Files and directories

```
├── server.py              Flask web server + background Piper TTS loader
├── realtime.py            WebSocket pipeline (VAD → STT → LLM → TTS)
├── setup.sh               One-time installation (nginx, models, venv)
├── start.sh               Start all services (parakeet → llama → Flask/WS → nginx)
├── run.sh                 Legacy start script (no nginx, direct Flask ports)
├── requirements.txt       Python dependencies
│
├── static/
│   ├── app.js             WebSocket client, mic capture, audio playback, settings UI
│   └── styles.css         Amber CRT pixel UI, orb animation, settings drawer
│
├── templates/
│   └── index.html         Single-page orb + transcript + settings drawer
│
├── docker/
│   └── nginx/
│       └── default.conf   Nginx config (proxies :7777 → :7778 + :7779)
│
├── bin/                   parakeet-server binary (created by setup.sh, gitignored)
├── models/                LLM, STT, Piper voice models (created by setup.sh, gitignored)
└── .venv/                 Python virtual environment (created by setup.sh, gitignored)
```

---

## Quick Start — Docker (alternative)

If you prefer Docker, the traditional method still works:

```bash
git clone git@github.com:wprudencio/STS-45.git
cd STS-45
git checkout piper-tts
docker compose up
```

Then open **[http://localhost:7777](http://localhost:7777)**.

See the original README sections below for Docker-specific details.

---

## Voices

15 Piper voices across 8 languages. Select from the Settings drawer (gear icon, top-right):

| Voice | Language | Gender |
|-------|----------|--------|
| `en_US-lessac-medium` | English (US) | Female |
| `en_US-amy-low` | English (US) | Female |
| `en_US-libritts-high` | English (US) | Female |
| `en_US-ryan-high` | English (US) | Male |
| `en_US-joe-medium` | English (US) | Male |
| `en_US-kusal-medium` | English (US) | Male |
| `en_GB-alan-medium` | English (UK) | Male |
| `en_GB-semaine-medium` | English (UK) | Female |
| `pt_BR-faber-medium` | Portuguese (BR) | Male |
| `es_ES-carlfm-x_low` | Spanish (ES) | Male |
| `fr_FR-siwis-medium` | French (FR) | Female |
| `de_DE-thorsten-medium` | German (DE) | Male |
| `it_IT-paola-medium` | Italian (IT) | Female |

New voices auto-download from HuggingFace on first use and are cached in `models/piper/`.

## Settings

Click the gear icon (top-right) to open the settings drawer. All fields auto-save to `localStorage`:

| Setting | Description |
|---------|-------------|
| **Voice** | Piper voice (determines spoken language) |
| **Language** | Used by parakeet STT for transcription |
| **Max tokens** | Maximum LLM response length |
| **System prompt** | Custom instructions for the assistant |

## Troubleshooting

### `llama-server` segfaults

Some models requires `--no-kv-offload` on CPU-only systems. The `start.sh` script already includes this flag. If starting manually:

```bash
llama-server serve -m models/gemma-4-E2B-UD-Q2.gguf --host 127.0.0.1 --port 8080 --no-kv-offload
```

### Nginx returns 502 Bad Gateway

Flask isn't running or crashed. Check:

```bash
# Is Flask running?
curl http://127.0.0.1:7778/api/health

# If not, check the Flask log (if redirected to a file) or restart:
./start.sh
```

### WebSocket disconnects immediately (1011 error)

This usually means the TTS model hasn't finished loading. Wait a few seconds after starting, or check:

```bash
curl http://localhost:7777/api/health   # should show "tts_ready": true
```

### "No module named piper"

The Python virtual environment isn't activated or wasn't created:

```bash
cd ~/STS-45
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### `parakeet-server` crashes with `libgomp1` error

Missing OpenMP library:

```bash
sudo apt install libgomp1
```

### STT model not found

```bash
cd ~/STS-45/models
curl -L -o tdt_ctc-110m-f16.gguf https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/tdt_ctc-110m-f16.gguf
```

### No audio from the assistant

1. Check the browser console (F12) for errors
2. Make sure the page is not muted
3. Click the page once to allow audio playback (browsers require user gesture)
4. Check the AudioContext state in console: `navigator.mediaDevices.getUserMedia({audio:true})`

### "LLM returned no content" toast

`llama-server` isn't running or the API URL is wrong:

```bash
# Verify it's running:
curl http://127.0.0.1:8080/v1/models

# If not, start it:
export PATH="$HOME/.local/bin:$PATH"
llama-server serve -m models/gemma-4-E2B-UD-Q2.gguf --host 127.0.0.1 --port 8080 --no-kv-offload
```

### Port already in use

```bash
# Find what's using the port:
sudo ss -tlnp | grep 7777

# Kill the process:
sudo kill <PID>
```

### LLM responds too slowly

The Gemma 4 E2B Q2 model runs at ~10 tokens/second on CPU. A typical 50-token response takes ~5 seconds. If it's slower:

- Close other CPU-intensive applications
- Reduce `--ctx-size` (e.g., `-c 1024`)
- Use a smaller model (try `Q4_0` quantization instead of `Q4_K_M`)

---

## Override the LLM model

To use a different GGUF model, edit the model path in `start.sh` or pass environment variables:

```bash
LLM_API=http://127.0.0.1:8080/v1/chat/completions LLM_MODEL=my-model ./start.sh
```

To download a different quantization:

```bash
cd models
wget https://huggingface.co/unsloth/gemma-4-E2B-it-qat-GGUF/resolve/main/gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf
# Then update start.sh to point to this file
```

Available quantizations: `Q4_0`, `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`, `F16`, `BF16`.

---

## Cloudflare Tunnel (public URL)

Expose your local STS-45 to the internet via **Cloudflare Tunnel** so you — or anyone you share the link with — can use it from anywhere. No static IP, no port forwarding, no DNS config needed.

### Quick tunnel (random URL, no account required)

```bash
./start.sh --cf
```

This starts a `cloudflared` tunnel that creates a random `*.trycloudflare.com` URL (e.g., `https://random-string.trycloudflare.com`). The URL appears in the terminal output once the tunnel is ready (~5 seconds).

Or with an environment variable:

```bash
CLOUDFLARE=1 ./start.sh
```

### Persistent tunnel (your own domain, requires a Cloudflare account)

1. **Log in** (one-time):
   ```bash
   cloudflared tunnel login
   ```

2. **Create a tunnel** (one-time):
   ```bash
   cloudflared tunnel create sts45
   ```
   This creates a credentials file and a tunnel ID.

3. **Create the config file** at `~/.cloudflared/sts45.yml`:
   ```yaml
   tunnel: <tunnel-id-from-step-2>
   credentials-file: /home/$USER/.cloudflared/<tunnel-id>.json
   
   ingress:
     - hostname: voice.yourdomain.com
       service: http://localhost:7777
     - service: http_status:404
   ```

4. **Route DNS** (replace `voice.yourdomain.com` with your domain):
   ```bash
   cloudflared tunnel route dns sts45 voice.yourdomain.com
   ```

5. **Run with your domain**:
   ```bash
   ./start.sh --cf voice.yourdomain.com
   ```

Now open `https://voice.yourdomain.com` anywhere.

### How it works

```
User's browser ──► Cloudflare Edge ──► cloudflared tunnel ──► localhost:7777
                        (TLS)              (persistent)         (nginx)
```

The tunnel handles TLS termination, DDoS protection, and WebSocket support automatically. The WebSocket connection for realtime audio works seamlessly through the tunnel.

### Notes

- The quick tunnel URL changes every restart. Use a persistent tunnel for a fixed URL.
- WebSocket connections (`wss://`) work automatically through Cloudflare — no extra config needed.
- The tunnel adds ~50-100 ms of latency, which is fine for voice conversations.

---

## Stack

| Role | Tech | Port |
|------|------|------|
| Reverse proxy | **Nginx** | `7777` |
| Cloudflare Tunnel | **cloudflared** | tunnel → `:7777` |
| STT | [parakeet.cpp](https://github.com/mudler/parakeet.cpp) v0.4.0 | `8081` |
| LLM | [llama.cpp](https://github.com/ggerganov/llama.cpp) (via [llama.app](https://llama.app)) + `unsloth/gemma-4-E2B-it-qat-GGUF` | `8080` |
| TTS | [Piper](https://github.com/rhasspy/piper) (ONNX Runtime) | in-process |
| UI | Flask + vanilla JS | `7778` (HTTP), `7779` (WS) |
