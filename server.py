#!/usr/bin/env python3
"""
STS-45 — VAD → parakeet STT → llama.cpp LLM → Inflect-Nano-v2 TTS → speaker.

    Browser mic ──16kHz PCM──► ws :PORT ──► VAD -> STT -> LLM -> TTS
    Browser speaker ◄──PCM────────────────────────────────────────┘

Open http://localhost:PORT, tap the orb, and talk.
"""

import argparse
import json
import os
import signal
import sys
import threading
from pathlib import Path

import requests
from flask import Flask, render_template, request

try:
    import realtime as _realtime
except Exception as _rt_err:
    _realtime = None
    print(f"  Realtime module unavailable (voice mode disabled): {_rt_err}")

LLAMA_API = "http://127.0.0.1:8080/v1/chat/completions"
STT_API = "http://localhost:8081"

SYS_PROMPT = (
    "You are a friendly, helpful assistant. Respond in the same language as the user. "
    "Keep answers concise and natural for text-to-speech. "
    "Avoid markdown, lists, URLs, or special formatting. "
    "Use short to medium sentences. Avoid asterisks and emojis. "
    "Do NOT show your thinking or reasoning process. Answer directly."
)

app = Flask(__name__)

# Inflect-Nano-v2 TTS engine (loaded in background)
tts = None  # InflectTTS instance
tts_lock = threading.Lock()
MODEL_DIR = Path("models/inflect-nano-v2")

config = {
    "lang": "en",
    "voice": "inflect-nano-v2",
    "api_url": LLAMA_API,
    "stt_api_url": STT_API,
    "model": "default",
}

RT_WS_PORT = 7778
CLIENT_WS_PORT = int(os.environ.get("WS_CLIENT_PORT", 0))

# --- graceful shutdown ------------------------------------------------------
# Flask's dev server (Werkzeug) installs a handler for SIGINT (KeyboardInterrupt)
# but NOT SIGTERM, which is what `docker stop` sends. Without a SIGTERM handler
# Werkzeug swallows the interruption while blocked in accept()/select(), so the
# container ignores `docker stop` until Docker's 10s SIGKILL timeout -- the
# "can't stop the app" symptom. We catch both and exit promptly.
_exiting = threading.Event()

def _on_signal(signum, _frame):
    print(f"\n🛑 Caught signal {signum}, exiting…", flush=True)
    _exiting.set()
    # werkzeug.serving runs in the main thread; raising SystemExit there is the
    # only thing that reliably breaks its serve_forever(). os._exit is too harsh
    # (skips the realtime daemon cleanup) -- SystemExit lets daemon threads die
    # normally and Python finishes promptly.
    raise SystemExit(0)


def _download_model():
    """Download Inflect-Nano-v2 model from HuggingFace via snapshot_download."""
    from huggingface_hub import snapshot_download
    marker = MODEL_DIR / "model.pth"
    if marker.exists():
        print(f"  Inflect-Nano-v2 model already at {MODEL_DIR}")
        return True
    print("  Downloading Inflect-Nano-v2 from HuggingFace (~16 MB)...")
    snapshot_download(
        "owensong/Inflect-Nano-v2",
        local_dir=str(MODEL_DIR),
        ignore_patterns=[
            "evaluation/*", "samples/*", "assets/*", "docs/*",
            "onnx/*", "*.md", "*.cff", "release_manifest.json",
        ],
    )
    print(f"  Inflect-Nano-v2 ready at {MODEL_DIR}")
    return True


@app.route("/")
def index():
    return render_template(
        "index.html",
        ws_port=CLIENT_WS_PORT or RT_WS_PORT,
    )


@app.route("/api/settings", methods=["POST"])
def api_settings():
    data = request.get_json() or {}
    for key in ("lang", "voice", "api_url", "stt_api_url", "model"):
        if key in data and data[key] is not None:
            config[key] = data[key]
    return {"status": "ok"}


@app.route("/api/health")
def api_health():
    return {
        "tts_ready": tts is not None,
        "realtime": _realtime is not None,
        "config": config,
    }


def _load_tts_background():
    global tts
    try:
        _download_model()
        # Add model dir + runtime to path for imports
        sys.path.insert(0, str(MODEL_DIR))
        sys.path.insert(0, str(MODEL_DIR / "runtime"))
        from inference import InflectTTS
        tts = InflectTTS(MODEL_DIR, device="cpu")
        print(f"  Inflect-Nano-v2 TTS loaded ({tts.deployed_parameters:,} params, {tts.sample_rate} Hz)")
    except Exception as e:
        print(f"  Inflect-Nano-v2 loading failed (will retry on first use): {e}")


def main():
    global config, RT_WS_PORT

    parser = argparse.ArgumentParser(description="STS-45 (Inflect-Nano-v2 TTS)")
    parser.add_argument("--host", default="0.0.0.0", help="Host (0.0.0.0 for LAN access)")
    parser.add_argument("--port", type=int, default=7777, help="HTTP port")
    parser.add_argument("--ws-port", type=int, default=0, help="WS port (default: HTTP port + 1)")
    parser.add_argument("--api", default=os.environ.get("LLM_API", LLAMA_API), help="LLM API URL")
    parser.add_argument("--stt-api", default=os.environ.get("STT_API", STT_API), help="Parakeet STT server URL")
    parser.add_argument("--model", default="default", help="Model name")
    parser.add_argument("--voice", default="inflect-nano-v2", help="TTS voice")
    parser.add_argument("--lang", default="en", help="Language")
    args = parser.parse_args()

    config.update({
        "lang": args.lang,
        "voice": args.voice,
        "api_url": args.api,
        "stt_api_url": args.stt_api,
        "model": args.model,
    })

    RT_WS_PORT = args.ws_port or (args.port + 1)

    print("🚀 Loading Inflect-Nano-v2 TTS in background...")
    threading.Thread(target=_load_tts_background, daemon=True).start()

    if _realtime is not None:
        try:
            _realtime.start(args.host, RT_WS_PORT, sys.modules[__name__])
        except Exception as e:
            print(f"  Realtime server failed to start: {e}")
    else:
        print("  Realtime mode unavailable (websockets not installed).")

    print(f"""
╔════════════════════════════════════════╗
║   🎤 STS-45 (Inflect-Nano-v2)              ║
║   Open: http://{args.host}:{args.port}          ║
║   WS:   ws://{args.host}:{RT_WS_PORT}/ws           ║
║   LLM:  {args.api}        ║
║   STT:  {args.stt_api}           ║
║   Voice: {args.voice}  |  Lang: {args.lang}              ║
╚══════════════════════════════════════════╝
""")

    # Install signal handlers BEFORE app.run() so we own the main thread's
    # signal delivery instead of relying on Werkzeug's SIGINT-only handling.
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # Use make_server() so we own the server object; our SIGTERM/SIGINT handler
    # above raises SystemExit in the main thread, which breaks serve_forever()
    # promptly. This is what makes `docker stop` actually stop instead of
    # hanging until SIGKILL.
    try:
        from werkzeug.serving import make_server
        srv = make_server(args.host, args.port, app, threaded=True)
        srv.serve_forever()
    except SystemExit:
        print("✅ Server stopped.", flush=True)
    except KeyboardInterrupt:
        print("✅ Server stopped.", flush=True)


if __name__ == "__main__":
    main()