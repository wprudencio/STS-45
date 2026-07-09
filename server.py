#!/usr/bin/env python3
"""
STS-45 — VAD → parakeet STT → llama.cpp LLM → Piper TTS → speaker.

    Browser mic ──16kHz PCM──► ws :PORT ──► VAD -> STT -> LLM -> TTS
    Browser speaker ◄──PCM────────────────────────────────────────┘

Open http://localhost:PORT, tap the orb, and talk.
"""

import argparse
import json
import sys
import threading
from pathlib import Path

import requests
from flask import Flask, render_template, request
from piper import PiperVoice

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
    "Use short to medium sentences. Avoid asterisks and emojis."
)

app = Flask(__name__)

tts = {}
tts_lock = threading.Lock()
VOICE_DIR = Path("models/piper")
VOICE_DIR.mkdir(parents=True, exist_ok=True)

config = {
    "lang": "en",
    "voice": "en_US-lessac-medium",
    "api_url": LLAMA_API,
    "stt_api_url": STT_API,
    "model": "default",
}

RT_WS_PORT = 7778


def _voice_url(voice_name):
    """Construct huggingface download URL from voice name like en_US-lessac-medium."""
    parts = voice_name.split("-")
    if len(parts) < 3:
        return None
    region = parts[0]  # en_US
    lang = region.split("_")[0]  # en
    speaker = parts[1]  # lessac
    quality = "-".join(parts[2:])  # medium or low or high
    base = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
    return f"{base}/{lang}/{region}/{speaker}/{quality}/{voice_name}"


def _download_voice(voice_name):
    onnx_path = VOICE_DIR / f"{voice_name}.onnx"
    json_path = VOICE_DIR / f"{voice_name}.onnx.json"
    if onnx_path.exists() and json_path.exists():
        return onnx_path
    url = _voice_url(voice_name)
    if url is None:
        return None
    print(f"  Downloading {voice_name}...")
    for ext, path in [(".onnx", onnx_path), (".onnx.json", json_path)]:
        if not path.exists():
            r = requests.get(f"{url}{ext}", timeout=300)
            r.raise_for_status()
            path.write_bytes(r.content)
    print(f"  {voice_name} ready")
    return onnx_path


@app.route("/")
def index():
    return render_template(
        "index.html",
        default_api_url=config["api_url"],
        default_stt_api_url=config["stt_api_url"],
        ws_port=RT_WS_PORT,
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
        "tts_ready": bool(tts),
        "realtime": _realtime is not None,
        "config": config,
    }


def _load_tts_background():
    global tts
    try:
        default_voice = config["voice"]
        path = _download_voice(default_voice)
        voice = PiperVoice.load(str(path))
        tts[default_voice] = (voice, 22050)
        print(f"  Piper TTS loaded ({default_voice})")
    except Exception as e:
        print(f"  Piper loading failed (will retry on first use): {e}")


def main():
    global config, RT_WS_PORT

    parser = argparse.ArgumentParser(description="STS-45 (Piper TTS)")
    parser.add_argument("--host", default="0.0.0.0", help="Host (0.0.0.0 for LAN access)")
    parser.add_argument("--port", type=int, default=7777, help="HTTP port")
    parser.add_argument("--ws-port", type=int, default=0, help="WS port (default: HTTP port + 1)")
    parser.add_argument("--api", default=LLAMA_API, help="LLM API URL")
    parser.add_argument("--stt-api", default=STT_API, help="Parakeet STT server URL")
    parser.add_argument("--model", default="default", help="Model name")
    parser.add_argument("--voice", default="en_US-lessac-medium", help="Piper voice")
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

    print("🚀 Loading Piper TTS in background...")
    threading.Thread(target=_load_tts_background, daemon=True).start()

    if _realtime is not None:
        try:
            _realtime.start(args.host, RT_WS_PORT, sys.modules[__name__])
        except Exception as e:
            print(f"  Realtime server failed to start: {e}")
    else:
        print("  Realtime mode unavailable (websockets not installed).")

    print(f"""
╔══════════════════════════════════════════╗
║   🎤 STS-45 (Piper TTS)                  ║
║   Open: http://{args.host}:{args.port}          ║
║   WS:   ws://{args.host}:{RT_WS_PORT}/ws           ║
║   LLM:  {args.api}        ║
║   STT:  {args.stt_api}           ║
║   Voice: {args.voice}  |  Lang: {args.lang}              ║
╚══════════════════════════════════════════╝
""")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()