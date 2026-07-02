#!/usr/bin/env python3
"""
Supertonic Voice Chat — Web UI com microfone + teclado + LLM + TTS local.
"""

import argparse
import json
import sys
import threading
import time
import base64
import io
import wave
from typing import Optional

import numpy as np
import requests
from flask import Flask, Response, request, render_template_string
from supertonic import TTS

try:
    import realtime as _realtime
except Exception as _rt_err:  # websockets missing -> core chat still works
    _realtime = None
    print(f"⚠️  Realtime module unavailable (voice mode disabled): {_rt_err}")

LLAMA_API = "http://127.0.0.1:8080/v1/chat/completions"
STT_API = "http://localhost:8080"

# Realtime WebSocket port (set in main(); one above the HTTP port by default).
RT_WS_PORT = 7778

SYS_PROMPT = (
    "You are a friendly, helpful assistant. Respond in the same language as the user. "
    "Keep answers concise and natural for text-to-speech. "
    "Avoid markdown, lists, URLs, or special formatting. "
    "Use short to medium sentences. Avoid asterisks and emojis."
)

app = Flask(__name__)

# Estado global (carregado no startup)
tts = None
style = None
tts_lock = threading.Lock()

# Configurações
config = {
    "lang": "en",
    "voice": "M1",
    "steps": 5,
    "speed": 1.15,
    "api_url": LLAMA_API,
    "stt_api_url": STT_API,
    "model": "default",
}

# Histórico do LLM
messages = [{"role": "system", "content": SYS_PROMPT}]


# --- Streaming chat endpoint ---
@app.route("/api/chat", methods=["POST"])
def api_chat():
    global messages

    data = request.get_json()
    user_msg = data.get("message", "").strip()
    if not user_msg:
        return {"error": "empty message"}, 400

    # Limpa histórico se pedir
    if user_msg.lower() in ("clear", "/clear"):
        messages = [{"role": "system", "content": SYS_PROMPT}]
        return {"status": "cleared", "message": "History cleared"}

    # Atualiza config se veio do frontend
    for key in ("lang", "voice", "steps", "speed"):
        if key in data:
            config[key] = data[key]
    # Aceita API URL, key, e system prompt do frontend
    api_url = data.get("api_url", config["api_url"]).strip()
    api_key = data.get("api_key", "").strip()
    sys_prompt = data.get("sys_prompt", "").strip()
    if sys_prompt:
        messages[0] = {"role": "system", "content": sys_prompt}
    if not api_url:
        api_url = config["api_url"]
    if "voice" in data and tts is not None:
        try:
            global style
            style = tts.get_voice_style(voice_name=data["voice"])
        except:
            pass

    def generate():
        global messages

        # Stream do LLM
        messages.append({"role": "user", "content": user_msg})
        payload = {
            "model": config["model"],
            "messages": messages,
            "stream": True,
            "max_tokens": int(data.get("max_tokens", 2048)),
            "temperature": 0.7,
        }

        # Headers com auth se tiver API key
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            r = requests.post(api_url, json=payload, headers=headers, stream=True, timeout=120)
            r.raise_for_status()
            r.encoding = 'utf-8'
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"
            return

        content_parts = []
        has_content = False

        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                d = line[6:]
                if d.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(d)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})

                reasoning = delta.get("reasoning_content")
                content = delta.get("content")

                if reasoning and not has_content:
                    yield f"data: {json.dumps({'type': 'reasoning', 'text': reasoning})}\n\n"
                elif content:
                    has_content = True
                    content_parts.append(content)
                    yield f"data: {json.dumps({'type': 'text', 'text': content})}\n\n"

        full = "".join(content_parts)
        messages.append({"role": "assistant", "content": full})

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


# --- Load conversation history ---
@app.route("/api/chat/load", methods=["POST"])
def api_chat_load():
    global messages
    data = request.get_json()
    msgs = data.get("messages", [])
    sys_prompt = data.get("sys_prompt", "").strip()
    if sys_prompt:
        messages = [{"role": "system", "content": sys_prompt}] + msgs
    else:
        messages = [{"role": "system", "content": SYS_PROMPT}] + msgs
    return {"status": "ok", "count": len(messages)}

# --- Standalone TTS endpoint (for play-message feature) ---
@app.route("/api/tts", methods=["POST"])
def api_tts():
    data = request.get_json()
    text = (data.get("text", "")).strip()
    if not text:
        return {"error": "empty text"}, 400
    lang = data.get("lang", config["lang"])
    voice = data.get("voice", config["voice"])
    steps = int(data.get("steps", config["steps"]))
    speed = float(data.get("speed", config["speed"]))
    if tts is None:
        return {"error": "TTS still loading (model download in progress)"}, 503
    try:
        vs = tts.get_voice_style(voice_name=voice)
        with tts_lock:
            wav, _dur = tts.synthesize(text=text, lang=lang, voice_style=vs, total_steps=steps, speed=speed)
        b64 = wav_to_base64(wav, tts.sample_rate)
        return {"audio": b64}
    except Exception as e:
        return {"error": str(e)}, 500

# --- STT endpoint (proxies to parakeet.cpp server) ---
@app.route("/api/stt", methods=["POST"])
def api_stt():
    if "file" not in request.files:
        return {"error": "no file"}, 400

    file = request.files["file"]
    lang = request.form.get("lang", "en")
    stt_api_url = request.form.get("stt_api", config.get("stt_api_url", STT_API)).strip()

    lang_map = {
        "en": "en", "pt": "pt", "es": "es", "fr": "fr",
        "de": "de", "ja": "ja", "ko": "ko",
    }
    parakeet_lang = lang_map.get(lang, "en")

    try:
        url = stt_api_url.rstrip("/") + "/v1/audio/transcriptions"
        resp = requests.post(
            url,
            files={"file": (file.filename or "audio.wav", file.read(), "audio/wav")},
            data={"language": parakeet_lang, "response_format": "json"},
            timeout=30,
        )
        resp.raise_for_status()
        return {"text": resp.json().get("text", "")}
    except Exception as e:
        return {"error": str(e)}, 500


def wav_to_base64(wav: np.ndarray, sample_rate: int) -> str:
    """Converte numpy WAV para base64 (WAV format)."""
    wav_mono = wav.squeeze()
    wav_int16 = (np.clip(wav_mono, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(wav_int16.tobytes())
    return base64.b64encode(buf.getvalue()).decode()


# --- HTML UI ---
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Supertonic — Voice Chat</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Geist+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg: #2a2a2a;
      --bg2: #303030;
      --surface: #383838;
      --surface-hover: #424242;
      --border: #4a4a4a;
      --border-strong: #5a5a5a;
      --charcoal: #f5f5f5;
      --char2: #d4d4d4;
      --mid: #a3a3a3;
      --light: #8a8a8a;
      --orange: #ea580c;
      --orange-soft: rgba(234,88,12,0.15);
      --soft: #fb923c;
      --success: #16a34a;
      --error: #dc2626;
      --code-bg: #2f2f2f;

      --sp1: 8px;
      --sp2: 16px;
      --sp3: 24px;
      --sp4: 40px;
      --r1: 6px;
      --r2: 10px;
      --r3: 14px;
      --font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      --mono: 'Geist Mono', 'JetBrains Mono', ui-monospace, monospace;
      --dur: 180ms;
      --ease: cubic-bezier(0.25, 0, 0, 1);
    }

    :root.light {
      --bg: #fafafa;
      --bg2: #ffffff;
      --surface: #ffffff;
      --surface-hover: #f5f5f5;
      --border: #ececec;
      --border-strong: #d4d4d4;
      --charcoal: #0a0a0a;
      --char2: #404040;
      --mid: #737373;
      --light: #a3a3a3;
      --orange: #ea580c;
      --orange-soft: #fff7ed;
      --soft: #fb923c;
      --success: #16a34a;
      --error: #dc2626;
      --code-bg: #f4f4f5;
    }

    html, body { height: 100%; overflow: hidden; }

    body {
      background: var(--bg);
      color: var(--charcoal);
      font-family: var(--font);
      font-weight: 400;
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
      display: flex;
      flex-direction: column;
    }

    button { font-family: var(--font); }

    /* TOPBAR */
    .topbar {
      height: 52px;
      flex-shrink: 0;
      background: var(--bg);
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 var(--sp3);
      z-index: 50;
    }

    .brand { display: flex; align-items: center; gap: 10px; }

    .brand-mark {
      width: 20px;
      height: 20px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      grid-template-rows: 1fr 1fr;
      gap: 2px;
    }
    .brand-mark span {
      background: var(--orange);
      border-radius: 2px;
    }
    .brand-mark span:nth-child(2),
    .brand-mark span:nth-child(3) { background: var(--border); }

    .brand-name {
      font-size: 13px;
      font-weight: 600;
      color: var(--charcoal);
      letter-spacing: -0.02em;
    }

    .topbar-center { display: flex; align-items: center; gap: 8px; }

    .status-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--light);
      transition: background var(--dur);
    }
    .status-dot.active {
      background: var(--orange);
      box-shadow: 0 0 0 4px rgba(234,88,12,0.15);
      animation: pulse 2.5s ease-in-out infinite;
    }
    .status-dot.rec {
      background: var(--error);
      box-shadow: 0 0 0 4px rgba(220,38,38,0.15);
      animation: pulse 0.9s ease-in-out infinite;
    }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

    .status-text {
      font-size: 11px;
      font-weight: 500;
      color: var(--mid);
      letter-spacing: 0.02em;
    }

    .topbar-right { display: flex; align-items: center; gap: 2px; }

    .icon-btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 30px;
      height: 30px;
      background: none;
      border: 1px solid transparent;
      color: var(--mid);
      cursor: pointer;
      border-radius: var(--r1);
      transition: all var(--dur);
    }
    .icon-btn:hover { background: var(--surface); color: var(--charcoal); }
    .icon-btn.active { color: var(--orange); background: var(--orange-soft); }
    .icon-btn svg { width: 15px; height: 15px; }

    .live-pill {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 3px 8px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 999px;
      font-family: var(--mono);
      font-size: 10px;
      color: var(--mid);
      letter-spacing: 0.02em;
    }
    .live-pill strong { color: var(--charcoal); font-weight: 500; }

    /* SHELL */
    .shell {
      flex: 1;
      min-height: 0;
      display: grid;
      grid-template-columns: 260px 1fr;
      overflow: hidden;
    }

    /* PANEL LEFT */
    .panel-left {
      background: var(--bg2);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .panel-section {
      padding: 16px var(--sp3);
      border-bottom: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .section-label {
      font-size: 10px;
      font-weight: 600;
      color: var(--light);
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }

    .section-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .section-link {
      font-size: 11px;
      color: var(--orange);
      cursor: pointer;
      font-weight: 500;
      letter-spacing: 0.02em;
    }
    .section-link:hover { text-decoration: underline; }

    .select-wrap { position: relative; }
    .select-wrap::after {
      content: '';
      position: absolute;
      right: 10px;
      top: 50%;
      width: 5px;
      height: 5px;
      border-right: 1.5px solid var(--mid);
      border-bottom: 1.5px solid var(--mid);
      transform: translateY(-70%) rotate(45deg);
      pointer-events: none;
    }

    select, .field-input, .sys-prompt, .user-input {
      width: 100%;
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--charcoal);
      outline: none;
      transition: border-color var(--dur), box-shadow var(--dur);
      border-radius: var(--r1);
    }
    select, .field-input {
      padding: 7px 28px 7px 10px;
      font-size: 12px;
      cursor: pointer;
      appearance: none;
      -webkit-appearance: none;
    }
    select { font-family: var(--font); }
    .field-input { padding: 7px 10px; font-family: var(--mono); font-size: 11px; }
    select:focus, .field-input:focus {
      border-color: var(--orange);
      box-shadow: 0 0 0 2px rgba(234,88,12,0.12);
    }
    select option { background: var(--surface); color: var(--charcoal); }

    .sys-prompt {
      padding: 8px 10px;
      font-family: var(--font);
      font-size: 12px;
      resize: none;
      line-height: 1.5;
      min-height: 60px;
    }

    .slider-row { display: flex; flex-direction: column; gap: 5px; }
    .slider-label { display: flex; justify-content: space-between; align-items: baseline; }
    .slider-name {
      font-size: 11px;
      color: var(--mid);
      letter-spacing: 0.02em;
    }
    .slider-val {
      font-family: var(--mono);
      font-size: 11px;
      color: var(--charcoal);
      font-weight: 500;
    }

    input[type=range] {
      -webkit-appearance: none;
      appearance: none;
      width: 100%;
      height: 3px;
      background: var(--border);
      border-radius: 2px;
      outline: none;
      cursor: pointer;
    }
    input[type=range]::-webkit-slider-thumb {
      -webkit-appearance: none;
      appearance: none;
      width: 12px;
      height: 12px;
      background: var(--orange);
      cursor: pointer;
      border-radius: 50%;
      border: 2px solid var(--bg2);
    }
    input[type=range]::-moz-range-thumb {
      width: 12px;
      height: 12px;
      background: var(--orange);
      cursor: pointer;
      border-radius: 50%;
      border: 2px solid var(--bg2);
    }

    .conv-header {
      padding: 12px var(--sp3) 8px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      border-bottom: 1px solid var(--border);
    }
    .conv-search-wrap { position: relative; }
    .conv-search {
      width: 100%;
      padding: 6px 8px 6px 28px;
      font-size: 11px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--r1);
      color: var(--charcoal);
      font-family: var(--font);
      outline: none;
    }
    .conv-search:focus { border-color: var(--orange); box-shadow: 0 0 0 2px rgba(234,88,12,0.12); }
    .conv-search-icon {
      position: absolute;
      left: 8px;
      top: 50%;
      transform: translateY(-50%);
      color: var(--light);
      pointer-events: none;
    }
    .conv-search-icon svg { width: 12px; height: 12px; }

    .new-chat-btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      width: 100%;
      background: var(--orange);
      color: #fff;
      border: none;
      padding: 7px 10px;
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
      border-radius: var(--r1);
      transition: background var(--dur);
      letter-spacing: 0.02em;
    }
    .new-chat-btn:hover { background: #c2410c; }
    .new-chat-btn svg { width: 12px; height: 12px; }

    .conv-list {
      flex: 1;
      overflow-y: auto;
      padding: 4px 0;
      scrollbar-width: thin;
    }
    .conv-group {
      padding: 6px var(--sp3) 3px;
      font-size: 9px;
      font-weight: 600;
      color: var(--light);
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .conv-item {
      position: relative;
      padding: 7px var(--sp3);
      cursor: pointer;
      transition: background var(--dur);
    }
    .conv-item:hover { background: var(--surface-hover); }
    .conv-item:hover .conv-del { opacity: 1; }
    .conv-item.active { background: var(--orange-soft); }
    .conv-item.active::before {
      content: '';
      position: absolute;
      left: 0; top: 0; bottom: 0;
      width: 2px;
      background: var(--orange);
    }
    .conv-title {
      font-size: 12px;
      color: var(--charcoal);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      margin-bottom: 1px;
    }
    .conv-item.active .conv-title { color: var(--orange); font-weight: 600; }
    .conv-meta {
      font-family: var(--mono);
      font-size: 9px;
      color: var(--light);
    }
    .conv-del {
      position: absolute;
      right: 8px;
      top: 50%;
      transform: translateY(-50%);
      width: 20px;
      height: 20px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border-radius: 4px;
      color: var(--mid);
      cursor: pointer;
      opacity: 0;
      transition: all var(--dur);
    }
    .conv-del:hover { background: var(--surface); color: var(--error); }
    .conv-del svg { width: 11px; height: 11px; }
    .conv-empty {
      padding: 20px var(--sp3);
      text-align: center;
      color: var(--light);
      font-size: 11px;
    }

    .panel-spacer { flex: 1; min-height: 0; }

    /* PANEL CENTER */
    .panel-center {
      display: flex;
      flex-direction: column;
      background: var(--bg);
      min-height: 0;
      overflow: hidden;
    }

    .chat-header {
      height: 44px;
      flex-shrink: 0;
      border-bottom: 1px solid var(--border);
      background: var(--bg);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 var(--sp3);
    }
    .chat-header-title {
      font-size: 13px;
      font-weight: 600;
      color: var(--charcoal);
      letter-spacing: -0.01em;
    }
    .chat-header-meta {
      font-family: var(--mono);
      font-size: 10px;
      color: var(--light);
    }

    /* Per-conversation system prompt */
    .conv-prompt {
      flex-shrink: 0;
      background: var(--bg);
      border-bottom: 1px solid var(--border);
    }
    .conv-prompt-header {
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 6px var(--sp3);
      cursor: pointer;
      user-select: none;
      transition: background var(--dur);
    }
    .conv-prompt-header:hover { background: var(--surface); }
    .conv-prompt-header svg { width: 13px; height: 13px; color: var(--mid); }
    .conv-prompt-label {
      font-size: 11px;
      font-weight: 600;
      color: var(--charcoal);
    }
    .conv-prompt-badge {
      font-size: 9px;
      color: var(--mid);
      background: var(--surface);
      border: 1px solid var(--border);
      padding: 1px 5px;
      border-radius: 999px;
    }
    .conv-prompt-toggle {
      margin-left: auto;
      color: var(--light);
      display: inline-flex;
      transition: transform var(--dur);
    }
    .conv-prompt-toggle svg { width: 11px; height: 11px; }
    .conv-prompt.open .conv-prompt-toggle { transform: rotate(180deg); }
    .conv-prompt-body {
      max-height: 0;
      overflow: hidden;
      transition: max-height 200ms ease;
    }
    .conv-prompt.open .conv-prompt-body { max-height: 140px; }
    .conv-prompt .sys-prompt {
      border: none !important;
      border-top: 1px solid var(--border) !important;
      border-radius: 0 !important;
      box-shadow: none !important;
      padding: 8px var(--sp3) !important;
      min-height: 50px !important;
      max-height: 120px !important;
      background: var(--bg) !important;
    }
    .conv-prompt.has-value .conv-prompt-label { color: var(--orange); }
    .conv-prompt.has-value .conv-prompt-header svg { color: var(--orange); }

    /* Init overlay */
    .init-overlay {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: var(--sp2);
      padding: var(--sp4);
    }
    .init-overlay.hidden { display: none; }

    .init-glyph {
      display: grid;
      grid-template-columns: repeat(4, 12px);
      gap: 3px;
      margin-bottom: var(--sp2);
    }
    .init-cell {
      width: 12px;
      height: 12px;
      background: var(--border);
      border-radius: 2px;
    }
    .init-cell:nth-child(3n) { background: var(--border-strong); }
    .init-cell:nth-child(7) { background: var(--orange); animation: cell-pulse 1.6s ease-in-out infinite; }

    @keyframes cell-pulse { 0%,100%{opacity:1; transform:scale(1)} 50%{opacity:.4; transform:scale(0.85)} }

    .init-title {
      font-size: 22px;
      font-weight: 600;
      color: var(--charcoal);
      letter-spacing: -0.03em;
      text-align: center;
      line-height: 1.25;
    }
    .init-sub {
      font-size: 12px;
      color: var(--mid);
      text-align: center;
      line-height: 1.6;
      max-width: 320px;
    }

    /* Messages */
    .messages {
      flex: 1;
      overflow-y: auto;
      min-height: 0;
      padding: var(--sp3) var(--sp3) 40px;
      display: flex;
      flex-direction: column;
      gap: var(--sp3);
    }
    .messages.hidden { display: none; }

    .message {
      display: flex;
      gap: 10px;
      animation: msg-in 180ms var(--ease);
    }
    @keyframes msg-in {
      from { opacity: 0; transform: translateY(3px) }
      to { opacity: 1; transform: none }
    }
    .message.user { flex-direction: row-reverse; }

    .msg-avatar {
      width: 24px;
      height: 24px;
      border-radius: 50%;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
      font-size: 10px;
      font-weight: 600;
      background: var(--surface);
      color: var(--mid);
      border: 1px solid var(--border);
      margin-top: 1px;
    }
    .message.user .msg-avatar { background: var(--orange); color: #fff; border-color: var(--orange); }
    .message.reasoning .msg-avatar { color: var(--soft); }

    .msg-body { display: flex; flex-direction: column; gap: 3px; max-width: 75%; min-width: 0; }
    .message.user .msg-body { align-items: flex-end; }

    .msg-header { display: flex; gap: 8px; align-items: center; }
    .message.user .msg-header { flex-direction: row-reverse; }

    .msg-name {
      font-size: 11px;
      font-weight: 600;
      color: var(--charcoal);
    }
    .message.reasoning .msg-name { color: var(--soft); }

    .msg-time {
      font-family: var(--mono);
      font-size: 9px;
      color: var(--light);
    }

    .msg-play {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 18px;
      height: 18px;
      cursor: pointer;
      color: var(--light);
      border-radius: 3px;
      transition: all var(--dur);
    }
    .msg-play:hover { color: var(--orange); background: var(--orange-soft); }
    .msg-play svg { width: 11px; height: 11px; }
    .msg-play.playing { color: var(--orange); }

    .msg-copy {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 18px;
      height: 18px;
      cursor: pointer;
      color: var(--light);
      border-radius: 3px;
      transition: all var(--dur);
    }
    .msg-copy:hover { color: var(--orange); background: var(--orange-soft); }
    .msg-copy svg { width: 11px; height: 11px; }
    .msg-copy.copied { color: var(--success); background: rgba(40,200,64,0.1); }

    .msg-content {
      font-size: 14px;
      line-height: 1.6;
      color: var(--char2);
      word-break: break-word;
    }
    .msg-content p { margin-bottom: 0.5em; }
    .msg-content p:last-child { margin-bottom: 0; }
    .msg-content code {
      font-family: var(--mono);
      background: var(--code-bg);
      padding: 1px 4px;
      border-radius: 3px;
      font-size: 12px;
    }
    .message.reasoning { margin: 4px 0; }
    .message.reasoning .msg-content {
      font-family: var(--mono);
      font-size: 11px;
      color: var(--mid);
      background: var(--surface);
      border: 1px solid var(--border);
      border-left: 2px solid var(--soft);
      padding: 8px 12px;
      border-radius: var(--r1);
      letter-spacing: 0.01em;
    }
    .message.user .msg-content {
      background: var(--surface);
      border: 1px solid var(--border);
      padding: 8px 12px;
      color: var(--charcoal);
      white-space: pre-wrap;
      border-radius: var(--r2);
    }

    /* Markdown rendered content */
    .message.assistant .msg-content h1,
    .message.assistant .msg-content h2,
    .message.assistant .msg-content h3,
    .message.assistant .msg-content h4 {
      margin: 0.8em 0 0.4em;
      font-weight: 600;
      line-height: 1.3;
    }
    .message.assistant .msg-content h1 { font-size: 1.3em; }
    .message.assistant .msg-content h2 { font-size: 1.15em; }
    .message.assistant .msg-content h3 { font-size: 1.05em; }
    .message.assistant .msg-content ul,
    .message.assistant .msg-content ol {
      margin: 0.4em 0 0.6em 1.5em;
    }
    .message.assistant .msg-content li { margin-bottom: 0.2em; }
    .message.assistant .msg-content blockquote {
      border-left: 3px solid var(--orange);
      padding: 4px 12px;
      margin: 0.5em 0;
      color: var(--mid);
      background: var(--surface);
      border-radius: 0 var(--r1) var(--r1) 0;
    }
    .message.assistant .msg-content pre {
      background: var(--code-bg);
      border: 1px solid var(--border);
      border-radius: var(--r1);
      padding: 10px 14px;
      margin: 0.5em 0;
      overflow-x: auto;
      position: relative;
    }
    .message.assistant .msg-content pre code {
      background: none;
      padding: 0;
      border-radius: 0;
      font-size: 12px;
      line-height: 1.5;
    }
    .message.assistant .msg-content hr {
      border: none;
      border-top: 1px solid var(--border);
      margin: 0.8em 0;
    }
    .message.assistant .msg-content table {
      width: 100%;
      border-collapse: collapse;
      margin: 0.5em 0;
      font-size: 13px;
    }
    .message.assistant .msg-content th,
    .message.assistant .msg-content td {
      border: 1px solid var(--border);
      padding: 6px 10px;
      text-align: left;
    }
    .message.assistant .msg-content th {
      background: var(--surface);
      font-weight: 600;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .message.assistant .msg-content a {
      color: var(--orange);
      text-decoration: none;
    }
    .message.assistant .msg-content a:hover { text-decoration: underline; }
    .message.assistant .msg-content strong { font-weight: 600; }
    .message.assistant .msg-content img { max-width: 100%; border-radius: var(--r1); }

    .code-copy-btn {
      position: absolute;
      top: 6px;
      right: 6px;
      width: 22px;
      height: 22px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 4px;
      color: var(--mid);
      cursor: pointer;
      opacity: 0;
      transition: all var(--dur);
    }
    .code-copy-btn svg { width: 11px; height: 11px; }
    pre:hover .code-copy-btn { opacity: 1; }
    .code-copy-btn:hover { color: var(--orange); border-color: var(--orange); }
    .code-copy-btn.copied { color: var(--success); border-color: var(--success); }

    .typing-cursor {
      display: inline-block;
      width: 2px;
      height: 13px;
      background: var(--orange);
      margin-left: 2px;
      vertical-align: middle;
      animation: blink 1s step-end infinite;
    }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }

    /* Input area */
    .input-area {
      flex-shrink: 0;
      padding: 10px var(--sp3) 8px;
      background: var(--bg);
      border-top: 1px solid var(--border);
    }
    .input-wrap {
      display: flex;
      align-items: flex-end;
      gap: 6px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--r2);
      padding: 6px 6px 6px 5px;
      transition: border-color var(--dur), box-shadow var(--dur);
    }
    .input-wrap:focus-within {
      border-color: var(--orange);
      box-shadow: 0 0 0 2px rgba(234,88,12,0.10);
    }

    .ptt-btn {
      width: 36px;
      height: 36px;
      flex-shrink: 0;
      background: var(--bg);
      border: 1px solid var(--border);
      color: var(--charcoal);
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border-radius: var(--r1);
      transition: all var(--dur);
    }
    .ptt-btn:hover { color: var(--orange); border-color: var(--orange); }
    .ptt-btn svg { width: 15px; height: 15px; }
    .ptt-btn.recording {
      background: var(--error);
      border-color: var(--error);
      color: white;
      animation: pttPulse 1.2s ease-in-out infinite;
    }
    @keyframes pttPulse {
      0%,100% { box-shadow: 0 0 0 0 rgba(220,38,38,0.5); }
      50% { box-shadow: 0 0 0 6px rgba(220,38,38,0); }
    }

    .user-input {
      flex: 1;
      border: none !important;
      background: transparent !important;
      padding: 8px 4px;
      font-family: var(--font);
      font-size: 14px;
      color: var(--charcoal);
      resize: none;
      outline: none;
      line-height: 1.5;
      min-height: 36px;
      max-height: 180px;
      overflow-y: auto;
      box-shadow: none !important;
    }
    .user-input::placeholder { color: var(--light); }

    .send-btn {
      width: 36px;
      height: 36px;
      flex-shrink: 0;
      background: var(--orange);
      color: #fff;
      border: none;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border-radius: var(--r1);
      transition: background var(--dur), opacity var(--dur);
    }
    .send-btn:hover:not(:disabled) { background: #c2410c; }
    .send-btn:disabled { background: var(--border); color: var(--light); cursor: not-allowed; }
    .send-btn svg { width: 15px; height: 15px; }

    .input-hint {
      margin-top: 5px;
      font-size: 10px;
      color: var(--light);
      text-align: center;
      letter-spacing: 0.02em;
    }
    .input-hint kbd {
      font-family: var(--mono);
      font-size: 9px;
      background: var(--surface);
      border: 1px solid var(--border);
      padding: 1px 4px;
      border-radius: 3px;
    }

    /* Footer log */
    .footer-log {
      flex-shrink: 0;
      border-top: 1px solid var(--border);
      background: var(--bg2);
    }
    .footer-log-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 5px var(--sp3);
      cursor: pointer;
      user-select: none;
      transition: background var(--dur);
    }
    .footer-log-header:hover { background: var(--surface); }
    .footer-log-label {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      font-size: 10px;
      font-weight: 600;
      color: var(--light);
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    .footer-log-label svg { width: 11px; height: 11px; }
    .footer-log-toggle {
      display: inline-flex;
      align-items: center;
      gap: 3px;
      font-size: 10px;
      color: var(--mid);
      background: none;
      border: none;
      cursor: pointer;
      padding: 2px 5px;
      border-radius: 3px;
    }
    .footer-log-toggle:hover { color: var(--charcoal); }
    .footer-log-toggle svg { width: 9px; height: 9px; transition: transform var(--dur); }
    .footer-log.open .footer-log-toggle svg { transform: rotate(180deg); }

    .footer-log-body {
      max-height: 0;
      overflow: hidden;
      transition: max-height 200ms ease;
    }
    .footer-log.open .footer-log-body { max-height: 160px; }

    .log-stream {
      height: 160px;
      overflow-y: auto;
      background: var(--bg);
      border-top: 1px solid var(--border);
      padding: 6px var(--sp3);
      display: flex;
      flex-direction: column;
      gap: 1px;
      font-family: var(--mono);
      font-size: 10px;
    }
    .log-entry {
      display: flex;
      gap: 6px;
      color: var(--mid);
      line-height: 1.5;
      animation: log-in 100ms var(--ease);
    }
    @keyframes log-in { from {opacity:0; transform:translateX(-2px)} to {opacity:1; transform:none} }
    .log-entry .ts { color: var(--light); flex-shrink: 0; }
    .log-entry.ok { color: var(--success); }
    .log-entry.warn { color: var(--error); }
    .log-entry.hl { color: var(--orange); }

    /* Modal */
    .modal-overlay {
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.6);
      backdrop-filter: blur(4px);
      z-index: 1000;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: var(--sp3);
    }
    .modal-overlay.hidden { display: none; }
    .modal {
      background: var(--bg2);
      border: 1px solid var(--border);
      width: 480px;
      max-width: 100%;
      max-height: 90vh;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      box-shadow: 0 20px 60px rgba(0,0,0,0.4);
      border-radius: var(--r2);
    }
    .modal-header {
      padding: 16px var(--sp3);
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .modal-title {
      font-size: 14px;
      font-weight: 600;
      color: var(--charcoal);
    }
    .modal-close {
      background: none;
      border: none;
      color: var(--mid);
      cursor: pointer;
      padding: 4px;
      border-radius: 3px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }
    .modal-close:hover { background: var(--surface); color: var(--charcoal); }
    .modal-close svg { width: 15px; height: 15px; }

    .modal-body { padding: 16px var(--sp3); display: flex; flex-direction: column; gap: var(--sp2); }
    .modal-footer {
      padding: 16px var(--sp3);
      border-top: 1px solid var(--border);
      display: flex;
      justify-content: flex-end;
      gap: 6px;
    }
    .input-lbl {
      font-size: 10px;
      font-weight: 600;
      color: var(--mid);
      letter-spacing: 0.06em;
      text-transform: uppercase;
      margin-bottom: 5px;
    }
    .field-input { padding: 7px 10px; font-family: var(--mono); font-size: 11px; }
    .field-input::placeholder { color: var(--light); }

    .btn {
      padding: 7px 14px;
      font-size: 12px;
      font-weight: 500;
      border-radius: var(--r1);
      cursor: pointer;
      border: 1px solid transparent;
      transition: all var(--dur);
    }
    .btn-primary {
      background: var(--orange);
      color: #fff;
    }
    .btn-primary:hover { background: #c2410c; }
    .btn-ghost {
      background: transparent;
      color: var(--mid);
      border-color: var(--border);
    }
    .btn-ghost:hover { color: var(--charcoal); border-color: var(--border-strong); }

    hr.divider {
      border: none;
      border-top: 1px solid var(--border);
      margin: 0;
    }

    /* Scrollbars */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--mid); }

    /* Mobile drawer */
    .drawer-backdrop {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.5);
      backdrop-filter: blur(2px);
      z-index: 998;
    }
    .drawer-backdrop.open { display: block; }

    .shell.collapsed { grid-template-columns: 0 1fr; }
    .shell.collapsed .panel-left { overflow: hidden; padding: 0; border: none; }

    /* Responsive */
    @media (max-width: 900px) {
      .shell { grid-template-columns: 1fr; grid-template-rows: 1fr; }
      .panel-left {
        position: fixed;
        top: 0; left: 0; bottom: 0;
        width: 270px;
        max-height: none;
        z-index: 999;
        transform: translateX(-100%);
        transition: transform 200ms var(--ease);
        border-right: 1px solid var(--border);
        box-shadow: 4px 0 24px rgba(0,0,0,0.3);
      }
      .panel-left.open { transform: translateX(0); }
      .shell.collapsed { grid-template-columns: 1fr; }
      .drawer-btn { display: inline-flex; }
      .topbar-right .live-pill { display: none; }
      .messages { padding: 12px; }
      .message { gap: 8px; }
      .msg-body { max-width: 88%; }
    }
    @media (max-width: 600px) {
      .brand-name { display: none; }
      .input-area { padding: 8px 10px 6px; }
    }

    /* --- Realtime conversation mode (orb + live transcript) --- */
    .rt-view { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: flex-end; padding: 24px 18px 18px; z-index: 5; background: var(--bg); }
    .rt-view.hidden { display: none; }
    .rt-stage { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 22px; width: 100%; }
    .rt-orb { width: 132px; height: 132px; border-radius: 50%; display: grid; place-items: center; position: relative; background: radial-gradient(circle at 50% 40%, var(--surface-hover), var(--surface)); border: 1px solid var(--border); transition: box-shadow .4s var(--ease), border-color .4s var(--ease); }
    .rt-orb::after { content: ''; position: absolute; inset: -10px; border-radius: 50%; border: 1px solid var(--border); opacity: .5; }
    .rt-orb-core { width: 46%; height: 46%; border-radius: 50%; background: var(--mid); transition: background .4s var(--ease), transform .12s linear; }
    .rt-orb.listening { border-color: var(--orange); box-shadow: 0 0 0 6px rgba(234,88,12,.10), 0 0 60px 4px rgba(234,88,12,.30); }
    .rt-orb.listening .rt-orb-core { background: var(--orange); animation: rt-pulse 1.6s ease-in-out infinite; }
    .rt-orb.thinking { border-color: var(--soft); box-shadow: 0 0 0 6px rgba(251,146,60,.10), 0 0 60px 4px rgba(251,146,60,.28); }
    .rt-orb.thinking .rt-orb-core { background: var(--soft); animation: rt-spin 1.1s linear infinite; }
    .rt-orb.speaking { border-color: var(--success); box-shadow: 0 0 0 6px rgba(22,163,74,.12), 0 0 70px 6px rgba(22,163,74,.32); }
    .rt-orb.speaking .rt-orb-core { background: var(--success); animation: rt-breathe 1s ease-in-out infinite; }
    .rt-orb.connecting .rt-orb-core { background: var(--light); animation: rt-breathe 1.4s ease-in-out infinite; }
    .rt-orb.idle .rt-orb-core { background: var(--mid); }
    @keyframes rt-pulse { 0%,100% { transform: scale(1); } 50% { transform: scale(1.14); } }
    @keyframes rt-breathe { 0%,100% { transform: scale(.92); opacity: .85; } 50% { transform: scale(1.08); opacity: 1; } }
    @keyframes rt-spin { 0% { transform: scale(.9) rotate(0); } 100% { transform: scale(.9) rotate(360deg); } }
    .rt-state { font-family: var(--mono); font-size: 11px; letter-spacing: .14em; text-transform: uppercase; color: var(--light); }
    .rt-transcript { flex: 1 1 0; width: 100%; max-width: 680px; overflow-y: auto; padding: 10px 4px; display: flex; flex-direction: column; gap: 12px; }
    .rt-transcript::-webkit-scrollbar { width: 6px; }
    .rt-transcript::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    .rt-line { font-size: 15px; line-height: 1.5; color: var(--charcoal); white-space: pre-wrap; word-break: break-word; }
    .rt-line.user { color: var(--char2); }
    .rt-line.assistant { color: var(--charcoal); }
    .rt-line .rt-role { display: block; font-family: var(--mono); font-size: 10px; letter-spacing: .12em; text-transform: uppercase; color: var(--light); margin-bottom: 3px; }
    .rt-line.user .rt-role { color: var(--orange); }
    .rt-line.assistant .rt-role { color: var(--success); }
    .rt-line.live::after { content: '▋'; color: var(--orange); animation: rt-blink 1s steps(2) infinite; }
    @keyframes rt-blink { 50% { opacity: 0; } }
    .rt-controls { display: flex; gap: 10px; margin-top: 14px; }
    .rt-end-btn { display: inline-flex; align-items: center; gap: 8px; padding: 10px 20px; border-radius: 999px; border: 1px solid var(--border-strong); background: var(--surface); color: var(--charcoal); font-family: var(--font); font-size: 14px; cursor: pointer; transition: all var(--dur) var(--ease); }
    .rt-end-btn:hover { background: var(--surface-hover); border-color: var(--error); color: var(--error); }
    .rt-end-btn svg { width: 16px; height: 16px; }
    .rt-badge { position: fixed; right: 16px; bottom: 16px; z-index: 6; display: inline-flex; align-items: center; gap: 8px; padding: 7px 14px; border-radius: 999px; background: var(--surface); border: 1px solid var(--border); color: var(--char2); font-family: var(--mono); font-size: 11px; letter-spacing: .1em; text-transform: uppercase; }
    .rt-badge .rt-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--success); box-shadow: 0 0 8px 1px rgba(22,163,74,.6); }
    .rt-badge.listening .rt-dot { background: var(--orange); box-shadow: 0 0 8px 1px rgba(234,88,12,.6); }
    .rt-badge.thinking .rt-dot { background: var(--soft); }
    .rt-badge.idle .rt-dot { background: var(--light); box-shadow: none; }
    @media (max-width: 600px) { .rt-orb { width: 108px; height: 108px; } .rt-transcript { font-size: 14px; } }
  </style>
</head>
<body>

  <header class="topbar">
    <button class="icon-btn drawer-btn" id="drawerBtn" onclick="toggleDrawer()" title="Menu" aria-label="Open menu">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <line x1="3" y1="6" x2="21" y2="6"/>
        <line x1="3" y1="12" x2="21" y2="12"/>
        <line x1="3" y1="18" x2="21" y2="18"/>
      </svg>
    </button>
    <div class="brand">
      <div class="brand-mark">
        <span></span><span></span><span></span><span></span>
      </div>
      <span class="brand-name">Supertonic &middot; Voice Chat</span>
    </div>
    <div class="topbar-center">
      <div class="status-dot" id="statusDot"></div>
      <span class="status-text" id="statusText">Ready</span>
    </div>
    <div class="topbar-right">
      <button class="icon-btn" id="rtBtn" onclick="toggleRealtime()" title="Realtime conversation" aria-label="Realtime conversation">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
          <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
          <line x1="12" y1="19" x2="12" y2="23"/>
          <line x1="8" y1="23" x2="16" y2="23"/>
        </svg>
      </button>
      <button class="icon-btn" id="themeBtn" onclick="toggleTheme()" title="Toggle theme" aria-label="Toggle theme">
        <svg id="themeIcon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="4"/>
          <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/>
        </svg>
      </button>
    </div>
  </header>

  <div class="drawer-backdrop" id="drawerBackdrop" onclick="closeDrawer()"></div>

  <div class="shell">

    <aside class="panel-left">
      <div class="conv-header">
        <button class="new-chat-btn" onclick="newConversation()">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <line x1="12" y1="5" x2="12" y2="19"/>
            <line x1="5" y1="12" x2="19" y2="12"/>
          </svg>
          New Chat
        </button>
        <div class="conv-search-wrap">
          <span class="conv-search-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="11" cy="11" r="8"/>
              <line x1="21" y1="21" x2="16.65" y2="16.65"/>
            </svg>
          </span>
          <input type="text" class="conv-search" id="convSearch" placeholder="Search conversations..." oninput="filterConvs()">
        </div>
      </div>
      <div class="conv-list" id="conversationList"></div>
      <div class="panel-section">
        <div class="section-row">
          <span class="section-label">Settings</span>
          <span class="section-link" onclick="openModal()">Configure</span>
        </div>
      </div>
    </aside>

    <main class="panel-center">
      <div class="rt-view hidden" id="rtView">
        <div class="rt-stage">
          <div class="rt-orb idle" id="rtOrb"><span class="rt-orb-core"></span></div>
          <div class="rt-state" id="rtState">Tap to start</div>
        </div>
        <div class="rt-transcript" id="rtTranscript"></div>
        <div class="rt-controls">
          <button class="rt-end-btn" id="rtEndBtn" onclick="toggleRealtime()">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" stroke="none"/></svg>
            End conversation
          </button>
        </div>
      </div>
      <div class="chat-header">
        <span class="chat-header-title" id="chatTitle">Voice Session</span>
        <span class="chat-header-meta" id="sessionMeta">SES &middot; 00:00</span>
      </div>

      <div class="conv-prompt" id="convPromptWrap">
        <div class="conv-prompt-header" onclick="toggleConvPrompt()">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
          </svg>
          <span class="conv-prompt-label">System Prompt</span>
          <span class="conv-prompt-badge">this conversation</span>
          <span class="conv-prompt-toggle" id="convPromptChevron">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="6 9 12 15 18 9"/>
            </svg>
          </span>
        </div>
        <div class="conv-prompt-body" id="convPromptBody">
          <textarea class="sys-prompt" id="sysPrompt" rows="2" placeholder="Set instructions for this conversation only. Leave empty for default behavior." oninput="onSysPromptChange()"></textarea>
        </div>
      </div>

      <div class="init-overlay" id="initOverlay">
        <div class="init-glyph">
          <div class="init-cell"></div><div class="init-cell"></div><div class="init-cell"></div><div class="init-cell"></div>
          <div class="init-cell"></div><div class="init-cell"></div><div class="init-cell"></div><div class="init-cell"></div>
          <div class="init-cell"></div><div class="init-cell"></div><div class="init-cell"></div><div class="init-cell"></div>
        </div>
        <div class="init-title">Hold mic to talk<br>or type a message.</div>
        <div class="init-sub">TTS runs locally on your device with streaming synthesis, sentence by sentence. No audio data leaves your machine.</div>
      </div>

      <div class="messages hidden" id="messages"></div>

      <div class="input-area">
        <div class="input-wrap">
          <button class="ptt-btn" id="pttBtn" title="Hold to talk · ⇧⇧ start · ⇧ stop" aria-label="Push to talk">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
              <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
              <line x1="12" y1="19" x2="12" y2="23"/>
              <line x1="8" y1="23" x2="16" y2="23"/>
            </svg>
          </button>
          <textarea class="user-input" id="userInput" placeholder="Type a message..." rows="1"
            onkeydown="handleKey(event)" oninput="autoResize(this); onInputChange()"></textarea>
          <button class="send-btn" id="sendBtn" onclick="sendText()" disabled aria-label="Send message">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <line x1="5" y1="12" x2="19" y2="12"/>
              <polyline points="12 5 19 12 12 19"/>
            </svg>
          </button>
        </div>
        <div class="input-hint"><kbd>&#x21E7;&#x21E7;</kbd> start mic &middot; <kbd>&#x21E7;</kbd> stop &middot; <kbd>&#x23CE;</kbd> send &middot; <kbd>&#x21E7;&#x23CE;</kbd> new line</div>
      </div>
    </main>
  </div>

  <div class="footer-log" id="footerLog">
    <div class="footer-log-header" onclick="toggleLog()">
      <span class="footer-log-label">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
        Event Log
      </span>
      <button class="footer-log-toggle" aria-label="Toggle log">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="18 15 12 9 6 15"/>
        </svg>
      </button>
    </div>
    <div class="footer-log-body">
      <div class="log-stream" id="logStream"></div>
    </div>
  </div>

  <div id="settingsModal" class="modal-overlay hidden">
    <div class="modal">
      <div class="modal-header">
        <span class="modal-title">Settings</span>
        <button class="modal-close" onclick="closeModal()" aria-label="Close">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="18" y1="6" x2="6" y2="18"/>
            <line x1="6" y1="6" x2="18" y2="18"/>
          </svg>
        </button>
      </div>
      <div class="modal-body">
        <div>
          <div class="input-lbl">Voice</div>
          <div class="select-wrap" style="margin-bottom:0">
            <select class="model-select" id="voice" onchange="onVoiceChange()">
              <option value="M1">M1 &middot; Male deep</option>
              <option value="M2">M2 &middot; Male mid</option>
              <option value="M3">M3 &middot; Male bright</option>
              <option value="M4">M4 &middot; Male warm</option>
              <option value="M5">M5 &middot; Male neutral</option>
              <option value="F1">F1 &middot; Female deep</option>
              <option value="F2">F2 &middot; Female mid</option>
              <option value="F3">F3 &middot; Female bright</option>
              <option value="F4">F4 &middot; Female warm</option>
              <option value="F5">F5 &middot; Female neutral</option>
            </select>
          </div>
        </div>
        <div>
          <div class="input-lbl">Language</div>
          <div class="select-wrap" style="margin-bottom:0">
            <select class="model-select" id="lang" onchange="onLangChange()">
              <option value="en">EN &middot; English</option>
              <option value="pt">PT &middot; Portuguese</option>
              <option value="es">ES &middot; Spanish</option>
              <option value="fr">FR &middot; French</option>
              <option value="de">DE &middot; German</option>
              <option value="ja">JP &middot; Japanese</option>
              <option value="ko">KO &middot; Korean</option>
            </select>
          </div>
        </div>
        <div>
          <div class="slider-row" style="margin-bottom:0">
            <div class="slider-label">
              <span class="slider-name">Diffusion Steps</span>
              <span class="slider-val" id="stepsVal">5</span>
            </div>
            <input type="range" id="steps" min="2" max="12" step="1" value="5"
              oninput="document.getElementById('stepsVal').textContent=this.value; updateSlider(this)">
          </div>
          <div class="slider-row" style="margin-bottom:0">
            <div class="slider-label">
              <span class="slider-name">Speed</span>
              <span class="slider-val" id="speedVal">1.15</span>
            </div>
            <input type="range" id="speed" min="0.7" max="2.0" step="0.05" value="1.15"
              oninput="document.getElementById('speedVal').textContent=parseFloat(this.value).toFixed(2); updateSlider(this)">
          </div>
        </div>
        <hr style="border:none; border-top:1px solid #d4d4d4; margin:0">
        <div>
          <div class="input-lbl">Max Tokens (LLM response length)</div>
          <input class="field-input" type="number" id="maxTokens" min="64" max="32768" step="64" value="2048" placeholder="2048">
        </div>
        <div>
          <div class="input-lbl">LLM API URL</div>
          <input class="field-input" type="text" id="apiUrl" placeholder="http://127.0.0.1:8080/v1/chat/completions">
        </div>
        <div>
          <div class="input-lbl">LLM API Key</div>
          <input class="field-input" type="password" id="apiKey" placeholder="(leave empty for local)">
        </div>
        <div>
          <div class="input-lbl">STT API URL (parakeet.cpp)</div>
          <input class="field-input" type="text" id="sttApiUrl" placeholder="http://localhost:8080">
        </div>
        <div style="font-family:var(--mono); font-size:10px; color:var(--light); margin-top:5px; letter-spacing:0.04em;">
          // Settings are stored in localStorage.
        </div>
      </div>
      <div class="modal-footer">
        <button class="btn btn-ghost" onclick="closeModal()">Cancel</button>
        <button class="btn btn-primary" onclick="saveConnection()">Save</button>
      </div>
    </div>
  </div>

  <script type="module">
    const sessionStart = Date.now();
    let currentPlayingAudio = null;
    let isBusy = false;
    let messageQueue = [];
    let currentAssistantEl = null;
    let currentReasoningEl = null;

    const DEFAULT_API_URL = '{{ default_api_url }}';
    const DEFAULT_STT_API_URL = '{{ default_stt_api_url }}';

    document.querySelectorAll('input[type=range]').forEach(el => {
      const min = +el.min, max = +el.max, val = +el.value;
      el.style.setProperty('--fill', ((val - min) / (max - min) * 100) + '%');
    });
    window.updateSlider = el => {
      const min = +el.min, max = +el.max, val = +el.value;
      el.style.setProperty('--fill', ((val - min) / (max - min) * 100) + '%');
    };

    setInterval(() => {
      const s = Math.floor((Date.now() - sessionStart) / 1000);
      const m = String(Math.floor(s / 60)).padStart(2, '0');
      document.getElementById('sessionMeta').textContent = 'SES \\u00b7 ' + m + ':' + String(s % 60).padStart(2, '0');
    }, 1000);

    function log(text, type = '') {
      const ls = document.getElementById('logStream');
      const ts = new Date().toISOString().slice(11, 19);
      const el = document.createElement('div');
      el.className = 'log-entry' + (type ? ' ' + type : '');
      el.innerHTML = '<span class="ts">' + ts + '</span>' + text;
      ls.appendChild(el);
      ls.scrollTop = ls.scrollHeight;
      if (ls.children.length > 120) ls.firstChild.remove();
    }
    window.log = log;

    function loadSettings() {
      const url = localStorage.getItem('supertonic_api_url') || DEFAULT_API_URL;
      const key = localStorage.getItem('supertonic_api_key') || '';
      const stt = localStorage.getItem('supertonic_stt_api_url') || DEFAULT_STT_API_URL;
      const voice = localStorage.getItem('supertonic_voice') || 'M1';
      const lang = localStorage.getItem('supertonic_lang') || 'en';
      const steps = parseInt(localStorage.getItem('supertonic_steps') || '5', 10);
      const speed = parseFloat(localStorage.getItem('supertonic_speed') || '1.15');
      const maxTokens = parseInt(localStorage.getItem('supertonic_max_tokens') || '2048', 10);
      document.getElementById('apiUrl').value = url;
      document.getElementById('apiKey').value = key;
      document.getElementById('sttApiUrl').value = stt;
      document.getElementById('voice').value = voice;
      document.getElementById('lang').value = lang;
      document.getElementById('steps').value = steps;
      document.getElementById('stepsVal').textContent = steps;
      updateSlider(document.getElementById('steps'));
      document.getElementById('speed').value = speed;
      document.getElementById('speedVal').textContent = speed.toFixed(2);
      updateSlider(document.getElementById('speed'));
      document.getElementById('maxTokens').value = maxTokens;
      syncTopbar();
    }
    window.saveConnection = function() {
      const url = document.getElementById('apiUrl').value.trim();
      const key = document.getElementById('apiKey').value.trim();
      const stt = document.getElementById('sttApiUrl').value.trim();
      const voice = document.getElementById('voice').value;
      const lang = document.getElementById('lang').value;
      const steps = document.getElementById('steps').value;
      const speed = document.getElementById('speed').value;
      const maxTokens = parseInt(document.getElementById('maxTokens').value) || 2048;
      localStorage.setItem('supertonic_api_url', url);
      localStorage.setItem('supertonic_api_key', key);
      localStorage.setItem('supertonic_stt_api_url', stt);
      localStorage.setItem('supertonic_voice', voice);
      localStorage.setItem('supertonic_lang', lang);
      localStorage.setItem('supertonic_steps', steps);
      localStorage.setItem('supertonic_speed', speed);
      localStorage.setItem('supertonic_max_tokens', maxTokens);
      closeModal();
      log('Settings saved.', 'ok');
    };
    window.openModal = function() {
      document.getElementById('settingsModal').classList.remove('hidden');
    };
    window.closeModal = function() {
      document.getElementById('settingsModal').classList.add('hidden');
    };
    document.getElementById('settingsModal').addEventListener('click', e => {
      if (e.target.id === 'settingsModal') closeModal();
    });
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape') closeModal();
    });
    loadSettings();

    // --- IndexedDB conversation history ---
    let db = null;
    let currentConvId = null;
    let currentConvMessages = [];
    let currentReasoningText = '';

    function escapeHtml(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

    function formatTimeAgo(iso) {
      const diff = Date.now() - new Date(iso).getTime();
      const min = Math.floor(diff / 60000);
      if (min < 1) return 'Just now';
      if (min < 60) return min + 'm ago';
      const h = Math.floor(min / 60);
      if (h < 24) return h + 'h ago';
      return new Date(iso).toLocaleDateString();
    }

    function openDB() {
      return new Promise((resolve, reject) => {
        const req = indexedDB.open('supertonic_history', 1);
        req.onupgradeneeded = e => {
          const db2 = e.target.result;
          if (!db2.objectStoreNames.contains('conversations')) {
            const store = db2.createObjectStore('conversations', { keyPath: 'id', autoIncrement: true });
            store.createIndex('updatedAt', 'updatedAt', { unique: false });
          }
        };
        req.onsuccess = e => resolve(e.target.result);
        req.onerror = e => reject(e.target.error);
      });
    }

    function dbAdd(conv) {
      return new Promise((resolve, reject) => {
        const tx = db.transaction('conversations', 'readwrite');
        const store = tx.objectStore('conversations');
        const req = store.add(conv);
        req.onsuccess = () => resolve(req.result);
        req.onerror = e => reject(e.target.error);
      });
    }

    function dbUpdate(conv) {
      return new Promise((resolve, reject) => {
        const tx = db.transaction('conversations', 'readwrite');
        const store = tx.objectStore('conversations');
        const req = store.put(conv);
        req.onsuccess = () => resolve();
        req.onerror = e => reject(e.target.error);
      });
    }

    function dbGet(id) {
      return new Promise((resolve, reject) => {
        const tx = db.transaction('conversations', 'readonly');
        const store = tx.objectStore('conversations');
        const req = store.get(id);
        req.onsuccess = () => resolve(req.result);
        req.onerror = e => reject(e.target.error);
      });
    }

    function dbGetAll() {
      return new Promise((resolve, reject) => {
        const tx = db.transaction('conversations', 'readonly');
        const store = tx.objectStore('conversations');
        const index = store.index('updatedAt');
        const req = index.openCursor(null, 'prev');
        const results = [];
        req.onsuccess = e => {
          const cursor = e.target.result;
          if (cursor) { results.push(cursor.value); cursor.continue(); }
          else resolve(results);
        };
        req.onerror = e => reject(e.target.error);
      });
    }

    async function saveCurrentConv() {
      if (!currentConvId || currentConvMessages.length === 0) return;
      const conv = await dbGet(currentConvId);
      if (!conv) return;
      conv.messages = currentConvMessages;
      conv.updatedAt = new Date().toISOString();
      if (conv.title === 'New Chat' && currentConvMessages.length > 0) {
        conv.title = currentConvMessages[0].content.substring(0, 50);
      }
      conv.sysPrompt = document.getElementById('sysPrompt').value;
      await dbUpdate(conv);
    }

    async function loadConversation(conv) {
      if (isBusy) return;
      await saveCurrentConv();
      clearChatUI();
      currentConvId = conv.id;
      currentConvMessages = conv.messages || [];
      localStorage.setItem('activeConvId', currentConvId);
      const sp = document.getElementById('sysPrompt');
      sp.value = conv.sysPrompt || '';
      updateSysPromptUI();
      const titleEl = document.getElementById('chatTitle');
      if (titleEl) titleEl.textContent = conv.title || 'Voice Session';
      if (currentConvMessages.length > 0) {
        document.getElementById('initOverlay').classList.add('hidden');
        document.getElementById('messages').classList.remove('hidden');
        currentConvMessages.forEach(m => appendMessage(m.role, m.content));
      }
      try {
        await fetch('/api/chat/load', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            messages: conv.messages ? conv.messages.filter(m => m.role !== 'reasoning') : [],
            sys_prompt: conv.sysPrompt || ''
          })
        });
      } catch(e) {}
      await renderConvList();
    }

    function clearChatUI() {
      document.getElementById('messages').innerHTML = '';
      document.getElementById('messages').classList.add('hidden');
      document.getElementById('initOverlay').classList.remove('hidden');
      currentAssistantEl = null; currentReasoningEl = null;
      currentReasoningText = '';
      const sp = document.getElementById('sysPrompt');
      if (sp) { sp.value = ''; updateSysPromptUI(); }
    }

    function updateSysPromptUI() {
      const sp = document.getElementById('sysPrompt');
      const wrap = document.getElementById('convPromptWrap');
      if (!sp || !wrap) return;
      const has = sp.value.trim().length > 0;
      wrap.classList.toggle('has-value', has);
      if (has) wrap.classList.add('open');
    }

    window.toggleConvPrompt = function() {
      document.getElementById('convPromptWrap').classList.toggle('open');
    };

    let sysPromptSaveTimer = null;
    window.onSysPromptChange = function() {
      updateSysPromptUI();
      clearTimeout(sysPromptSaveTimer);
      sysPromptSaveTimer = setTimeout(() => { saveCurrentConv(); }, 400);
    };

    function dbDelete(id) {
      return new Promise((resolve, reject) => {
        const tx = db.transaction('conversations', 'readwrite');
        const store = tx.objectStore('conversations');
        const req = store.delete(id);
        req.onsuccess = () => resolve();
        req.onerror = e => reject(e.target.error);
      });
    }

    function groupLabel(iso) {
      const d = new Date(iso);
      const now = new Date();
      const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
      const yesterday = new Date(today); yesterday.setDate(today.getDate() - 1);
      const weekAgo = new Date(today); weekAgo.setDate(today.getDate() - 7);
      if (d >= today) return 'Today';
      if (d >= yesterday) return 'Yesterday';
      if (d >= weekAgo) return 'This Week';
      return 'Earlier';
    }

    function convItemHTML(c) {
      return '<div class="conv-item' + (c.id === currentConvId ? ' active' : '') + '" data-id="' + c.id + '">' +
        '<div style="display:flex; justify-content:space-between; align-items:center; gap:8px">' +
          '<div style="flex:1; min-width:0">' +
            '<div class="conv-title">' + escapeHtml(c.title) + '</div>' +
            '<div class="conv-meta">' + formatTimeAgo(c.updatedAt) + '</div>' +
          '</div>' +
          '<span class="conv-del" data-id="' + c.id + '" title="Delete conversation"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/></svg></span>' +
        '</div>' +
      '</div>';
    }

    let allConvsCache = [];
    async function renderConvList() {
      const container = document.getElementById('conversationList');
      if (!container) return;
      allConvsCache = await dbGetAll();
      const query = (document.getElementById('convSearch')?.value || '').toLowerCase().trim();
      const filtered = query ? allConvsCache.filter(c => c.title.toLowerCase().includes(query)) : allConvsCache;
      if (filtered.length === 0) {
        container.innerHTML = '<div class="conv-empty">' + (query ? 'No matches found.' : 'No conversations yet.') + '</div>';
        return;
      }
      // Group
      const groups = {};
      filtered.forEach(c => {
        const g = groupLabel(c.updatedAt);
        (groups[g] = groups[g] || []).push(c);
      });
      const order = ['Today', 'Yesterday', 'This Week', 'Earlier'];
      let html = '';
      for (const g of order) {
        if (groups[g] && groups[g].length) {
          html += '<div class="conv-group">' + g + '</div>';
          html += groups[g].map(convItemHTML).join('');
        }
      }
      container.innerHTML = html;
      container.querySelectorAll('.conv-item').forEach(el => {
        const id = parseInt(el.dataset.id);
        el.addEventListener('click', async e => {
          if (e.target.closest('.conv-del')) return;
          const conv = await dbGet(id);
          if (conv) loadConversation(conv);
        });
        el.addEventListener('dblclick', () => startRename(id, el));
      });
      container.querySelectorAll('.conv-del').forEach(el => {
        el.addEventListener('click', async e => {
          e.stopPropagation();
          const id = parseInt(el.dataset.id);
          await dbDelete(id);
          if (currentConvId === id) {
            clearChatUI();
            currentConvMessages = [];
            const convs = await dbGetAll();
            if (convs.length > 0) {
              await loadConversation(convs[0]);
            } else {
              await newConversation();
            }
          } else {
            await renderConvList();
          }
        });
      });
    }

    window.filterConvs = function() { renderConvList(); };

    async function startRename(id, el) {
      const conv = await dbGet(id);
      if (!conv) return;
      const titleEl = el.querySelector('.conv-title');
      const original = conv.title;
      const input = document.createElement('input');
      input.type = 'text';
      input.value = original;
      input.className = 'conv-search';
      input.style.padding = '4px 8px';
      input.style.margin = '0';
      titleEl.replaceWith(input);
      input.focus();
      input.select();
      let done = false;
      const finish = async (save) => {
        if (done) return;
        done = true;
        if (save && input.value.trim() && input.value.trim() !== original) {
          conv.title = input.value.trim().substring(0, 80);
          conv.updatedAt = new Date().toISOString();
          await dbUpdate(conv);
        }
        await renderConvList();
      };
      input.addEventListener('blur', () => finish(true));
      input.addEventListener('keydown', e => {
        if (e.key === 'Enter') { e.preventDefault(); input.blur(); }
        if (e.key === 'Escape') { e.preventDefault(); finish(false); }
      });
      e?.stopPropagation?.();
    }

    window.newConversation = async function() {
      if (isBusy) return;
      await saveCurrentConv();
      clearChatUI();
      const now = new Date().toISOString();
      const id = await dbAdd({
        title: 'New Chat',
        sysPrompt: '',
        messages: [],
        createdAt: now,
        updatedAt: now
      });
      currentConvId = id;
      currentConvMessages = [];
      localStorage.setItem('activeConvId', id);
      const titleEl = document.getElementById('chatTitle');
      if (titleEl) titleEl.textContent = 'New Chat';
      await renderConvList();
    };

    async function initConversationHistory() {
      db = await openDB();
      const savedId = parseInt(localStorage.getItem('activeConvId') || '0');
      if (savedId) {
        const conv = await dbGet(savedId);
        if (conv) { await loadConversation(conv); return; }
      }
      await newConversation();
    }

    function syncTopbar() {
      // Voice and lang pills removed from topbar; nothing to sync visually.
    }
    window.onVoiceChange = function() {
      syncTopbar();
      log('Voice set to ' + document.getElementById('voice').value);
    };
    window.onLangChange = function() {
      syncTopbar();
      log('Language set to ' + document.getElementById('lang').value.toUpperCase());
    };
    syncTopbar();

    window.toggleLog = function() {
      document.getElementById('footerLog').classList.toggle('open');
    };

    window.toggleDrawer = function() {
      const isMobile = window.matchMedia('(max-width: 900px)').matches;
      if (isMobile) {
        const panel = document.querySelector('.panel-left');
        const backdrop = document.getElementById('drawerBackdrop');
        const isOpen = panel.classList.toggle('open');
        backdrop.classList.toggle('open', isOpen);
      } else {
        const collapsed = document.querySelector('.shell').classList.toggle('collapsed');
        localStorage.setItem('supertonic_sidebar_collapsed', collapsed ? '1' : '');
      }
    };

    window.closeDrawer = function() {
      const isMobile = window.matchMedia('(max-width: 900px)').matches;
      if (isMobile) {
        document.querySelector('.panel-left').classList.remove('open');
        document.getElementById('drawerBackdrop').classList.remove('open');
      } else {
        document.querySelector('.shell').classList.remove('collapsed');
        localStorage.setItem('supertonic_sidebar_collapsed', '');
      }
    };

    // Close drawer when clicking a conversation item
    document.addEventListener('click', e => {
      if (e.target.closest('.conv-item')) closeDrawer();
    });

    const ICONS = {
      sun: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/></svg>',
      moon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>',
      play: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3" fill="currentColor"/></svg>',
      stop: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" fill="currentColor"/></svg>',
      trash: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/></svg>',
    };

    function setIcon(id, key) { document.getElementById(id).innerHTML = ICONS[key]; }

    // Theme toggle (default = dark, toggle to light)
    window.toggleTheme = function() {
      const isLight = document.documentElement.classList.toggle('light');
      setIcon('themeIcon', isLight ? 'moon' : 'sun');
      localStorage.setItem('supertonic_theme', isLight ? 'light' : 'dark');
    };
    if (localStorage.getItem('supertonic_theme') === 'light') {
      document.documentElement.classList.add('light');
      setIcon('themeIcon', 'moon');
    } else {
      setIcon('themeIcon', 'sun');
    }

    if (localStorage.getItem('supertonic_sidebar_collapsed') === '1') {
      document.querySelector('.shell').classList.add('collapsed');
    }

    function setStatus(text, state) {
      document.getElementById('statusText').textContent = text;
      const dot = document.getElementById('statusDot');
      dot.className = 'status-dot' + (state === 'active' ? ' active' : state === 'rec' ? ' rec' : '');
    }

    // Configure marked for safe markdown rendering
    if (typeof marked !== 'undefined') {
      marked.setOptions({ breaks: true, gfm: true });
    }

    function renderMarkdown(text) {
      if (typeof marked !== 'undefined') {
        return marked.parse(text);
      }
      return text.split('<').join('&lt;').split('>').join('&gt;').split('\\n').join('<br>');
    }

    function addCodeCopyButtons(container) {
      container.querySelectorAll('pre').forEach(pre => {
        if (pre.querySelector('.code-copy-btn')) return;
        const btn = document.createElement('span');
        btn.className = 'code-copy-btn';
        btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
        btn.addEventListener('click', () => {
          const code = pre.querySelector('code');
          navigator.clipboard.writeText(code ? code.textContent : pre.textContent).then(() => {
            btn.classList.add('copied');
            setTimeout(() => btn.classList.remove('copied'), 1200);
          });
        });
        pre.style.position = 'relative';
        pre.appendChild(btn);
      });
    }

    function appendMessage(role, content) {
      const container = document.getElementById('messages');
      const div = document.createElement('div');
      div.className = 'message ' + role;
      const now = new Date().toTimeString().slice(0, 8);
      let avatar = 'AI', name = 'ASSISTANT';
      if (role === 'user') { avatar = 'USR'; name = 'OPERATOR'; }
      else if (role === 'reasoning') { avatar = 'THK'; name = 'REASONING'; }
      let extra = '';
      if (role === 'assistant' || role === 'reasoning') {
        extra = '<span class="msg-play" title="Read aloud"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3" fill="currentColor" stroke="none"/></svg></span>';
      }
      extra += '<span class="msg-copy" title="Copy text"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg></span>';
      div.innerHTML = '<div class="msg-avatar">' + avatar + '</div><div class="msg-body"><div class="msg-header"><span class="msg-name">' + name + '</span><span class="msg-time">' + now + '</span>' + extra + '</div><div class="msg-content"></div></div>';
      const contentDiv = div.querySelector('.msg-content');
      if (role === 'assistant' && content) {
        contentDiv.innerHTML = renderMarkdown(content);
        addCodeCopyButtons(contentDiv);
      } else {
        contentDiv.textContent = content || '';
      }
      const playBtn = div.querySelector('.msg-play');
      if (playBtn) {
        playBtn.addEventListener('click', e => {
          e.stopPropagation();
          const text = contentDiv.textContent.trim();
          playMessage(text, playBtn);
        });
      }
      const copyBtn = div.querySelector('.msg-copy');
      if (copyBtn) {
        copyBtn.addEventListener('click', e => {
          e.stopPropagation();
          const text = contentDiv.textContent;
          navigator.clipboard.writeText(text).then(() => {
            copyBtn.classList.add('copied');
            setTimeout(() => copyBtn.classList.remove('copied'), 1200);
          }).catch(() => { log('Copy failed', 'warn'); });
        });
      }
      container.appendChild(div);
      container.scrollTop = container.scrollHeight;
      return div;
    }

    async function playMessage(text, btn) {
      if (!text) return;
      if (btn && btn.classList.contains('playing')) {
        if (currentPlayingAudio) { currentPlayingAudio.pause(); currentPlayingAudio.currentTime = 0; currentPlayingAudio = null; }
        btn.classList.remove('playing');
        btn.innerHTML = ICONS.play;
        return;
      }
      if (currentPlayingAudio) { currentPlayingAudio.pause(); currentPlayingAudio.currentTime = 0; }
      document.querySelectorAll('.msg-play.playing').forEach(b => { b.classList.remove('playing'); b.innerHTML = ICONS.play; });
      try {
        const resp = await fetch('/api/tts', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            text: text,
            lang: document.getElementById('lang').value,
            voice: document.getElementById('voice').value,
            steps: parseInt(document.getElementById('steps').value),
            speed: parseFloat(document.getElementById('speed').value),
          })
        });
        const data = await resp.json();
        if (data.error) { log('TTS error: ' + data.error, 'warn'); return; }
        const audio = new Audio('data:audio/wav;base64,' + data.audio);
        currentPlayingAudio = audio;
        if (btn) {
          btn.classList.add('playing');
          btn.innerHTML = ICONS.stop;
          audio.onended = () => { btn.classList.remove('playing'); btn.innerHTML = ICONS.play; currentPlayingAudio = null; };
          audio.onerror = () => { btn.classList.remove('playing'); btn.innerHTML = ICONS.play; currentPlayingAudio = null; };
        }
        audio.play();
      } catch (e) {
        log('TTS error: ' + e.message, 'warn');
        if (btn) { btn.classList.remove('playing'); btn.innerHTML = ICONS.play; }
        currentPlayingAudio = null;
      }
    }

    function addOrUpdateAssistant(text) {
      if (!currentAssistantEl) {
        document.getElementById('initOverlay').classList.add('hidden');
        document.getElementById('messages').classList.remove('hidden');
        currentAssistantEl = appendMessage('assistant', '');
      }
      currentAssistantEl.querySelector('.msg-content').textContent += text;
      document.getElementById('messages').scrollTop = document.getElementById('messages').scrollHeight;
    }

    function addReasoning(text) {
      if (!currentReasoningEl) {
        currentReasoningEl = appendMessage('reasoning', '');
      }
      currentReasoningEl.querySelector('.msg-content').textContent += text;
      document.getElementById('messages').scrollTop = document.getElementById('messages').scrollHeight;
    }

    function attachCursor() {
      if (!currentAssistantEl) return;
      const content = currentAssistantEl.querySelector('.msg-content');
      const old = content.querySelector('.typing-cursor');
      if (old) old.remove();
      const cursor = document.createElement('span');
      cursor.className = 'typing-cursor';
      content.appendChild(cursor);
    }

    window.onInputChange = function() {
      document.getElementById('sendBtn').disabled = !document.getElementById('userInput').value.trim();
    };
    window.autoResize = function(el) {
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight, 140) + 'px';
    };
    window.handleKey = function(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendText();
      }
    };

    async function sendToServer(message) {
      isBusy = true;
      setStatus('Streaming', 'active');
      currentAssistantEl = null;
      currentReasoningEl = null;
      currentReasoningText = '';
      let firstText = true;

      let resp;
      try {
        resp = await fetch('/api/chat', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            message: message,
            lang: document.getElementById('lang').value,
            voice: document.getElementById('voice').value,
            steps: parseInt(document.getElementById('steps').value),
            speed: parseFloat(document.getElementById('speed').value),
            max_tokens: parseInt(document.getElementById('maxTokens').value) || 2048,
            api_url: document.getElementById('apiUrl').value.trim(),
            api_key: document.getElementById('apiKey').value.trim(),
            sys_prompt: document.getElementById('sysPrompt').value,
          })
        });
      } catch (err) {
        setStatus('Error', '');
        log('Connection error: ' + err.message, 'warn');
        isBusy = false;
        onInputChange();
        processQueue();
        return;
      }

      if (!resp.ok) {
        setStatus('Error ' + resp.status, '');
        log('Server error ' + resp.status, 'warn');
        isBusy = false;
        onInputChange();
        processQueue();
        return;
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split('\\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.type === 'text') {
                if (firstText) { firstText = false; log('LLM streaming...'); }
                currentReasoningEl = null;
                addOrUpdateAssistant(data.text);
                attachCursor();
              } else if (data.type === 'reasoning') {
                addReasoning(data.text);
                currentReasoningText += data.text;
              } else if (data.type === 'error') {
                log('TTS error: ' + data.text, 'warn');
                setStatus('TTS Error', '');
              } else if (data.type === 'done') {
                currentReasoningEl = null;
                let asstText = '';
                if (currentAssistantEl) {
                  const c = currentAssistantEl.querySelector('.typing-cursor');
                  if (c) c.remove();
                  const contentEl = currentAssistantEl.querySelector('.msg-content');
                  asstText = contentEl.textContent;
                  // Re-render as markdown after streaming completes
                  contentEl.innerHTML = renderMarkdown(asstText);
                  addCodeCopyButtons(contentEl);
                }
                setStatus('Ready', '');
                log('Response complete', 'ok');
                if (currentReasoningText) {
                  currentConvMessages.push({ role: 'reasoning', content: currentReasoningText });
                  currentReasoningText = '';
                }
                if (asstText) {
                  currentConvMessages.push({ role: 'assistant', content: asstText });
                }
                saveCurrentConv().then(() => renderConvList());
              }
            } catch(e) { /* ignore */ }
          }
        }
      }

      isBusy = false;
      onInputChange();
    }

    window.sendText = function() {
      const input = document.getElementById('userInput');
      const text = input.value.trim();
      if (!text) return;
      document.getElementById('initOverlay').classList.add('hidden');
      document.getElementById('messages').classList.remove('hidden');
      appendMessage('user', text);
      currentConvMessages.push({ role: 'user', content: text });
      input.value = '';
      autoResize(input);
      onInputChange();
      messageQueue.push(text);
      if (messageQueue.length > 1) {
        log('Queued · ' + (messageQueue.length - 1) + ' waiting', 'hl');
      }
      processQueue();
    };

    async function processQueue() {
      if (isBusy || messageQueue.length === 0) return;
      const text = messageQueue.shift();
      await sendToServer(text);
      processQueue();
    }

    window.clearChat = async function() {
      document.getElementById('messages').innerHTML = '';
      document.getElementById('messages').classList.add('hidden');
      document.getElementById('initOverlay').classList.remove('hidden');
      currentAssistantEl = null;
      currentReasoningEl = null;
      try { await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({message:'/clear'})}); } catch(e){}
      log('Conversation cleared.', 'ok');
    };

    // STT via parakeet.cpp (local, on-device) — real-time rolling chunks.
    // The parakeet.cpp-server is a one-shot WAV transcriber (no streaming
    // endpoint), so we drive "text appears while talking" from the browser:
    //   - ScriptProcessor collects 16 kHz mono PCM.
    //   - A VAD detects pauses; each time you stop talking for ~650 ms the
    //     buffered utterance is sent off and its transcript is *appended* to
    //     the textarea, so committed text never gets re-transcribed.
    //   - While you are still talking, short rolling partials of the current
    //     (uncommitted) utterance are sent too and shown as a live tail,
    //     replaced on each tick — so words appear as you speak.
    //   - Requests are serialized (one in flight + at most one queued commit),
    //     to avoid pile-up on slow CPUs. This is also why long audios no
    //     longer feel slow: we never send the whole recording.
    let pttHeld = false;
    let shiftTapCount = 0;
    let shiftTapTimer = null;
    let shiftIsDown = false;
    let sttPrefix = '';
    let mediaStream = null;
    let audioContext = null;
    let scriptProcessor = null;

    const STT_SR = 16000;
    const VAD_RMS = 0.012;
    const SILENCE_MS = 650;        // trailing silence that finalizes an utterance
    const MIN_UTT_SAMPLES = Math.floor(STT_SR * 0.30);
    const MAX_UTT_SAMPLES = Math.floor(STT_SR * 12);    // force-split an over-long run
    const MAX_PARTIAL_SAMPLES = Math.floor(STT_SR * 6); // stop live partials past this
    const PARTIAL_DELTA_SAMPLES = Math.floor(STT_SR * 0.35);
    const TICK_MS = 350;

    let sttTick = null;
    let uttChunks = [];
    let uttCount = 0;
    let lastVoiceAt = 0;
    let lastSentPartialCount = 0;
    let committedText = '';
    let livePartial = '';
    let partialGen = 0;       // bump to invalidate in-flight partial results
    let sttActive = null;     // a Promise while a transcription is running
    let commitPending = false; // schedule a commit once the current job finishes
    let stopDeferred = null;  // {resolve} to finish stopPTT after queued work

    function encodeWAV(samples, sampleRate) {
      const buf = new ArrayBuffer(44 + samples.length * 2);
      const v = new DataView(buf);
      const w = (s, o) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
      w('RIFF', 0); v.setUint32(4, 36 + samples.length * 2, true); w('WAVE', 8);
      w('fmt ', 12); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
      v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
      w('data', 36); v.setUint32(40, samples.length * 2, true);
      for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      }
      return new Blob([buf], { type: 'audio/wav' });
    }

    function joinSpace(a, b) {
      if (!a) return b;
      if (!b) return a;
      return a + (a.endsWith(' ') ? '' : ' ') + b;
    }

    function collectUtt() {
      const out = new Float32Array(uttCount);
      let off = 0;
      for (const c of uttChunks) { out.set(c, off); off += c.length; }
      return out;
    }

    function renderSTT() {
      const input = document.getElementById('userInput');
      input.value = joinSpace(committedText, livePartial);
      autoResize(input);
      onInputChange();
    }

    async function sttTranscribe(samples) {
      try {
        const wav = encodeWAV(samples, STT_SR);
        const fd = new FormData();
        fd.append('file', wav, 'utt.wav');
        fd.append('lang', document.getElementById('lang').value);
        fd.append('stt_api', document.getElementById('sttApiUrl').value.trim());
        const resp = await fetch('/api/stt', { method: 'POST', body: fd });
        const data = await resp.json();
        if (data.error) { log('STT error: ' + data.error, 'warn'); return ''; }
        return (data.text || '').replace(/^\\s+|\\s+$/g, '');
      } catch (e) {
        log('STT error: ' + e.message, 'warn');
        return '';
      }
    }

    async function flushCommit() {
      const samples = collectUtt();
      uttChunks = []; uttCount = 0; lastSentPartialCount = 0;
      partialGen++;
      livePartial = '';
      renderSTT();
      const text = await sttTranscribe(samples);
      if (text) committedText = joinSpace(committedText, text);
      renderSTT();
    }

    async function sendPartial() {
      const samples = collectUtt();
      lastSentPartialCount = uttCount;
      const gen = partialGen;
      const text = await sttTranscribe(samples);
      if (gen !== partialGen) return;          // superseded by a commit / stop
      livePartial = text;
      renderSTT();
    }

    function enqueue(promise) {
      sttActive = promise.then(() => { sttActive = null; pump(); });
    }

    function pump() {
      if (sttActive) return;
      if (commitPending) { commitPending = false; enqueue(flushCommit()); return; }
      if (stopDeferred) { const d = stopDeferred; stopDeferred = null; d.resolve(); }
    }

    function sttTickFn() {
      if (!pttHeld) return;
      const needCommit = uttCount >= MIN_UTT_SAMPLES &&
        (performance.now() - lastVoiceAt >= SILENCE_MS || uttCount >= MAX_UTT_SAMPLES);
      if (needCommit) {
        if (sttActive) commitPending = true;
        else enqueue(flushCommit());
        return;
      }
      if (sttActive || commitPending) return;
      if (uttCount < MIN_UTT_SAMPLES) return;
      if (uttCount >= MAX_PARTIAL_SAMPLES) return;
      if (uttCount - lastSentPartialCount < PARTIAL_DELTA_SAMPLES) return;
      enqueue(sendPartial());
    }

    async function startPTT() {
      if (isBusy || pttHeld) return;
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        log('Mic unavailable: page must be served over HTTPS or from localhost', 'warn');
        return;
      }
      const current = document.getElementById('userInput').value;
      sttPrefix = current ? (current.endsWith(' ') ? current : current + ' ') : '';
      committedText = sttPrefix;
      livePartial = '';
      uttChunks = []; uttCount = 0;
      lastSentPartialCount = 0;
      partialGen = 0;
      sttActive = null; commitPending = false; stopDeferred = null;
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
          audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true }
        });
        audioContext = new AudioContext({ sampleRate: 16000 });
        const source = audioContext.createMediaStreamSource(mediaStream);
        scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
        scriptProcessor.onaudioprocess = (e) => {
          const d = new Float32Array(e.inputBuffer.getChannelData(0));
          let sum = 0;
          for (let i = 0; i < d.length; i++) sum += d[i] * d[i];
          if (Math.sqrt(sum / d.length) > VAD_RMS) lastVoiceAt = performance.now();
          uttChunks.push(d);
          uttCount += d.length;
        };
        source.connect(scriptProcessor);
        scriptProcessor.connect(audioContext.destination);
        pttHeld = true;
        lastVoiceAt = performance.now();
        document.getElementById('pttBtn').classList.add('recording');
        setStatus('Recording', 'rec');
        sttTick = setInterval(sttTickFn, TICK_MS);
        log('Recording (real-time STT)...', 'hl');
      } catch (e) {
        log('Mic error: ' + e.message, 'warn');
      }
    }

    async function stopPTT() {
      if (!pttHeld) return;
      pttHeld = false;
      if (sttTick) { clearInterval(sttTick); sttTick = null; }
      if (scriptProcessor) { scriptProcessor.disconnect(); scriptProcessor = null; }
      if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); }
      const hadTail = uttCount >= MIN_UTT_SAMPLES;
      partialGen++;                  // invalidate any in-flight partial
      livePartial = '';
      renderSTT();
      if (hadTail) {
        if (sttActive) commitPending = true; else enqueue(flushCommit());
      } else {
        commitPending = false;
      }
      if (audioContext) { try { await audioContext.close(); } catch (e) {} audioContext = null; }
      if (mediaStream) { mediaStream = null; }
      document.getElementById('pttBtn').classList.remove('recording');
      setStatus('Transcribing', 'active');
      if (sttActive || commitPending) {
        await new Promise(res => { stopDeferred = { resolve: res }; pump(); });
      }
      livePartial = ''; renderSTT();
      const input = document.getElementById('userInput');
      input.value = input.value.replace(/^\\s+|\\s+$/g, '');
      autoResize(input); onInputChange();
      sttPrefix = ''; committedText = '';
      if (input.value) log('STT: ' + input.value.substring(0, 50) + (input.value.length > 50 ? '...' : ''), 'ok');
      setStatus('Ready', '');
    }

    const pttBtn = document.getElementById('pttBtn');
    pttBtn.addEventListener('click', () => {
      if (pttHeld) stopPTT();
      else startPTT();
    });
    pttBtn.addEventListener('touchstart', e => { e.preventDefault(); }, {passive: false});

    document.addEventListener('keydown', e => {
      if (e.key === 'Shift' && !shiftIsDown) {
        shiftIsDown = true;
        shiftTapCount++;
        if (pttHeld) {
          // Single Shift stops the mic immediately
          shiftTapCount = 0;
          if (shiftTapTimer) { clearTimeout(shiftTapTimer); shiftTapTimer = null; }
          stopPTT();
        } else if (shiftTapCount === 1) {
          shiftTapTimer = setTimeout(() => { shiftTapCount = 0; }, 400);
        } else if (shiftTapCount >= 2) {
          // Double-tap Shift starts the mic
          clearTimeout(shiftTapTimer);
          shiftTapCount = 0;
          shiftTapTimer = null;
          e.preventDefault();
          startPTT();
        }
      }
    });
    document.addEventListener('keyup', e => {
      if (e.key === 'Shift') {
        shiftIsDown = false;
      }
    });

    // --- Realtime conversation mode (local speech-to-speech over WebSocket) ---
    const RT_WS_PORT = {{ ws_port }};
    let rtWS = null, rtCtx = null, rtMicStream = null, rtScript = null, rtRunning = false;
    let rtPlayCtx = null, rtTTSsr = 24000, rtPlaySources = [], rtPlayTime = 0;
    let rtState = 'idle', rtAsstEl = null, rtAsstText = '';

    function setRTState(s) {
      rtState = s;
      document.getElementById('rtOrb').className = 'rt-orb ' + s;
      const labels = {idle:'Tap to start', connecting:'Connecting…', listening:'Listening', thinking:'Thinking', speaking:'Speaking'};
      document.getElementById('rtState').textContent = labels[s] || s;
    }

    function rtSettings() {
      return {
        lang: document.getElementById('lang').value,
        voice: document.getElementById('voice').value,
        steps: parseInt(document.getElementById('steps').value),
        speed: parseFloat(document.getElementById('speed').value),
        max_tokens: parseInt(document.getElementById('maxTokens').value) || 512,
        api_url: document.getElementById('apiUrl').value.trim(),
        api_key: document.getElementById('apiKey').value.trim(),
        sys_prompt: document.getElementById('sysPrompt').value,
      };
    }

    function rtSendStart() {
      if (!rtWS || rtWS.readyState !== 1) return;
      rtWS.send(JSON.stringify(Object.assign({type:'start'}, rtSettings())));
    }

    function rtEscapeHtml(s) {
      return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }

    function rtAppendUser(text) {
      const t = document.getElementById('rtTranscript');
      const el = document.createElement('div');
      el.className = 'rt-line user';
      el.innerHTML = '<span class="rt-role">You</span>' + rtEscapeHtml(text);
      t.appendChild(el); t.scrollTop = t.scrollHeight;
      rtAsstEl = null; rtAsstText = '';
    }

    function rtAppendAssistantDelta(tok) {
      const t = document.getElementById('rtTranscript');
      if (!rtAsstEl) {
        rtAsstEl = document.createElement('div');
        rtAsstEl.className = 'rt-line assistant live';
        rtAsstEl.innerHTML = '<span class="rt-role">Assistant</span><span class="rt-text"></span>';
        t.appendChild(rtAsstEl);
        rtAsstText = '';
      }
      rtAsstText += tok;
      rtAsstEl.querySelector('.rt-text').textContent = rtAsstText;
      t.scrollTop = t.scrollHeight;
    }

    function rtFinalizeAssistant() {
      if (rtAsstEl) { rtAsstEl.classList.remove('live'); rtAsstEl = null; rtAsstText = ''; }
    }

    function rtClearPlayback() {
      rtPlaySources.forEach(s => { try { s.stop(); } catch(e){} });
      rtPlaySources = [];
      if (rtPlayCtx) rtPlayTime = rtPlayCtx.currentTime;
    }

    function onRTAudio(buf) {
      if (!rtPlayCtx) return;
      const i16 = new Int16Array(buf);
      const f32 = new Float32Array(i16.length);
      for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768;
      const ab = rtPlayCtx.createBuffer(1, f32.length, rtTTSsr);
      ab.copyToChannel(f32, 0);
      const src = rtPlayCtx.createBufferSource();
      src.buffer = ab; src.connect(rtPlayCtx.destination);
      const now = rtPlayCtx.currentTime;
      if (rtPlayTime < now) rtPlayTime = now + 0.02;
      src.start(rtPlayTime);
      rtPlayTime += ab.duration;
      rtPlaySources.push(src);
      src.onended = () => { rtPlaySources = rtPlaySources.filter(s => s !== src); };
    }

    function onRTMsg(m) {
      if (m.type === 'ready') { rtTTSsr = m.sampleRate || 24000; setRTState('listening'); }
      else if (m.type === 'state') setRTState(m.state);
      else if (m.type === 'transcript') {
        if (m.role === 'user' && m.final) rtAppendUser(m.text);
        else if (m.role === 'assistant') {
          if (m.final) rtFinalizeAssistant();
          else if (m.text) rtAppendAssistantDelta(m.text);
        }
      } else if (m.type === 'clear') rtClearPlayback();
      else if (m.type === 'error') {
        log('RT: ' + m.text, 'warn');
        if (/loading|TTS/i.test(m.text)) { setRTState('connecting'); setTimeout(rtSendStart, 2500); }
      }
    }

    async function startRealtime() {
      if (rtRunning) return;
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        log('Mic unavailable (page must be HTTPS or localhost)', 'warn'); return;
      }
      setRTState('connecting');
      document.getElementById('rtView').classList.remove('hidden');
      document.querySelector('.input-area').classList.add('hidden');
      document.getElementById('initOverlay').classList.add('hidden');
      document.getElementById('messages').classList.add('hidden');
      const url = 'ws://' + location.hostname + ':' + RT_WS_PORT + '/ws';
      try { rtWS = new WebSocket(url); }
      catch (e) { log('WS error: ' + e.message, 'warn'); stopRealtimeUI('Tap to start'); return; }
      rtWS.binaryType = 'arraybuffer';
      rtWS.onopen = () => { rtSendStart(); log('Realtime connected', 'hl'); };
      rtWS.onmessage = (e) => {
        if (e.data instanceof ArrayBuffer) onRTAudio(e.data);
        else { try { onRTMsg(JSON.parse(e.data)); } catch(_){} }
      };
      rtWS.onclose = () => { if (rtRunning) { log('Realtime disconnected', 'warn'); stopRealtimeUI('Disconnected'); } };
      rtWS.onerror = () => { log('Realtime connection error', 'warn'); };
      try {
        rtMicStream = await navigator.mediaDevices.getUserMedia({
          audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true }
        });
      } catch (e) { log('Mic error: ' + e.message, 'warn'); stopRealtimeUI('Tap to start'); return; }
      rtCtx = new AudioContext({ sampleRate: 16000 });
      const src = rtCtx.createMediaStreamSource(rtMicStream);
      rtScript = rtCtx.createScriptProcessor(2048, 1, 1);
      rtScript.onaudioprocess = (e) => {
        if (!rtWS || rtWS.readyState !== 1) return;
        const d = e.inputBuffer.getChannelData(0);
        const buf = new Int16Array(d.length);
        for (let i = 0; i < d.length; i++) { let s = Math.max(-1, Math.min(1, d[i])); buf[i] = s < 0 ? s * 0x8000 : s * 0x7FFF; }
        rtWS.send(buf.buffer);
      };
      src.connect(rtScript); rtScript.connect(rtCtx.destination);
      rtPlayCtx = new AudioContext();
      rtPlayTime = rtPlayCtx.currentTime;
      rtRunning = true;
      document.getElementById('rtBtn').classList.add('active');
      log('Realtime conversation started — just talk (use headphones to avoid echo).', 'hl');
    }

    function stopRealtimeUI(msg) {
      rtRunning = false;
      if (rtScript) { try { rtScript.disconnect(); } catch(e){} rtScript = null; }
      if (rtMicStream) { rtMicStream.getTracks().forEach(t => t.stop()); rtMicStream = null; }
      if (rtCtx) { try { rtCtx.close(); } catch(e){} rtCtx = null; }
      rtClearPlayback();
      if (rtPlayCtx) { try { rtPlayCtx.close(); } catch(e){} rtPlayCtx = null; }
      if (rtWS) { try { rtWS.close(); } catch(e){} rtWS = null; }
      document.getElementById('rtView').classList.add('hidden');
      document.querySelector('.input-area').classList.remove('hidden');
      document.getElementById('rtBtn').classList.remove('active');
      setRTState('idle');
      document.getElementById('rtState').textContent = msg || 'Tap to start';
      document.getElementById('rtTranscript').innerHTML = '';
      rtAsstEl = null; rtAsstText = '';
    }

    async function stopRealtime() {
      if (rtWS && rtWS.readyState === 1) { try { rtWS.send(JSON.stringify({type:'stop'})); } catch(e){} }
      stopRealtimeUI('Ended');
      setStatus('Ready', '');
    }

    window.toggleRealtime = function() {
      if (rtRunning) stopRealtime();
      else { setStatus('Realtime', 'active'); startRealtime(); }
    };

    initConversationHistory().then(() => {
      setStatus('Ready', '');
      log('Supertonic voice chat ready \u2014 STT via parakeet.cpp.', 'ok');
      document.getElementById('userInput').focus();
    });
  </script>
</body>
</html>"""



@app.route("/")
def index():
    return render_template_string(
        HTML,
        default_api_url=config["api_url"],
        default_stt_api_url=config["stt_api_url"],
        ws_port=RT_WS_PORT,
    )


def _load_tts_background():
    global tts, style, config
    try:
        tts = TTS(auto_download=True)
        style = tts.get_voice_style(voice_name=config["voice"])
        print("✅ TTS loaded")
    except Exception as e:
        print(f"⚠️ TTS loading failed (will retry on first use): {e}")


def main():
    global tts, style, config, RT_WS_PORT

    parser = argparse.ArgumentParser(description="Supertonic Voice Chat Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host")
    parser.add_argument("--port", type=int, default=7777, help="Port")
    parser.add_argument("--ws-port", type=int, default=0, help="Realtime WebSocket port (default: HTTP port + 1)")
    parser.add_argument("--api", default=LLAMA_API, help="LLM API URL")
    parser.add_argument("--stt-api", default=STT_API, help="Parakeet STT server URL")
    parser.add_argument("--model", default="default", help="Model name")
    parser.add_argument("--voice", default="M1", help="Voice")
    parser.add_argument("--lang", default="en", help="Language")
    parser.add_argument("--steps", type=int, default=5, help="TTS steps")
    parser.add_argument("--speed", type=float, default=1.15, help="Speed")
    args = parser.parse_args()

    config.update({
        "lang": args.lang,
        "voice": args.voice,
        "steps": args.steps,
        "speed": args.speed,
        "api_url": args.api,
        "stt_api_url": args.stt_api,
        "model": args.model,
    })

    RT_WS_PORT = args.ws_port or (args.port + 1)

    print("🚀 Loading Supertonic TTS in background...")
    t = threading.Thread(target=_load_tts_background, daemon=True)
    t.start()

    if _realtime is not None:
        try:
            import sys
            _realtime.start(args.host, RT_WS_PORT, sys.modules[__name__])
        except Exception as e:
            print(f"⚠️  Realtime server failed to start: {e}")
    else:
        print("⚠️  Realtime mode unavailable (websockets not installed).")

    print(f"""
╔══════════════════════════════════════════╗
║   🎤 Supertonic Voice Chat              ║
║   Open: http://{args.host}:{args.port}          ║
║   WS:   ws://{args.host}:{RT_WS_PORT}/ws           ║
║   LLM:  {args.api}        ║
║   STT:  {args.stt_api}           ║
║   Voice: {args.voice}  |  Lang: {args.lang}      ║
║   Steps: {args.steps}  |  Speed: {args.speed}          ║
╚══════════════════════════════════════════╝
""")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
