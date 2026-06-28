#!/usr/bin/env python3
"""
Supertonic Voice Chat — Web UI com microfone + teclado + LLM + TTS local.
"""

import argparse
import json
import re
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

LLAMA_API = "http://127.0.0.1:8080/v1/chat/completions"
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
    "model": "default",
}

# Histórico do LLM
messages = [{"role": "system", "content": SYS_PROMPT}]

# --- Sentence Buffer ---
class SentenceBuffer:
    def __init__(self):
        self.buffer = ""

    def add(self, token: str) -> Optional[str]:
        self.buffer += token
        if len(self.buffer) > 180:
            s = self.buffer.strip()
            self.buffer = ""
            return s
        if re.search(r"[.!?…]\s*$", self.buffer):
            if not re.search(r"(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|etc)\.$", self.buffer):
                s = self.buffer.strip()
                self.buffer = ""
                return s
        return None

    def flush(self) -> Optional[str]:
        if self.buffer.strip():
            s = self.buffer.strip()
            self.buffer = ""
            return s
        return None


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
    # Aceita API URL e key do frontend (sobrescreve defaults locais)
    api_url = data.get("api_url", config["api_url"]).strip()
    api_key = data.get("api_key", "").strip()
    if not api_url:
        api_url = config["api_url"]
    if "voice" in data:
        try:
            global style
            style = tts.get_voice_style(voice_name=data["voice"])
        except:
            pass

    def generate():
        global messages
        buf = SentenceBuffer()

        # Stream do LLM
        messages.append({"role": "user", "content": user_msg})
        payload = {
            "model": config["model"],
            "messages": messages,
            "stream": True,
            "max_tokens": 512,
            "temperature": 0.7,
        }

        # Headers com auth se tiver API key
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            r = requests.post(api_url, json=payload, headers=headers, stream=True, timeout=120)
            r.raise_for_status()
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

                    # Detecta sentença completa → sintetiza TTS
                    sentence = buf.add(content)
                    if sentence is not None and len(sentence) > 3:
                        try:
                            with tts_lock:
                                wav, dur = tts.synthesize(
                                    text=sentence,
                                    lang=config["lang"],
                                    voice_style=style,
                                    total_steps=config["steps"],
                                    speed=config["speed"],
                                )
                            b64 = wav_to_base64(wav, tts.sample_rate)
                            yield f"data: {json.dumps({'type': 'audio', 'data': b64})}\n\n"
                        except Exception as e:
                            yield f"data: {json.dumps({'type': 'error', 'text': f'TTS error: {e}'})}\n\n"

        # Flush final
        sentence = buf.flush()
        if sentence and len(sentence) > 3:
            try:
                with tts_lock:
                    wav, dur = tts.synthesize(
                        text=sentence,
                        lang=config["lang"],
                        voice_style=style,
                        total_steps=config["steps"],
                        speed=config["speed"],
                    )
                b64 = wav_to_base64(wav, tts.sample_rate)
                yield f"data: {json.dumps({'type': 'audio', 'data': b64})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'text': f'TTS error: {e}'})}\n\n"

        full = "".join(content_parts)
        messages.append({"role": "assistant", "content": full})

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


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
<title>SUPERTONIC // VOICE CHAT</title>
<link href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@400;500;600;700&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<style>
/* ===== CYBERPUNK NEON DESIGN SYSTEM ===== */
:root {
  --base-bg: #0D1117;
  --base-panel: #161B22;
  --base-card: #1C1F26;
  --base-input: #21262D;
  --base-border: #2A2F38;
  --base-border-bright: #3A4050;
  --text-primary: #F0F6E8;
  --text-secondary: #D4DFC8;
  --text-muted: #9AAF88;
  --text-dim: #6A7A5A;
  --neon-green: #C0FC14;
  --neon-pink: #FF2D7C;
  --neon-blue: #2B7FFF;
  --neon-cyan: #14FCEB;
  --neon-purple: #B829FF;
  --green-dim: rgba(192,252,20,0.12);
  --green-glow: rgba(192,252,20,0.45);
  --pink-dim: rgba(255,45,124,0.12);
  --pink-glow: rgba(255,45,124,0.45);
  --blue-dim: rgba(43,127,255,0.12);
  --blue-glow: rgba(43,127,255,0.45);
  --cyan-dim: rgba(20,252,235,0.12);
  --cyan-glow: rgba(20,252,235,0.45);
}

* { box-sizing: border-box; margin: 0; padding: 0; border-radius: 0; }

body {
  font-family: 'Chakra Petch', 'Share Tech Mono', monospace;
  background: var(--base-bg);
  background-image:
    linear-gradient(rgba(192,252,20,0.02) 1px, transparent 1px),
    linear-gradient(90deg, rgba(43,127,255,0.02) 1px, transparent 1px);
  background-size: 40px 40px;
  color: var(--text-primary);
  min-height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  letter-spacing: 0.04em;
}

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--base-bg); }
::-webkit-scrollbar-thumb { background: var(--base-border-bright); }
::-webkit-scrollbar-thumb:hover { background: var(--neon-green); }

*:focus-visible {
  outline: 2px solid var(--neon-green);
  outline-offset: -2px;
  box-shadow: 0 0 8px var(--green-dim);
}

.container {
  width: 100%;
  max-width: 720px;
  height: 100vh;
  display: flex;
  flex-direction: column;
  padding: 0 16px;
}

/* ===== HEADER ===== */
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 0;
  border-bottom: 1px solid var(--base-border);
  margin-bottom: 8px;
}

.brand-stamp {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: var(--neon-green);
  color: var(--base-bg);
  font-family: 'Share Tech Mono', monospace;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  padding: 4px 10px;
  line-height: 1;
  box-shadow: 0 0 8px var(--green-glow);
}

.header-title {
  font-family: 'Chakra Petch', sans-serif;
  font-weight: 600;
  font-size: 0.95em;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-secondary);
}

.header-right {
  font-family: 'Share Tech Mono', monospace;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-dim);
}

/* ===== STATUS BAR ===== */
.status-bar {
  font-family: 'Share Tech Mono', monospace;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-dim);
  text-align: center;
  padding: 6px 0;
  min-height: 22px;
  transition: color 0.2s;
}
.status-bar.active { color: var(--neon-green); }
.status-bar.warn { color: var(--neon-pink); }

/* ===== SETTINGS ROW ===== */
.settings {
  display: flex;
  gap: 6px;
  padding: 6px 0 10px;
  flex-wrap: wrap;
  align-items: center;
}

.ctrl-wrap {
  position: relative;
  background: var(--base-card);
  border: 1px solid var(--base-border);
  transition: border-color 200ms ease, box-shadow 200ms ease;
}
.ctrl-wrap:hover {
  border-color: var(--neon-green);
  box-shadow: 0 0 12px var(--green-dim), inset 0 0 12px var(--green-dim);
}
.ctrl-wrap:focus-within {
  border-color: var(--neon-green);
  box-shadow: 0 0 8px var(--green-dim);
}

.ctrl-label {
  font-family: 'Share Tech Mono', monospace;
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--text-muted);
  padding: 0 6px 2px;
}

.ctrl-select {
  appearance: none;
  -webkit-appearance: none;
  background: transparent;
  border: none;
  padding: 8px 28px 8px 8px;
  font-family: 'Share Tech Mono', monospace;
  font-size: 12px;
  font-weight: 700;
  color: var(--text-primary);
  text-align: center;
  text-transform: uppercase;
  cursor: pointer;
  outline: none;
  min-height: 36px;
  letter-spacing: 0.1em;
}

.ctrl-select option {
  background: var(--base-card);
  color: var(--text-primary);
  text-align: center;
}

.ctrl-input {
  background: transparent;
  border: none;
  padding: 8px;
  font-family: 'Share Tech Mono', monospace;
  font-size: 12px;
  font-weight: 700;
  color: var(--text-primary);
  text-align: center;
  outline: none;
  min-height: 36px;
  letter-spacing: 0.1em;
  width: 52px;
}
.ctrl-input::-webkit-inner-spin-button,
.ctrl-input::-webkit-outer-spin-button { -webkit-appearance: none; margin: 0; }

/* ===== CHAT AREA ===== */
.chat-panel {
  flex: 1;
  position: relative;
  margin: 4px 0;
  overflow: hidden;
}

.chat-area {
  height: 100%;
  overflow-y: auto;
  overflow-x: hidden;
  padding: 10px 4px;
  background: var(--base-panel);
  background-image:
    linear-gradient(rgba(192,252,20,0.015) 1px, transparent 1px),
    linear-gradient(90deg, rgba(43,127,255,0.015) 1px, transparent 1px);
  background-size: 28px 28px;
  border: 1px solid var(--base-border);
  position: relative;
}

/* Scanline overlay */
.chat-area::before {
  content: '';
  position: absolute;
  inset: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent, transparent 2px,
    rgba(192,252,20,0.015) 2px,
    rgba(43,127,255,0.01) 4px
  );
  pointer-events: none;
  z-index: 1;
}

/* Corner brackets */
.chat-panel::before,
.chat-panel::after {
  content: '';
  position: absolute;
  width: 14px;
  height: 14px;
  border-style: solid;
  pointer-events: none;
  z-index: 3;
  transition: all 200ms ease;
}
.chat-panel::before {
  top: -1px;
  left: -1px;
  border-width: 2px 0 0 2px;
  border-color: var(--neon-green);
}
.chat-panel::after {
  bottom: -1px;
  right: -1px;
  border-width: 0 2px 2px 0;
  border-color: var(--neon-pink);
}
.chat-panel:hover::before,
.chat-panel:hover::after {
  width: 22px;
  height: 22px;
}

/* ===== MESSAGES ===== */
.msg {
  margin: 6px 0;
  padding: 10px 14px;
  max-width: 82%;
  word-wrap: break-word;
  animation: msgIn 0.25s ease;
  position: relative;
  z-index: 2;
  font-size: 0.9em;
  line-height: 1.5;
}
.msg.user {
  background: var(--base-card);
  border: 1px solid var(--neon-blue);
  color: var(--text-primary);
  margin-left: auto;
  box-shadow: 0 0 10px var(--blue-dim);
}
.msg.user::before {
  content: '\\25B8';
  color: var(--neon-blue);
  margin-right: 6px;
  font-family: 'Share Tech Mono', monospace;
}
.msg.assistant {
  background: var(--base-card);
  border: 1px solid var(--neon-green);
  color: var(--text-primary);
  margin-right: auto;
  box-shadow: 0 0 10px var(--green-dim);
}
.msg.assistant::before {
  content: '\\25C0';
  color: var(--neon-green);
  margin-right: 6px;
  font-family: 'Share Tech Mono', monospace;
}
.msg.reasoning {
  background: var(--base-card);
  border: 1px solid var(--neon-cyan);
  color: var(--text-muted);
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.78em;
  font-style: italic;
  margin-right: auto;
  box-shadow: 0 0 6px var(--cyan-dim);
}

@keyframes msgIn {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}

/* Thinking indicator */
.think-line {
  display: none;
  font-family: 'Share Tech Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--text-dim);
  padding: 8px 14px;
  position: relative;
  z-index: 2;
}
.think-line.active { display: block; }
.think-dots::after {
  content: '';
  display: inline;
  animation: thinkDots 1.2s steps(4,end) infinite;
  color: var(--neon-green);
  text-shadow: 0 0 6px var(--green-glow);
}
@keyframes thinkDots {
  0% { content: ''; }
  25% { content: '.'; }
  50% { content: '..'; }
  75% { content: '...'; }
  100% { content: ''; }
}

/* ===== CONTROLS ===== */
.controls {
  padding: 10px 0 14px;
  display: flex;
  gap: 8px;
  align-items: center;
  border-top: 1px solid var(--base-border);
  margin-top: 4px;
}

.ctrl-input-text {
  flex: 1;
  padding: 10px 14px;
  border: 1px solid var(--base-border);
  background: var(--base-input);
  color: var(--text-primary);
  font-family: 'Share Tech Mono', monospace;
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.06em;
  outline: none;
  transition: border-color 200ms ease, box-shadow 200ms ease;
}
.ctrl-input-text::placeholder {
  color: var(--text-dim);
  text-transform: uppercase;
  font-size: 11px;
}
.ctrl-input-text:focus {
  border-color: var(--neon-green);
  box-shadow: 0 0 14px var(--green-dim);
}

/* Buttons */
.btn {
  padding: 10px 16px;
  border: 1px solid var(--base-border);
  cursor: pointer;
  font-family: 'Share Tech Mono', monospace;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  transition: all 150ms ease;
  min-height: 42px;
  display: flex;
  align-items: center;
  gap: 6px;
  outline: none;
  background: var(--base-card);
  color: var(--text-secondary);
}
.btn:hover {
  border-color: var(--neon-green);
  box-shadow: 0 0 12px var(--green-dim), inset 0 0 12px var(--green-dim);
  color: var(--text-primary);
}
.btn:active { transform: translateY(1px); }
.btn:disabled {
  opacity: 0.35;
  cursor: not-allowed;
  box-shadow: none !important;
  border-color: var(--base-border) !important;
}

.btn-mic {
  border-color: var(--neon-blue);
  color: var(--neon-blue);
  font-size: 1.1em;
  padding: 10px 13px;
  box-shadow: 0 0 8px var(--blue-dim);
}
.btn-mic:hover {
  border-color: var(--neon-cyan);
  color: var(--neon-cyan);
  box-shadow: 0 0 16px var(--cyan-dim);
}
.btn-mic.recording {
  background: var(--neon-pink);
  border-color: var(--neon-pink);
  color: #fff;
  animation: micPulse 0.8s ease-in-out infinite;
}
.btn-mic.recording:hover {
  color: #fff;
  box-shadow: 0 0 20px var(--pink-glow);
}
@keyframes micPulse {
  0%, 100% { box-shadow: 0 0 0 0 var(--pink-glow), 0 0 8px var(--pink-dim); }
  50% { box-shadow: 0 0 0 8px var(--pink-dim), 0 0 20px var(--pink-glow); }
}

.btn-send {
  background: var(--neon-green);
  border-color: var(--neon-green);
  color: var(--base-bg);
  box-shadow: 0 0 10px var(--green-dim);
}
.btn-send:hover {
  background: var(--neon-green);
  color: var(--base-bg);
  box-shadow: 0 0 20px var(--green-glow), 0 0 40px var(--green-dim);
}

/* ===== SETTINGS DRAWER ===== */
.settings-drawer {
  border-top: 1px solid var(--base-border);
  margin-top: 6px;
}
.settings-toggle {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 0;
  cursor: pointer;
  font-family: 'Share Tech Mono', monospace;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text-dim);
  transition: color 0.2s;
  background: none;
  border: none;
  width: 100%;
}
.settings-toggle:hover { color: var(--neon-green); }
.settings-toggle::before {
  content: '[+]';
  color: var(--neon-green);
  width: 24px;
}
.settings-toggle.open::before { content: '[-]'; }
.settings-body {
  display: none;
  padding: 8px 0 12px;
  flex-direction: column;
  gap: 8px;
}
.settings-body.open { display: flex; }
.settings-row {
  display: flex;
  align-items: center;
  gap: 8px;
}
.settings-row label {
  font-family: 'Share Tech Mono', monospace;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-muted);
  min-width: 70px;
}
.settings-row input {
  flex: 1;
  padding: 8px 10px;
  border: 1px solid var(--base-border);
  background: var(--base-input);
  color: var(--text-primary);
  font-family: 'Share Tech Mono', monospace;
  font-size: 12px;
  letter-spacing: 0.04em;
  outline: none;
  transition: border-color 200ms ease;
}
.settings-row input:focus {
  border-color: var(--neon-green);
  box-shadow: 0 0 8px var(--green-dim);
}
.settings-row input::placeholder {
  color: var(--text-faint, #4A5A3A);
  font-size: 11px;
}</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div class="brand-stamp">S3</div>
    <div class="header-title">Supertonic Voice Chat</div>
    <div class="header-right">ON-DEVICE</div>
  </div>

  <div class="status-bar" id="status">[ SYS ] READY // PRESS MIC OR TYPE</div>

  <div class="settings">
    <div class="ctrl-wrap">
      <div class="ctrl-label">LANG</div>
      <select class="ctrl-select" id="lang">
        <option value="en">EN</option>
        <option value="pt">PT</option>
        <option value="es">ES</option>
        <option value="fr">FR</option>
        <option value="de">DE</option>
        <option value="ja">JP</option>
        <option value="ko">KR</option>
      </select>
    </div>
    <div class="ctrl-wrap">
      <div class="ctrl-label">VOICE</div>
      <select class="ctrl-select" id="voice">
        <option value="M1">M1</option>
        <option value="M2">M2</option>
        <option value="M3">M3</option>
        <option value="M4">M4</option>
        <option value="M5">M5</option>
        <option value="F1">F1</option>
        <option value="F2">F2</option>
        <option value="F3">F3</option>
        <option value="F4">F4</option>
        <option value="F5">F5</option>
      </select>
    </div>
    <div class="ctrl-wrap">
      <div class="ctrl-label">STEPS</div>
      <input class="ctrl-input" type="number" id="steps" value="5" min="3" max="12">
    </div>
    <div class="ctrl-wrap">
      <div class="ctrl-label">SPEED</div>
      <input class="ctrl-input" type="number" id="speed" value="1.15" min="0.7" max="2.0" step="0.05">
    </div>
  </div>

  <div class="settings-drawer">
    <button class="settings-toggle" id="settingsToggle">LLM CONNECTION</button>
    <div class="settings-body" id="settingsBody">
      <div class="settings-row">
        <label>API URL</label>
        <input type="text" id="apiUrl" placeholder="http://127.0.0.1:8080/v1/chat/completions">
      </div>
      <div class="settings-row">
        <label>API KEY</label>
        <input type="password" id="apiKey" placeholder="(leave empty for local)">
      </div>
    </div>
  </div>

  <div class="chat-panel">
    <div class="chat-area" id="chat">
      <div class="think-line" id="typing">PROCESSING<span class="think-dots"></span></div>
    </div>
  </div>

  <div class="controls">
    <button class="btn btn-mic" id="micBtn">[ REC ]</button>
    <input class="ctrl-input-text" type="text" id="input" placeholder="// TYPE MESSAGE..." autofocus>
    <button class="btn btn-send" id="sendBtn">SEND</button>
  </div>
</div>

<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const sendBtn = document.getElementById('sendBtn');
const micBtn = document.getElementById('micBtn');
const statusEl = document.getElementById('status');
const typing = document.getElementById('typing');
const langSel = document.getElementById('lang');
const voiceSel = document.getElementById('voice');
const stepsInp = document.getElementById('steps');
const speedInp = document.getElementById('speed');
const apiUrlInp = document.getElementById('apiUrl');
const apiKeyInp = document.getElementById('apiKey');
const settingsToggle = document.getElementById('settingsToggle');
const settingsBody = document.getElementById('settingsBody');

// --- Settings persistence ---
const DEFAULT_API_URL = 'http://127.0.0.1:8080/v1/chat/completions';
function loadSettings() {
    apiUrlInp.value = localStorage.getItem('supertone_api_url') || DEFAULT_API_URL;
    apiKeyInp.value = localStorage.getItem('supertone_api_key') || '';
}
function saveSettings() {
    localStorage.setItem('supertone_api_url', apiUrlInp.value.trim());
    localStorage.setItem('supertone_api_key', apiKeyInp.value.trim());
}
apiUrlInp.addEventListener('change', saveSettings);
apiKeyInp.addEventListener('input', saveSettings);
loadSettings();

// Settings drawer toggle
settingsToggle.addEventListener('click', () => {
    const open = settingsBody.classList.toggle('open');
    settingsToggle.classList.toggle('open', open);
    saveSettings();
});

let audioQueue = [];
let isPlaying = false;
let currentAssistantDiv = null;
let currentReasoningDiv = null;

// --- Audio playback ---
function playNext() {
    if (audioQueue.length === 0) { isPlaying = false; return; }
    isPlaying = true;
    const b64 = audioQueue.shift();
    const audio = new Audio('data:audio/wav;base64,' + b64);
    audio.onended = () => playNext();
    audio.onerror = () => playNext();
    audio.play().catch(() => playNext());
}

function queueAudio(b64) {
    audioQueue.push(b64);
    if (!isPlaying) playNext();
}

// --- Chat ---
function addMessage(role, text) {
    const div = document.createElement('div');
    div.className = 'msg ' + role;
    div.textContent = text;
    const chatArea = chat;
    chatArea.appendChild(div);
    chatArea.scrollTop = chatArea.scrollHeight;
    return div;
}

function addOrUpdateAssistant(text) {
    if (!currentAssistantDiv) {
        currentAssistantDiv = addMessage('assistant', '');
    }
    currentAssistantDiv.textContent += text;
    chat.scrollTop = chat.scrollHeight;
}

function addReasoning(text) {
    if (!currentReasoningDiv) {
        currentReasoningDiv = document.createElement('div');
        currentReasoningDiv.className = 'msg reasoning';
        chat.appendChild(currentReasoningDiv);
    }
    currentReasoningDiv.textContent += text;
    chat.scrollTop = chat.scrollHeight;
}

function setStatus(text, cls) {
    statusEl.textContent = '[ SYS ] ' + text.toUpperCase();
    statusEl.className = 'status-bar' + (cls ? ' ' + cls : '');
}

function setTyping(active) {
    typing.className = active ? 'think-line active' : 'think-line';
}

function clearHistory() {
    chat.querySelectorAll('.msg').forEach(m => m.remove());
    audioQueue = [];
    currentAssistantDiv = null;
    currentReasoningDiv = null;
    sendToServer('clear').catch(() => {});
}

// --- Send to server (SSE) ---
async function sendToServer(message) {
    sendBtn.disabled = true;
    micBtn.disabled = true;
    setTyping(true);
    setStatus('PROCESSING', 'active');
    currentAssistantDiv = null;
    currentReasoningDiv = null;

    let resp;
    try {
        resp = await fetch('/api/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            message: message,
            lang: langSel.value,
            voice: voiceSel.value,
            steps: parseInt(stepsInp.value),
            speed: parseFloat(speedInp.value),
            api_url: apiUrlInp.value.trim(),
            api_key: apiKeyInp.value.trim(),
        })
        });
    } catch (err) {
        setStatus('CONNECTION ERROR: ' + err.message, 'warn');
        typing.className = 'think-line';
        sendBtn.disabled = false;
        micBtn.disabled = false;
        return;
    }

    if (!resp.ok) {
        setStatus('SERVER ERROR: ' + resp.status, 'warn');
        typing.className = 'think-line';
        sendBtn.disabled = false;
        micBtn.disabled = false;
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
                        currentReasoningDiv = null;
                        addOrUpdateAssistant(data.text);
                    } else if (data.type === 'audio') {
                        queueAudio(data.data);
                    } else if (data.type === 'reasoning') {
                        addReasoning(data.text);
                    } else if (data.type === 'error') {
                        setStatus('ERROR: ' + data.text, 'warn');
                    } else if (data.type === 'done') {
                        currentReasoningDiv = null;
                        setTyping(false);
                        setStatus('READY');
                    }
                } catch(e) {}
            }
        }
    }

    sendBtn.disabled = false;
    micBtn.disabled = false;
    setTyping(false);
    input.focus();
}

// --- Send text ---
function sendText() {
    const text = input.value.trim();
    if (!text) return;
    if (text === '/clear') { clearHistory(); input.value = ''; return; }
    addMessage('user', text);
    input.value = '';
    sendToServer(text);
}

sendBtn.addEventListener('click', sendText);
input.addEventListener('keydown', e => {
    if (e.key === 'Enter') sendText();
});

// --- Microphone (Web Speech API) ---
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let isRecording = false;

if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onresult = (event) => {
        let final = '';
        let interim = '';
        for (let i = 0; i < event.results.length; i++) {
            if (event.results[i].isFinal) {
                final += event.results[i][0].transcript;
            } else {
                interim += event.results[i][0].transcript;
            }
        }
        if (final) {
            input.value = final;
        } else {
            input.value = interim;
        }
    };

    recognition.onend = () => {
        micBtn.classList.remove('recording');
        isRecording = false;
        setStatus('READY');
        if (input.value.trim()) {
            sendText();
        }
    };

    recognition.onerror = (e) => {
        micBtn.classList.remove('recording');
        isRecording = false;
        setStatus('MIC ERROR: ' + e.error, 'warn');
    };
} else {
    micBtn.style.display = 'none';
    setStatus('SPEECH API NOT SUPPORTED', 'warn');
}

micBtn.addEventListener('click', () => {
    if (!recognition) return;
    if (isRecording) {
        recognition.stop();
    } else {
        try {
            recognition.lang = langSel.value === 'pt' ? 'pt-BR' :
                               langSel.value === 'es' ? 'es-ES' :
                               langSel.value === 'fr' ? 'fr-FR' :
                               langSel.value === 'de' ? 'de-DE' :
                               langSel.value === 'ja' ? 'ja-JP' :
                               langSel.value === 'ko' ? 'ko-KR' : 'en-US';
            recognition.start();
            isRecording = true;
            micBtn.classList.add('recording');
            micBtn.textContent = '[ LIVE ]';
            setStatus('LISTENING', 'active');
            input.value = '';
            input.placeholder = '// SPEAK NOW...';
        } catch(e) {
            setStatus('MIC START ERROR', 'warn');
        }
    }
});

setStatus('READY');
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


def main():
    global tts, style, config

    parser = argparse.ArgumentParser(description="Supertonic Voice Chat Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host")
    parser.add_argument("--port", type=int, default=7777, help="Port")
    parser.add_argument("--api", default=LLAMA_API, help="LLM API URL")
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
        "model": args.model,
    })

    print("🚀 Loading Supertonic TTS...")
    tts = TTS(auto_download=True)
    style = tts.get_voice_style(voice_name=config["voice"])

    print(f"""
╔══════════════════════════════════════════╗
║   🎤 Supertonic Voice Chat              ║
║   Open: http://{args.host}:{args.port}          ║
║   LLM:  {args.api}        ║
║   Voice: {args.voice}  |  Lang: {args.lang}      ║
║   Steps: {args.steps}  |  Speed: {args.speed}          ║
╚══════════════════════════════════════════╝
""")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
