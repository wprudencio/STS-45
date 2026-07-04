"""
Realtime voice conversation mode for Supertonic.

A small asyncio WebSocket server (run in a daemon thread alongside the Flask
app) that mirrors the Hugging Face "hf-realtime-voice" speech-to-speech loop,
but fully local:

    you speak -> VAD -> parakeet STT -> llama.cpp LLM -> Supertonic TTS -> orb replies

The browser streams 16 kHz PCM16 mono over the socket; the server runs an
energy VAD to find utterance boundaries, transcribes each utterance, streams
the LLM reply, and speaks it back sentence-by-sentence as PCM16 frames so audio
starts before the full reply is generated. Barge-in (interrupting the assistant
mid-speech) is supported both server- and client-side.

Protocol (text frames are JSON; binary frames are raw PCM16 little-endian):

  client -> server
    {type:"start", lang,voice,steps,speed,api_url,api_key,sys_prompt,max_tokens}
    {type:"stop"}              clean teardown
    {type:"barge"}             request cancellation of the current turn
    <bytes>                    16 kHz PCM16 mono mic audio

  server -> client
    {type:"ready", sampleRate}               session accepted; tts.sample_rate
    {type:"state", state}                    listening | thinking | speaking
    {type:"transcript", role, text, final}   user (final) / assistant (delta)
    {type:"clear"}                           drop the TTS playback queue (barge)
    {type:"error", text}
    <bytes>                                  TTS PCM16 mono at sampleRate
"""

import asyncio
import io
import json
import re
import threading
import time
import wave

import numpy as np
import requests
import websockets
from websockets.asyncio.server import serve

# --- Voice activity detection (mirrors the browser PTT thresholds) ----------
SR_IN = 16000
VAD_RMS = 0.012          # low: detects the user's voice while listening (no playback)
BARGE_RMS = 0.05        # higher: resists speaker bleed when interrupting the assistant
SILENCE_MS = 650
MIN_UTT = int(0.30 * SR_IN)        # ~0.3s minimum utterance
MAX_UTT = int(12 * SR_IN)          # force-split an over-long run
BARGE_CONFIRM = 3                  # consecutive voiced frames -> barge-in

# --- Streaming TTS chunking -------------------------------------------------
# We synthesize per clause so the first audio appears early and barge-in stays
# responsive (cancel is checked between chunks, not once per paragraph).
_CHUNK_MAX = 140                    # chars; split longer sentences further
_CLAUSE_RE = re.compile(
    r"[^.!?;:,\n]+[.!?;:,\n]*\s*",
    re.S,
)


def _rms(pcm16_bytes):
    a = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if a.size == 0:
        return 0.0, a
    return float(np.sqrt(np.mean(a * a))), a


def int16_to_wav_bytes(pcm16_bytes, sr=SR_IN):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16_bytes)
    return buf.getvalue()


def _chunk_text(text):
    """Yield speakable pieces, preferring clause punctuation then spaces."""
    text = text.strip()
    if not text:
        return
    pieces = _CLAUSE_RE.findall(text)
    if not pieces:
        pieces = [text]
    acc = ""
    for p in pieces:
        acc += p
        if len(acc) >= _CHUNK_MAX:
            # split a long accumulated piece on spaces
            while len(acc) > _CHUNK_MAX:
                cut = acc.rfind(" ", 0, _CHUNK_MAX)
                if cut <= 0:
                    yield acc[:_CHUNK_MAX]
                    acc = acc[_CHUNK_MAX:]
                else:
                    yield acc[:cut] + " "
                    acc = acc[cut:]
        if len(acc) >= 8 and acc[-1] in ".!?;:,\n":
            yield acc
            acc = ""
    if acc.strip():
        yield acc


class _Session:
    """Per-connection state. Audio handling runs on the loop thread; each turn's
    STT/LLM/TTS pipeline runs in a worker thread (blocking calls are dispatched
    there). An asyncio queue bridges worker -> loop -> socket for sends."""

    def __init__(self, ws, app, loop):
        self.ws = ws
        self.app = app
        self.loop = loop

        self.cfg = {}
        self.history = []
        self.tts_sr = 24000

        self.utt = bytearray()       # buffered mic PCM16 for the current utterance
        self.last_voice = 0.0
        self.last_partial_time = 0.0  # timestamp of last partial STT send
        self.last_partial_samples = 0  # sample count at last partial
        self.barge_count = 0
        self.barging = False         # capturing the post-barge utterance

        self.state = "idle"          # idle | listening | thinking | speaking
        self.turn_busy = False
        self.cancel = False
        self.closed = False

        self.out_q = asyncio.Queue()  # str(json) or bytes; drained by sender
        self.turn_q = asyncio.Queue()  # utterance bytes; drained by turn_loop

    # --- output (thread-safe enqueue onto the loop) -------------------------
    def emit(self, item):
        if self.closed:
            return
        try:
            self.loop.call_soon_threadsafe(self.out_q.put_nowait, item)
        except RuntimeError:
            pass

    def send_json(self, obj):
        self.emit(json.dumps(obj))

    def send_audio(self, pcm16_bytes):
        self.emit(pcm16_bytes)

    def set_state(self, s):
        if self.state != s:
            self.state = s
            self.send_json({"type": "state", "state": s})

    # --- loops --------------------------------------------------------------
    async def sender(self):
        try:
            while True:
                item = await self.out_q.get()
                if item is None:
                    break
                await self.ws.send(item)
        except Exception:
            pass

    async def turn_loop(self):
        while True:
            utt = await self.turn_q.get()
            if utt is None:
                break
            try:
                await self.loop.run_in_executor(None, self.run_turn, utt)
            except Exception as e:
                self.send_json({"type": "error", "text": str(e)})
            finally:
                self.turn_busy = False
                self.barging = False
                self.cancel = False
                if not self.closed:
                    self.set_state("listening")

    # --- mic audio (called on the loop thread) ------------------------------
    def on_audio(self, pcm16_bytes):
        if self.closed or not self.cfg:
            return
        rms, _ = _rms(pcm16_bytes)
        now = time.monotonic()

        # While a turn is running we only watch for barge-in (and, once it has
        # fired, capture the new utterance). The higher barge threshold resists
        # speaker->mic bleed so the assistant doesn't interrupt itself; use
        # headphones to all but eliminate echo.
        if self.turn_busy:
            if rms >= BARGE_RMS:
                self.barge_count += 1
                if self.barge_count >= BARGE_CONFIRM and not self.cancel:
                    self.cancel = True
                    self.barging = True
                    self.send_json({"type": "clear"})
                    self.utt = bytearray()
                    self.last_voice = now
            else:
                self.barge_count = 0
            if self.barging:
                if rms >= VAD_RMS:
                    self.last_voice = now
                self.utt.extend(pcm16_bytes)
            return

        voiced = rms >= VAD_RMS
        if voiced:
            self.last_voice = now
        self.utt.extend(pcm16_bytes)

        n = len(self.utt) // 2
        silence = (now - self.last_voice) * 1000

        # Periodic partial STT while user is speaking (every ~500ms, min 0.5s speech)
        partial_interval = 0.50  # seconds
        partial_min_samples = int(0.5 * SR_IN)  # don't send partials for <0.5s speech
        if n >= partial_min_samples and (now - self.last_partial_time) >= partial_interval \
                and (n - self.last_partial_samples) >= int(0.35 * SR_IN) \
                and silence < SILENCE_MS:
            self.last_partial_time = now
            self.last_partial_samples = n
            # Run partial STT in executor (non-blocking)
            utt_snap = bytes(self.utt)
            lang = self.cfg.get("lang", "en")
            stt_api = self.cfg.get("stt_api_url", "")
            async def _send_partial():
                try:
                    loop = asyncio.get_running_loop()
                    text = await loop.run_in_executor(
                        None, transcribe_wav,
                        int16_to_wav_bytes(utt_snap, SR_IN), "partial.wav", lang, stt_api
                    )
                    if text and not self.cancel and not self.turn_busy:
                        self.send_json({"type": "transcript", "role": "user", "text": text, "final": False})
                except Exception:
                    pass
            asyncio.create_task(_send_partial())

        if n >= MIN_UTT and (silence >= SILENCE_MS or n >= MAX_UTT):
            utt = bytes(self.utt)
            self.utt = bytearray()
            self.last_partial_time = 0.0
            self.last_partial_samples = 0
            self.turn_busy = True
            self.barging = False
            self.set_state("thinking")
            self.turn_q.put_nowait(utt)

    # --- one conversation turn (runs in a worker thread) --------------------
    def run_turn(self, utt_bytes):
        rms, _ = _rms(utt_bytes)
        n = len(utt_bytes) // 2
        if rms < VAD_RMS * 0.5 or n < MIN_UTT:
            return  # too quiet / too short; silently resume listening

        self.set_state("thinking")
        text = transcribe_wav(
            int16_to_wav_bytes(utt_bytes, SR_IN),
            "utt.wav",
            self.cfg.get("lang", "en"),
            self.cfg.get("stt_api_url", ""),
        )
        if self.cancel:
            return
        if not text:
            return
        self.send_json({"type": "transcript", "role": "user", "text": text, "final": True})
        self.history.append({"role": "user", "content": text})

        acc = ""
        buf = ""
        for kind, tok in stream_llm(self.history, self.cfg):
            if self.cancel:
                break
            if kind != "text":
                continue
            self.send_json({"type": "transcript", "role": "assistant", "text": tok, "final": False})
            acc += tok
            buf += tok
            # Flush complete clauses as speech as they arrive.
            while True:
                piece, buf = _take_one_clause(buf)
                if piece is None:
                    break
                self._speak(piece)
                if self.cancel:
                    break
        if not self.cancel and buf.strip():
            self._speak(buf)
        if not self.cancel:
            self.send_json({"type": "transcript", "role": "assistant", "text": "", "final": True})
            if acc.strip():
                self.history.append({"role": "assistant", "content": acc})

    def _speak(self, text):
        for chunk in _chunk_text(text):
            if self.cancel:
                return
            pcm = synth_to_pcm16(self.app, self.cfg, chunk)
            if pcm is None:
                return
            if self.cancel:
                return
            self.set_state("speaking")
            self.send_audio(pcm.tobytes())


_SENT_END = re.compile(r"[.!?]", re.S)


def _take_one_clause(buf):
    """Split off the first complete sentence from `buf`.

    Returns (piece|None, rest). We only flush once a sentence terminator (.!?)
    has arrived, so the first TTS call happens on a full clause, not a fragment.
    A very long buffer with no terminator (a rambling model) is force-flushed so
    synthesis can start and barge-in stays responsive."""
    m = _SENT_END.search(buf)
    if m:
        end = m.end()
        # absorb trailing whitespace and closing quotes so they aren't stranded
        while end < len(buf) and buf[end] in " \t\n\")\']}":
            end += 1
        return buf[:end], buf[end:]
    if len(buf) >= _CHUNK_MAX * 4:
        return buf, ""
    return None, buf


# --- Local pipeline helpers (thin wrappers over the existing services) -----
def transcribe_wav(wav_bytes, filename, lang, stt_api_url):
    """POST a WAV blob to the parakeet.cpp server and return the transcript."""
    lang_map = {
        "en": "en", "pt": "pt", "es": "es", "fr": "fr",
        "de": "de", "ja": "ja", "ko": "ko",
    }
    parakeet_lang = lang_map.get(lang, "en")
    url = (stt_api_url or "").rstrip("/") + "/v1/audio/transcriptions"
    try:
        resp = requests.post(
            url,
            files={"file": (filename, wav_bytes, "audio/wav")},
            data={"language": parakeet_lang, "response_format": "json"},
            timeout=30,
        )
        resp.raise_for_status()
        return (resp.json().get("text", "") or "").strip()
    except Exception:
        return ""


def stream_llm(history, cfg):
    """Stream an OpenAI-style chat completion from llama.cpp.

    Yields (kind, token) tuples where kind is 'text' or 'reasoning'.
    """
    api_url = (cfg.get("api_url") or "").strip()
    if not api_url:
        return
    payload = {
        "model": cfg.get("model", "default"),
        "messages": history,
        "stream": True,
        "max_tokens": int(cfg.get("max_tokens", 512)),
        "temperature": 0.7,
    }
    headers = {"Content-Type": "application/json"}
    api_key = (cfg.get("api_key") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        r = requests.post(api_url, json=payload, headers=headers, stream=True, timeout=120)
        r.raise_for_status()
        r.encoding = "utf-8"
    except Exception:
        return
    has_content = False
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
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
            yield ("reasoning", reasoning)
        elif content:
            has_content = True
            yield ("text", content)


def synth_to_pcm16(app, cfg, text):
    """Synthesize `text` with the shared Supertonic engine; return int16 PCM."""
    tts = getattr(app, "tts", None)
    if tts is None:
        return None
    try:
        vs = tts.get_voice_style(voice_name=cfg.get("voice", "M1"))
        with app.tts_lock:
            wav, _dur = tts.synthesize(
                text=text,
                lang=cfg.get("lang", "en"),
                voice_style=vs,
                total_steps=int(cfg.get("steps", 5)),
                speed=float(cfg.get("speed", 1.15)),
            )
        w = np.clip(wav.squeeze().astype(np.float32), -1.0, 1.0)
        return (w * 32767).astype(np.int16)
    except Exception:
        return None


# --- Server -----------------------------------------------------------------
_app = None


async def _handler(ws):
    loop = asyncio.get_running_loop()
    sess = _Session(ws, _app, loop)
    sender = asyncio.create_task(sess.sender())
    turns = asyncio.create_task(sess.turn_loop())
    try:
        async for msg in ws:
            if isinstance(msg, bytes):
                sess.on_audio(msg)
                continue
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue
            t = data.get("type")
            if t == "start":
                sess.cfg = {
                    "lang": data.get("lang", _app.config.get("lang", "en")),
                    "voice": data.get("voice", _app.config.get("voice", "M1")),
                    "steps": data.get("steps", _app.config.get("steps", 5)),
                    "speed": data.get("speed", _app.config.get("speed", 1.15)),
                    "max_tokens": int(data.get("max_tokens", 512)),
                    "model": _app.config.get("model", "default"),
                    "api_url": (data.get("api_url") or "").strip() or _app.config.get("api_url", ""),
                    "api_key": (data.get("api_key") or "").strip(),
                    "stt_api_url": _app.config.get("stt_api_url", ""),
                }
                sp = (data.get("sys_prompt") or "").strip()
                sess.history = [{"role": "system", "content": sp or _app.SYS_PROMPT}]
                tts = getattr(_app, "tts", None)
                if tts is None:
                    sess.send_json({"type": "error", "text": "TTS model still loading — retrying…"})
                    continue
                sess.tts_sr = getattr(tts, "sample_rate", 24000)
                sess.cancel = False
                sess.set_state("listening")
                await ws.send(json.dumps({"type": "ready", "sampleRate": sess.tts_sr}))
            elif t == "barge":
                sess.cancel = True
            elif t == "stop":
                break
    except Exception:
        pass
    finally:
        sess.closed = True
        sess.cancel = True
        sess.out_q.put_nowait(None)
        sess.turn_q.put_nowait(None)
        try:
            await asyncio.wait_for(asyncio.gather(sender, turns, return_exceptions=True), timeout=2)
        except asyncio.TimeoutError:
            pass
        try:
            await ws.close()
        except Exception:
            pass


def start(host, port, app_module):
    """Start the realtime WebSocket server in a daemon thread."""
    global _app
    _app = app_module
    loop = asyncio.new_event_loop()

    async def _serve():
        async with serve(_handler, host, port, max_size=2 ** 24):
            print(f"  Realtime WS : ws://{host}:{port}/ws")
            await asyncio.Future()  # run forever

    def _run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_serve())

    t = threading.Thread(target=_run, daemon=True, name="supertonic-realtime")
    t.start()
    return t
