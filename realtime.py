"""
Realtime voice conversation mode for STS-45.

A small asyncio WebSocket server (run in a daemon thread alongside the Flask
app) that mirrors the Hugging Face "hf-realtime-voice" speech-to-speech loop,
but fully local:

    you speak -> VAD -> parakeet STT -> llama.cpp LLM -> Piper TTS -> orb replies

The browser streams 16 kHz PCM16 mono over the socket; the server runs an
energy VAD to find utterance boundaries, transcribes each utterance, streams
the LLM reply, and speaks it back sentence-by-sentence as PCM16 frames so audio
starts before the full reply is generated.

Protocol (text frames are JSON; binary frames are raw PCM16 little-endian):

  client -> server
    {type:"start", lang,voice,steps,speed,api_url,api_key,sys_prompt,max_tokens}
    {type:"stop"}              clean teardown
    <bytes>                    16 kHz PCM16 mono mic audio

  server -> client
    {type:"ready", sampleRate}               session accepted; tts.sample_rate
    {type:"state", state}                    listening | thinking | speaking
    {type:"transcript", role, text, final}   user (final) / assistant (delta)
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

# --- logging (stdout → Docker console) ---
_SESS_CTR = [0]  # mutable counter shared across sessions

def _now_ts():
    return time.strftime("%H:%M:%S", time.localtime()) + f".{int(time.time() * 1000) % 1000:03d}"

def _log(sid, msg):
    print(f"[{_now_ts()}] [sess:{sid}] {msg}", flush=True)

# --- Voice activity detection (mirrors the browser PTT thresholds) ----------
SR_IN = 16000
VAD_RMS = 0.022          # voice activity detection threshold (was 0.012)
SILENCE_MS = 650
MIN_UTT = int(0.30 * SR_IN)        # ~0.3s minimum utterance
MAX_UTT = int(18 * SR_IN)          # force-split an over-long run (was 12s)

# --- Memory safety: bounded history + queues ------------------------------
# The conversation history is sent to the LLM every turn, so unbounded growth
# makes each response slower and eats memory. We keep the system prompt plus a
# sliding window of recent turns; old turns are dropped. Tune HISTORY_MAX_TURNS
# (= number of user+assistant pairs retained) to taste.
HISTORY_MAX_TURNS = 4               # pairs of (user, assistant) kept
OUT_Q_MAX = 64                     # backpressure: don't queue infinite audio
PARTIAL_STT_TASKS = 1               # at most one in-flight partial transcription

# --- Streaming TTS chunking -------------------------------------------------
# We synthesize per clause so the first audio appears early.
# responsive (cancel is checked between chunks, not once per paragraph).
_CHUNK_MAX = 140                    # chars; split longer sentences further
_CLAUSE_RE = re.compile(
    r"[^.!?;:,\n]+[.!?;:,\n]*\s*",
    re.S,
)
# Fast-first-phrase: the first reply chunk is flushed as soon as a soft pause
# (comma/colon/semicolon/newline) arrives instead of waiting for a full
# sentence terminator. This shaves the better part of the LLM's first-sentence
# latency off the time-to-first-audio. Only the very first chunk of each turn
# uses this; subsequent chunks keep clause boundaries for natural prosody.
_FIRST_PHRASE_MIN = 4               # min chars before flushing on a soft break
_FIRST_PHRASE_MAX = 80              # force-flush the first phrase by this length
_SOFT_END = re.compile(r"[,;:\n]")


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

        _SESS_CTR[0] += 1
        self.sid = _SESS_CTR[0]  # stable integer session id for logs
        _log(self.sid, "session created")

        self.cfg = {}
        self.history = []
        self.tts_sr = 24000

        self.utt = bytearray()       # buffered mic PCM16 for the current utterance
        self.last_voice = 0.0
        self.last_partial_time = 0.0  # timestamp of last partial STT send
        self.last_partial_samples = 0  # sample count at last partial

        self.state = "idle"          # idle | listening | thinking | speaking
        self.turn_busy = False
        self.cancel = False
        self.closed = False

        self.out_q = asyncio.Queue(maxsize=OUT_Q_MAX)  # str(json) or bytes; drained by sender
        self.turn_q = asyncio.Queue()  # utterance bytes; drained by turn_loop
        self._partial_task = None      # in-flight partial STT task (cancellable)
        self._partial_seq = 0          # monotonically increasing partial id
        self._last_rms_log = 0.0       # throttle periodic RMS logs
        self._loop_tid = threading.get_ident()  # event-loop thread id for emit

    # --- output (thread-safe enqueue onto the loop) -------------------------
    # run_turn() executes in a ThreadPoolExecutor thread and calls
    # send_json / send_audio / set_state. asyncio.Queue.put_nowait is NOT
    # thread-safe, so we route through call_soon_threadsafe when called from
    # a non-event-loop thread. When already on the loop thread (on_audio,
    # handler), call_soon_threadsafe adds unnecessary delay.
    def emit(self, item):
        if self.closed:
            return
        if threading.get_ident() == self._loop_tid:
            # Already on event-loop thread — call directly, no scheduling overhead
            self._emit_on_loop(item)
        else:
            # Executor thread (run_turn) — must go through the event loop
            self.loop.call_soon_threadsafe(self._emit_on_loop, item)

    def _emit_on_loop(self, item):
        """Called on the event-loop thread — direct queue put."""
        if self.closed:
            return
        try:
            self.out_q.put_nowait(item)
        except asyncio.QueueFull:
            try:
                self.out_q.get_nowait()
                self.out_q.put_nowait(item)
            except Exception:
                pass

    def send_json(self, obj):
        self.emit(json.dumps(obj))

    def send_audio(self, pcm16_bytes):
        self.emit(pcm16_bytes)

    def set_state(self, s):
        if self.state != s:
            _log(self.sid, f"state {self.state} -> {s}")
            self.state = s
            self.send_json({"type": "state", "state": s})

    def _cancel_partial(self):
        """Cancel any in-flight partial STT and bump the sequence so its
        result is discarded. Prevents stale/racing partials from overwriting
        the final transcript or keeping the STT server busy."""
        if self._partial_task is not None and not self._partial_task.done():
            _log(self.sid, "cancel in-flight partial STT")
            self._partial_task.cancel()
            self._partial_seq += 1
        self._partial_task = None

    def _try_submit_utterance(self, now=None):
        """If the buffered utterance has reached an endpoint (silence or max
        length), cancel any partial STT and hand the audio to the turn queue.
        Safe to call repeatedly: it is a no-op when no endpoint is reached or
        when a turn is already in flight."""
        if self.turn_busy:
            return
        now = now or time.monotonic()
        n = len(self.utt) // 2
        if n < MIN_UTT:
            return
        silence = (now - self.last_voice) * 1000
        if silence < SILENCE_MS and n < MAX_UTT:
            return
        _log(self.sid, f"submit utterance: {n} samples ({n/SR_IN:.1f}s), silence={silence:.0f}ms, rms={_rms(bytes(self.utt))[0]:.4f}")
        self._cancel_partial()
        utt = bytes(self.utt)
        self.utt = bytearray()
        self.last_partial_time = 0.0
        self.last_partial_samples = 0
        self.turn_busy = True
        self.set_state("thinking")
        _log(self.sid, f"DEBUG: before put_nowait, turn_q empty={self.turn_q.empty()}, loop running={self.loop.is_running()}")
        self.turn_q.put_nowait(utt)
        _log(self.sid, f"DEBUG: put_nowait OK, turn_q size={self.turn_q.qsize()}")

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
            _log(self.sid, f"DEBUG: turn_loop waiting, qempty={self.turn_q.empty()}, alive={asyncio.current_task().done()}")
            utt = await self.turn_q.get()
            if utt is None:
                _log(self.sid, "turn_loop stopping")
                break
            self.turn_busy = True   # guard: old turn finally may have cleared it
            _log(self.sid, f"turn start — utt={len(utt)//2}spl pcm")
            try:
                await self.loop.run_in_executor(None, self.run_turn, utt)
            except Exception as e:
                _log(self.sid, f"turn exception: {e}")
                self.send_json({"type": "error", "text": str(e)})
            finally:
                _log(self.sid, f"turn end — busy={self.turn_busy} cancel={self.cancel}")
                self.turn_busy = False
                self.cancel = False
                self._tts_warned = False
                # The user may have kept talking while we were busy (STT/LLM/TTS).
                # Drain any completed utterance immediately; if they are still
                # talking, the next audio frame will trigger the boundary check.
                self._try_submit_utterance()
                if not self.closed and not self.turn_busy:
                    self.set_state("listening")

    # --- mic audio (called on the loop thread) ------------------------------
    def on_audio(self, pcm16_bytes):
        if self.closed or not self.cfg:
            return
        rms, _ = _rms(pcm16_bytes)
        now = time.monotonic()

        # Buffer every single frame. Previously frames received while a turn was
        # in flight were dropped, so users who kept talking lost everything
        # said during STT/LLM/TTS. The VAD boundary is only evaluated when no
        # turn is running.
        if rms >= VAD_RMS:
            self.last_voice = now
        self.utt.extend(pcm16_bytes)

        # throttle: log RMS + buffer size roughly every 1s
        if now - self._last_rms_log >= 1.0:
            self._last_rms_log = now
            _log(self.sid, f"audio rms={rms:.4f} buf={len(self.utt)//2}spl turn_busy={self.turn_busy}")

        # While a turn is in progress, just buffer audio. The drain after
        # run_turn finishes will submit any completed utterance.
        if self.turn_busy:
            return

        n = len(self.utt) // 2
        silence = (now - self.last_voice) * 1000

        # Periodic partial STT (only while freely listening).
        # We cap the buffer length: re-transcribing a 15+ second monologue every
        # half second wastes STT time and the result is discarded anyway.
        partial_interval = 0.75  # seconds
        partial_min_samples = int(0.5 * SR_IN)
        partial_max_samples = int(8 * SR_IN)
        if n <= partial_max_samples and n >= partial_min_samples \
                and (now - self.last_partial_time) >= partial_interval \
                and (n - self.last_partial_samples) >= int(0.35 * SR_IN) \
                and silence < SILENCE_MS:
            self.last_partial_time = now
            self.last_partial_samples = n
            self._cancel_partial()
            utt_snap = bytes(self.utt)
            seq = self._partial_seq + 1
            self._partial_seq = seq
            lang = self.cfg.get("lang", "en")
            stt_api = self.cfg.get("stt_api_url", "")
            async def _send_partial(my_seq=seq, snap=utt_snap):
                try:
                    loop = asyncio.get_running_loop()
                    text = await loop.run_in_executor(
                        None, transcribe_wav,
                        int16_to_wav_bytes(snap, SR_IN), "partial.wav", lang, stt_api
                    )
                    # discard stale results (a newer partial superseded us)
                    if my_seq != self._partial_seq:
                        return
                    if text and not self.cancel and not self.turn_busy:
                        self.send_json({"type": "transcript", "role": "user", "text": text, "final": False})
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
            self._partial_task = asyncio.create_task(_send_partial())

        # --- utterance complete? submit to turn queue ---
        self._try_submit_utterance(now)
        _log(self.sid, f"DEBUG: on_audio done turn_busy={self.turn_busy} utt={len(self.utt)//2}spl")

    # --- one conversation turn (runs in a worker thread) --------------------
    def run_turn(self, utt_bytes):
        rms, _ = _rms(utt_bytes)
        n = len(utt_bytes) // 2
        if rms < VAD_RMS * 0.5 or n < MIN_UTT:
            _log(self.sid, f"turn skipped: rms={rms:.4f} n={n} (too quiet/short)")
            return  # too quiet / too short; silently resume listening

        _log(self.sid, f"STT start — {n} samples ({n/SR_IN:.1f}s) rms={rms:.4f}")
        self.set_state("thinking")
        t0 = time.monotonic()
        text = transcribe_wav(
            int16_to_wav_bytes(utt_bytes, SR_IN),
            "utt.wav",
            self.cfg.get("lang", "en"),
            self.cfg.get("stt_api_url", ""),
        )
        _log(self.sid, f"STT done in {time.monotonic()-t0:.1f}s → {repr(text)}")
        if self.cancel:
            _log(self.sid, "turn cancelled after STT")
            return
        if not text:
            _log(self.sid, "STT returned empty — aborting turn (no response)")
            return
        self.send_json({"type": "transcript", "role": "user", "text": text, "final": True})
        self.history.append({"role": "user", "content": text})

        _log(self.sid, f"LLM stream start — history={len(self.history)} msgs")
        acc = ""
        buf = ""
        first_spoken = False
        chunk_count = 0
        for kind, tok in stream_llm(self.history, self.cfg):
            if self.cancel:
                _log(self.sid, "turn cancelled during LLM stream")
                break
            if kind == "error":
                _log(self.sid, f"LLM error: {tok}")
                self.send_json({"type": "error", "text": tok})
                continue
            if kind != "text":
                continue
            self.send_json({"type": "transcript", "role": "assistant", "text": tok, "final": False})
            acc += tok
            buf += tok
            # Flush the first phrase as early as a soft pause so the user hears
            # audio before the first sentence is fully generated; after that,
            # fall back to full-clause flushing for natural prosody.
            if not first_spoken:
                piece, buf = _take_first_phrase(buf)
                if piece is None:
                    continue
                _log(self.sid, f"TTS first phrase ({len(piece)} chars): {repr(piece[:80])}")
                self._speak(piece)
                first_spoken = True
                chunk_count += 1
                if self.cancel:
                    break
                continue
            # Flush complete clauses as speech as they arrive.
            while True:
                piece, buf = _take_one_clause(buf)
                if piece is None:
                    break
                self._speak(piece)
                chunk_count += 1
                if self.cancel:
                    break
        if not self.cancel and buf.strip():
            self._speak(buf)
            chunk_count += 1
        _log(self.sid, f"LLM stream done — acc={len(acc)} chars, {chunk_count} TTS chunks")
        if not self.cancel:
            self.send_json({"type": "transcript", "role": "assistant", "text": "", "final": True})
            if acc.strip():
                self.history.append({"role": "assistant", "content": acc})
                self._prune_history()
            else:
                _log(self.sid, "WARNING: LLM returned zero text content — no response spoken")

    def _prune_history(self):
        """Keep system prompt + recent turns. Caps message count AND total
        character count to prevent prompt eval from ballooning on CPU-only
        inference (e.g. Docker with llama-server)."""
        if not self.history:
            return
        sys_prefix = 1 if self.history[0].get("role") == "system" else 0
        max_msgs = sys_prefix + 2 * HISTORY_MAX_TURNS
        if len(self.history) > max_msgs:
            self.history = self.history[:sys_prefix] + self.history[-(max_msgs - sys_prefix):]
        # Hard-cap total content length. On CPU, a 4K+ token prompt easily
        # takes 5+ seconds to evaluate before the first token is generated.
        MAX_HIST_CHARS = 1500
        total = sum(len(m.get("content", "")) for m in self.history)
        while total > MAX_HIST_CHARS and len(self.history) > sys_prefix + 2:
            self.history.pop(sys_prefix)        # oldest user
            if sys_prefix < len(self.history):
                self.history.pop(sys_prefix)    # matching assistant
            total = sum(len(m.get("content", "")) for m in self.history)

    def _speak(self, text):
        for chunk in _chunk_text(text):
            if self.cancel:
                return
            pcm = synth_to_pcm16(self.app, self.cfg, chunk)
            if pcm is None:
                if not getattr(self, "_tts_warned", False):
                    self._tts_warned = True
                    if getattr(self.app, "tts", None) is None:
                        self.send_json({"type": "error", "text": "TTS model still loading — retrying…"})
                    else:
                        self.send_json({"type": "error", "text": "TTS synthesis failed (see server console)."})
                return
            if self.cancel:
                return
            self.set_state("speaking")
            self.send_audio(pcm.tobytes())


_SENT_END = re.compile(r"[.!?]", re.S)


def _take_first_phrase(buf):
    """Like `_take_one_clause` but tuned for the very first reply chunk: it
    also flushes on a soft pause (comma/colon/semicolon/newline) so audio starts
    before the first sentence is fully generated. Sentence terminators still
    flush as before, so a short "Yes." reply never regresses. Returns
    (piece|None, rest)."""
    # 1. Flush on a sentence terminator first (preserves prior latency for
    #    short first sentences that have no qualifying soft pause).
    m = _SENT_END.search(buf)
    if m:
        end = m.end()
        while end < len(buf) and buf[end] in " \t\n\")\']}":
            end += 1
        return buf[:end], buf[end:]
    # 2. Force-flush once the buffer is long even without any punctuation.
    if len(buf) >= _FIRST_PHRASE_MAX:
        cut = buf.rfind(" ")
        if cut < _FIRST_PHRASE_MIN - 1:
            return buf, ""
        return buf[:cut] + " ", buf[cut:]
    # 3. Early flush on a soft pause once we have enough text.
    sm = _SOFT_END.search(buf)
    if sm and sm.start() >= _FIRST_PHRASE_MIN - 1:
        end = sm.end()
        while end < len(buf) and buf[end] in " \t":
            end += 1
        return buf[:end], buf[end:]
    return None, buf


def _take_one_clause(buf):
    """Split off the first complete sentence from `buf`.

    Returns (piece|None, rest). We only flush once a sentence terminator (.!?)
    has arrived, so the first TTS call happens on a full clause, not a fragment.
    A very long buffer with no terminator (a rambling model) is force-flushed so
    synthesis can start and stay responsive."""
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
    # Scale the HTTP timeout with audio length so long utterances (and any
    # queueing behind a partial STT request) don't get killed mid-transcription.
    audio_seconds = max(1.0, len(wav_bytes) / (2 * SR_IN))
    timeout = max(30, int(audio_seconds * 4) + 5)
    t0 = time.monotonic()
    try:
        resp = requests.post(
            url,
            files={"file": (filename, wav_bytes, "audio/wav")},
            data={"language": parakeet_lang, "response_format": "json"},
            timeout=timeout,
        )
        elapsed = time.monotonic() - t0
        resp.raise_for_status()
        text = (resp.json().get("text", "") or "").strip()
        print(f"[{_now_ts()}] STT {filename} → {elapsed:.1f}s lang={parakeet_lang} text={repr(text)}", flush=True)
        return text
    except Exception as e:
        elapsed = time.monotonic() - t0
        print(f"[{_now_ts()}] STT FAIL ({filename}, {audio_seconds:.1f}s, timeout={timeout}s, elapsed={elapsed:.1f}s): {e}", flush=True)
        return ""


def stream_llm(history, cfg):
    """Stream an OpenAI-style chat completion from llama.cpp.

    Yields (kind, token) tuples where kind is 'text'.
    """
    api_url = (cfg.get("api_url") or "").strip()
    if not api_url:
        print(f"[{_now_ts()}] LLM SKIP: no api_url configured", flush=True)
        return
    payload = {
        "model": cfg.get("model", "default"),
        "messages": history,
        "stream": True,
        "max_tokens": int(cfg.get("max_tokens", 256)),
        "temperature": 0.7,
    }
    headers = {"Content-Type": "application/json"}
    api_key = (cfg.get("api_key") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    t0 = time.monotonic()
    print(f"[{_now_ts()}] LLM POST → {api_url} model={payload['model']} history={len(history)}", flush=True)
    try:
        r = requests.post(api_url, json=payload, headers=headers, stream=True, timeout=120)
        r.raise_for_status()
        r.encoding = "utf-8"
        print(f"[{_now_ts()}] LLM connected in {time.monotonic()-t0:.2f}s", flush=True)
    except Exception as e:
        print(f"[{_now_ts()}] LLM FAIL ({time.monotonic()-t0:.2f}s): {e}", flush=True)
        yield ("error", f"LLM request failed: {e}")
        return
    # Important: must close the streaming response even when the consumer
    # breaks early (e.g. teardown). Without this the underlying
    # and urllib3 connections leak, eventually exhausting the pool and making
    # every request slower/hang — the "gets slow over time" symptom.
    try:
        has_content = False
        has_reasoning = False
        char_count = 0
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
            finish = choices[0].get("finish_reason")
            delta = choices[0].get("delta", {})
            reasoning = delta.get("reasoning_content")
            content = delta.get("content")
            if reasoning and not has_content:
                has_reasoning = True
            if content:
                has_content = True
                char_count += len(content)
                yield ("text", content)
            if finish:
                print(f"[{_now_ts()}] LLM finish_reason={finish} chars={char_count}", flush=True)
        elapsed = time.monotonic() - t0
        if not has_content and not has_reasoning:
            print(f"[{_now_ts()}] LLM WARN: zero content after {elapsed:.1f}s", flush=True)
            yield ("error", "LLM returned no content (check that llama-server is running and the API URL is correct).")
        else:
            print(f"[{_now_ts()}] LLM DONE: {char_count} chars in {elapsed:.1f}s", flush=True)
    finally:
        # runs on normal exit, break, or generator close (teardown)
        try:
            r.close()
        except Exception:
            pass


def synth_to_pcm16(app, cfg, text):
    """Synthesize `text` with Piper TTS; return int16 PCM numpy array."""
    voice_map = getattr(app, "tts", None)
    if not voice_map:
        print(f"[{_now_ts()}] TTS SKIP: no voice map loaded yet", flush=True)
        return None
    try:
        from pathlib import Path
        from piper import PiperVoice

        voice_name = cfg.get("voice", "en_US-lessac-medium")
        with app.tts_lock:
            if voice_name not in voice_map:
                onnx_path = app._download_voice(voice_name)
                if onnx_path is None:
                    return None
                voice = PiperVoice.load(str(onnx_path))
                import json
                cfg_json = json.loads(Path(str(onnx_path) + ".json").read_text())
                sr = cfg_json.get("audio", {}).get("sample_rate", 22050)
                voice_map[voice_name] = (voice, sr)
            voice, sr = voice_map[voice_name]
        gen = voice.synthesize(text)
        pcm = b"".join(chunk.audio_int16_bytes for chunk in gen)
        arr = np.frombuffer(pcm, dtype=np.int16)
        print(f"[{_now_ts()}] TTS synth OK: {len(text)} chars → {len(arr)} samples ({len(arr)/sr:.2f}s @ {sr}Hz)", flush=True)
        return arr
    except Exception as e:
        print(f"[{_now_ts()}] TTS FAIL: {e} (text={repr(text[:60])})", flush=True)
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
                    "voice": data.get("voice", _app.config.get("voice", "en_US-lessac-medium")),
                    "max_tokens": int(data.get("max_tokens", 512)),
                    "model": _app.config.get("model", "default"),
                    "api_url": (data.get("api_url") or "").strip() or _app.config.get("api_url", ""),
                    "api_key": (data.get("api_key") or "").strip(),
                    "stt_api_url": _app.config.get("stt_api_url", ""),
                }
                sp = (data.get("sys_prompt") or "").strip()
                sess.history = [{"role": "system", "content": sp or _app.SYS_PROMPT}]
                _log(sess.sid, f"start cfg: voice={sess.cfg['voice']} lang={sess.cfg['lang']} model={sess.cfg['model']}")
                tts_dict = getattr(_app, "tts", None)
                if not tts_dict:
                    _log(sess.sid, "start rejected: TTS model not loaded")
                    sess.send_json({"type": "error", "text": "TTS model still loading — retrying…"})
                    continue
                sess.tts_sr = 22050  # Piper sample rate
                sess.cancel = False
                sess.set_state("listening")
                await ws.send(json.dumps({"type": "ready", "sampleRate": sess.tts_sr}))
            elif t == "stop":
                _log(sess.sid, "stop requested")
                break
    except Exception as e:
        _log(sess.sid, f"handler exception: {e}")
    finally:
        _log(sess.sid, "session closing")
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

    t = threading.Thread(target=_run, daemon=True, name="sts45-realtime")
    t.start()
    return t
