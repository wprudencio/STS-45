// STS-45 Realtime
const BOOT = JSON.parse(document.getElementById('boot').textContent);
const RT_WS_PORT = BOOT.ws_port;

// ---------- Settings ----------
function loadSettings() {
  const g = (k, d) => localStorage.getItem(k) || d;
  document.getElementById('voice').value = g('sts45_voice', 'en_US-lessac-medium');
  document.getElementById('lang').value = g('sts45_lang', 'en');
  const maxTokens = parseInt(g('sts45_max_tokens', '512'), 10);
  document.getElementById('maxTokens').value = maxTokens;
  document.getElementById('maxTokensVal').textContent = maxTokens;
  updateSlider(document.getElementById('maxTokens'));
  document.getElementById('sysPrompt').value = g('sts45_sys_prompt', '');
}
function readSettings() {
  return {
    lang: document.getElementById('lang').value,
    voice: document.getElementById('voice').value,
    max_tokens: parseInt(document.getElementById('maxTokens').value, 10) || 512,
    sys_prompt: document.getElementById('sysPrompt').value,
  };
}
let settingsSaveTimer = null;
function persistSettingsDebounced() {
  clearTimeout(settingsSaveTimer);
  settingsSaveTimer = setTimeout(persistSettings, 500);
}
function persistSettings() {
  const s = (k, v) => localStorage.setItem(k, v);
  const v = readSettings();
  s('sts45_voice', v.voice);
  s('sts45_lang', v.lang);
  s('sts45_max_tokens', v.max_tokens);
  s('sts45_sys_prompt', v.sys_prompt);
  try { fetch('/api/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(v) }); } catch (e) { }
  flashHint();
}
function flashHint() {
  const el = document.getElementById('settingsHint');
  if (!el) return;
  el.textContent = 'saved ✓';
  el.style.color = 'var(--neon-green)';
  setTimeout(() => { el.textContent = 'saved automatically'; el.style.color = ''; }, 1200);
}
function updateSlider(el) {
  const min = +el.min, max = +el.max, val = +el.value;
  el.style.setProperty('--fill', ((val - min) / (max - min) * 100) + '%');
}
window.updateSlider = updateSlider;

window.toggleSettings = function () {
  document.getElementById('settingsPanel').classList.toggle('hidden');
};
document.querySelectorAll('input, select, textarea').forEach(el => {
  if (el.id && ['voice', 'lang', 'maxTokens', 'sysPrompt'].includes(el.id)) {
    el.addEventListener('input', persistSettingsDebounced);
    el.addEventListener('change', persistSettingsDebounced);
  }
});
document.querySelectorAll('input[type=range]').forEach(updateSlider);
loadSettings();

// ---------- Toast ----------
let toastTimer = null;
function showToast(msg, type) {
  const old = document.querySelector('.toast');
  if (old) old.remove();
  const t = document.createElement('div');
  t.className = 'toast' + (type ? ' ' + type : '');
  const icon = type === 'error'
    ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>'
    : '';
  t.innerHTML = icon + '<span>' + escapeHtml(msg) + '</span>';
  const close = document.createElement('button');
  close.className = 'toast-close'; close.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
  close.onclick = () => t.remove();
  t.appendChild(close);
  document.body.appendChild(t);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.classList.add('hide'); setTimeout(() => t.remove(), 200); }, 6000);
}
window.showToast = showToast;
function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }

// ============================================================
// Realtime WebSocket session
// ============================================================
let rtWS = null, rtCtx = null, rtMicStream = null, rtScript = null, rtRunning = false;
let rtPlayCtx = null, rtTTSsr = 24000, rtPlaySources = [], rtPlayTime = 0, rtAudioMuted = false;
let rtState = 'idle', rtAsstEl = null, rtAsstText = '', rtUserEl = null;
const RT_MAX_LINES = 40;  // cap transcript DOM growth; old lines trimmed to keep long sessions fast
// client-side barge-in: stop AI playback the moment the user starts talking,
// without waiting for the server round-trip.
let rtBargeCount = 0, rtBargeArmed = true, rtBargeArmedTimer = null;
const RT_BARGE_RMS = 0.030;     // mic input energy that counts as "user talking"
const RT_BARGE_CONFIRM = 2;     // consecutive voiced 128ms frames -> trigger
const RT_BARGE_COOLDOWN = 500;  // ms before another barge can fire

function rtBarge() {
  // stop local playback instantly and tell server to drop the current turn
  if (!rtBargeArmed) return;
  rtBargeArmed = false;
  clearTimeout(rtBargeArmedTimer);
  rtBargeArmedTimer = setTimeout(() => { rtBargeArmed = true; rtBargeCount = 0; }, RT_BARGE_COOLDOWN);
  rtClearPlayback();
  if (rtWS && rtWS.readyState === 1) {
    try { rtWS.send(JSON.stringify({ type: 'barge' })); } catch (e) { }
  }
  setRTState('listening');
}

function setRTState(s) {
  rtState = s;
  const labels = { idle: 'click to start', connecting: 'Connecting…', listening: '', thinking: '', speaking: '' };
  const el = document.getElementById('rtState');
  el.textContent = s in labels ? labels[s] : s;
  el.className = 'rt-state ' + s;
  const wrap = document.getElementById('rtVizWrap');
  if (wrap) {
    wrap.className = 'rt-viz-wrap'
      + (s === 'listening' ? ' listening' : '')
      + (s === 'thinking' ? ' thinking' : '')
      + (s === 'speaking' ? ' speaking' : '')
      + (rtRunning ? ' running' : '');
  }
  if (s === 'speaking' || s === 'listening') rtAudioMuted = false;
  document.getElementById('rtEndBtn').classList.toggle('hidden', !rtRunning);
}

function rtSendStart() { if (rtWS && rtWS.readyState === 1) rtWS.send(JSON.stringify(Object.assign({ type: 'start' }, readSettings()))); }

function rtAppendUser(text) {
  const t = document.getElementById('rtTranscript');
  rtTrimTranscript(t);
  if (rtUserEl) {
    rtUserEl.querySelector('.rt-text').textContent = text;
    rtUserEl.classList.remove('live');
    rtUserEl = null;
  } else {
    const el = document.createElement('div'); el.className = 'rt-line user';
    el.innerHTML = '<span class="rt-role">You</span><span class="rt-text"></span>';
    el.querySelector('.rt-text').textContent = text;
    t.appendChild(el);
  }
  t.scrollTop = t.scrollHeight; rtAsstEl = null; rtAsstText = '';
}
function rtUpdateUserPartial(text) {
  const t = document.getElementById('rtTranscript');
  rtTrimTranscript(t);
  if (!rtUserEl) {
    rtUserEl = document.createElement('div'); rtUserEl.className = 'rt-line user live';
    rtUserEl.innerHTML = '<span class="rt-role">You</span><span class="rt-text"></span>';
    t.appendChild(rtUserEl);
  }
  rtUserEl.querySelector('.rt-text').textContent = text;
  t.scrollTop = t.scrollHeight;
}
function rtAppendAssistantDelta(tok) {
  const t = document.getElementById('rtTranscript');
  if (!rtAsstEl) {
    rtTrimTranscript(t);
    rtAsstEl = document.createElement('div'); rtAsstEl.className = 'rt-line assistant live';
    rtAsstEl.innerHTML = '<span class="rt-role">Assistant</span><span class="rt-text"></span>';
    t.appendChild(rtAsstEl); rtAsstText = '';
  }
  rtAsstText += tok; rtAsstEl.querySelector('.rt-text').textContent = rtAsstText;
  t.scrollTop = t.scrollHeight;
}
function rtFinalizeAssistant() { if (rtAsstEl) { rtAsstEl.classList.remove('live'); rtAsstEl = null; rtAsstText = ''; } }
function rtTrimTranscript(t) {
  // drop the oldest lines once we exceed the cap; keeps the DOM small so a
  // long session never turns into a giant, slow-to-paint/scroll tree.
  while (t.childElementCount > RT_MAX_LINES) t.removeChild(t.firstElementChild);
}

function rtClearPlayback() {
  rtAudioMuted = true;
  rtPlaySources.forEach(s => { try { s.stop(); } catch (e) { } });
  rtPlaySources = []; if (rtPlayCtx) rtPlayTime = rtPlayCtx.currentTime;
}

function onRTAudio(buf) {
  if (!rtPlayCtx || rtAudioMuted) return;
  if (rtPlayCtx.state === 'suspended') { try { rtPlayCtx.resume(); } catch (e) { } }
  if (rtPlayCtx.state !== 'running') { showToast('Playback audio blocked by browser (state: ' + rtPlayCtx.state + ')', 'error'); }
  const i16 = new Int16Array(buf); const f32 = new Float32Array(i16.length);
  let sumSq = 0;
  for (let i = 0; i < i16.length; i++) { const s = i16[i] / 32768; f32[i] = s; sumSq += s * s; }
  rtPulsePushOut(Math.min(0.7, Math.sqrt(sumSq / i16.length) * 5));
  const ab = rtPlayCtx.createBuffer(1, f32.length, rtTTSsr);
  ab.copyToChannel(f32, 0);
  const src = rtPlayCtx.createBufferSource(); src.buffer = ab; src.connect(rtPlayCtx.destination);
  const now = rtPlayCtx.currentTime;
  if (rtPlayTime < now) rtPlayTime = now + 0.02;
  src.start(rtPlayTime); rtPlayTime += ab.duration; rtPlaySources.push(src);
  src.onended = () => { rtPlaySources = rtPlaySources.filter(s => s !== src); };
}

function onRTMsg(m) {
  if (m.type === 'ready') { rtTTSsr = m.sampleRate || 24000; setRTState('listening'); }
  else if (m.type === 'state') setRTState(m.state);
  else if (m.type === 'transcript') {
    if (m.role === 'user') {
      if (m.final) rtAppendUser(m.text);
      else rtUpdateUserPartial(m.text);
    } else if (m.role === 'assistant') {
      if (m.final) rtFinalizeAssistant();
      else if (m.text) rtAppendAssistantDelta(m.text);
    }
  } else if (m.type === 'clear') rtClearPlayback();
  else if (m.type === 'error') {
    showToast('Realtime: ' + m.text, 'error');
    if (/loading/i.test(m.text)) { setRTState('connecting'); setTimeout(rtSendStart, 2500); }
  }
}

async function startRealtime() {
  if (rtRunning) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showToast('Microphone unavailable (needs HTTPS or localhost)', 'error'); return;
  }
  setRTState('connecting');
  // Same port? Use relative URL (Docker/nginx single-port). Otherwise explicit WS port (local dev).
  const pagePort = String(location.port || (location.protocol === 'https:' ? '443' : '80'));
  const wsHost = (!location.hostname || location.hostname === '0.0.0.0') ? '127.0.0.1' : location.hostname;
  const url = (pagePort === String(RT_WS_PORT))
    ? (location.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + location.host + '/ws'
    : 'ws://' + wsHost + ':' + RT_WS_PORT + '/ws';
  try { rtWS = new WebSocket(url); }
  catch (e) { showToast('WS error: ' + e.message, 'error'); stopRealtimeUI(''); return; }
  rtWS.binaryType = 'arraybuffer';
  rtWS.onopen = () => { rtSendStart(); };
  rtWS.onmessage = (e) => { if (e.data instanceof ArrayBuffer) onRTAudio(e.data); else { try { onRTMsg(JSON.parse(e.data)); } catch (_) { } } };
  rtWS.onclose = () => { if (rtRunning) { showToast('Realtime disconnected', 'error'); stopRealtimeUI(''); } };
  rtWS.onerror = () => { showToast('Realtime connection error', 'error'); };
  try { rtMicStream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true } }); }
  catch (e) { showToast('Mic error: ' + e.message, 'error'); stopRealtimeUI(''); return; }
  rtCtx = new AudioContext({ sampleRate: 16000 });
  const src = rtCtx.createMediaStreamSource(rtMicStream);
  rtScript = rtCtx.createScriptProcessor(2048, 1, 1);
  rtScript.onaudioprocess = (e) => {
    if (!rtWS || rtWS.readyState !== 1) return;
    const d = e.inputBuffer.getChannelData(0); const buf = new Int16Array(d.length);
    let sumSq = 0;
    for (let i = 0; i < d.length; i++) { let s = Math.max(-1, Math.min(1, d[i])); buf[i] = s < 0 ? s * 0x8000 : s * 0x7FFF; sumSq += s * s; }
    const inRms = Math.sqrt(sumSq / d.length);
    rtPulsePushIn(Math.min(1, inRms * 6));
    // client-side instant barge-in: while the AI is speaking or thinking, the
    // user's voice (echo-cancelled by the browser) stops playback immediately.
    if (rtBargeArmed && (rtState === 'speaking' || rtState === 'thinking')) {
      if (inRms >= RT_BARGE_RMS) {
        rtBargeCount++;
        if (rtBargeCount >= RT_BARGE_CONFIRM) rtBarge();
      } else {
        rtBargeCount = 0;
      }
    } else {
      rtBargeCount = 0;
    }
    rtWS.send(buf.buffer);
  };
  src.connect(rtScript); rtScript.connect(rtCtx.destination);
  rtPlayCtx = new AudioContext(); rtPlayTime = rtPlayCtx.currentTime;
  if (rtPlayCtx.state === 'suspended') { try { await rtPlayCtx.resume(); } catch (e) { showToast('Could not resume playback audio: ' + e.message, 'error'); } }
  if (rtCtx && rtCtx.state === 'suspended') { try { await rtCtx.resume(); } catch (e) { } }
  rtRunning = true;
  setRTState(rtState); // refresh end button + classes
  rtWaveStart();
}

function stopRealtimeUI(msg) {
  rtRunning = false;
  rtBargeCount = 0; rtBargeArmed = true; clearTimeout(rtBargeArmedTimer); rtBargeArmedTimer = null;
  if (rtScript) { try { rtScript.disconnect(); } catch (e) { } rtScript = null; }
  if (rtMicStream) { rtMicStream.getTracks().forEach(t => t.stop()); rtMicStream = null; }
  if (rtCtx) { try { rtCtx.close(); } catch (e) { } rtCtx = null; }
  rtClearPlayback();
  if (rtPlayCtx) { try { rtPlayCtx.close(); } catch (e) { } rtPlayCtx = null; }
  if (rtWS) { try { rtWS.close(); } catch (e) { } rtWS = null; }
  rtWaveStop();
  setRTState('idle');
  document.getElementById('rtState').textContent = msg || '';
  document.getElementById('rtTranscript').innerHTML = '';
  rtAsstEl = null; rtAsstText = ''; rtUserEl = null;
}

async function stopRealtime() {
  if (rtWS && rtWS.readyState === 1) { try { rtWS.send(JSON.stringify({ type: 'stop' })); } catch (e) { } }
  stopRealtimeUI('');
}

window.toggleRealtime = function () { if (rtRunning) stopRealtime(); else startRealtime(); };

// ============================================================
// Pulse envelope viz — dual rings
// ============================================================
let rtPulseIn = null, rtPulseOut = null;
let rtEnergyIn = 0, rtTargetIn = 0, rtEnergyOut = 0, rtTargetOut = 0;
let rtPulseRaf = null;
const RT_ATTACK = 0.40;
const RT_RELEASE = 0.07;

function rtPulseTick() {
  rtPulseRaf = requestAnimationFrame(rtPulseTick);
  const a = RT_ATTACK, r = RT_RELEASE;
  rtEnergyIn  += (rtTargetIn  - rtEnergyIn)  * (rtTargetIn  > rtEnergyIn  ? a : r);
  rtEnergyOut += (rtTargetOut - rtEnergyOut) * (rtTargetOut > rtEnergyOut ? a : r);
  if (rtPulseIn)  rtPulseIn.style.setProperty('--pin',  rtEnergyIn.toFixed(3));
  if (rtPulseOut) rtPulseOut.style.setProperty('--pout', rtEnergyOut.toFixed(3));
  if (rtEnergyIn < 0.002 && rtTargetIn < 0.002 && rtEnergyOut < 0.002 && rtTargetOut < 0.002) {
    cancelAnimationFrame(rtPulseRaf); rtPulseRaf = null;
  }
}
function rtWaveStart() {
  rtPulseIn = document.getElementById('rtPulseIn');
  rtPulseOut = document.getElementById('rtPulseOut');
  rtEnergyIn = 0; rtTargetIn = 0; rtEnergyOut = 0; rtTargetOut = 0;
  if (!rtPulseRaf) rtPulseRaf = requestAnimationFrame(rtPulseTick);
}
function rtWaveStop() {
  if (rtPulseRaf) { cancelAnimationFrame(rtPulseRaf); rtPulseRaf = null; }
  rtEnergyIn = 0; rtTargetIn = 0; rtEnergyOut = 0; rtTargetOut = 0;
  rtPulseIn = null; rtPulseOut = null;
}
function rtPulsePushIn(v)  { rtTargetIn  = v; if (!rtPulseRaf) rtPulseRaf = requestAnimationFrame(rtPulseTick); }
function rtPulsePushOut(v) { rtTargetOut = v; if (!rtPulseRaf) rtPulseRaf = requestAnimationFrame(rtPulseTick); }

// ---------- Init ----------
setRTState('idle');