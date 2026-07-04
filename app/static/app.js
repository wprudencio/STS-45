// Supertonic Voice Chat — UI logic
const BOOT = JSON.parse(document.getElementById('boot').textContent);
const DEFAULT_API_URL = BOOT.default_api_url;
const DEFAULT_STT_API_URL = BOOT.default_stt_api_url;
const RT_WS_PORT = BOOT.ws_port;

let isBusy = false;
let currentPlayingAudio = null;
let currentAssistantEl = null;
let currentReasoningEl = null;
let currentReasoningText = '';
let lastStats = null;
let abortCtrl = null;

// ---------- Icons ----------
const ICONS = {
  sun: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/></svg>',
  moon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>',
  copy: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>',
  play: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3" fill="currentColor"/></svg>',
  stop: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor"/></svg>',
  regen: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>',
  branch: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="6" y1="3" x2="6" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/></svg>',
  pin: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="17" x2="12" y2="22"/><path d="M9 8h6l1 4-3 2-3-2z"/><path d="M8 8a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v0L9 8z"/><path d="M5 4l2 4"/></svg>',
  pinon: '<svg viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="17" x2="12" y2="22"/><path d="M9 8h6l1 4-3 2-3-2z"/></svg>',
  edit: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>',
  bot: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/><path d="M12 7v4"/><line x1="8" y1="16" x2="8" y2="16"/><line x1="16" y1="16" x2="16" y2="16"/></svg>',
  user: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
  think: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.66 18h6a2 2 0 0 0 1.92-2.56l-2-8A2 2 0 0 0 13.66 5.64h-2.7a2 2 0 0 0-1.94 1.8l-.46 3.7"/><path d="M7 13h1.5a2 2 0 0 1 1.6.8l.9 1.2"/></svg>',
  bolt: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
  clock: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
  hash: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="9" x2="20" y2="9"/><line x1="4" y1="15" x2="20" y2="15"/><line x1="10" y1="3" x2="8" y2="21"/><line x1="16" y1="3" x2="14" y2="21"/></svg>',
};

// ---------- Logging ----------
function log(text, type) {
  const ls = document.getElementById('logStream');
  const ts = new Date().toISOString().slice(11, 19);
  const el = document.createElement('div');
  el.className = 'log-entry' + (type ? ' ' + type : '');
  const span = document.createElement('span'); span.className = 'ts'; span.textContent = ts;
  el.appendChild(span); el.appendChild(document.createTextNode(text));
  ls.appendChild(el); ls.scrollTop = ls.scrollHeight;
  if (ls.children.length > 120) ls.firstChild.remove();
}
window.log = log;

// ---------- Toast banner ----------
let toastTimer = null;
let sttFailCount = 0;
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

// ---------- Settings ----------
function loadSettings() {
  const g = (k, d) => localStorage.getItem(k) || d;
  document.getElementById('apiUrl').value = g('supertonic_api_url', DEFAULT_API_URL);
  document.getElementById('apiKey').value = g('supertonic_api_key', '');
  document.getElementById('sttApiUrl').value = g('supertonic_stt_api_url', DEFAULT_STT_API_URL);
  document.getElementById('modelId').value = g('supertonic_model', 'default');
  document.getElementById('voice').value = g('supertonic_voice', 'M1');
  document.getElementById('lang').value = g('supertonic_lang', 'en');
  const steps = parseInt(g('supertonic_steps', '5'), 10);
  document.getElementById('steps').value = steps;
  document.getElementById('stepsVal').textContent = steps;
  updateSlider(document.getElementById('steps'));
  const speed = parseFloat(g('supertonic_speed', '1.15'));
  document.getElementById('speed').value = speed;
  document.getElementById('speedVal').textContent = speed.toFixed(2);
  updateSlider(document.getElementById('speed'));
  document.getElementById('maxTokens').value = parseInt(g('supertonic_max_tokens', '2048'), 10);
}
window.saveConnection = function () {
  const s = (k, v) => localStorage.setItem(k, v);
  s('supertonic_api_url', document.getElementById('apiUrl').value.trim());
  s('supertonic_api_key', document.getElementById('apiKey').value.trim());
  s('supertonic_stt_api_url', document.getElementById('sttApiUrl').value.trim());
  s('supertonic_model', document.getElementById('modelId').value.trim() || 'default');
  s('supertonic_voice', document.getElementById('voice').value);
  s('supertonic_lang', document.getElementById('lang').value);
  s('supertonic_steps', document.getElementById('steps').value);
  s('supertonic_speed', document.getElementById('speed').value);
  s('supertonic_max_tokens', parseInt(document.getElementById('maxTokens').value) || 2048);
  closeModal(); log('Settings saved.', 'ok');
};
window.openModal = function () { document.getElementById('settingsModal').classList.remove('hidden'); };
window.closeModal = function () { document.getElementById('settingsModal').classList.add('hidden'); };
window.closeModalById = function (id) { document.getElementById(id).classList.add('hidden'); };
document.querySelectorAll('.modal-overlay').forEach(m => {
  m.addEventListener('click', e => { if (e.target === m) m.classList.add('hidden'); });
});
window.onVoiceChange = function () { log('Voice set to ' + document.getElementById('voice').value); };
window.onLangChange = function () { log('Language set to ' + document.getElementById('lang').value.toUpperCase()); };
loadSettings();

// ---------- Slider fill ----------
function updateSlider(el) {
  const min = +el.min, max = +el.max, val = +el.value;
  el.style.setProperty('--fill', ((val - min) / (max - min) * 100) + '%');
}
window.updateSlider = updateSlider;
document.querySelectorAll('input[type=range]').forEach(updateSlider);

// ---------- Theme ----------
function setIcon(id, key) { document.getElementById(id).innerHTML = ICONS[key]; }
window.toggleTheme = function () {
  const isLight = document.documentElement.classList.toggle('light');
  setIcon('themeIcon', isLight ? 'moon' : 'sun');
  localStorage.setItem('supertonic_theme', isLight ? 'light' : 'dark');
};
if (localStorage.getItem('supertonic_theme') === 'light') {
  document.documentElement.classList.add('light'); setIcon('themeIcon', 'moon');
} else { setIcon('themeIcon', 'sun'); }

if (localStorage.getItem('supertonic_sidebar_collapsed') === '1') {
  document.querySelector('.shell').classList.add('collapsed');
}

window.toggleLog = function () { document.getElementById('footerLog').classList.toggle('open'); };

// ---------- Status ----------
function setStatus(text, state) {
  document.getElementById('statusText').textContent = text;
  const dot = document.getElementById('statusDot');
  dot.className = 'status-dot' + (state === 'active' ? ' active' : state === 'rec' ? ' rec' : '');
}

// ---------- Markdown ----------
if (typeof marked !== 'undefined') { marked.setOptions({ breaks: true, gfm: true }); }
function renderMarkdown(text) {
  if (typeof marked !== 'undefined') { try { return marked.parse(text); } catch (e) { } }
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\n/g, '<br>');
}
function addCodeCopyButtons(container) {
  container.querySelectorAll('pre').forEach(pre => {
    if (pre.querySelector('.code-copy-btn')) return;
    const btn = document.createElement('span');
    btn.className = 'code-copy-btn'; btn.innerHTML = ICONS.copy;
    btn.addEventListener('click', () => {
      const code = pre.querySelector('code');
      navigator.clipboard.writeText(code ? code.textContent : pre.textContent).then(() => {
        btn.classList.add('copied'); setTimeout(() => btn.classList.remove('copied'), 1200);
      });
    });
    pre.style.position = 'relative'; pre.appendChild(btn);
  });
}

// ---------- IndexedDB ----------
let db = null;
let currentConvId = null;
let currentConvMessages = [];
let currentTitle = 'New chat';

function uid() { return Date.now().toString(36) + Math.random().toString(36).slice(2, 8); }
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
function dbAdd(c) { return new Promise((res, rej) => { const r = db.transaction('conversations', 'readwrite').objectStore('conversations').add(c); r.onsuccess = () => res(r.result); r.onerror = e => rej(e.target.error); }); }
function dbUpdate(c) { return new Promise((res, rej) => { const r = db.transaction('conversations', 'readwrite').objectStore('conversations').put(c); r.onsuccess = () => res(); r.onerror = e => rej(e.target.error); }); }
function dbGet(id) { return new Promise((res, rej) => { const r = db.transaction('conversations', 'readonly').objectStore('conversations').get(id); r.onsuccess = () => res(r.result); r.onerror = e => rej(e.target.error); }); }
function dbGetAll() { return new Promise((res, rej) => { const idx = db.transaction('conversations', 'readonly').objectStore('conversations').index('updatedAt'); const rq = idx.openCursor(null, 'prev'); const out = []; rq.onsuccess = e => { const c = e.target.result; if (c) { out.push(c.value); c.continue(); } else res(out); }; rq.onerror = e => rej(e.target.error); }); }
function dbDelete(id) { return new Promise((res, rej) => { const r = db.transaction('conversations', 'readwrite').objectStore('conversations').delete(id); r.onsuccess = () => res(); r.onerror = e => rej(e.target.error); }); }

function llmHistory() {
  return currentConvMessages.filter(m => m.role === 'user' || m.role === 'assistant').map(m => ({ role: m.role, content: m.content }));
}
async function saveCurrentConv() {
  if (!currentConvId || currentConvMessages.length === 0) return;
  const conv = await dbGet(currentConvId); if (!conv) return;
  conv.messages = currentConvMessages;
  conv.updatedAt = new Date().toISOString();
  if (conv.title === 'New chat' && currentConvMessages.length > 0) {
    const firstUser = currentConvMessages.find(m => m.role === 'user');
    if (firstUser) conv.title = firstUser.content.substring(0, 50);
  }
  conv.sysPrompt = document.getElementById('sysPrompt').value;
  await dbUpdate(conv);
}

// ---------- Sidebar ----------
function formatTimeAgo(iso) {
  const min = Math.floor((Date.now() - new Date(iso).getTime()) / 60000);
  if (min < 1) return 'Just now';
  if (min < 60) return min + 'm ago';
  const h = Math.floor(min / 60);
  if (h < 24) return h + 'h ago';
  return new Date(iso).toLocaleDateString();
}
function groupLabel(iso) {
  const d = new Date(iso); const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterday = new Date(today); yesterday.setDate(today.getDate() - 1);
  const weekAgo = new Date(today); weekAgo.setDate(today.getDate() - 7);
  if (d >= today) return 'Today';
  if (d >= yesterday) return 'Yesterday';
  if (d >= weekAgo) return 'This week';
  return 'Earlier';
}
function escapeHtml(s) { return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'); }

function convItemHTML(c) {
  return '<div class="conv-item' + (c.id === currentConvId ? ' active' : '') + '" data-id="' + c.id + '">' +
    '<div style="display:flex; justify-content:space-between; align-items:center; gap:8px">' +
      '<div style="flex:1; min-width:0">' +
        '<div class="conv-title">' + escapeHtml(c.title || 'New chat') + '</div>' +
        '<div class="conv-meta">' + formatTimeAgo(c.updatedAt) + '</div>' +
      '</div>' +
      '<span class="conv-del" data-id="' + c.id + '" title="Delete"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/></svg></span>' +
    '</div></div>';
}

let allConvsCache = [];
async function renderConvList() {
  const container = document.getElementById('conversationList'); if (!container) return;
  allConvsCache = await dbGetAll();
  const q = (document.getElementById('convSearch').value || '').toLowerCase().trim();
  const filtered = q ? allConvsCache.filter(c => (c.title || '').toLowerCase().includes(q)) : allConvsCache;
  if (filtered.length === 0) {
    container.innerHTML = '<div class="conv-empty">' + (q ? 'No matches found.' : 'No conversations yet.') + '</div>';
    return;
  }
  const groups = {};
  filtered.forEach(c => { const g = groupLabel(c.updatedAt); (groups[g] = groups[g] || []).push(c); });
  const order = ['Today', 'Yesterday', 'This week', 'Earlier'];
  let html = '';
  for (const g of order) { if (groups[g] && groups[g].length) { html += '<div class="conv-group">' + g + '</div>'; html += groups[g].map(convItemHTML).join(''); } }
  container.innerHTML = html;
  container.querySelectorAll('.conv-item').forEach(el => {
    const id = parseInt(el.dataset.id);
    el.addEventListener('click', async e => { if (e.target.closest('.conv-del')) return; const conv = await dbGet(id); if (conv) loadConversation(conv); });
    el.addEventListener('dblclick', () => startRename(id, el));
  });
  container.querySelectorAll('.conv-del').forEach(el => {
    el.addEventListener('click', async e => {
      e.stopPropagation(); const id = parseInt(el.dataset.id); await dbDelete(id);
      if (currentConvId === id) {
        clearChatUI(); currentConvMessages = [];
        const convs = await dbGetAll();
        if (convs.length > 0) await loadConversation(convs[0]); else await newConversation();
      } else { await renderConvList(); }
    });
  });
}
window.filterConvs = function () { renderConvList(); };

async function startRename(id, el) {
  const conv = await dbGet(id); if (!conv) return;
  const titleEl = el.querySelector('.conv-title'); const original = conv.title || 'New chat';
  const input = document.createElement('input'); input.type = 'text'; input.value = original;
  input.className = 'conv-search'; input.style.padding = '4px 8px'; input.style.margin = '0';
  titleEl.replaceWith(input); input.focus(); input.select();
  let done = false;
  const finish = async (save) => {
    if (done) return; done = true;
    if (save && input.value.trim() && input.value.trim() !== original) {
      conv.title = input.value.trim().substring(0, 80); conv.updatedAt = new Date().toISOString();
      if (conv.id === currentConvId) { currentTitle = conv.title; document.getElementById('chatTitle').textContent = conv.title; }
      await dbUpdate(conv);
    }
    await renderConvList();
  };
  input.addEventListener('blur', () => finish(true));
  input.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); input.blur(); } if (e.key === 'Escape') { e.preventDefault(); finish(false); } });
}

window.newConversation = async function () {
  if (isBusy) return;
  await saveCurrentConv();
  clearChatUI();
  const now = new Date().toISOString();
  const id = await dbAdd({ title: 'New chat', sysPrompt: '', messages: [], createdAt: now, updatedAt: now });
  currentConvId = id; currentConvMessages = []; currentTitle = 'New chat';
  localStorage.setItem('activeConvId', id);
  document.getElementById('chatTitle').textContent = 'New chat';
  await renderConvList(); closeDrawer();
  document.getElementById('userInput').focus();
};

async function loadConversation(conv) {
  if (isBusy) return;
  await saveCurrentConv();
  clearChatUI();
  currentConvId = conv.id; currentConvMessages = conv.messages || []; currentTitle = conv.title || 'New chat';
  localStorage.setItem('activeConvId', conv.id);
  document.getElementById('sysPrompt').value = conv.sysPrompt || '';
  updateSysPromptUI();
  document.getElementById('chatTitle').textContent = currentTitle;
  if (currentConvMessages.length > 0) {
    document.getElementById('initOverlay').classList.add('hidden');
    document.getElementById('messages').classList.remove('hidden');
    renderMessages();
  }
  try {
    await fetch('/api/chat/load', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: llmHistory(), sys_prompt: conv.sysPrompt || '' })
    });
  } catch (e) { }
  await renderConvList(); closeDrawer();
}

function clearChatUI() {
  document.getElementById('messages').innerHTML = '';
  document.getElementById('messages').classList.add('hidden');
  document.getElementById('initOverlay').classList.remove('hidden');
  currentAssistantEl = null; currentReasoningEl = null; currentReasoningText = '';
  const sp = document.getElementById('sysPrompt'); if (sp) { sp.value = ''; updateSysPromptUI(); }
  document.getElementById('headerStats').innerHTML = '';
}

function updateSysPromptUI() {
  const sp = document.getElementById('sysPrompt'); const wrap = document.getElementById('convPromptWrap');
  if (!sp || !wrap) return;
  const has = sp.value.trim().length > 0;
  wrap.classList.toggle('has-value', has);
  if (has) wrap.classList.add('open');
}
window.toggleConvPrompt = function () { document.getElementById('convPromptWrap').classList.toggle('open'); };
let sysPromptSaveTimer = null;
window.onSysPromptChange = function () {
  updateSysPromptUI();
  clearTimeout(sysPromptSaveTimer);
  sysPromptSaveTimer = setTimeout(() => { saveCurrentConv(); }, 400);
};

// ---------- Title edit ----------
const titleEl = document.getElementById('chatTitle');
titleEl.addEventListener('dblclick', startTitleEdit);
function startTitleEdit() {
  const el = titleEl; el.classList.add('editing');
  const original = el.textContent;
  el.contentEditable = 'true'; el.focus();
  const range = document.createRange(); range.selectNodeContents(el);
  const sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(range);
  let done = false;
  const finish = async (save) => {
    if (done) return; done = true;
    el.contentEditable = 'false'; el.classList.remove('editing');
    const v = el.textContent.trim();
    if (save && v && v !== original && currentConvId) {
      currentTitle = v.substring(0, 80);
      el.textContent = currentTitle;
      const conv = await dbGet(currentConvId); if (conv) { conv.title = currentTitle; conv.updatedAt = new Date().toISOString(); await dbUpdate(conv); await renderConvList(); }
    } else { el.textContent = original; }
  };
  el.addEventListener('blur', () => finish(true), { once: true });
  el.addEventListener('keydown', function onKey(e) {
    if (e.key === 'Enter') { e.preventDefault(); el.blur(); }
    if (e.key === 'Escape') { e.preventDefault(); finish(false); }
  });
}
window.startTitleEdit = startTitleEdit;

// ---------- Message rendering ----------
function renderMessages() {
  const container = document.getElementById('messages');
  container.innerHTML = '';
  currentConvMessages.forEach((m, idx) => appendMessage(m.role, m.content, idx, m));
  if (currentConvMessages.length > 0) {
    document.getElementById('initOverlay').classList.add('hidden');
    document.getElementById('messages').classList.remove('hidden');
  }
  container.scrollTop = container.scrollHeight;
}

function actBtn(key, onclick, on) {
  const b = document.createElement('button');
  b.className = 'act-btn' + (on ? ' on' : ''); b.innerHTML = ICONS[key];
  b.addEventListener('click', e => { e.stopPropagation(); onclick(); });
  return b;
}

function appendMessage(role, content, idx, meta) {
  meta = meta || {};
  const container = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'message ' + role;
  if (meta.pinned) div.classList.add('pinned');
  if (idx != null) div.dataset.index = idx;
  const now = meta.time ? new Date(meta.time).toTimeString().slice(0, 8) : new Date().toTimeString().slice(0, 8);

  let avatarIcon = ICONS.bot, name = 'Assistant';
  if (role === 'user') { avatarIcon = ICONS.user; name = 'You'; }
  else if (role === 'reasoning') { avatarIcon = ICONS.think; name = 'Reasoning'; }

  const avatar = document.createElement('div'); avatar.className = 'msg-avatar'; avatar.innerHTML = avatarIcon;
  const main = document.createElement('div'); main.className = 'msg-main';
  const head = document.createElement('div'); head.className = 'msg-head';
  const nameEl = document.createElement('span'); nameEl.className = 'msg-name'; nameEl.textContent = name;
  const timeEl = document.createElement('span'); timeEl.className = 'msg-time'; timeEl.textContent = now;
  head.appendChild(nameEl); head.appendChild(timeEl);
  if (meta.pinned) { const pf = document.createElement('span'); pf.className = 'msg-pin-flag'; pf.innerHTML = ICONS.pinon; head.appendChild(pf); }

  if (role === 'reasoning') {
    head.style.cursor = 'pointer';
    head.addEventListener('click', () => div.classList.toggle('collapsed'));
  }

  const contentDiv = document.createElement('div'); contentDiv.className = 'msg-content';
  if (role === 'assistant' && content) { contentDiv.innerHTML = renderMarkdown(content); addCodeCopyButtons(contentDiv); }
  else { contentDiv.textContent = content || ''; }

  main.appendChild(head); main.appendChild(contentDiv);

  if (role === 'assistant' && meta.stats) main.appendChild(buildStatsLine(meta.stats));

  // actions (skip while streaming)
  if (!meta.streaming) {
    const actions = document.createElement('div'); actions.className = 'msg-actions';
    if (role === 'user') {
      actions.appendChild(actBtn('edit', () => startEdit(idx)));
      actions.appendChild(actBtn('copy', () => copyText(content)));
    } else if (role === 'assistant') {
      actions.appendChild(actBtn('copy', () => copyText(content)));
      actions.appendChild(actBtn('play', () => playMessage(content, actions.querySelector('[data-act=play]'))));
      actions.appendChild(actBtn('regen', regenerate));
      actions.appendChild(actBtn('branch', () => branchFrom(idx)));
      const pinBtn = actBtn(meta.pinned ? 'pinon' : 'pin', () => togglePin(idx, pinBtn, div), meta.pinned);
      pinBtn.dataset.act = 'pin'; actions.appendChild(pinBtn);
      const playBtn = actions.children[1]; playBtn.dataset.act = 'play';
    }
    if (actions.children.length) main.appendChild(actions);
  }

  div.appendChild(avatar); div.appendChild(main);
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return div;
}

function buildStatsLine(s) {
  const line = document.createElement('div'); line.className = 'msg-stats';
  line.appendChild(makeChip(ICONS.hash, s.tokens + ' tok'));
  line.appendChild(makeChip(ICONS.bolt, s.tps + ' tok/s'));
  line.appendChild(makeChip(ICONS.clock, (s.total_ms / 1000).toFixed(1) + 's'));
  line.appendChild(makeChip('', 'TTFT ' + (s.ttft_ms / 1000).toFixed(2) + 's'));
  return line;
}
function makeChip(icon, text) {
  const c = document.createElement('span'); c.className = 'stat-chip';
  if (icon) c.innerHTML = icon;
  const b = document.createElement('b'); b.textContent = text; c.appendChild(b);
  return c;
}

function copyText(text) {
  navigator.clipboard.writeText(text).then(() => log('Copied', 'ok')).catch(() => log('Copy failed', 'warn'));
}

async function playMessage(text, btn) {
  if (!text) return;
  if (btn && btn.classList.contains('on')) {
    if (currentPlayingAudio) { currentPlayingAudio.pause(); currentPlayingAudio = null; }
    btn.classList.remove('on'); btn.innerHTML = ICONS.play; return;
  }
  if (currentPlayingAudio) { currentPlayingAudio.pause(); currentPlayingAudio = null; }
  document.querySelectorAll('.act-btn.on[data-act=play]').forEach(b => { b.classList.remove('on'); b.innerHTML = ICONS.play; });
  try {
    const resp = await fetch('/api/tts', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, lang: document.getElementById('lang').value, voice: document.getElementById('voice').value, steps: parseInt(document.getElementById('steps').value), speed: parseFloat(document.getElementById('speed').value) })
    });
    const data = await resp.json();
    if (data.error) { log('TTS error: ' + data.error, 'warn'); return; }
    const audio = new Audio('data:audio/wav;base64,' + data.audio); currentPlayingAudio = audio;
    if (btn) { btn.classList.add('on'); btn.innerHTML = ICONS.stop; audio.onended = () => { btn.classList.remove('on'); btn.innerHTML = ICONS.play; currentPlayingAudio = null; }; audio.onerror = () => { btn.classList.remove('on'); btn.innerHTML = ICONS.play; currentPlayingAudio = null; }; }
    audio.play();
  } catch (e) { log('TTS error: ' + e.message, 'warn'); currentPlayingAudio = null; }
}

function togglePin(idx, btn, div) {
  if (idx == null || idx < 0 || idx >= currentConvMessages.length) return;
  const m = currentConvMessages[idx]; m.pinned = !m.pinned;
  div.classList.toggle('pinned', m.pinned);
  if (btn) { btn.innerHTML = m.pinned ? ICONS.pinon : ICONS.pin; btn.classList.toggle('on', m.pinned); }
  saveCurrentConv();
}

async function branchFrom(idx) {
  if (idx == null || isBusy) return;
  const branchMsgs = currentConvMessages.slice(0, idx + 1)
    .filter(m => m.role === 'user' || m.role === 'assistant')
    .map(m => ({ role: m.role, content: m.content, id: uid(), pinned: false }));
  await saveCurrentConv();
  const now = new Date().toISOString();
  const title = '↪ ' + ((branchMsgs.find(m => m.role === 'user') || {}).content || 'Branch').substring(0, 50);
  const id = await dbAdd({ title, sysPrompt: document.getElementById('sysPrompt').value, messages: branchMsgs, createdAt: now, updatedAt: now });
  const conv = await dbGet(id);
  if (conv) await loadConversation(conv);
  log('Branched conversation from message #' + (idx + 1), 'hl');
}

// ---------- Inline edit & resend ----------
function startEdit(idx) {
  if (isBusy) return;
  if (idx == null || idx < 0 || idx >= currentConvMessages.length) return;
  const m = currentConvMessages[idx]; if (m.role !== 'user') return;
  const msgEl = document.querySelector('.message[data-index="' + idx + '"]'); if (!msgEl) return;
  const contentEl = msgEl.querySelector('.msg-content');
  const actionsEl = msgEl.querySelector('.msg-actions'); if (actionsEl) actionsEl.style.display = 'none';
  const box = document.createElement('div'); box.className = 'edit-box';
  const ta = document.createElement('textarea'); ta.value = m.content;
  const row = document.createElement('div'); row.className = 'edit-row';
  const cancel = document.createElement('button'); cancel.className = 'btn btn-ghost'; cancel.textContent = 'Cancel';
  const save = document.createElement('button'); save.className = 'btn btn-primary'; save.textContent = 'Send';
  row.appendChild(cancel); row.appendChild(save);
  box.appendChild(ta); box.appendChild(row);
  contentEl.replaceWith(box);
  ta.focus(); ta.style.height = 'auto'; ta.style.height = Math.min(200, ta.scrollHeight) + 'px';
  ta.addEventListener('input', () => { ta.style.height = 'auto'; ta.style.height = Math.min(200, ta.scrollHeight) + 'px'; });
  let done = false;
  const finish = async (apply) => {
    if (done) return; done = true;
    if (!apply) { renderMessages(); return; }
    const newText = ta.value.trim();
    if (!newText) { renderMessages(); return; }
    currentConvMessages = currentConvMessages.slice(0, idx);
    currentConvMessages.push({ role: 'user', content: newText, id: uid(), pinned: false });
    renderMessages();
    await streamTurn({ message: '', append: false, history: llmHistory() });
  };
  save.addEventListener('click', () => finish(true));
  cancel.addEventListener('click', () => finish(false));
  ta.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); finish(true); } if (e.key === 'Escape') { e.preventDefault(); finish(false); } });
}

// ---------- Streaming turn ----------
async function streamTurn(body) {
  isBusy = true;
  setStatus('Generating', 'active');
  document.getElementById('sendBtn').hidden = true;
  document.getElementById('stopBtn').hidden = false;
  currentAssistantEl = null; currentReasoningEl = null; currentReasoningText = '';
  lastStats = null;

  abortCtrl = new AbortController();
  const fullBody = Object.assign({
    lang: document.getElementById('lang').value,
    voice: document.getElementById('voice').value,
    steps: parseInt(document.getElementById('steps').value),
    speed: parseFloat(document.getElementById('speed').value),
    max_tokens: parseInt(document.getElementById('maxTokens').value) || 2048,
    api_url: document.getElementById('apiUrl').value.trim(),
    api_key: document.getElementById('apiKey').value.trim(),
    model: document.getElementById('modelId').value.trim() || 'default',
    sys_prompt: document.getElementById('sysPrompt').value,
  }, body);

  let resp;
  try {
    resp = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(fullBody), signal: abortCtrl.signal });
  } catch (err) {
    if (err.name !== 'AbortError') { setStatus('Error', ''); log('Connection error: ' + err.message, 'warn'); }
    finishTurn(); return;
  }
  if (!resp.ok) { setStatus('Error ' + resp.status, ''); log('Server error ' + resp.status, 'warn'); finishTurn(); return; }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = ''; let asstText = ''; let firstText = true;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        let data; try { data = JSON.parse(line.slice(6)); } catch (e) { continue; }
        if (data.type === 'text') {
          if (firstText) { firstText = false; log('LLM streaming...'); }
          currentReasoningEl = null;
          asstText += data.text;
          addOrUpdateAssistant(data.text);
          attachCursor();
        } else if (data.type === 'reasoning') {
          addReasoning(data.text);
          currentReasoningText += data.text;
        } else if (data.type === 'error') {
          log('Error: ' + data.text, 'warn'); setStatus('Error', '');
        } else if (data.type === 'stats') {
          lastStats = data; updateHeaderStats(data);
        } else if (data.type === 'done') {
          currentReasoningEl = null;
        }
      }
    }
  } catch (err) {
    if (err.name !== 'AbortError') log('Stream error: ' + err.message, 'warn');
  }

  // finalize
  if (currentReasoningText) currentConvMessages.push({ role: 'reasoning', content: currentReasoningText, id: uid() });
  if (asstText || lastStats) currentConvMessages.push({ role: 'assistant', content: asstText, stats: lastStats || undefined, id: uid(), pinned: false });
  renderMessages();
  saveCurrentConv().then(() => renderConvList());
  setStatus('Ready', '');
  log('Response complete', 'ok');
  finishTurn();
}

function finishTurn() {
  isBusy = false; abortCtrl = null;
  document.getElementById('sendBtn').hidden = false;
  document.getElementById('stopBtn').hidden = true;
  onInputChange();
}

function addOrUpdateAssistant(text) {
  if (!currentAssistantEl) {
    document.getElementById('initOverlay').classList.add('hidden');
    document.getElementById('messages').classList.remove('hidden');
    currentAssistantEl = appendMessage('assistant', '', null, { streaming: true });
  }
  currentAssistantEl.querySelector('.msg-content').textContent += text;
  document.getElementById('messages').scrollTop = document.getElementById('messages').scrollHeight;
}
function addReasoning(text) {
  if (!currentReasoningEl) currentReasoningEl = appendMessage('reasoning', '', null, { streaming: true });
  currentReasoningEl.querySelector('.msg-content').textContent += text;
  document.getElementById('messages').scrollTop = document.getElementById('messages').scrollHeight;
}
function attachCursor() {
  if (!currentAssistantEl) return;
  const content = currentAssistantEl.querySelector('.msg-content');
  const old = content.querySelector('.typing-cursor'); if (old) old.remove();
  const c = document.createElement('span'); c.className = 'typing-cursor'; content.appendChild(c);
}

window.stopGeneration = function () {
  if (abortCtrl) { try { abortCtrl.abort(); } catch (e) { } log('Generation stopped', 'hl'); }
};

async function regenerate() {
  if (isBusy) return;
  let lastIdx = -1;
  for (let i = currentConvMessages.length - 1; i >= 0; i--) { if (currentConvMessages[i].role === 'assistant') { lastIdx = i; break; } }
  if (lastIdx < 0) return;
  const hist = currentConvMessages.slice(0, lastIdx).filter(m => m.role === 'user' || m.role === 'assistant').map(m => ({ role: m.role, content: m.content }));
  currentConvMessages = currentConvMessages.slice(0, lastIdx);
  renderMessages();
  await streamTurn({ message: '', append: false, regenerate: true, history: hist });
}
window.regenerate = regenerate;

// ---------- Input ----------
window.onInputChange = function () { document.getElementById('sendBtn').disabled = !document.getElementById('userInput').value.trim(); };
window.autoResize = function (el) { el.style.height = 'auto'; el.style.height = Math.min(el.scrollHeight, 200) + 'px'; };
window.handleKey = function (e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (slashSel >= 0 && slashMenuOpen()) { applySlashSelection(); return; }
    sendText();
  } else if (e.key === 'ArrowDown' && slashMenuOpen()) { e.preventDefault(); moveSlash(1); }
  else if (e.key === 'ArrowUp' && slashMenuOpen()) { e.preventDefault(); moveSlash(-1); }
  else if (e.key === 'Escape' && slashMenuOpen()) { closeSlash(); }
};
window.sendText = function () {
  const input = document.getElementById('userInput');
  const text = input.value.trim();
  if (!text || isBusy) return;
  closeSlash();
  currentConvMessages.push({ role: 'user', content: text, id: uid(), pinned: false });
  renderMessages();
  input.value = ''; autoResize(input); onInputChange();
  streamTurn({ message: text, append: true });
};

function updateHeaderStats(s) {
  const c = document.getElementById('headerStats'); c.innerHTML = '';
  c.appendChild(makeChip(ICONS.bolt, s.tps + ' tok/s'));
  c.appendChild(makeChip(ICONS.clock, (s.total_ms / 1000).toFixed(1) + 's'));
}

// ---------- Prompt templates / slash ----------
const BUILTIN_TEMPLATES = [
  { trigger: 'summarize', text: 'Summarize the following in concise bullet points:\n\n' },
  { trigger: 'explain', text: 'Explain in simple terms: ' },
  { trigger: 'translate', text: 'Translate the following to English:\n\n' },
  { trigger: 'rewrite', text: 'Rewrite the following more clearly and concisely:\n\n' },
  { trigger: 'code', text: 'Write clean, commented code for: ' },
  { trigger: 'eli5', text: 'Explain like I am 5 years old: ' },
  { trigger: 'fix', text: 'Find and fix issues in the following:\n\n' },
];
function getCustomTemplates() { try { return JSON.parse(localStorage.getItem('supertonic_templates') || '[]'); } catch (e) { return []; } }
function setCustomTemplates(t) { localStorage.setItem('supertonic_templates', JSON.stringify(t)); }
function allTemplates() { return BUILTIN_TEMPLATES.concat(getCustomTemplates()); }

let slashSel = -1;
function slashMenuOpen() { return document.getElementById('slashMenu').classList.contains('open'); }
window.onSlashInput = function () {
  const input = document.getElementById('userInput');
  const v = input.value;
  if (!v.startsWith('/')) { closeSlash(); return; }
  const q = v.slice(1).toLowerCase();
  const list = allTemplates().filter(t => t.trigger.toLowerCase().includes(q));
  const menu = document.getElementById('slashMenu');
  if (list.length === 0) {
    menu.innerHTML = '<div class="slash-empty">No templates match. Type a name and press Enter to send as-is.</div>';
    menu.classList.add('open'); slashSel = -1; return;
  }
  menu.innerHTML = '';
  list.forEach((t, i) => {
    const item = document.createElement('div'); item.className = 'slash-item' + (i === 0 ? ' sel' : '');
    item.innerHTML = '<span class="sl-trig">/' + escapeHtml(t.trigger) + '</span><span class="sl-desc">' + escapeHtml(t.text.replace(/\n/g, ' ')) + '</span>';
    item.addEventListener('click', () => applyTemplate(t));
    item.addEventListener('mouseenter', () => { slashSel = i; renderSlashSel(); });
    menu.appendChild(item);
  });
  menu.classList.add('open'); slashSel = 0;
};
function renderSlashSel() { document.querySelectorAll('.slash-item').forEach((it, i) => it.classList.toggle('sel', i === slashSel)); }
function moveSlash(dir) {
  const items = document.querySelectorAll('.slash-item'); if (!items.length) return;
  slashSel = (slashSel + dir + items.length) % items.length;
  renderSlashSel(); items[slashSel].scrollIntoView({ block: 'nearest' });
}
function applySlashSelection() {
  const items = document.querySelectorAll('.slash-item');
  if (slashSel >= 0 && items[slashSel]) { items[slashSel].click(); }
  else { closeSlash(); }
}
function applyTemplate(t) {
  const input = document.getElementById('userInput'); input.value = t.text;
  closeSlash(); autoResize(input); onInputChange(); input.focus();
  const end = input.value.length; input.setSelectionRange(end, end);
}
function closeSlash() { document.getElementById('slashMenu').classList.remove('open'); slashSel = -1; }

window.openTemplates = function () { renderTemplatesList(); document.getElementById('templatesModal').classList.remove('hidden'); closeMore(); };
function renderTemplatesList() {
  const builtin = document.getElementById('builtinTplList'); builtin.innerHTML = '';
  BUILTIN_TEMPLATES.forEach(t => {
    const row = document.createElement('div'); row.className = 'tpl-row';
    row.innerHTML = '<span class="tpl-trig">/' + escapeHtml(t.trigger) + '</span><span class="tpl-text">' + escapeHtml(t.text) + '</span>';
    builtin.appendChild(row);
  });
  const custom = document.getElementById('customTplList'); custom.innerHTML = '';
  const list = getCustomTemplates();
  if (list.length === 0) { custom.innerHTML = '<div class="slash-empty">No custom templates yet.</div>'; return; }
  list.forEach((t, i) => {
    const row = document.createElement('div'); row.className = 'tpl-row';
    row.innerHTML = '<span class="tpl-trig">/' + escapeHtml(t.trigger) + '</span><span class="tpl-text">' + escapeHtml(t.text) + '</span>';
    const del = document.createElement('button'); del.className = 'tpl-del';
    del.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/></svg>';
    del.addEventListener('click', () => { const l = getCustomTemplates(); l.splice(i, 1); setCustomTemplates(l); renderTemplatesList(); });
    row.appendChild(del); custom.appendChild(row);
  });
}
window.addCustomTemplate = function () {
  const tr = document.getElementById('newTplTrigger').value.trim();
  const tx = document.getElementById('newTplText').value.trim();
  if (!tr || !tx) return;
  const list = getCustomTemplates(); list.push({ trigger: tr, text: tx }); setCustomTemplates(list);
  document.getElementById('newTplTrigger').value = ''; document.getElementById('newTplText').value = '';
  renderTemplatesList();
};

// ---------- Shortcuts ----------
const SHORTCUTS = [
  { keys: ['↵'], desc: 'Send message' },
  { keys: ['⇧', '↵'], desc: 'New line' },
  { keys: ['⇧', '⇧'], desc: 'Start microphone' },
  { keys: ['⇧'], desc: 'Stop mic (while recording)' },
  { keys: ['/'], desc: 'Prompt templates' },
  { keys: ['?'], desc: 'Show shortcuts' },
  { keys: ['Esc'], desc: 'Close dialog / menu' },
  { keys: ['⌘', 'K'], desc: 'New chat' },
  { keys: ['⌘', 'B'], desc: 'Toggle sidebar' },
  { keys: ['⌘', '⇧', 'O'], desc: 'Open shortcuts' },
];
window.openShortcuts = function () {
  const grid = document.getElementById('scGrid'); grid.innerHTML = '';
  SHORTCUTS.forEach(s => {
    const row = document.createElement('div'); row.className = 'sc-row';
    const desc = document.createElement('span'); desc.className = 'sc-desc'; desc.textContent = s.desc;
    const keys = document.createElement('span'); keys.className = 'sc-keys';
    s.keys.forEach(k => { const kd = document.createElement('kbd'); kd.textContent = k; keys.appendChild(kd); });
    row.appendChild(desc); row.appendChild(keys); grid.appendChild(row);
  });
  document.getElementById('shortcutsModal').classList.remove('hidden'); closeMore();
};

// ---------- More menu ----------
window.toggleMore = function (e) { e.stopPropagation(); document.getElementById('moreMenu').hidden = !document.getElementById('moreMenu').hidden; };
function closeMore() { document.getElementById('moreMenu').hidden = true; }
document.addEventListener('click', e => { if (!e.target.closest('.menu-wrap')) closeMore(); });

// ---------- Export / Import ----------
function download(name, content, mime) {
  const blob = new Blob([content], { type: mime });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
window.exportConv = function (fmt) {
  closeMore();
  const title = (currentTitle || 'conversation').replace(/[^a-z0-9-_ ]/gi, '').trim() || 'conversation';
  if (fmt === 'json') {
    const content = JSON.stringify({ title: currentTitle, sysPrompt: document.getElementById('sysPrompt').value, messages: currentConvMessages, exportedAt: new Date().toISOString() }, null, 2);
    download(title + '.json', content, 'application/json'); log('Exported JSON', 'ok');
  } else {
    let md = '# ' + currentTitle + '\n\n';
    if (document.getElementById('sysPrompt').value) md += '> _System prompt:_ ' + document.getElementById('sysPrompt').value + '\n\n';
    currentConvMessages.forEach(m => {
      if (m.role === 'reasoning') md += '> _Reasoning:_ ' + m.content + '\n\n';
      else md += '## ' + (m.role === 'user' ? 'You' : 'Assistant') + '\n\n' + m.content + '\n\n';
    });
    download(title + '.md', md, 'text/markdown'); log('Exported Markdown', 'ok');
  }
};
window.importConv = function () {
  closeMore();
  const fileInput = document.getElementById('importFile');
  fileInput.value = '';
  fileInput.onchange = async () => {
    const f = fileInput.files[0]; if (!f) return;
    try {
      const text = await f.text(); const data = JSON.parse(text);
      const msgs = (data.messages || []).map(m => ({ role: m.role, content: m.content, id: uid(), pinned: !!m.pinned }));
      const now = new Date().toISOString();
      const id = await dbAdd({ title: data.title || 'Imported', sysPrompt: data.sysPrompt || '', messages: msgs, createdAt: now, updatedAt: now });
      const conv = await dbGet(id); if (conv) await loadConversation(conv);
      log('Imported conversation', 'ok');
    } catch (e) { log('Import failed: ' + e.message, 'warn'); }
  };
  fileInput.click();
};

// ---------- Suggestions ----------
const SUGGESTIONS = [
  'Explain quantum computing simply',
  'Write a haiku about the sea',
  'Give me 3 dinner ideas with eggs',
  'Translate "good morning" to Japanese',
];
const suggEl = document.getElementById('suggestions');
SUGGESTIONS.forEach(s => {
  const b = document.createElement('button'); b.className = 'sugg'; b.textContent = s;
  b.addEventListener('click', () => { const input = document.getElementById('userInput'); input.value = s; autoResize(input); onInputChange(); input.focus(); });
  suggEl.appendChild(b);
});

// ---------- Drawer ----------
window.toggleDrawer = function () {
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
window.closeDrawer = function () {
  const isMobile = window.matchMedia('(max-width: 900px)').matches;
  if (isMobile) {
    document.querySelector('.panel-left').classList.remove('open');
    document.getElementById('drawerBackdrop').classList.remove('open');
  } else {
    document.querySelector('.shell').classList.remove('collapsed');
    localStorage.setItem('supertonic_sidebar_collapsed', '');
  }
};
document.addEventListener('click', e => { if (e.target.closest('.conv-item')) closeDrawer(); });

// ---------- Global keyboard shortcuts ----------
document.addEventListener('keydown', e => {
  const mod = e.metaKey || e.ctrlKey;
  if (mod && e.key.toLowerCase() === 'k') { e.preventDefault(); newConversation(); return; }
  if (mod && e.key.toLowerCase() === 'b') { e.preventDefault(); toggleDrawer(); return; }
  if (mod && e.shiftKey && e.key.toLowerCase() === 'o') { e.preventDefault(); openShortcuts(); return; }
  if (e.key === '?' && document.activeElement !== document.getElementById('userInput') && !isInInput()) { e.preventDefault(); openShortcuts(); }
  if (e.key === 'Escape') { closeMore(); }
});
function isInInput() { const t = document.activeElement; return t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable); }

// ============================================================
// STT via parakeet.cpp (push-to-talk) — ported from original
// ============================================================
let pttHeld = false, shiftTapCount = 0, shiftTapTimer = null, shiftIsDown = false;
let sttPrefix = '', mediaStream = null, audioContext = null, scriptProcessor = null;

const STT_SR = 16000;
const VAD_RMS = 0.012;
const SILENCE_MS = 650;
const MIN_UTT_SAMPLES = Math.floor(STT_SR * 0.30);
const MAX_UTT_SAMPLES = Math.floor(STT_SR * 12);
const MAX_PARTIAL_SAMPLES = Math.floor(STT_SR * 6);
const PARTIAL_DELTA_SAMPLES = Math.floor(STT_SR * 0.35);
const TICK_MS = 350;

let sttTick = null, uttChunks = [], uttCount = 0, lastVoiceAt = 0;
let lastSentPartialCount = 0, committedText = '', livePartial = '';
let partialGen = 0, sttActive = null, commitPending = false, stopDeferred = null;

function encodeWAV(samples, sampleRate) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const w = (s, o) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w('RIFF', 0); v.setUint32(4, 36 + samples.length * 2, true); w('WAVE', 8);
  w('fmt ', 12); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  w('data', 36); v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) { const s = Math.max(-1, Math.min(1, samples[i])); v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true); }
  return new Blob([buf], { type: 'audio/wav' });
}
function joinSpace(a, b) { if (!a) return b; if (!b) return a; return a + (a.endsWith(' ') ? '' : ' ') + b; }
function collectUtt() { const out = new Float32Array(uttCount); let off = 0; for (const c of uttChunks) { out.set(c, off); off += c.length; } return out; }
function renderSTT() { const input = document.getElementById('userInput'); input.value = joinSpace(committedText, livePartial); autoResize(input); onInputChange(); }
async function sttTranscribe(samples) {
  try {
    const wav = encodeWAV(samples, STT_SR);
    const fd = new FormData(); fd.append('file', wav, 'utt.wav');
    fd.append('lang', document.getElementById('lang').value);
    fd.append('stt_api', document.getElementById('sttApiUrl').value.trim());
    const resp = await fetch('/api/stt', { method: 'POST', body: fd });
    const data = await resp.json();
    if (data.error) {
      log('STT error: ' + data.error, 'warn');
      sttFailCount++;
      if (sttFailCount === 1) {
        showToast('Speech recognition failed: ' + data.error + ' — check STT server URL in Settings', 'error');
      }
      return '';
    }
    sttFailCount = 0;
    return (data.text || '').replace(/^\s+|\s+$/g, '');
  } catch (e) {
    log('STT error: ' + e.message, 'warn');
    sttFailCount++;
    if (sttFailCount === 1) {
      showToast('Speech recognition error: ' + e.message, 'error');
    }
    return '';
  }
}
async function flushCommit() {
  const samples = collectUtt(); uttChunks = []; uttCount = 0; lastSentPartialCount = 0; partialGen++; livePartial = ''; renderSTT();
  const text = await sttTranscribe(samples);
  if (text) committedText = joinSpace(committedText, text);
  renderSTT();
}
async function sendPartial() {
  const samples = collectUtt(); lastSentPartialCount = uttCount; const gen = partialGen;
  const text = await sttTranscribe(samples);
  if (gen !== partialGen) return;
  livePartial = text; renderSTT();
}
function enqueue(promise) { sttActive = promise.then(() => { sttActive = null; pump(); }); }
function pump() {
  if (sttActive) return;
  if (commitPending) { commitPending = false; enqueue(flushCommit()); return; }
  if (stopDeferred) { const d = stopDeferred; stopDeferred = null; d.resolve(); }
}
function sttTickFn() {
  if (!pttHeld) return;
  const needCommit = uttCount >= MIN_UTT_SAMPLES && (performance.now() - lastVoiceAt >= SILENCE_MS || uttCount >= MAX_UTT_SAMPLES);
  if (needCommit) { if (sttActive) commitPending = true; else enqueue(flushCommit()); return; }
  if (sttActive || commitPending) return;
  if (uttCount < MIN_UTT_SAMPLES) return;
  if (uttCount >= MAX_PARTIAL_SAMPLES) return;
  if (uttCount - lastSentPartialCount < PARTIAL_DELTA_SAMPLES) return;
  enqueue(sendPartial());
}
async function startPTT() {
  if (isBusy || pttHeld) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { log('Mic unavailable: page must be served over HTTPS or from localhost', 'warn'); return; }
  const current = document.getElementById('userInput').value;
  sttPrefix = current ? (current.endsWith(' ') ? current : current + ' ') : '';
  committedText = sttPrefix; livePartial = '';
  uttChunks = []; uttCount = 0; lastSentPartialCount = 0; partialGen = 0;
  sttActive = null; commitPending = false; stopDeferred = null;
  sttFailCount = 0;
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true } });
    audioContext = new AudioContext({ sampleRate: 16000 });
    const source = audioContext.createMediaStreamSource(mediaStream);
    scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
    scriptProcessor.onaudioprocess = (e) => {
      const d = new Float32Array(e.inputBuffer.getChannelData(0));
      let sum = 0; for (let i = 0; i < d.length; i++) sum += d[i] * d[i];
      if (Math.sqrt(sum / d.length) > VAD_RMS) lastVoiceAt = performance.now();
      uttChunks.push(d); uttCount += d.length;
    };
    source.connect(scriptProcessor); scriptProcessor.connect(audioContext.destination);
    pttHeld = true; lastVoiceAt = performance.now();
    document.getElementById('pttBtn').classList.add('recording');
    setStatus('Recording', 'rec');
    sttTick = setInterval(sttTickFn, TICK_MS);
    log('Recording (real-time STT)...', 'hl');
  } catch (e) { log('Mic error: ' + e.message, 'warn'); }
}
async function stopPTT() {
  if (!pttHeld) return;
  pttHeld = false;
  if (sttTick) { clearInterval(sttTick); sttTick = null; }
  if (scriptProcessor) { scriptProcessor.disconnect(); scriptProcessor = null; }
  if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); }
  const hadTail = uttCount >= MIN_UTT_SAMPLES;
  partialGen++; livePartial = ''; renderSTT();
  if (hadTail) { if (sttActive) commitPending = true; else enqueue(flushCommit()); }
  else { commitPending = false; }
  if (audioContext) { try { await audioContext.close(); } catch (e) { } audioContext = null; }
  if (mediaStream) { mediaStream = null; }
  document.getElementById('pttBtn').classList.remove('recording');
  setStatus('Transcribing', 'active');
  if (sttActive || commitPending) { await new Promise(res => { stopDeferred = { resolve: res }; pump(); }); }
  livePartial = ''; renderSTT();
  const input = document.getElementById('userInput');
  input.value = input.value.replace(/^\s+|\s+$/g, '');
  autoResize(input); onInputChange();
  sttPrefix = ''; committedText = '';
  if (input.value) log('STT: ' + input.value.substring(0, 50) + (input.value.length > 50 ? '...' : ''), 'ok');
  setStatus('Ready', '');
}
const pttBtnEl = document.getElementById('pttBtn');
pttBtnEl.addEventListener('click', () => { if (pttHeld) stopPTT(); else startPTT(); });
pttBtnEl.addEventListener('touchstart', e => { e.preventDefault(); }, { passive: false });
document.addEventListener('keydown', e => {
  if (e.key === 'Shift' && !shiftIsDown) {
    shiftIsDown = true; shiftTapCount++;
    if (pttHeld) { shiftTapCount = 0; if (shiftTapTimer) { clearTimeout(shiftTapTimer); shiftTapTimer = null; } stopPTT(); }
    else if (shiftTapCount === 1) { shiftTapTimer = setTimeout(() => { shiftTapCount = 0; }, 400); }
    else if (shiftTapCount >= 2) { clearTimeout(shiftTapTimer); shiftTapCount = 0; shiftTapTimer = null; e.preventDefault(); startPTT(); }
  }
});
document.addEventListener('keyup', e => { if (e.key === 'Shift') shiftIsDown = false; });

// ============================================================
// Realtime conversation mode — ported from original
// ============================================================
let rtWS = null, rtCtx = null, rtMicStream = null, rtScript = null, rtRunning = false;
let rtPlayCtx = null, rtTTSsr = 24000, rtPlaySources = [], rtPlayTime = 0;
let rtState = 'idle', rtAsstEl = null, rtAsstText = '';

function setRTState(s) {
  rtState = s;
  document.getElementById('rtOrb').className = 'rt-orb ' + s;
  const labels = { idle: 'Tap to start', connecting: 'Connecting…', listening: 'Listening', thinking: 'Thinking', speaking: 'Speaking' };
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
function rtSendStart() { if (rtWS && rtWS.readyState === 1) rtWS.send(JSON.stringify(Object.assign({ type: 'start' }, rtSettings()))); }
function rtEscapeHtml(s) { return s.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }
let rtUserEl = null;
function rtAppendUser(text) {
  const t = document.getElementById('rtTranscript');
  if (rtUserEl) {
    // Finalize the existing partial element
    rtUserEl.querySelector('.rt-text').textContent = text;
    rtUserEl.classList.remove('live');
    rtUserEl = null;
  } else {
    const el = document.createElement('div'); el.className = 'rt-line user';
    el.innerHTML = '<span class="rt-role">You</span>' + rtEscapeHtml(text);
    t.appendChild(el);
  }
  t.scrollTop = t.scrollHeight; rtAsstEl = null; rtAsstText = '';
}
function rtUpdateUserPartial(text) {
  const t = document.getElementById('rtTranscript');
  if (!rtUserEl) {
    rtUserEl = document.createElement('div'); rtUserEl.className = 'rt-line user live';
    rtUserEl.innerHTML = '<span class="rt-role">You</span><span class="rt-text"></span>';
    t.appendChild(rtUserEl);
  }
  rtUserEl.querySelector('.rt-text').textContent = text;
  t.scrollTop = t.scrollHeight;
}
function rtFinalizeUserPartial() {
  if (rtUserEl) { rtUserEl.classList.remove('live'); rtUserEl = null; }
}
function rtAppendAssistantDelta(tok) {
  const t = document.getElementById('rtTranscript');
  if (!rtAsstEl) {
    rtAsstEl = document.createElement('div'); rtAsstEl.className = 'rt-line assistant live';
    rtAsstEl.innerHTML = '<span class="rt-role">Assistant</span><span class="rt-text"></span>';
    t.appendChild(rtAsstEl); rtAsstText = '';
  }
  rtAsstText += tok; rtAsstEl.querySelector('.rt-text').textContent = rtAsstText;
  t.scrollTop = t.scrollHeight;
}
function rtFinalizeAssistant() { if (rtAsstEl) { rtAsstEl.classList.remove('live'); rtAsstEl = null; rtAsstText = ''; } }
function rtClearPlayback() {
  rtPlaySources.forEach(s => { try { s.stop(); } catch (e) { } });
  rtPlaySources = []; if (rtPlayCtx) rtPlayTime = rtPlayCtx.currentTime;
}
function onRTAudio(buf) {
  if (!rtPlayCtx) return;
  const i16 = new Int16Array(buf); const f32 = new Float32Array(i16.length);
  for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768;
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
    } else if (m.role === 'assistant') { if (m.final) rtFinalizeAssistant(); else if (m.text) rtAppendAssistantDelta(m.text); }
  } else if (m.type === 'clear') rtClearPlayback();
  else if (m.type === 'error') { log('RT: ' + m.text, 'warn'); showToast('Realtime: ' + m.text, 'error'); if (/loading|TTS/i.test(m.text)) { setRTState('connecting'); setTimeout(rtSendStart, 2500); } }
}
async function startRealtime() {
  if (rtRunning) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { log('Mic unavailable (page must be HTTPS or localhost)', 'warn'); return; }
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
  rtWS.onmessage = (e) => { if (e.data instanceof ArrayBuffer) onRTAudio(e.data); else { try { onRTMsg(JSON.parse(e.data)); } catch (_) { } } };
  rtWS.onclose = () => { if (rtRunning) { log('Realtime disconnected', 'warn'); stopRealtimeUI('Disconnected'); } };
  rtWS.onerror = () => { log('Realtime connection error', 'warn'); };
  try { rtMicStream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true } }); }
  catch (e) { log('Mic error: ' + e.message, 'warn'); stopRealtimeUI('Tap to start'); return; }
  rtCtx = new AudioContext({ sampleRate: 16000 });
  const src = rtCtx.createMediaStreamSource(rtMicStream);
  rtScript = rtCtx.createScriptProcessor(2048, 1, 1);
  rtScript.onaudioprocess = (e) => {
    if (!rtWS || rtWS.readyState !== 1) return;
    const d = e.inputBuffer.getChannelData(0); const buf = new Int16Array(d.length);
    for (let i = 0; i < d.length; i++) { let s = Math.max(-1, Math.min(1, d[i])); buf[i] = s < 0 ? s * 0x8000 : s * 0x7FFF; }
    rtWS.send(buf.buffer);
  };
  src.connect(rtScript); rtScript.connect(rtCtx.destination);
  rtPlayCtx = new AudioContext(); rtPlayTime = rtPlayCtx.currentTime;
  rtRunning = true;
  document.getElementById('rtBtn').classList.add('active');
  log('Realtime conversation started — just talk (use headphones to avoid echo).', 'hl');
}
function stopRealtimeUI(msg) {
  rtRunning = false;
  if (rtScript) { try { rtScript.disconnect(); } catch (e) { } rtScript = null; }
  if (rtMicStream) { rtMicStream.getTracks().forEach(t => t.stop()); rtMicStream = null; }
  if (rtCtx) { try { rtCtx.close(); } catch (e) { } rtCtx = null; }
  rtClearPlayback();
  if (rtPlayCtx) { try { rtPlayCtx.close(); } catch (e) { } rtPlayCtx = null; }
  if (rtWS) { try { rtWS.close(); } catch (e) { } rtWS = null; }
  document.getElementById('rtView').classList.add('hidden');
  document.querySelector('.input-area').classList.remove('hidden');
  const hasMsgs = currentConvMessages.length > 0;
  document.getElementById('initOverlay').classList.toggle('hidden', hasMsgs);
  document.getElementById('messages').classList.toggle('hidden', !hasMsgs);
  document.getElementById('rtBtn').classList.remove('active');
  setRTState('idle');
  document.getElementById('rtState').textContent = msg || 'Tap to start';
  document.getElementById('rtTranscript').innerHTML = '';
  rtAsstEl = null; rtAsstText = ''; rtUserEl = null;
}
async function stopRealtime() {
  if (rtWS && rtWS.readyState === 1) { try { rtWS.send(JSON.stringify({ type: 'stop' })); } catch (e) { } }
  stopRealtimeUI('Ended'); setStatus('Ready', '');
}
window.toggleRealtime = function () { if (rtRunning) stopRealtime(); else { setStatus('Realtime', 'active'); startRealtime(); } };

// ============================================================
// Init
// ============================================================
(async function init() {
  db = await openDB();
  const savedId = parseInt(localStorage.getItem('activeConvId') || '0');
  if (savedId) { const conv = await dbGet(savedId); if (conv) { await loadConversation(conv); } else { await newConversation(); } }
  else { await newConversation(); }
  setStatus('Ready', '');
  log('Supertonic voice chat ready — STT via parakeet.cpp.', 'ok');
  document.getElementById('userInput').focus();
})();
