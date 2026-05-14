const state = {
  lastResponse: null,
  dataset: [],
  filteredDataset: [],
};

const $ = (id) => document.getElementById(id);
const has = (id) => $(id) !== null;
const valueOf = (id, fallback = '') => {
  const element = $(id);
  return element ? element.value : fallback;
};
const checkedOf = (id, fallback = false) => {
  const element = $(id);
  return element ? element.checked : fallback;
};
const setValue = (id, value) => {
  const element = $(id);
  if (element) element.value = value;
};
const setText = (id, value) => {
  const element = $(id);
  if (element) element.textContent = value;
};
const setClass = (id, value) => {
  const element = $(id);
  if (element) element.className = value;
};

function setDefaultPrompt() {
  setValue('roleInput', 'user');
  setValue('promptInput', '');
  setValue('datasetSelect', '');
  setText('datasetMeta', 'После выбора текст будет подставлен в поле запроса.');
}

function asYesNo(value) {
  if (value === true) return 'да';
  if (value === false) return 'нет';
  return '—';
}

function setBadge(decision) {
  const badge = $('decisionBadge');
  if (!badge) return;
  badge.className = 'badge';

  if (!decision) {
    badge.classList.add('badge-idle');
    badge.textContent = 'нет данных';
    return;
  }

  const normalized = String(decision).toLowerCase();
  badge.classList.add(`badge-${normalized}`);
  badge.textContent = normalized;
}

function updateRisk(score) {
  const value = Number.isFinite(Number(score)) ? Math.max(0, Math.min(1, Number(score))) : null;
  setText('riskValue', value === null ? '—' : value.toFixed(2));
  const riskBar = $('riskBar');
  if (riskBar) riskBar.style.width = value === null ? '0%' : `${Math.round(value * 100)}%`;
}

function renderReasons(reasons = [], mlFeatures = []) {
  const target = $('reasonsList');

  if (!reasons.length && !mlFeatures.length) {
    target.className = 'empty-state';
    target.textContent = 'Причины отсутствуют.';
    return;
  }

  target.className = '';
  const reasonHtml = reasons.map((reason) => `
    <article class="reason-item">
      <div class="reason-meta">
        <span class="pill">${escapeHtml(reason.source || 'source')}</span>
        <span class="pill">${escapeHtml(reason.code || 'code')}</span>
        <span class="pill">severity: ${escapeHtml(reason.severity || 'info')}</span>
        ${reason.action ? `<span class="pill">action: ${escapeHtml(reason.action)}</span>` : ''}
      </div>
      <div>${escapeHtml(reason.message || 'Без описания')}</div>
    </article>
  `).join('');

  const featuresHtml = mlFeatures.length ? `
    <article class="reason-item">
      <div class="reason-meta"><span class="pill">ml_features</span></div>
      <div>${mlFeatures.map(escapeHtml).join(', ')}</div>
    </article>
  ` : '';

  target.innerHTML = reasonHtml + featuresHtml;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function buildRequestPayload() {
  return {
    messages: [
      {
        role: valueOf('roleInput', 'user'),
        content: valueOf('promptInput').trim(),
      },
    ],
    metadata: {
      source: 'gateway-static-ui',
      created_at: new Date().toISOString(),
    },
    debug_filter: {
      enable_rule_detection: checkedOf('enableRules', true),
      enable_heuristics: checkedOf('enableHeuristics', true),
      enable_rewrite: checkedOf('enableRewrite', true),
      llm_enabled: checkedOf('llmEnabled', false),
      ml_model: valueOf('mlModel', 'tfidf_logreg'),
    },
  };
}

function renderResponse(data) {
  state.lastResponse = data;
  setBadge(data.decision);
  updateRisk(data.risk_score);
  setText('blockedValue', asYesNo(data.blocked));
  setText('llmCalledValue', asYesNo(data.llm_called));
  setText('modelValue', data.model || '—');
  setText('answerOutput', data.response || data.llm_error || 'Ответ LLM отсутствует.');
  setText('sanitizedOutput', data.sanitized_request
    ? JSON.stringify(data.sanitized_request, null, 2)
    : 'Sanitized request отсутствует.');
  setText('jsonOutput', JSON.stringify(data, null, 2));
  renderReasons(data.reasons || [], data.ml_features || []);
}

function renderError(error) {
  setBadge('error');
  updateRisk(null);
  setText('blockedValue', '—');
  setText('llmCalledValue', '—');
  setText('modelValue', '—');
  setText('answerOutput', String(error.message || error));
  setText('sanitizedOutput', '—');
  setText('jsonOutput', String(error.stack || error.message || error));
  setClass('reasonsList', 'empty-state');
  setText('reasonsList', 'Ошибка при обращении к API.');
}

async function checkHealth() {
  const dot = $('healthDot');
  const text = $('healthText');
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), 2500);

  try {
    if (dot) dot.className = 'status-dot checking';
    if (text) text.textContent = 'проверяется…';

    const response = await fetch('/health', {
      cache: 'no-store',
      signal: controller.signal,
      headers: { Accept: 'application/json' },
    });

    const raw = await response.text();
    let data = {};
    if (raw) {
      try {
        data = JSON.parse(raw);
      } catch (_error) {
        data = { status: raw.trim() };
      }
    }

    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const status = data.status || 'ok';
    const service = data.service ? ` / ${data.service}` : '';
    if (dot) dot.className = 'status-dot ok';
    if (text) text.textContent = `${status}${service}`;
  } catch (error) {
    if (dot) dot.className = 'status-dot fail';
    if (text) {
      text.textContent = error.name === 'AbortError'
        ? 'тайм-аут /health'
        : 'недоступен';
    }
  } finally {
    window.clearTimeout(timeoutId);
  }
}
async function submitPrompt(event) {
  if (event) event.preventDefault();

  const prompt = valueOf('promptInput').trim();
  if (!prompt) {
    renderError(new Error('Введите текст запроса.'));
    return false;
  }

  const button = $('submitBtn');
  if (button) {
    button.disabled = true;
    button.textContent = 'Проверяется…';
  }

  try {
    const response = await fetch('/v1/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify(buildRequestPayload()),
    });

    const raw = await response.text();
    let data = {};
    if (raw) {
      try {
        data = JSON.parse(raw);
      } catch (_error) {
        data = { response: raw };
      }
    }

    if (!response.ok) {
      throw new Error(data.detail ? JSON.stringify(data.detail, null, 2) : `HTTP ${response.status}`);
    }

    renderResponse(data);
    const resultCard = document.querySelector('.result-card');
    if (resultCard) resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (error) {
    renderError(error);
  } finally {
    if (button) {
      button.disabled = false;
      button.textContent = 'Проверить запрос';
    }
    checkHealth();
  }

  return false;
}
function setupTabs() {
  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach((item) => item.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach((item) => item.classList.remove('active'));
      tab.classList.add('active');
      const panel = $(`tab-${tab.dataset.tab}`);
      if (panel) panel.classList.add('active');
    });
  });
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = '';
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];

    if (char === '"') {
      if (inQuotes && next === '"') {
        cell += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (char === ',' && !inQuotes) {
      row.push(cell);
      cell = '';
      continue;
    }

    if ((char === '\n' || char === '\r') && !inQuotes) {
      if (char === '\r' && next === '\n') i += 1;
      row.push(cell);
      if (row.some((value) => value.length > 0)) rows.push(row);
      row = [];
      cell = '';
      continue;
    }

    cell += char;
  }

  row.push(cell);
  if (row.some((value) => value.length > 0)) rows.push(row);

  const headers = rows.shift() || [];
  return rows.map((values, index) => {
    const item = { rowNumber: index + 1 };
    headers.forEach((header, headerIndex) => {
      item[header] = values[headerIndex] || '';
    });
    return item;
  });
}

function uniqueSorted(values) {
  return [...new Set(values.filter(Boolean))].sort((a, b) => a.localeCompare(b));
}

function fillFilter(selectId, values, allLabel) {
  const select = $(selectId);
  if (!select) return;
  select.innerHTML = `<option value="all" selected>${allLabel}</option>`;
  uniqueSorted(values).forEach((value) => {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  });
}

function languageLabel(value) {
  const labels = {
    ru: 'Русский',
    en: 'Английский',
  };
  return labels[value] || value || '—';
}

function fillLanguageFilter() {
  const select = $('languageFilter');
  if (!select) return;
  select.innerHTML = '<option value="all" selected>Все языки</option>';

  [
    ['ru', 'Русский'],
    ['en', 'Английский'],
  ].forEach(([value, label]) => {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = label;
    select.appendChild(option);
  });
}

function makeDatasetOption(item) {
  const labelText = item.label === '1' ? 'атака' : 'безопасный';
  const snippet = (item.text || '').replace(/\s+/g, ' ').trim().slice(0, 130);
  return `#${String(item.rowNumber).padStart(4, '0')} | ${item.attack_type || 'unknown'} | ${labelText} | ${languageLabel(item.language)} | ${snippet}`;
}

function renderDatasetList() {
  const type = valueOf('attackTypeFilter', 'all');
  const language = valueOf('languageFilter', 'all');

  state.filteredDataset = state.dataset.filter((item) => {
    const matchesType = type === 'all' || item.attack_type === type;
    const matchesLanguage = language === 'all' || item.language === language;
    return matchesType && matchesLanguage;
  });

  const select = $('datasetSelect');
  if (!select) return;
  select.innerHTML = '';

  const emptyOption = document.createElement('option');
  emptyOption.value = '';
  emptyOption.textContent = state.filteredDataset.length
    ? 'Выберите пример из CSV'
    : 'Нет примеров по выбранным фильтрам';
  select.appendChild(emptyOption);

  state.filteredDataset.forEach((item, index) => {
    const option = document.createElement('option');
    option.value = String(index);
    option.textContent = makeDatasetOption(item);
    select.appendChild(option);
  });

  setText('datasetCount', `${state.filteredDataset.length} из ${state.dataset.length}`);
}

function applyDatasetSelection() {
  const index = Number(valueOf('datasetSelect', ''));
  const item = state.filteredDataset[index];
  if (!item) {
    setText('datasetMeta', 'После выбора текст будет подставлен в поле запроса.');
    return;
  }

  setValue('roleInput', 'user');
  setValue('promptInput', item.text || '');
  setText('datasetMeta', `Строка #${item.rowNumber}; label=${item.label}; attack_type=${item.attack_type}; language=${languageLabel(item.language)}; source=${item.source || '—'}.`);
}


function pickRandomDatasetItemIfPromptEmpty() {
  if (valueOf('promptInput').trim()) return;

  const baseCandidates = state.filteredDataset.length ? state.filteredDataset : state.dataset;
  const russianCandidates = baseCandidates.filter((item) => item.language === 'ru');
  const candidates = russianCandidates.length ? russianCandidates : baseCandidates;
  if (!candidates.length) return;

  const randomIndex = Math.floor(Math.random() * candidates.length);
  const item = candidates[randomIndex];

  const select = $('datasetSelect');
  if (select) {
    const optionIndex = state.filteredDataset.indexOf(item);
    select.value = optionIndex >= 0 ? String(optionIndex) : '';
  }

  setValue('roleInput', 'user');
  setValue('promptInput', item.text || '');
  setText('datasetMeta', `Случайно выбранная строка #${item.rowNumber}; label=${item.label}; attack_type=${item.attack_type}; language=${languageLabel(item.language)}; source=${item.source || '—'}.`);
}

async function loadDataset() {
  const select = $('datasetSelect');
  if (!select) return;

  try {
    const response = await fetch('/data/test.csv', { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const csv = await response.text();
    state.dataset = parseCsv(csv);
    fillFilter('attackTypeFilter', state.dataset.map((item) => item.attack_type), 'Все типы');
    fillLanguageFilter();
    renderDatasetList();
    if (valueOf('promptInput').trim()) {
      setText('datasetMeta', 'CSV загружен. Выберите строку, чтобы подставить её в запрос.');
    }
    pickRandomDatasetItemIfPromptEmpty();
  } catch (error) {
    select.innerHTML = '<option value="">CSV не найден</option>';
    setText('datasetCount', 'ошибка');
    setText('datasetMeta', `Не удалось загрузить /data/test.csv: ${error.message}`);
  }
}

function setupDatasetControls() {
  ['attackTypeFilter', 'languageFilter'].forEach((id) => {
    const control = $(id);
    if (!control) return;
    control.addEventListener('input', renderDatasetList);
    control.addEventListener('change', renderDatasetList);
  });

  const datasetSelect = $('datasetSelect');
  if (datasetSelect) datasetSelect.addEventListener('change', applyDatasetSelection);
}

function initApp() {
  setDefaultPrompt();
  setupTabs();
  setupDatasetControls();
  loadDataset();

  const promptForm = $('promptForm');
  const resetBtn = $('resetBtn');

  if (promptForm) {
    promptForm.setAttribute('action', 'javascript:void(0)');
    promptForm.addEventListener('submit', submitPrompt);
  }

  if (resetBtn) resetBtn.addEventListener('click', setDefaultPrompt);

  checkHealth();
  window.setInterval(checkHealth, 15000);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}
