/* ==========================================================================
   LLM Eval Dashboard — dependency-free frontend
   ========================================================================== */
(() => {
  'use strict';

  const CATEGORY_ORDER = [
    'reasoning', 'coding', 'math', 'japanese', 'summarization', 'instruction'
  ];
  const CATEGORY_LABELS = {
    reasoning: '推論',
    coding: 'コーディング',
    math: '数学',
    japanese: '日本語',
    summarization: '要約',
    instruction: '指示追従',
    general: '一般'
  };
  const FONT = '"Segoe UI", "Hiragino Sans", "Noto Sans JP", system-ui, sans-serif';
  const MONO = 'ui-monospace, "Cascadia Code", Consolas, monospace';

  const state = {
    bootstrap: null,
    runs: [],
    run: null,
    config: null,
    dataset: null,
    datasetItems: new Map(),
    job: null,
    jobTimer: null,
    jobHandledFor: null,
    radarSelection: new Set(),
    sortKey: 'overall',
    sortDir: -1,
    detailModel: 'all',
    detailCategory: 'all',
    failOnly: false,
    search: '',
    theme: 'dark',
    chartHits: new Map(),
    toastTimer: null
  };

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  // ---------------------------------------------------------------- helpers
  function esc(value) {
    return String(value ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function fmt(value, digits = 1) {
    return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '—';
  }

  function fmtMs(value) {
    const v = Number(value);
    if (!Number.isFinite(v) || v <= 0) return '—';
    return v >= 1000 ? `${(v / 1000).toFixed(2)}s` : `${Math.round(v)}ms`;
  }

  function fmtPct(value, digits = 1) {
    return `${fmt(value, digits)}%`;
  }

  function fmtUsd(value) {
    const v = Number(value);
    if (!Number.isFinite(v) || v <= 0) return '$0';
    if (v < 0.01) return `$${v.toFixed(4)}`;
    if (v < 1) return `$${v.toFixed(3)}`;
    return `$${v.toFixed(2)}`;
  }

  function fmtCostPer1k(value) {
    const v = Number(value);
    if (!Number.isFinite(v) || v <= 0) return '無料';
    return fmtUsd(v);
  }

  function fmtDuration(ms) {
    const v = Number(ms);
    if (!Number.isFinite(v) || v <= 0) return '—';
    if (v < 1000) return `${Math.round(v)}ms`;
    if (v < 60000) return `${(v / 1000).toFixed(2)}s`;
    return `${(v / 60000).toFixed(1)}min`;
  }

  function fmtDate(iso) {
    if (!iso) return '—';
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return iso;
    return date.toLocaleString('ja-JP', {
      year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit'
    });
  }

  function fmtShortDate(iso) {
    if (!iso) return '';
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return iso;
    return date.toLocaleDateString('ja-JP', { month: '2-digit', day: '2-digit' });
  }

  function categoryLabel(category) {
    return CATEGORY_LABELS[category] || category || '一般';
  }

  function providerLabel(provider) {
    return provider === 'mock' ? 'MOCK' : 'API';
  }

  function providerBadge(provider) {
    return `<span class="badge ${esc(provider)}">${esc(providerLabel(provider))}</span>`;
  }

  function debounce(fn, wait) {
    let timer = null;
    return (...args) => {
      clearTimeout(timer);
      timer = setTimeout(() => fn(...args), wait);
    };
  }

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function hexToRgba(hex, alpha) {
    let value = String(hex || '#38bdf8').replace('#', '');
    if (value.length === 3) {
      value = value.split('').map((c) => c + c).join('');
    }
    const number = parseInt(value, 16);
    if (Number.isNaN(number)) return `rgba(56,189,248,${alpha})`;
    const r = (number >> 16) & 255;
    const g = (number >> 8) & 255;
    const b = number & 255;
    return `rgba(${r},${g},${b},${alpha})`;
  }

  function roundRect(ctx, x, y, w, h, r) {
    const radius = Math.min(r, Math.abs(w) / 2, Math.abs(h) / 2);
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.arcTo(x + w, y, x + w, y + h, radius);
    ctx.arcTo(x + w, y + h, x, y + h, radius);
    ctx.arcTo(x, y + h, x, y, radius);
    ctx.arcTo(x, y, x + w, y, radius);
    ctx.closePath();
  }

  function fitText(ctx, text, maxWidth) {
    const value = String(text ?? '');
    if (ctx.measureText(value).width <= maxWidth) return value;
    let result = value;
    while (result.length > 1 && ctx.measureText(`${result}…`).width > maxWidth) {
      result = result.slice(0, -1);
    }
    return `${result}…`;
  }

  // ---------------------------------------------------------------- api
  async function api(path, options = {}) {
    const response = await fetch(path, {
      headers: { 'Content-Type': 'application/json' },
      ...options
    });
    const text = await response.text();
    let data = {};
    try {
      data = text ? JSON.parse(text) : {};
    } catch {
      data = {};
    }
    if (!response.ok) {
      throw new Error(data.error || `${response.status} ${response.statusText}`);
    }
    return data;
  }

  // ---------------------------------------------------------------- init
  document.addEventListener('DOMContentLoaded', init);

  async function init() {
    state.theme = localStorage.getItem('llm-eval-theme') || 'dark';
    applyTheme(state.theme);
    bindEvents();
    try {
      await refreshBootstrap();
    } catch (error) {
      renderFatal(error);
    }
    if (state.job && state.job.state === 'running') {
      startJobPolling();
    }
  }

  async function refreshBootstrap(selectRunId) {
    const data = await api('/api/bootstrap');
    state.bootstrap = data;
    state.runs = data.runs || [];
    state.config = data.config || { models: [] };
    state.dataset = data.dataset || {};
    state.datasetItems = new Map();
    (data.dataset?.items || []).forEach((item) => state.datasetItems.set(item.id, item));
    state.job = data.job;

    renderRunSelect();
    renderConfigModels();

    const id = selectRunId
      || state.run?.id
      || data.latest?.id
      || state.runs[0]?.id;

    if (id) {
      const previousId = state.run?.id;
      if (data.latest && data.latest.id === id) {
        state.run = data.latest;
      } else {
        state.run = await api(`/api/runs/${encodeURIComponent(id)}`);
      }
      if (previousId !== id) {
        state.radarSelection = new Set();
        state.detailModel = 'all';
        state.detailCategory = 'all';
      }
      renderAll();
    } else {
      state.run = null;
      renderEmpty();
    }
    if (state.job && state.job.state === 'running') {
      updateJobPanel(state.job);
    }
  }

  async function loadRun(id) {
    if (!id) return;
    try {
      $('#run-meta').textContent = 'LOADING RUN';
      state.run = await api(`/api/runs/${encodeURIComponent(id)}`);
      state.radarSelection = new Set();
      state.detailModel = 'all';
      state.detailCategory = 'all';
      renderAll();
    } catch (error) {
      toast(error.message, true);
    }
  }

  // ---------------------------------------------------------------- render
  function renderAll() {
    renderRunSelect();
    renderMeta();
    renderKpis();
    renderLeaderboard();
    renderRadarLegend();
    renderCharts();
    renderHeatmap();
    renderHistoryLegend();
    renderDetailControls();
    renderDetails();
    renderFooter();
  }

  function renderRunSelect() {
    const select = $('#run-select');
    if (!select) return;
    if (!state.runs.length) {
      select.innerHTML = '<option>実行結果なし</option>';
      return;
    }
    select.innerHTML = state.runs.map((run) => {
      const suffix = run.source === 'demo' ? ' · demo' : '';
      return `<option value="${esc(run.id)}">${esc(run.name || run.id)} · ${esc(fmtShortDate(run.created_at))}${suffix}</option>`;
    }).join('');
    if (state.run) select.value = state.run.id;
  }

  function renderMeta() {
    const meta = $('#run-meta');
    if (!state.run) {
      meta.textContent = 'NO RUNS YET';
      return;
    }
    const run = state.run;
    const source = run.source === 'demo' ? 'DEMO RUN' : 'LIVE RUN';
    meta.textContent = `${source} · ${fmtDate(run.created_at)} · ${run.dataset_name || 'benchmark'}`;
    const exportLink = $('#export-link');
    exportLink.href = `/api/runs/${encodeURIComponent(run.id)}/csv`;
    exportLink.style.opacity = '1';
    exportLink.style.pointerEvents = 'auto';
  }

  function renderKpis() {
    const host = $('#kpis');
    if (!state.run || !state.run.summaries?.length) {
      host.innerHTML = '';
      return;
    }
    const summaries = [...state.run.summaries].sort((a, b) => b.overall - a.overall);
    const best = summaries[0];
    const fastest = [...summaries].sort((a, b) => a.p50_latency_ms - b.p50_latency_ms)[0];
    const avgQuality = summaries.reduce((sum, item) => sum + item.quality, 0) / summaries.length;
    const totalCost = summaries.reduce((sum, item) => sum + (item.total_cost_usd || 0), 0);
    const passThreshold = (state.run.pass_threshold ?? 0.6) * 100;

    const cards = [
      {
        label: 'Best overall',
        value: fmt(best.overall, 1),
        sub: best.model_id,
        color: best.color,
        wide: false
      },
      {
        label: 'Models',
        value: String(summaries.length),
        sub: `${state.run.total_requests || 0} requests`,
        color: '#a78bfa',
        wide: false
      },
      {
        label: 'Avg quality',
        value: fmt(avgQuality, 1),
        sub: `pass line ${fmt(passThreshold, 0)}%`,
        color: '#34d399',
        wide: false
      },
      {
        label: 'Fastest p50',
        value: fmtMs(fastest.p50_latency_ms),
        sub: fastest.model_id,
        color: '#fbbf24',
        wide: false
      },
      {
        label: 'Total cost',
        value: fmtUsd(totalCost),
        sub: `duration ${fmtDuration(state.run.duration_ms)} · ${state.run.total_requests || 0} calls`,
        color: '#fb7185',
        wide: true
      }
    ];

    host.innerHTML = cards.map((card, index) => `
      <div class="kpi ${card.wide ? 'wide' : ''}" style="--kpi-color:${esc(card.color)};animation-delay:${index * 40}ms">
        <div class="kpi-label">${esc(card.label)}</div>
        <div class="kpi-value">${esc(card.value)}</div>
        <div class="kpi-sub">${esc(card.sub)}</div>
      </div>
    `).join('');
  }

  function renderLeaderboard() {
    const tbody = $('#leaderboard tbody');
    if (!state.run || !state.run.summaries?.length) {
      tbody.innerHTML = '<tr><td colspan="10"><div class="empty">実行結果がありません</div></td></tr>';
      return;
    }
    const summaries = [...state.run.summaries].sort((a, b) => {
      const av = Number(a[state.sortKey] ?? 0);
      const bv = Number(b[state.sortKey] ?? 0);
      return (av - bv) * state.sortDir;
    });

    tbody.innerHTML = summaries.map((summary, index) => {
      const passRatio = summary.total > 0 ? summary.passed / summary.total : 0;
      return `
        <tr data-model="${esc(summary.model_id)}">
          <td class="rank ${index === 0 ? 'top' : ''}">${index + 1}</td>
          <td>
            <div class="model-cell">
              <span class="dot" style="color:${esc(summary.color)};background:${esc(summary.color)}"></span>
              <span class="model-name">${esc(summary.model_id)}</span>
              ${providerBadge(summary.provider)}
            </div>
          </td>
          <td>
            <div class="score-cell">
              <span class="score-num">${fmt(summary.overall, 1)}</span>
              <span class="bar"><i style="width:${clamp(summary.overall, 0, 100)}%;background:linear-gradient(90deg, ${esc(summary.color)}, ${esc(summary.color)}aa)"></i></span>
            </div>
          </td>
          <td>${fmt(summary.quality, 1)}</td>
          <td>${fmtPct(summary.success_rate, 1)}</td>
          <td>${fmt(summary.tokens_per_sec, 1)}</td>
          <td>${fmtMs(summary.p50_latency_ms)}</td>
          <td>${fmtMs(summary.avg_ttft_ms)}</td>
          <td>${esc(fmtCostPer1k(summary.cost_per_1k_usd))}</td>
          <td class="pass"><span class="pill ${passRatio >= 0.8 ? 'ok' : 'fail'}"><b>${summary.passed}</b>/${summary.total}</span></td>
        </tr>
      `;
    }).join('');

    $$('#leaderboard thead th.sortable').forEach((th) => {
      th.classList.remove('sorted-asc', 'sorted-desc');
      if (th.dataset.sort === state.sortKey) {
        th.classList.add(state.sortDir > 0 ? 'sorted-asc' : 'sorted-desc');
      }
    });
  }

  function renderRadarLegend() {
    const host = $('#radar-legend');
    if (!state.run || !state.run.summaries?.length) {
      host.innerHTML = '';
      return;
    }
    const summaries = [...state.run.summaries].sort((a, b) => b.overall - a.overall);
    if (state.radarSelection.size === 0) {
      summaries.slice(0, Math.min(4, summaries.length)).forEach((summary) => {
        state.radarSelection.add(summary.model_id);
      });
    }
    const valid = new Set(summaries.map((summary) => summary.model_id));
    state.radarSelection.forEach((id) => {
      if (!valid.has(id)) state.radarSelection.delete(id);
    });

    host.innerHTML = summaries.map((summary) => {
      const active = state.radarSelection.has(summary.model_id);
      return `
        <button type="button" class="legend-item ${active ? 'active' : 'off'}" data-model="${esc(summary.model_id)}">
          <span class="dot" style="color:${esc(summary.color)};background:${esc(summary.color)}"></span>
          ${esc(summary.model_id)}
        </button>
      `;
    }).join('');
  }

  function renderHistoryLegend() {
    const host = $('#history-legend');
    if (!state.run || !state.run.summaries?.length || state.runs.length === 0) {
      host.innerHTML = '';
      return;
    }
    const models = [...state.run.summaries].sort((a, b) => b.overall - a.overall).slice(0, 5);
    host.innerHTML = models.map((model) => `
      <span class="legend-item active" style="cursor:default">
        <span class="dot" style="color:${esc(model.color)};background:${esc(model.color)}"></span>
        ${esc(model.model_id)}
      </span>
    `).join('');
  }

  function renderHeatmap() {
    const host = $('#heatmap');
    if (!state.run || !state.run.summaries?.length) {
      host.innerHTML = '<div class="empty">データがありません</div>';
      return;
    }
    const categories = collectCategories();
    const models = [...state.run.summaries].sort((a, b) => b.overall - a.overall);
    if (!categories.length) {
      host.innerHTML = '<div class="empty">カテゴリデータがありません</div>';
      return;
    }
    host.style.gridTemplateColumns = `minmax(104px, 132px) repeat(${categories.length}, minmax(42px, 1fr))`;
    const header = ['<div class="heat-head"></div>']
      .concat(categories.map((category) => `<div class="heat-head">${esc(categoryLabel(category))}</div>`))
      .join('');
    const rows = models.map((model) => {
      const label = `
        <div class="heat-row-label">
          <span class="dot" style="color:${esc(model.color)};background:${esc(model.color)}"></span>
          <span title="${esc(model.model_id)}">${esc(model.model_id)}</span>
        </div>`;
      const cells = categories.map((category) => {
        const value = Number(model.category_scores?.[category] ?? 0);
        const { r, g, b } = heatColor(value);
        const alpha = 0.12 + 0.55 * clamp(value / 100, 0, 1);
        const textColor = value >= 68 ? '#04111f' : 'var(--text)';
        return `<div class="heat-cell" title="${esc(model.model_id)} · ${esc(categoryLabel(category))}: ${fmt(value, 1)}"
          style="background:rgba(${r},${g},${b},${alpha.toFixed(3)});color:${textColor}">${fmt(value, 0)}</div>`;
      }).join('');
      return label + cells;
    }).join('');
    host.innerHTML = header + rows;
  }

  function renderDetailControls() {
    if (!state.run || !state.run.summaries?.length) {
      $('#detail-model').innerHTML = '';
      $('#detail-category').innerHTML = '';
      return;
    }
    const summaries = [...state.run.summaries].sort((a, b) => b.overall - a.overall);
    const modelSelect = $('#detail-model');
    const modelIds = summaries.map((summary) => summary.model_id);
    if (state.detailModel !== 'all' && !modelIds.includes(state.detailModel)) {
      state.detailModel = 'all';
    }
    modelSelect.innerHTML = '<option value="all">すべてのモデル</option>' +
      summaries.map((summary) => `<option value="${esc(summary.model_id)}">${esc(summary.model_id)}</option>`).join('');
    modelSelect.value = state.detailModel;

    const categories = collectCategories();
    const categorySelect = $('#detail-category');
    if (state.detailCategory !== 'all' && !categories.includes(state.detailCategory)) {
      state.detailCategory = 'all';
    }
    categorySelect.innerHTML = '<option value="all">すべてのカテゴリ</option>' +
      categories.map((category) => `<option value="${esc(category)}">${esc(categoryLabel(category))}</option>`).join('');
    categorySelect.value = state.detailCategory;
  }

  function renderDetails() {
    const host = $('#details');
    if (!state.run || !state.run.results?.length) {
      host.innerHTML = `
        <div class="empty">
          <div>実行結果がありません。<br>デモを生成するか、ベンチマークを実行してください。</div>
          <button type="button" class="primary-btn" id="empty-demo" style="margin-top:14px">デモを実行</button>
        </div>`;
      const button = $('#empty-demo');
      if (button) button.addEventListener('click', startDemo);
      return;
    }

    const search = state.search.trim().toLowerCase();
    const results = state.run.results.filter((result) => {
      if (state.detailModel !== 'all' && result.model_id !== state.detailModel) return false;
      if (state.detailCategory !== 'all' && result.category !== state.detailCategory) return false;
      if (state.failOnly && result.ok) return false;
      if (!search) return true;
      const item = state.datasetItems.get(result.item_id);
      const haystack = [
        result.item_id, result.category, result.output, result.error,
        item?.prompt || '', (result.matched_keywords || []).join(' ')
      ].join(' ').toLowerCase();
      return haystack.includes(search);
    });

    if (!results.length) {
      host.innerHTML = '<div class="empty">条件に一致する結果がありません</div>';
      return;
    }

    host.innerHTML = results.map((result) => {
      const item = state.datasetItems.get(result.item_id);
      const expected = item?.expected_keywords || [];
      const matched = new Set(result.matched_keywords || []);
      const chips = expected.length
        ? expected.map((keyword) => `<span class="chip ${matched.has(keyword) ? 'hit' : ''}">${esc(keyword)}</span>`).join('')
        : '<span class="chip">—</span>';
      const quality = clamp(result.quality * 100, 0, 100);
      return `
        <details class="result">
          <summary>
            <span class="status-dot ${result.ok ? 'ok' : 'fail'}"></span>
            <span class="item-id" title="${esc(result.item_id)}">${esc(result.item_id)}</span>
            <span class="cat-badge">${esc(categoryLabel(result.category))}</span>
            <span class="mini-quality">
              <span class="bar"><i style="width:${quality}%;background:linear-gradient(90deg,#38bdf8,#a78bfa)"></i></span>
              <span>${fmt(quality, 0)}</span>
            </span>
            <span class="result-metric">${fmtMs(result.latency_ms)}</span>
            <span class="result-metric">${result.completion_tokens || 0} tok</span>
            <span class="result-metric"><span class="pill ${result.passed ? 'ok' : 'fail'}">${result.passed ? 'PASS' : 'FAIL'}</span></span>
          </summary>
          <div class="result-body">
            <h4>Prompt</h4>
            <p class="prompt-text">${esc(item?.prompt || '（プロンプト情報なし）')}</p>
            ${result.error ? `<div class="error-box">${esc(result.error)}</div>` : ''}
            <h4>Output</h4>
            <pre class="output-pre">${esc(result.output || '（出力なし）')}</pre>
            <h4>Expected keywords</h4>
            <div class="chips">${chips}</div>
            <div class="meta-grid">
              <div>TTFT<b>${fmtMs(result.ttft_ms)}</b></div>
              <div>Throughput<b>${fmt(result.tokens_per_sec, 1)} tok/s</b></div>
              <div>Prompt tokens<b>${result.prompt_tokens || 0}</b></div>
              <div>Completion tokens<b>${result.completion_tokens || 0}</b></div>
              <div>Cost<b>${fmtUsd(result.cost_usd)}</b></div>
              <div>Quality<b>${fmt(quality, 1)} / 100</b></div>
            </div>
          </div>
        </details>
      `;
    }).join('');
  }

  function renderFooter() {
    const host = $('#footer-stats');
    if (!state.run) {
      host.textContent = 'No runs';
      return;
    }
    host.textContent = `${state.run.total_requests || 0} requests · ${fmtDuration(state.run.duration_ms)} · pass ≥ ${fmt((state.run.pass_threshold ?? 0.6) * 100, 0)}% · ${state.run.dataset_name || ''}`;
  }

  function renderEmpty() {
    $('#run-meta').textContent = 'NO RUNS YET';
    $('#kpis').innerHTML = '';
    $('#leaderboard tbody').innerHTML = '<tr><td colspan="10"><div class="empty">実行結果がありません</div></td></tr>';
    $('#radar-legend').innerHTML = '';
    $('#history-legend').innerHTML = '';
    $('#heatmap').innerHTML = '<div class="empty">データがありません</div>';
    $('#details').innerHTML = `
      <div class="empty">
        <div>まだ評価結果がありません。<br>デモを生成するか、右上の「ベンチマーク実行」から開始してください。</div>
        <button type="button" class="primary-btn" id="empty-demo" style="margin-top:14px">デモを実行</button>
      </div>`;
    const button = $('#empty-demo');
    if (button) button.addEventListener('click', startDemo);
    $('#export-link').style.opacity = '0.4';
    $('#export-link').style.pointerEvents = 'none';
    renderFooter();
    ['#radar', '#scatter', '#latency', '#history'].forEach((selector) => {
      const canvas = $(selector);
      const { ctx, w, h } = setupCanvas(canvas);
      drawEmptyState(ctx, w, h, '実行結果がありません');
    });
  }

  function renderFatal(error) {
    $('#run-meta').textContent = 'CONNECTION ERROR';
    $('#kpis').innerHTML = '';
    $('#details').innerHTML = `<div class="empty">ダッシュボードの読み込みに失敗しました。<br>${esc(error.message)}</div>`;
  }

  function collectCategories() {
    const set = new Set();
    (state.run?.summaries || []).forEach((summary) => {
      Object.keys(summary.category_scores || {}).forEach((category) => set.add(category));
    });
    (state.run?.results || []).forEach((result) => set.add(result.category));
    return Array.from(set).sort((a, b) => {
      const ai = CATEGORY_ORDER.indexOf(a);
      const bi = CATEGORY_ORDER.indexOf(b);
      if (ai === -1 && bi === -1) return a.localeCompare(b);
      if (ai === -1) return 1;
      if (bi === -1) return -1;
      return ai - bi;
    });
  }

  // ---------------------------------------------------------------- charts
  function setupCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(20, rect.width);
    const height = Math.max(20, rect.height);
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    return { ctx, w: width, h: height };
  }

  function drawEmptyState(ctx, w, h, text) {
    ctx.fillStyle = cssVar('--faint');
    ctx.font = `12px ${FONT}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, w / 2, h / 2);
  }

  function renderCharts() {
    if (!state.run) return;
    drawRadar();
    drawScatter();
    drawLatency();
    drawHistory();
  }

  function drawRadar() {
    const canvas = $('#radar');
    const { ctx, w, h } = setupCanvas(canvas);
    state.chartHits.set(canvas, { points: [] });

    const summaries = state.run?.summaries || [];
    const categories = collectCategories();
    if (!summaries.length || categories.length < 3) {
      drawEmptyState(ctx, w, h, 'カテゴリデータがありません');
      return;
    }
    const selected = summaries.filter((summary) => state.radarSelection.has(summary.model_id));
    const models = selected.length ? selected : summaries.slice(0, 4);
    const cx = w / 2;
    const cy = h / 2 + 4;
    const radius = Math.max(40, Math.min(w, h) / 2 - 52);
    const textColor = cssVar('--chart-text');
    const grid = cssVar('--chart-grid');
    const axis = cssVar('--chart-axis');
    const points = [];
    const angleFor = (index) => -Math.PI / 2 + (index * Math.PI * 2) / categories.length;

    for (let level = 1; level <= 4; level += 1) {
      ctx.beginPath();
      categories.forEach((_, index) => {
        const angle = angleFor(index);
        const r = (radius * level) / 4;
        const x = cx + Math.cos(angle) * r;
        const y = cy + Math.sin(angle) * r;
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.strokeStyle = grid;
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    categories.forEach((category, index) => {
      const angle = angleFor(index);
      const x = cx + Math.cos(angle) * radius;
      const y = cy + Math.sin(angle) * radius;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(x, y);
      ctx.strokeStyle = axis;
      ctx.stroke();

      const lx = cx + Math.cos(angle) * (radius + 24);
      const ly = cy + Math.sin(angle) * (radius + 24);
      ctx.fillStyle = textColor;
      ctx.font = `10.5px ${FONT}`;
      ctx.textAlign = Math.abs(Math.cos(angle)) < 0.18 ? 'center' : (Math.cos(angle) > 0 ? 'left' : 'right');
      ctx.textBaseline = Math.abs(Math.sin(angle)) < 0.18 ? 'middle' : (Math.sin(angle) > 0 ? 'top' : 'bottom');
      ctx.fillText(categoryLabel(category), lx, ly);
    });

    models.forEach((summary) => {
      const color = summary.color || '#38bdf8';
      ctx.beginPath();
      categories.forEach((category, index) => {
        const value = clamp(Number(summary.category_scores?.[category] ?? 0), 0, 100);
        const angle = angleFor(index);
        const r = (value / 100) * radius;
        const x = cx + Math.cos(angle) * r;
        const y = cy + Math.sin(angle) * r;
        points.push({ x, y, r: 12, html: `<b>${esc(summary.model_id)}</b><span>${esc(categoryLabel(category))}: ${fmt(value, 1)}</span>` });
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.fillStyle = hexToRgba(color, 0.10);
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.stroke();

      categories.forEach((category, index) => {
        const value = clamp(Number(summary.category_scores?.[category] ?? 0), 0, 100);
        const angle = angleFor(index);
        const r = (value / 100) * radius;
        const x = cx + Math.cos(angle) * r;
        const y = cy + Math.sin(angle) * r;
        ctx.beginPath();
        ctx.arc(x, y, 2.4, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
      });
    });

    state.chartHits.set(canvas, { points });
  }

  function drawScatter() {
    const canvas = $('#scatter');
    const { ctx, w, h } = setupCanvas(canvas);
    const summaries = state.run?.summaries || [];
    if (!summaries.length) {
      drawEmptyState(ctx, w, h, 'データがありません');
      state.chartHits.set(canvas, { points: [] });
      return;
    }

    const margin = { left: 46, right: 18, top: 18, bottom: 40 };
    const plotW = w - margin.left - margin.right;
    const plotH = h - margin.top - margin.bottom;
    const maxTps = Math.max(20, ...summaries.map((summary) => Number(summary.tokens_per_sec) || 0));
    const xMax = niceMax(maxTps * 1.15);
    const textColor = cssVar('--chart-text');
    const grid = cssVar('--chart-grid');
    const axis = cssVar('--chart-axis');
    const accent = cssVar('--accent');
    const bg = cssVar('--panel-solid');

    const xOf = (value) => margin.left + (clamp(value, 0, xMax) / xMax) * plotW;
    const yOf = (value) => margin.top + plotH - (clamp(value, 0, 100) / 100) * plotH;

    ctx.font = `10px ${FONT}`;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= 5; i += 1) {
      const value = (i / 5) * 100;
      const y = yOf(value);
      ctx.beginPath();
      ctx.moveTo(margin.left, y);
      ctx.lineTo(w - margin.right, y);
      ctx.strokeStyle = grid;
      ctx.stroke();
      ctx.fillStyle = textColor;
      ctx.fillText(String(value), margin.left - 8, y);
    }
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let i = 0; i <= 4; i += 1) {
      const value = (i / 4) * xMax;
      const x = xOf(value);
      ctx.beginPath();
      ctx.moveTo(x, margin.top);
      ctx.lineTo(x, h - margin.bottom);
      ctx.strokeStyle = grid;
      ctx.stroke();
      ctx.fillStyle = textColor;
      ctx.fillText(fmt(value, 0), x, h - margin.bottom + 8);
    }

    ctx.fillStyle = textColor;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText('品質', margin.left, margin.top - 4);
    ctx.textAlign = 'right';
    ctx.fillText('tokens/sec', w - margin.right, h - 4);

    const sorted = [...summaries].sort((a, b) => a.tokens_per_sec - b.tokens_per_sec);
    const frontier = [];
    let bestY = -1;
    sorted.forEach((summary) => {
      if (summary.quality > bestY) {
        frontier.push(summary);
        bestY = summary.quality;
      }
    });
    if (frontier.length > 1) {
      ctx.beginPath();
      frontier.forEach((summary, index) => {
        const x = xOf(summary.tokens_per_sec);
        const y = yOf(summary.quality);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = accent;
      ctx.setLineDash([4, 5]);
      ctx.lineWidth = 1.2;
      ctx.globalAlpha = 0.7;
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }

    const hits = [];
    summaries.forEach((summary) => {
      const x = xOf(summary.tokens_per_sec);
      const y = yOf(summary.quality);
      const radius = 6 + (1 - clamp(summary.success_rate / 100, 0, 1)) * 4;
      const color = summary.color || '#38bdf8';
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fillStyle = hexToRgba(color, 0.85);
      ctx.fill();
      ctx.strokeStyle = bg;
      ctx.lineWidth = 1.5;
      ctx.stroke();

      if (summaries.length <= 8) {
        ctx.fillStyle = textColor;
        ctx.font = `10px ${FONT}`;
        ctx.textAlign = x > w * 0.7 ? 'right' : 'left';
        ctx.textBaseline = 'middle';
        const labelX = x > w * 0.7 ? x - radius - 5 : x + radius + 5;
        ctx.fillText(fitText(ctx, summary.model_id, 120), labelX, y);
      }

      hits.push({
        x, y, r: radius + 8,
        html: `<b>${esc(summary.model_id)}</b><span>品質 ${fmt(summary.quality, 1)} · ${fmt(summary.tokens_per_sec, 1)} tok/s</span><span>p50 ${fmtMs(summary.p50_latency_ms)} · 成功率 ${fmtPct(summary.success_rate)}</span>`
      });
    });
    state.chartHits.set(canvas, { points: hits });
  }

  function drawLatency() {
    const canvas = $('#latency');
    const summaries = [...(state.run?.summaries || [])].sort((a, b) => b.overall - a.overall);
    if (!summaries.length) {
      const { ctx, w, h } = setupCanvas(canvas);
      drawEmptyState(ctx, w, h, 'データがありません');
      state.chartHits.set(canvas, { points: [] });
      return;
    }
    const rowHeight = 42;
    const height = Math.max(220, summaries.length * rowHeight + 60);
    canvas.style.height = `${height}px`;
    const { ctx, w, h } = setupCanvas(canvas);

    const margin = { left: Math.min(150, w * 0.3), right: 78, top: 30, bottom: 30 };
    const plotW = w - margin.left - margin.right;
    const plotH = h - margin.top - margin.bottom;
    const maxLatency = Math.max(100, ...summaries.map((summary) => Number(summary.p95_latency_ms) || 0));
    const xMax = niceMax(maxLatency * 1.12);
    const textColor = cssVar('--chart-text');
    const grid = cssVar('--chart-grid');
    const axis = cssVar('--chart-axis');
    const accent = cssVar('--accent');

    const xOf = (value) => margin.left + (clamp(value, 0, xMax) / xMax) * plotW;

    ctx.font = `10px ${FONT}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let i = 0; i <= 4; i += 1) {
      const value = (i / 4) * xMax;
      const x = xOf(value);
      ctx.beginPath();
      ctx.moveTo(x, margin.top - 6);
      ctx.lineTo(x, h - margin.bottom);
      ctx.strokeStyle = grid;
      ctx.stroke();
      ctx.fillStyle = textColor;
      ctx.fillText(fmtMs(value), x, h - margin.bottom + 6);
    }

    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = textColor;
    ctx.fillRect(margin.left, 6, 8, 8);
    ctx.fillText('p50', margin.left + 13, 10);
    ctx.fillStyle = hexToRgba(accent, 0.4);
    ctx.fillRect(margin.left + 46, 6, 8, 8);
    ctx.fillStyle = textColor;
    ctx.fillText('p95', margin.left + 59, 10);
    ctx.textAlign = 'right';
    ctx.fillText('tok/s', w - 10, 10);

    const hits = [];
    summaries.forEach((summary, index) => {
      const centerY = margin.top + index * rowHeight + rowHeight / 2;
      const color = summary.color || '#38bdf8';
      const labelMax = margin.left - 18;
      ctx.fillStyle = textColor;
      ctx.font = `11px ${FONT}`;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(fitText(ctx, summary.model_id, labelMax), margin.left - 10, centerY);

      const p50 = clamp(Number(summary.p50_latency_ms) || 0, 0, xMax);
      const p95 = clamp(Number(summary.p95_latency_ms) || 0, 0, xMax);
      const barX = margin.left;
      const barW50 = Math.max(2, xOf(p50) - margin.left);
      const barW95 = Math.max(2, xOf(p95) - margin.left);

      roundRect(ctx, barX, centerY - 10, barW95, 8, 4);
      ctx.fillStyle = hexToRgba(color, 0.28);
      ctx.fill();

      roundRect(ctx, barX, centerY + 1, barW50, 8, 4);
      ctx.fillStyle = color;
      ctx.fill();

      ctx.fillStyle = textColor;
      ctx.font = `9.5px ${FONT}`;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'bottom';
      ctx.fillText(fmtMs(p95), barX + barW95 + 4, centerY - 2);
      ctx.textBaseline = 'top';
      ctx.fillText(fmtMs(p50), barX + barW50 + 4, centerY + 1);

      ctx.fillStyle = accent;
      ctx.font = `11px ${FONT}`;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(fmt(summary.tokens_per_sec, 1), w - 10, centerY);

      hits.push({
        x: barX, y: centerY - 12, w: Math.max(barW95, 12), h: 30,
        html: `<b>${esc(summary.model_id)}</b><span>p50 ${fmtMs(summary.p50_latency_ms)} · p95 ${fmtMs(summary.p95_latency_ms)}</span><span>平均 ${fmtMs(summary.avg_latency_ms)} · TTFT ${fmtMs(summary.avg_ttft_ms)}</span>`
      });
    });

    state.chartHits.set(canvas, { points: [], rects: hits });
  }

  function drawHistory() {
    const canvas = $('#history');
    const { ctx, w, h } = setupCanvas(canvas);
    const runs = [...state.runs].sort((a, b) => String(a.created_at).localeCompare(String(b.created_at)));
    if (!runs.length) {
      drawEmptyState(ctx, w, h, '実行履歴がありません');
      state.chartHits.set(canvas, { points: [] });
      return;
    }
    const latestModels = [...(state.run?.summaries || [])]
      .sort((a, b) => b.overall - a.overall)
      .slice(0, 5);
    if (!latestModels.length) {
      drawEmptyState(ctx, w, h, '実行履歴がありません');
      state.chartHits.set(canvas, { points: [] });
      return;
    }

    const margin = { left: 42, right: 18, top: 18, bottom: 34 };
    const plotW = w - margin.left - margin.right;
    const plotH = h - margin.top - margin.bottom;
    const textColor = cssVar('--chart-text');
    const grid = cssVar('--chart-grid');
    const bg = cssVar('--panel-solid');

    const xOf = (index) => runs.length === 1
      ? margin.left + plotW / 2
      : margin.left + (index / (runs.length - 1)) * plotW;
    const yOf = (value) => margin.top + plotH - (clamp(value, 0, 100) / 100) * plotH;

    ctx.font = `10px ${FONT}`;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= 4; i += 1) {
      const value = i * 25;
      const y = yOf(value);
      ctx.beginPath();
      ctx.moveTo(margin.left, y);
      ctx.lineTo(w - margin.right, y);
      ctx.strokeStyle = grid;
      ctx.stroke();
      ctx.fillStyle = textColor;
      ctx.fillText(String(value), margin.left - 8, y);
    }

    const step = Math.max(1, Math.ceil(runs.length / 6));
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    runs.forEach((run, index) => {
      if (index % step !== 0 && index !== runs.length - 1) return;
      ctx.fillStyle = textColor;
      ctx.fillText(fmtShortDate(run.created_at), xOf(index), h - margin.bottom + 8);
    });

    const hits = [];
    latestModels.forEach((model) => {
      const color = model.color || '#38bdf8';
      const points = [];
      runs.forEach((run, index) => {
        const entry = (run.models || []).find((item) => item.id === model.model_id);
        if (!entry) return;
        points.push({ index, value: Number(entry.overall) || 0, run });
      });
      ctx.beginPath();
      points.forEach((point, index) => {
        const x = xOf(point.index);
        const y = yOf(point.value);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.globalAlpha = 0.85;
      ctx.stroke();
      ctx.globalAlpha = 1;

      points.forEach((point) => {
        const x = xOf(point.index);
        const y = yOf(point.value);
        ctx.beginPath();
        ctx.arc(x, y, 3.4, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = bg;
        ctx.lineWidth = 1;
        ctx.stroke();
        hits.push({
          x, y, r: 12,
          html: `<b>${esc(model.model_id)}</b><span>${esc(point.run.name)} · ${fmtShortDate(point.run.created_at)}</span><span>Overall ${fmt(point.value, 1)}</span>`
        });
      });
    });

    if (runs.length === 1) {
      ctx.fillStyle = textColor;
      ctx.font = `10px ${FONT}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText('履歴が 1 件のみです。ベンチマークを繰り返すと推移が表示されます。', w / 2, margin.top + 4);
    }

    state.chartHits.set(canvas, { points: hits });
  }

  function niceMax(value) {
    if (!Number.isFinite(value) || value <= 0) return 10;
    const exponent = Math.floor(Math.log10(value));
    const fraction = value / (10 ** exponent);
    let nice;
    if (fraction <= 1) nice = 1;
    else if (fraction <= 2) nice = 2;
    else if (fraction <= 5) nice = 5;
    else nice = 10;
    return nice * (10 ** exponent);
  }

  function heatColor(value) {
    const t = clamp(value, 0, 100) / 100;
    if (t < 0.55) {
      const u = t / 0.55;
      return {
        r: Math.round(248 + (251 - 248) * u),
        g: Math.round(113 + (191 - 113) * u),
        b: Math.round(113 + (36 - 113) * u)
      };
    }
    const u = (t - 0.55) / 0.45;
    return {
      r: Math.round(251 + (52 - 251) * u),
      g: Math.round(191 + (211 - 191) * u),
      b: Math.round(36 + (153 - 36) * u)
    };
  }

  // ---------------------------------------------------------------- tooltip
  function handleCanvasMove(event) {
    const canvas = event.currentTarget;
    const hits = state.chartHits.get(canvas);
    if (!hits) return;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    let found = null;
    let best = Infinity;
    (hits.points || []).forEach((point) => {
      const distance = Math.hypot(point.x - x, point.y - y);
      if (distance <= (point.r || 14) && distance < best) {
        best = distance;
        found = point;
      }
    });
    if (!found) {
      (hits.rects || []).forEach((box) => {
        if (x >= box.x && x <= box.x + box.w && y >= box.y && y <= box.y + box.h) {
          found = box;
        }
      });
    }
    if (found) showTooltip(found.html, event.clientX, event.clientY);
    else hideTooltip();
  }

  function showTooltip(html, clientX, clientY) {
    const tooltip = $('#tooltip');
    tooltip.innerHTML = html;
    tooltip.classList.remove('hidden');
    const rect = tooltip.getBoundingClientRect();
    let left = clientX + 14;
    let top = clientY + 14;
    if (left + rect.width > window.innerWidth - 10) left = clientX - rect.width - 14;
    if (top + rect.height > window.innerHeight - 10) top = clientY - rect.height - 14;
    tooltip.style.left = `${Math.max(8, left)}px`;
    tooltip.style.top = `${Math.max(8, top)}px`;
  }

  function hideTooltip() {
    $('#tooltip').classList.add('hidden');
  }

  // ---------------------------------------------------------------- modal
  function renderConfigModels() {
    const host = $('#model-list');
    const models = state.config?.models || [];
    if (!models.length) {
      host.innerHTML = '<div class="empty">設定されたモデルがありません</div>';
      return;
    }
    host.innerHTML = models.map((model) => {
      const meta = model.provider === 'mock'
        ? `profile: ${model.profile || 'balanced'}`
        : `${model.base_url || ''} · ${model.model || ''}`;
      const price = Number(model.input_price_per_mtok) || Number(model.output_price_per_mtok)
        ? `$${fmt(model.input_price_per_mtok, 2)} / $${fmt(model.output_price_per_mtok, 2)}`
        : '無料';
      return `
        <label class="model-option">
          <input type="checkbox" value="${esc(model.id)}" ${model.enabled ? 'checked' : ''}>
          <span class="dot" style="color:${esc(model.color)};background:${esc(model.color)}"></span>
          <span>
            <div class="name">${esc(model.id)} ${providerBadge(model.provider)}</div>
            <div class="meta">${esc(meta)}</div>
          </span>
          <span class="price">${esc(price)}<br><small>per 1M tok</small></span>
        </label>
      `;
    }).join('');
  }

  function openModal() {
    if (!state.bootstrap) {
      toast('サーバーに接続できません', true);
      return;
    }
    const modal = $('#modal');
    modal.classList.remove('hidden');
    const now = new Date();
    $('#run-name').value = `Evaluation ${now.toLocaleDateString('ja-JP')}`;
    $('#run-concurrency').value = String(state.config?.concurrency || 4);
    const count = state.dataset?.count || 0;
    const select = $('#run-max-items');
    const allOption = select.querySelector('option[value="0"]');
    if (allOption) allOption.textContent = count ? `全件（${count}）` : '全件';
    if ([6, 12, 24].includes(count)) select.value = String(count);
    else select.value = count > 0 && count <= 24 ? String(count) : '0';
    if (![6, 12, 24, 0].includes(Number(select.value))) {
      const option = document.createElement('option');
      option.value = String(count);
      option.textContent = `全件（${count}）`;
      select.appendChild(option);
      select.value = String(count);
    }
    ['#custom-id', '#custom-url', '#custom-model', '#custom-key', '#custom-in', '#custom-out'].forEach((selector) => {
      $(selector).value = '';
    });
    $('#custom-provider').value = 'openai';
    setTimeout(() => $('#run-name').focus(), 30);
  }

  function closeModal() {
    $('#modal').classList.add('hidden');
  }

  async function startRun() {
    const selected = $$('#model-list input[type="checkbox"]:checked').map((input) => input.value);
    const custom = {};
    const customId = $('#custom-id').value.trim();
    const customUrl = $('#custom-url').value.trim();
    const customModel = $('#custom-model').value.trim();
    const customProvider = $('#custom-provider').value;
    if (customId || customUrl || customModel) {
      if (!customId) {
        toast('カスタムモデルの表示名を入力してください', true);
        return;
      }
      if (customProvider === 'openai' && (!customUrl || !customModel)) {
        toast('Base URL とモデル名を入力してください', true);
        return;
      }
      custom.id = customId;
      custom.provider = customProvider;
      custom.base_url = customUrl || 'http://localhost:11434/v1';
      custom.model = customModel || customId;
      custom.api_key_env = $('#custom-key').value.trim();
      custom.input_price_per_mtok = Number($('#custom-in').value) || 0;
      custom.output_price_per_mtok = Number($('#custom-out').value) || 0;
    }

    if (!selected.length && !Object.keys(custom).length) {
      toast('少なくとも1つのモデルを選択してください', true);
      return;
    }

    const body = {
      name: $('#run-name').value.trim() || 'Evaluation',
      model_ids: selected,
      concurrency: Number($('#run-concurrency').value) || 4,
      max_items: Number($('#run-max-items').value) || 0
    };
    if (Object.keys(custom).length) body.custom_models = [custom];

    const button = $('#modal-start');
    button.disabled = true;
    button.textContent = '開始中...';
    try {
      const response = await api('/api/run', { method: 'POST', body: JSON.stringify(body) });
      closeModal();
      state.job = response.job;
      state.jobHandledFor = null;
      updateJobPanel(state.job);
      startJobPolling();
      toast('ベンチマークを開始しました');
    } catch (error) {
      toast(error.message, true);
    } finally {
      button.disabled = false;
      button.textContent = '実行する';
    }
  }

  async function startDemo() {
    try {
      const response = await api('/api/demo', { method: 'POST', body: '{}' });
      state.job = response.job;
      state.jobHandledFor = null;
      updateJobPanel(state.job);
      startJobPolling();
      toast('デモ評価を開始しました');
    } catch (error) {
      toast(error.message, true);
    }
  }

  // ---------------------------------------------------------------- jobs
  function startJobPolling() {
    stopJobPolling();
    state.jobHandledFor = null;
    pollJob();
  }

  function stopJobPolling() {
    if (state.jobTimer) {
      clearTimeout(state.jobTimer);
      state.jobTimer = null;
    }
  }

  async function pollJob() {
    try {
      const job = await api('/api/job');
      state.job = job;
      updateJobPanel(job);
      if (job.state === 'running') {
        state.jobTimer = setTimeout(pollJob, 800);
      } else {
        state.jobTimer = null;
        handleTerminalJob(job);
      }
    } catch {
      state.jobTimer = setTimeout(pollJob, 2000);
    }
  }

  function handleTerminalJob(job) {
    if (state.jobHandledFor === job.id) return;
    state.jobHandledFor = job.id;
    if (job.state === 'completed') {
      toast(`完了: ${job.completed} / ${job.total} requests`);
    } else if (job.state === 'cancelled') {
      toast('評価を中止しました', true);
    } else if (job.state === 'failed') {
      toast(`失敗: ${job.message}`, true);
    }
    setTimeout(async () => {
      try {
        await refreshBootstrap(job.run_id || undefined);
      } catch (error) {
        toast(error.message, true);
      }
      updateJobPanel(job);
    }, 700);
  }

  function updateJobPanel(job) {
    const panel = $('#job-panel');
    if (!job || !job.state || job.state === 'idle') {
      panel.classList.add('hidden');
      return;
    }
    panel.classList.remove('hidden');
    const titles = {
      running: 'ベンチマーク実行中',
      completed: '評価が完了しました',
      cancelled: '評価を中止しました',
      failed: '評価に失敗しました'
    };
    $('#job-title').textContent = titles[job.state] || job.state;
    $('#job-message').textContent = job.message || '';
    const percent = clamp(Number(job.percent) || 0, 0, 100);
    $('#job-bar').style.width = `${percent}%`;
    $('#job-percent').textContent = `${percent.toFixed(0)}% · ${job.completed || 0}/${job.total || 0}`;
    $('#job-elapsed').textContent = fmtDuration(job.elapsed_ms);

    const cancelButton = $('#job-cancel');
    if (job.state === 'running') {
      cancelButton.textContent = '中止';
      cancelButton.disabled = false;
      cancelButton.dataset.action = 'cancel';
    } else {
      cancelButton.textContent = '閉じる';
      cancelButton.disabled = false;
      cancelButton.dataset.action = 'close';
    }

    const models = job.models || [];
    $('#job-models').innerHTML = models.map((model) => {
      const ratio = model.total > 0 ? clamp((model.completed / model.total) * 100, 0, 100) : 0;
      const quality = model.completed > 0 ? `Q ${fmt(model.quality, 1)}` : '—';
      return `
        <div class="job-model">
          <span class="dot" style="color:${esc(model.color)};background:${esc(model.color)};box-shadow:none"></span>
          <span class="name" title="${esc(model.id)}">${esc(model.id)}</span>
          <span class="mini-bar"><i style="width:${ratio}%;background:${esc(model.color)}"></i></span>
          <span class="count">${model.completed || 0}/${model.total || 0}</span>
          <span class="q">${quality}</span>
        </div>
      `;
    }).join('');
  }

  async function cancelOrCloseJob() {
    const button = $('#job-cancel');
    if (button.dataset.action === 'close') {
      $('#job-panel').classList.add('hidden');
      return;
    }
    button.disabled = true;
    try {
      await api('/api/job/cancel', { method: 'POST', body: '{}' });
      toast('中止を要求しました');
      setTimeout(pollJob, 300);
    } catch (error) {
      toast(error.message, true);
      button.disabled = false;
    }
  }

  // ---------------------------------------------------------------- events
  function bindEvents() {
    $('#theme-toggle').addEventListener('click', () => {
      applyTheme(state.theme === 'dark' ? 'light' : 'dark');
      if (state.run) renderCharts();
    });

    $('#run-btn').addEventListener('click', openModal);
    $('#modal-close').addEventListener('click', closeModal);
    $('#modal-cancel').addEventListener('click', closeModal);
    $('#modal-start').addEventListener('click', startRun);
    $('#modal').addEventListener('click', (event) => {
      if (event.target === $('#modal')) closeModal();
    });

    $('#run-select').addEventListener('change', (event) => loadRun(event.target.value));

    $$('#leaderboard thead th.sortable').forEach((th) => {
      th.addEventListener('click', () => {
        const key = th.dataset.sort;
        if (state.sortKey === key) state.sortDir *= -1;
        else {
          state.sortKey = key;
          state.sortDir = key === 'p50_latency_ms' || key === 'cost_per_1k_usd' ? 1 : -1;
        }
        renderLeaderboard();
      });
    });

    $('#leaderboard tbody').addEventListener('click', (event) => {
      const row = event.target.closest('tr[data-model]');
      if (!row) return;
      state.detailModel = row.dataset.model;
      renderDetailControls();
      renderDetails();
      document.querySelector('#details')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });

    $('#radar-legend').addEventListener('click', (event) => {
      const button = event.target.closest('.legend-item');
      if (!button) return;
      const id = button.dataset.model;
      if (state.radarSelection.has(id)) {
        state.radarSelection.delete(id);
      } else {
        if (state.radarSelection.size >= 6) {
          toast('比較は最大6モデルまでです', true);
          return;
        }
        state.radarSelection.add(id);
      }
      renderRadarLegend();
      drawRadar();
    });

    $('#detail-model').addEventListener('change', (event) => {
      state.detailModel = event.target.value;
      renderDetails();
    });
    $('#detail-category').addEventListener('change', (event) => {
      state.detailCategory = event.target.value;
      renderDetails();
    });
    $('#fail-only').addEventListener('change', (event) => {
      state.failOnly = event.target.checked;
      renderDetails();
    });
    $('#detail-search').addEventListener('input', debounce((event) => {
      state.search = event.target.value;
      renderDetails();
    }, 120));

    $('#job-cancel').addEventListener('click', cancelOrCloseJob);

    ['#radar', '#scatter', '#latency', '#history'].forEach((selector) => {
      const canvas = $(selector);
      canvas.addEventListener('mousemove', handleCanvasMove);
      canvas.addEventListener('mouseleave', hideTooltip);
    });

    window.addEventListener('resize', debounce(() => {
      if (state.run) renderCharts();
    }, 160));

    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') closeModal();
    });
  }

  function applyTheme(theme) {
    state.theme = theme;
    document.documentElement.dataset.theme = theme;
    localStorage.setItem('llm-eval-theme', theme);
    const button = $('#theme-toggle');
    if (button) button.textContent = theme === 'dark' ? '☀' : '☾';
  }

  function toast(message, isError = false) {
    const host = $('#toast');
    host.textContent = message;
    host.classList.toggle('error', Boolean(isError));
    host.classList.remove('hidden');
    if (state.toastTimer) clearTimeout(state.toastTimer);
    state.toastTimer = setTimeout(() => host.classList.add('hidden'), 3600);
  }
})();
