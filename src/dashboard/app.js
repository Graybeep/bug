/* ==========================================================================
   Bug Management System — interactive dashboard runtime.
   Inlined into dashboards/index.html by src/08_dashboard.py.

   The whole bug table ships with the page as base64 typed arrays (one column
   per attribute, one byte per categorical value). Every filter change runs a
   single pass over those arrays and rebuilds every KPI, chart and table on the
   active view from the filtered slice — so the numbers on screen always agree
   with each other and with the filter chips above them.

   No CDN, no charting library, no network access: charts are SVG built with
   DOM calls, and every label goes in via textContent (labels are data).
   ========================================================================== */
(function () {
'use strict';

var D = window.__BUG_DATA__;
if (!D) { return; }

/* ══ 1. Payload decode ═══════════════════════════════════════════════════ */

function bytes(b64) {
  var bin = atob(b64), n = bin.length, out = new Uint8Array(n);
  for (var i = 0; i < n; i++) out[i] = bin.charCodeAt(i);
  return out;
}
function u16(b64) { var b = bytes(b64); return new Uint16Array(b.buffer, b.byteOffset, b.length >> 1); }
function i16(b64) { var b = bytes(b64); return new Int16Array(b.buffer, b.byteOffset, b.length >> 1); }
function u32(b64) { var b = bytes(b64); return new Uint32Array(b.buffer, b.byteOffset, b.length >> 2); }

var N    = D.meta.rows;
var DIM  = D.dims;
var LOOK = D.lookups;
var COL  = {};
Object.keys(D.cols).forEach(function (k) { COL[k] = bytes(D.cols[k]); });
COL.createdDay = u16(D.u16.createdDay);
COL.resDays    = i16(D.i16.resDays);          /* -1 => still open */
var BUG_NUM    = D.ids.seq ? null : u32(D.ids.data);

var SLA        = LOOK.slaByPriority;          /* index-aligned with DIM.priority */
var KLOC       = LOOK.klocByModule;
var HAS_MODEL  = !!D.meta.modelScored;

function bugId(i) {
  var num = BUG_NUM ? BUG_NUM[i] : i + 1;
  return 'BUG_' + String(num).padStart(6, '0');
}

/* Dimensions the filter engine and the aggregator both understand. Order is
   fixed: the aggregator indexes its accumulators by position in this list. */
var DIMS = ['module', 'feature', 'component', 'priority', 'severity', 'status',
            'resolution', 'sprint', 'release', 'category', 'domain',
            'environment', 'tech', 'errorCode', 'team', 'lifecycle', 'month'];
var DIMCOL = DIMS.map(function (d) { return COL[d]; });
var DIMK   = DIMS.map(function (d) { return DIM[d].length; });
var IDX    = {}; DIMS.forEach(function (d, i) { IDX[d] = i; });

var P_URGENT = DIM.priority.map(function (p) { return p === 'P1' || p === 'P2'; });
var REOPEN_K = DIM.status.indexOf('Reopened');
var nMonth   = DIM.month.length;

/* ══ 2. Formatting helpers ═══════════════════════════════════════════════ */

function nf(v)   { return Math.round(v).toLocaleString('en-US'); }
function nf1(v)  { return (v === null || v === undefined || isNaN(v)) ? '—'
                          : Number(v).toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 }); }
function pct(v)  { return (v === null || isNaN(v)) ? '—' : nf1(v) + '%'; }
function compact(v) {
  if (v >= 1e6) return (v / 1e6).toFixed(v >= 1e7 ? 0 : 1) + 'M';
  if (v >= 1e4) return Math.round(v / 1e3) + 'K';
  return nf(v);
}
function days(v) { return v === null || isNaN(v) ? '—' : nf1(v) + 'd'; }
function dayToDate(d) {
  var t = new Date(D.meta.originMs + d * 86400000);
  return t.toISOString().slice(0, 10);
}
function slaVerdict(p) {
  if (p === null || isNaN(p)) return ['neutral', 'No data'];
  if (p >= 85) return ['good', 'On track'];
  if (p >= 70) return ['warning', 'Watch'];
  return ['critical', 'At risk'];
}
var STATUS_ICON = { good: '✓', warning: '⚠', serious: '⚠', critical: '✕', neutral: '–' };

function el(tag, cls, text) {
  var n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined && text !== null) n.textContent = text;
  return n;
}
function badge(status, label) {
  var b = el('span', 'badge badge-' + status);
  b.appendChild(el('span', null, STATUS_ICON[status] || ''));
  b.appendChild(el('span', null, label));
  return b;
}
function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

/* ══ 3. Filter state ═════════════════════════════════════════════════════ */

var FILTERABLE = ['module', 'priority', 'severity', 'status', 'release', 'team',
                  'resolution', 'component', 'feature', 'category', 'domain',
                  'environment', 'lifecycle'];
var DIM_LABEL = {
  module: 'Module', priority: 'Priority', severity: 'Severity', status: 'Status',
  release: 'Release', team: 'Owner', resolution: 'Resolution', component: 'Component',
  feature: 'Feature', category: 'Bug category', domain: 'Domain',
  environment: 'Environment', lifecycle: 'Life cycle stage', sprint: 'Sprint',
  tech: 'Tech stack', errorCode: 'Error code', month: 'Month'
};

var state = {
  sel: {},                                   /* dim -> Set of value indices */
  sprint: [0, DIM.sprint.length - 1],
  life: 'all',                               /* all | open | closed */
  view: 'overview',
  split: 'all'                               /* model view: all | test | train | unseen */
};
FILTERABLE.forEach(function (d) { state.sel[d] = new Set(); });

var mask = new Uint8Array(N);

function buildMask() {
  var lo = state.sprint[0], hi = state.sprint[1], life = state.life;
  var active = [];
  FILTERABLE.forEach(function (d) {
    if (state.sel[d].size) active.push([COL[d], state.sel[d]]);
  });
  var sprintCol = COL.sprint, resCol = COL.resDays;
  var wholeSprint = (lo === 0 && hi === DIM.sprint.length - 1);
  for (var i = 0; i < N; i++) {
    var ok = 1;
    if (!wholeSprint) { var s = sprintCol[i]; if (s < lo || s > hi) ok = 0; }
    if (ok && life !== 'all') {
      var closed = resCol[i] >= 0;
      if (life === 'open' ? closed : !closed) ok = 0;
    }
    for (var a = 0; ok && a < active.length; a++) {
      if (!active[a][1].has(active[a][0][i])) ok = 0;
    }
    mask[i] = ok;
  }
}

function anyFilter() {
  if (state.life !== 'all') return true;
  if (state.sprint[0] !== 0 || state.sprint[1] !== DIM.sprint.length - 1) return true;
  return FILTERABLE.some(function (d) { return state.sel[d].size > 0; });
}

function toggleFilter(dim, k) {
  var s = state.sel[dim];
  if (s.has(k)) s.delete(k); else s.add(k);
  recompute();
}
function resetFilters() {
  FILTERABLE.forEach(function (d) { state.sel[d].clear(); });
  state.sprint = [0, DIM.sprint.length - 1];
  state.life = 'all';
  syncControls();
  recompute();
}

/* ══ 4. Aggregation — one pass over the filtered slice ═══════════════════ */

var A = null;

function aggregate() {
  var nd = DIMS.length, d, k, i;
  var cnt = [], cls = [], opn = [], res = [], urg = [], met = [];
  for (d = 0; d < nd; d++) {
    cnt.push(new Int32Array(DIMK[d]));
    cls.push(new Int32Array(DIMK[d]));
    opn.push(new Int32Array(DIMK[d]));
    res.push(new Float64Array(DIMK[d]));
    urg.push(new Int32Array(DIMK[d]));
    met.push(new Int32Array(DIMK[d]));
  }

  var nS = DIM.sprint.length, nP = DIM.priority.length, nM = DIM.module.length;
  var sprOpened = new Int32Array(nS), sprClosed = new Int32Array(nS);
  var sprResSum = new Float64Array(nS), sprResN = new Int32Array(nS);
  var monOpened = new Int32Array(nMonth), monClosed = new Int32Array(nMonth);
  var modPri    = new Int32Array(nM * nP);
  var resHist   = new Int32Array(RES_BINS.length);
  var ageHist   = new Int32Array(AGE_BINS.length);
  var confusion = new Int32Array(nP * nP);
  var confHist  = new Int32Array(10);
  var escalate  = [];

  var total = 0, closed = 0, slaMet = 0, resSum = 0, openLate = 0, ageSum = 0, reopen = 0;
  var resList = [];
  var monCol = COL.month, cmonCol = COL.closeMonth, sprCol = COL.sprint,
      cspCol = COL.closeSprint, priCol = COL.priority, modCol = COL.module,
      stCol = COL.status, resCol = COL.resDays, cdCol = COL.createdDay,
      pPri = COL.predPriority, pConf = COL.predConf, splitCol = COL.split;
  var snapDay = D.meta.snapshotDay;
  var splitWant = state.split === 'test' ? 2 : state.split === 'train' ? 1 : state.split === 'unseen' ? 0 : -1;

  for (i = 0; i < N; i++) {
    if (!mask[i]) continue;
    total++;
    var p = priCol[i], rd = resCol[i], isClosed = rd >= 0, urgent = P_URGENT[p];
    var slaOk = false;

    if (isClosed) {
      closed++;
      resSum += rd; resList.push(rd);
      slaOk = rd <= SLA[p];
      if (slaOk) slaMet++;
      sprClosed[cspCol[i]]++;
      sprResSum[cspCol[i]] += rd; sprResN[cspCol[i]]++;
      if (cmonCol[i] !== 255) monClosed[cmonCol[i]]++;
      resHist[binOf(RES_BINS, rd)]++;
    } else {
      var age = snapDay - cdCol[i];
      ageSum += age;
      if (age > SLA[p]) openLate++;
      ageHist[binOf(AGE_BINS, age)]++;
    }
    if (stCol[i] === REOPEN_K) reopen++;

    sprOpened[sprCol[i]]++;
    monOpened[monCol[i]]++;
    modPri[modCol[i] * nP + p]++;

    for (d = 0; d < nd; d++) {
      k = DIMCOL[d][i];
      cnt[d][k]++;
      if (isClosed) { cls[d][k]++; res[d][k] += rd; if (slaOk) met[d][k]++; }
      else opn[d][k]++;
      if (urgent) urg[d][k]++;
    }

    if (HAS_MODEL && (splitWant < 0 || splitCol[i] === splitWant)) {
      confusion[p * nP + pPri[i]]++;
      confHist[Math.min(9, Math.floor(pConf[i] / 10))]++;
      if (!isClosed && pPri[i] < p) escalate.push(i);
    }
  }

  resList.sort(function (a, b) { return a - b; });
  function q(f) { return resList.length ? resList[Math.min(resList.length - 1, Math.floor(resList.length * f))] : null; }

  /* Defect density only counts the KLOC of modules actually inside the slice —
     otherwise filtering to one module would divide by the whole codebase. */
  var kloc = 0;
  for (k = 0; k < nM; k++) if (cnt[IDX.module][k] > 0) kloc += KLOC[k];

  var backlog = new Int32Array(nS), run = 0;
  for (k = 0; k < nS; k++) { run += sprOpened[k] - sprClosed[k]; backlog[k] = run; }

  A = {
    cnt: cnt, cls: cls, opn: opn, res: res, urg: urg, met: met,
    total: total, closed: closed, open: total - closed,
    closeRate: total ? closed / total * 100 : 0,
    avgRes: closed ? resSum / closed : null,
    medRes: q(0.5), p90Res: q(0.9),
    slaPct: closed ? slaMet / closed * 100 : null,
    openLate: openLate,
    avgAge: (total - closed) ? ageSum / (total - closed) : null,
    reopenPct: total ? reopen / total * 100 : 0,
    kloc: kloc, density: kloc ? total / kloc : 0,
    sprOpened: sprOpened, sprClosed: sprClosed, backlog: backlog,
    sprAvgRes: Array.from(sprResN, function (n2, j) { return n2 ? sprResSum[j] / n2 : null; }),
    monOpened: monOpened, monClosed: monClosed,
    modPri: modPri, resHist: resHist, ageHist: ageHist,
    confusion: confusion, confHist: confHist, escalate: escalate
  };
}

var RES_BINS = [[0, 1], [2, 3], [4, 7], [8, 14], [15, 30], [31, 60], [61, 90], [91, 1e9]];
var AGE_BINS = [[0, 7], [8, 30], [31, 60], [61, 90], [91, 180], [181, 270], [271, 1e9]];
function binOf(bins, v) {
  for (var i = 0; i < bins.length; i++) if (v <= bins[i][1]) return i;
  return bins.length - 1;
}
function binLabel(b) { return b[1] >= 1e9 ? b[0] + '+' : (b[0] === b[1] ? String(b[0]) : b[0] + '–' + b[1]); }

/* Convenience readers over the aggregate */
function counts(dim)     { return A.cnt[IDX[dim]]; }
function closedOf(dim)   { return A.cls[IDX[dim]]; }
function openOf(dim)     { return A.opn[IDX[dim]]; }
function urgentOf(dim)   { return A.urg[IDX[dim]]; }
function avgDaysOf(dim, k) { var c = A.cls[IDX[dim]][k]; return c ? A.res[IDX[dim]][k] / c : null; }
function slaPctOf(dim, k)  { var c = A.cls[IDX[dim]][k]; return c ? A.met[IDX[dim]][k] / c * 100 : null; }

/* ranked(dim) -> [{k, label, value, ...}] sorted desc, zero rows dropped */
function ranked(dim, limit) {
  var c = counts(dim), out = [];
  for (var k = 0; k < c.length; k++) if (c[k] > 0) out.push({ k: k, label: DIM[dim][k], value: c[k] });
  out.sort(function (a, b) { return b.value - a.value; });
  return limit ? out.slice(0, limit) : out;
}
/* ordered(dim) keeps the dimension's own order (workflow states, P1..P5) */
function ordered(dim) {
  var c = counts(dim), out = [];
  for (var k = 0; k < c.length; k++) out.push({ k: k, label: DIM[dim][k], value: c[k] });
  return out;
}

/* ══ 5. SVG chart toolkit ════════════════════════════════════════════════ */

var NS = 'http://www.w3.org/2000/svg';
function s(tag, attrs) {
  var n = document.createElementNS(NS, tag);
  if (attrs) for (var k in attrs) n.setAttribute(k, attrs[k]);
  return n;
}
function stext(x, y, str, cls, anchor) {
  var t = s('text', { x: x, y: y, class: cls || 'tick-text' });
  if (anchor) t.setAttribute('text-anchor', anchor);
  t.textContent = str;
  return t;
}
function seriesColor(slot) {
  if (!slot) return 'var(--ink-soft)';
  return 'var(--series-' + (((slot - 1) % 8) + 1) + ')';
}

/* Rounded data-end, square at the baseline (4px radius, capped by the mark). */
function barPath(x, y, w, h, r, dir) {
  r = Math.max(0, Math.min(r, w / 2, h));
  if (h <= 0.4) return 'M' + x + ',' + y + 'h' + w;
  if (dir === 'h') {   /* grows right: rounded right edge */
    return 'M' + x + ',' + y + 'H' + (x + w - r) + 'Q' + (x + w) + ',' + y + ' ' + (x + w) + ',' + (y + r) +
           'V' + (y + h - r) + 'Q' + (x + w) + ',' + (y + h) + ' ' + (x + w - r) + ',' + (y + h) + 'H' + x + 'Z';
  }
  return 'M' + x + ',' + (y + h) + 'V' + (y + r) + 'Q' + x + ',' + y + ' ' + (x + r) + ',' + y +
         'H' + (x + w - r) + 'Q' + (x + w) + ',' + y + ' ' + (x + w) + ',' + (y + r) + 'V' + (y + h) + 'Z';
}

function niceMax(v, ticks) {
  if (v <= 0) return 1;
  var raw = v / (ticks || 4);
  var mag = Math.pow(10, Math.floor(Math.log10(raw)));
  var norm = raw / mag;
  var step = norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 2.5 ? 2.5 : norm <= 5 ? 5 : 10;
  return step * mag * (ticks || 4);
}

/* ── Tooltip singleton ───────────────────────────────────────────────── */
var tipEl = null;
function tip() {
  if (!tipEl) { tipEl = el('div', 'tip'); tipEl.setAttribute('role', 'status'); document.body.appendChild(tipEl); }
  return tipEl;
}
function tipShow(ev, spec) {
  var t = tip();
  clear(t);
  if (spec.title) t.appendChild(el('div', 'tip-title', spec.title));
  (spec.rows || []).forEach(function (r) {
    var row = el('div', 'tip-row');
    if (r.color) { var key = el('span', 'tip-key'); key.style.background = r.color; row.appendChild(key); }
    row.appendChild(el('span', 'tip-name', r.name));
    row.appendChild(el('span', 'tip-val', r.value));
    t.appendChild(row);
  });
  if (spec.foot) t.appendChild(el('div', 'tip-foot', spec.foot));
  t.classList.add('show');
  var pad = 14, w = t.offsetWidth, h = t.offsetHeight;
  var x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + w > window.innerWidth - 8) x = ev.clientX - w - pad;
  if (y + h > window.innerHeight - 8) y = ev.clientY - h - pad;
  t.style.left = Math.max(8, x) + 'px';
  t.style.top = Math.max(8, y) + 'px';
}
function tipHide() { if (tipEl) tipEl.classList.remove('show'); }
document.addEventListener('scroll', tipHide, true);

/* ── Column chart (grouped or stacked) ───────────────────────────────── */
function columnChart(host, cfg) {
  var W = Math.max(320, host.clientWidth || 640);
  var H = cfg.height || 260;
  var padL = cfg.padL || 52, padR = 14, padT = 12, padB = cfg.padB || 40;
  var series = cfg.series.filter(function (se) { return !se.off; });
  var n = cfg.labels.length;
  if (!n || !series.length) { host.appendChild(el('p', 'note', 'No data in the current filter.')); return; }

  var stacked = !!cfg.stacked;
  var maxV = 0, i, j;
  for (i = 0; i < n; i++) {
    if (stacked) { var sum = 0; for (j = 0; j < series.length; j++) sum += series[j].values[i] || 0; maxV = Math.max(maxV, sum); }
    else for (j = 0; j < series.length; j++) maxV = Math.max(maxV, series[j].values[i] || 0);
  }
  (cfg.thresholds || []).forEach(function (th) { th.values.forEach(function (v) { maxV = Math.max(maxV, v || 0); }); });
  var top = niceMax(maxV, 4);
  var plotW = W - padL - padR, plotH = H - padT - padB;
  var y = function (v) { return padT + plotH - (v / top) * plotH; };
  var band = plotW / n;
  var svg = s('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, role: 'img' });
  svg.setAttribute('aria-label', cfg.aria || cfg.title || 'column chart');

  for (i = 0; i <= 4; i++) {
    var gv = top * i / 4, gy = y(gv);
    svg.appendChild(s('line', { class: 'grid-line', x1: padL, x2: W - padR, y1: gy, y2: gy }));
    svg.appendChild(stext(padL - 8, gy + 3.5, compact(gv), 'tick-text', 'end'));
  }
  svg.appendChild(s('line', { class: 'axis-line', x1: padL, x2: W - padR, y1: y(0), y2: y(0) }));

  var GAP = 2;
  var slotW = stacked ? Math.min(26, band - 10) : Math.min(24, (band - 10 - GAP * (series.length - 1)) / series.length);
  slotW = Math.max(3, slotW);
  var groupW = stacked ? slotW : slotW * series.length + GAP * (series.length - 1);

  for (i = 0; i < n; i++) {
    var cx = padL + band * i + band / 2;
    var acc = 0;
    for (j = 0; j < series.length; j++) {
      var se = series[j], v = se.values[i] || 0;
      var bx = stacked ? cx - slotW / 2 : cx - groupW / 2 + j * (slotW + GAP);
      var bh, by;
      if (stacked) { bh = (v / top) * plotH; by = y(acc + v); acc += v; if (j > 0) { by += GAP; bh = Math.max(0, bh - GAP); } }
      else { bh = Math.max(0, y(0) - y(v)); by = y(v); }
      /* only the data-end of the whole column is rounded, not each segment */
      var radius = (!stacked || j === series.length - 1) ? 4 : 0;
      svg.appendChild(s('path', { d: barPath(bx, by, slotW, bh, radius, 'v'),
        fill: seriesColor(se.slot), class: 'mark' }));
    }
    /* one transparent hit target per category — bigger than any single bar */
    var hit = s('rect', { x: padL + band * i, y: padT, width: band, height: plotH, class: 'hit' });
    (function (idx, rect) {
      rect.addEventListener('pointermove', function (ev) {
        tipShow(ev, {
          title: cfg.labels[idx],
          rows: series.map(function (se) {
            return { name: se.name, value: (cfg.valueFmt || nf)(se.values[idx] || 0), color: seriesColor(se.slot) };
          }).concat((cfg.thresholds || []).map(function (th) {
            return { name: th.name, value: (cfg.valueFmt || nf)(th.values[idx]), color: 'var(--ink-soft)' };
          })),
          foot: cfg.tipFoot ? cfg.tipFoot(idx) : (cfg.onSelect ? 'Click to filter' : null)
        });
      });
      rect.addEventListener('pointerleave', tipHide);
      if (cfg.onSelect) rect.addEventListener('click', function () { tipHide(); cfg.onSelect(idx); });
    })(i, hit);
    svg.appendChild(hit);

    if (cfg.selected && cfg.selected.has(i)) {
      svg.appendChild(s('rect', { x: padL + band * i + 1, y: padT, width: band - 2, height: plotH,
        fill: 'none', stroke: 'var(--accent)', 'stroke-width': 1.5, rx: 5 }));
    }
  }

  /* SLA-style threshold: a short rule across each column, not a second axis */
  (cfg.thresholds || []).forEach(function (th) {
    for (i = 0; i < n; i++) {
      if (th.values[i] === null || th.values[i] === undefined) continue;
      var cx2 = padL + band * i + band / 2, ty = y(th.values[i]);
      svg.appendChild(s('line', { class: 'threshold', x1: cx2 - groupW / 2 - 3, x2: cx2 + groupW / 2 + 3, y1: ty, y2: ty,
        'stroke-dasharray': '5 4' }));
    }
  });

  /* Selective direct labels: only the extreme column of the first series */
  if (cfg.labelExtreme !== false && series.length && n > 1) {
    var best = 0;
    for (i = 1; i < n; i++) if ((series[0].values[i] || 0) > (series[0].values[best] || 0)) best = i;
    var bv = series[0].values[best] || 0;
    if (bv > 0) {
      var lx = padL + band * best + band / 2;
      svg.appendChild(stext(lx, y(stacked ? bv : bv) - 7, (cfg.valueFmt || nf)(bv), 'value-label', 'middle'));
    }
  }

  tickLabels(svg, cfg, n, H - padB + 15, function (i2) { return padL + band * i2 + band / 2; }, plotW);
  if (cfg.yTitle) {
    var yt = stext(0, 0, cfg.yTitle, 'axis-title', 'middle');
    yt.setAttribute('transform', 'translate(11,' + (padT + plotH / 2) + ') rotate(-90)');
    svg.appendChild(yt);
  }
  host.appendChild(svg);
}

/* Stepped x-axis labels that always include the last category — but drop the
   stepped label before it when the two would overlap, which is what produces
   the classic "SPR-25SPR-26" smear at the right edge. */
function tickLabels(svg, cfg, n, y, xOf, plotW) {
  var step = Math.max(1, Math.ceil(n / (cfg.maxTicks || Math.max(4, Math.floor(plotW / 64)))));
  var picks = [];
  for (var i = 0; i < n; i += step) picks.push(i);
  if (picks[picks.length - 1] !== n - 1) picks.push(n - 1);

  var minGap = 0;
  for (var j = 0; j < n; j++) minGap = Math.max(minGap, String(cfg.labels[j]).length);
  minGap = minGap * 6.4 + 8;                       /* ~6.4px per char at 10.5px */

  while (picks.length > 2 && xOf(picks[picks.length - 1]) - xOf(picks[picks.length - 2]) < minGap) {
    picks.splice(picks.length - 2, 1);
  }
  picks.forEach(function (i2) {
    svg.appendChild(stext(xOf(i2), y, cfg.labels[i2], 'tick-text', 'middle'));
  });
}

/* ── Line / area chart with crosshair ────────────────────────────────── */
function lineChart(host, cfg) {
  var W = Math.max(320, host.clientWidth || 640);
  var H = cfg.height || 250;
  var padL = cfg.padL || 52, padR = 16, padT = 12, padB = cfg.padB || 38;
  var series = cfg.series.filter(function (se) { return !se.off; });
  var n = cfg.labels.length;
  if (!n || !series.length) { host.appendChild(el('p', 'note', 'No data in the current filter.')); return; }

  var maxV = 0, minV = 0, i, j;
  series.forEach(function (se) { se.values.forEach(function (v) { if (v !== null && !isNaN(v)) { maxV = Math.max(maxV, v); minV = Math.min(minV, v); } }); });
  var top = niceMax(maxV, 4);
  var plotW = W - padL - padR, plotH = H - padT - padB;
  var x = function (i2) { return n === 1 ? padL + plotW / 2 : padL + (i2 / (n - 1)) * plotW; };
  var y = function (v) { return padT + plotH - (v / top) * plotH; };
  var svg = s('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, role: 'img' });
  svg.setAttribute('aria-label', cfg.aria || cfg.title || 'line chart');

  for (i = 0; i <= 4; i++) {
    var gv = top * i / 4, gy = y(gv);
    svg.appendChild(s('line', { class: 'grid-line', x1: padL, x2: W - padR, y1: gy, y2: gy }));
    svg.appendChild(stext(padL - 8, gy + 3.5, compact(gv), 'tick-text', 'end'));
  }
  svg.appendChild(s('line', { class: 'axis-line', x1: padL, x2: W - padR, y1: y(0), y2: y(0) }));

  series.forEach(function (se) {
    var col = seriesColor(se.slot), pts = [], area = [];
    for (i = 0; i < n; i++) {
      var v = se.values[i];
      if (v === null || v === undefined || isNaN(v)) continue;
      pts.push(x(i) + ',' + y(v));
      area.push((area.length ? 'L' : 'M') + x(i) + ',' + y(v));
    }
    if (!pts.length) return;
    if (se.area) {
      var firstX = x(0), lastX = x(n - 1);
      svg.appendChild(s('path', { d: area.join('') + 'L' + lastX + ',' + y(0) + 'L' + firstX + ',' + y(0) + 'Z',
        fill: col, opacity: 0.1 }));
    }
    svg.appendChild(s('polyline', { points: pts.join(' '), fill: 'none', stroke: col, 'stroke-width': 2,
      'stroke-linejoin': 'round', 'stroke-linecap': 'round', 'stroke-dasharray': se.dash || 'none', class: 'mark' }));
    /* end marker + 2px surface ring, selectively direct-labelled */
    var lastIdx = -1;
    for (i = n - 1; i >= 0; i--) { var lv = se.values[i]; if (lv !== null && lv !== undefined && !isNaN(lv)) { lastIdx = i; break; } }
    if (lastIdx >= 0) {
      svg.appendChild(s('circle', { cx: x(lastIdx), cy: y(se.values[lastIdx]), r: 4.5, fill: col,
        stroke: 'var(--surface)', 'stroke-width': 2 }));
      if (cfg.labelEnds !== false) {
        svg.appendChild(stext(x(lastIdx) - 8, y(se.values[lastIdx]) - 9,
          (cfg.valueFmt || nf)(se.values[lastIdx]), 'value-label', 'end'));
      }
    }
  });

  var cross = s('line', { class: 'crosshair', y1: padT, y2: padT + plotH, x1: -99, x2: -99 });
  svg.appendChild(cross);
  var overlay = s('rect', { x: padL, y: padT, width: plotW, height: plotH, class: 'hit' });
  overlay.addEventListener('pointermove', function (ev) {
    var box = svg.getBoundingClientRect();
    var rel = (ev.clientX - box.left - padL) / (plotW || 1);
    var idx = Math.max(0, Math.min(n - 1, Math.round(rel * (n - 1))));
    cross.setAttribute('x1', x(idx)); cross.setAttribute('x2', x(idx));
    tipShow(ev, {
      title: cfg.labels[idx],
      rows: series.map(function (se) {
        var v = se.values[idx];
        return { name: se.name, value: (v === null || v === undefined || isNaN(v)) ? '—' : (cfg.valueFmt || nf)(v), color: seriesColor(se.slot) };
      }),
      foot: cfg.tipFoot ? cfg.tipFoot(idx) : null
    });
  });
  overlay.addEventListener('pointerleave', function () { cross.setAttribute('x1', -99); cross.setAttribute('x2', -99); tipHide(); });
  svg.appendChild(overlay);

  tickLabels(svg, cfg, n, H - padB + 15, x, plotW);
  if (cfg.yTitle) {
    var yt = stext(0, 0, cfg.yTitle, 'axis-title', 'middle');
    yt.setAttribute('transform', 'translate(11,' + (padT + plotH / 2) + ') rotate(-90)');
    svg.appendChild(yt);
  }
  host.appendChild(svg);
}

/* ── Ranked horizontal bars (one series, one color) ──────────────────── */
function rankedBars(host, cfg) {
  var rows = cfg.rows;
  if (!rows.length) { host.appendChild(el('p', 'note', 'No data in the current filter.')); return; }
  var W = Math.max(300, host.clientWidth || 560);
  var rowH = cfg.rowH || 26, padT = 4, padB = 4;
  var labelW = Math.min(cfg.labelW || 150, Math.round(W * 0.42));
  var valueW = cfg.valueW || 92;
  var H = padT + padB + rows.length * rowH;
  var plotW = Math.max(40, W - labelW - valueW - 12);
  var maxV = Math.max.apply(null, rows.map(function (r) { return r.value; })) || 1;
  var svg = s('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, role: 'img' });
  svg.setAttribute('aria-label', cfg.aria || cfg.title || 'ranked bars');

  rows.forEach(function (r, i) {
    var yTop = padT + i * rowH, barH = Math.min(16, rowH - 10);
    var by = yTop + (rowH - barH) / 2;
    var w = Math.max(1, (r.value / maxV) * plotW);
    var lab = stext(labelW - 10, by + barH / 2 + 4, r.label, 'value-label-soft', 'end');
    lab.setAttribute('fill', 'var(--ink-soft)');
    svg.appendChild(lab);
    svg.appendChild(s('path', { d: barPath(labelW, by, w, barH, 4, 'h'),
      fill: seriesColor(cfg.slot || 1), class: 'mark' + (cfg.selected && cfg.selected.has(r.k) ? ' hot' : '') }));
    svg.appendChild(stext(labelW + w + 8, by + barH / 2 + 4, (cfg.valueFmt || nf)(r.value), 'value-label'));
    if (r.sub) svg.appendChild(stext(W - 4, by + barH / 2 + 4, r.sub, 'value-label-soft', 'end'));

    var hit = s('rect', { x: 0, y: yTop, width: W, height: rowH, class: 'hit' });
    hit.addEventListener('pointermove', function (ev) {
      tipShow(ev, {
        title: r.label,
        rows: (cfg.tipRows ? cfg.tipRows(r) : [{ name: cfg.measure || 'Bugs', value: nf(r.value), color: seriesColor(cfg.slot || 1) }]),
        foot: cfg.onSelect ? 'Click to filter' : null
      });
    });
    hit.addEventListener('pointerleave', tipHide);
    if (cfg.onSelect) hit.addEventListener('click', function () { tipHide(); cfg.onSelect(r.k); });
    svg.appendChild(hit);
  });
  host.appendChild(svg);
}

/* ── Heatmap (one-hue sequential ramp + scale legend) ─────────────────── */
var SEQ = ['var(--seq-1)', 'var(--seq-2)', 'var(--seq-3)', 'var(--seq-4)', 'var(--seq-5)', 'var(--seq-6)', 'var(--seq-7)'];
function heatmap(host, cfg) {
  var rowsN = cfg.rowLabels.length, colsN = cfg.colLabels.length;
  if (!rowsN || !colsN) { host.appendChild(el('p', 'note', 'No data in the current filter.')); return; }
  var W = Math.max(320, host.clientWidth || 620);
  var labelW = Math.min(cfg.labelW || 138, Math.round(W * 0.36));
  var cellH = cfg.cellH || 30, padT = 22, padB = 6, GAP = 2;
  var H = padT + padB + rowsN * cellH;
  var cellW = (W - labelW - 4) / colsN;
  var maxV = 0;
  cfg.values.forEach(function (v) { maxV = Math.max(maxV, v); });
  var svg = s('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, role: 'img' });
  svg.setAttribute('aria-label', cfg.aria || 'heatmap');

  cfg.colLabels.forEach(function (cl, c) {
    svg.appendChild(stext(labelW + cellW * c + cellW / 2, 13, cl, 'tick-text', 'middle'));
  });
  cfg.rowLabels.forEach(function (rl, r) {
    var yTop = padT + r * cellH;
    var lab = stext(labelW - 10, yTop + cellH / 2 + 4, rl, 'value-label-soft', 'end');
    lab.setAttribute('fill', 'var(--ink-soft)');
    svg.appendChild(lab);
    cfg.colLabels.forEach(function (cl, c) {
      var v = cfg.values[r * colsN + c] || 0;
      var t = maxV ? v / maxV : 0;
      var step = v === 0 ? -1 : Math.min(SEQ.length - 1, Math.floor(t * SEQ.length));
      var cx = labelW + cellW * c, cy = yTop;
      svg.appendChild(s('rect', { x: cx + GAP / 2, y: cy + GAP / 2, width: cellW - GAP, height: cellH - GAP, rx: 4,
        fill: step < 0 ? 'var(--track)' : SEQ[step], class: 'mark' }));
      /* label inside the cell only when it fits; ink flips on dark steps */
      var txt = compact(v);
      if (cellW - GAP > txt.length * 7.4 + 10) {
        var tn = stext(cx + cellW / 2, cy + cellH / 2 + 4, txt, null, 'middle');
        tn.setAttribute('fill', step >= 4 ? '#ffffff' : (step < 0 ? 'var(--ink-faint)' : '#0b0b0b'));
        tn.setAttribute('font-size', '11');
        tn.setAttribute('font-weight', '700');
        svg.appendChild(tn);
      }
      var hit = s('rect', { x: cx, y: cy, width: cellW, height: cellH, class: 'hit' });
      hit.addEventListener('pointermove', function (ev) {
        tipShow(ev, { title: rl + ' · ' + cl,
          rows: [{ name: cfg.measure || 'Bugs', value: nf(v), color: 'var(--seq-4)' }],
          foot: cfg.onSelect ? 'Click to filter' : null });
      });
      hit.addEventListener('pointerleave', tipHide);
      if (cfg.onSelect) hit.addEventListener('click', function () { tipHide(); cfg.onSelect(r, c); });
      svg.appendChild(hit);
    });
  });

  var leg = el('div', 'scale-legend');
  leg.appendChild(el('span', null, '0'));
  var ramp = el('div', 'ramp');
  SEQ.forEach(function (c) { var i2 = el('i'); i2.style.background = c; ramp.appendChild(i2); });
  leg.appendChild(ramp);
  leg.appendChild(el('span', null, nf(maxV)));
  leg.appendChild(el('span', null, cfg.measure || 'bugs'));
  host.appendChild(leg);
  host.appendChild(svg);
}

/* ── Legend (identity is never color-alone) ──────────────────────────── */
function legend(host, items, onToggle) {
  if (items.length < 2) return;
  var box = el('div', 'legend');
  items.forEach(function (se, i) {
    var b = el('button');
    b.type = 'button';
    b.setAttribute('aria-pressed', se.off ? 'false' : 'true');
    var key = el('span', se.dash ? 'key-dash' : (se.kind === 'line' ? 'key-line' : 'key-rect'));
    if (!se.dash) key.style.background = seriesColor(se.slot);
    else key.style.color = 'var(--ink-soft)';
    b.appendChild(key);
    b.appendChild(el('span', null, se.name));
    if (onToggle && !se.dash) b.addEventListener('click', function () { onToggle(i); });
    else b.disabled = true;
    box.appendChild(b);
  });
  host.appendChild(box);
}

/* ══ 6. Card scaffolding + chart/table twin ═════════════════════════════ */

function card(parent, opts) {
  var c = el('div', 'card' + (opts.span ? ' span-2' : ''));
  var head = el('div', 'card-head');
  var titles = el('div', 'titles');
  titles.appendChild(el('h2', null, opts.title));
  /* Always present, even when empty: several cards rewrite their own note on
     every update, and a conditionally-absent node would break that. */
  titles.appendChild(el('p', 'note', opts.note || ''));
  head.appendChild(titles);
  c.appendChild(head);
  var body = el('div');
  c.appendChild(body);
  parent.appendChild(c);
  return { card: c, head: head, body: body };
}

/* A chart card carries a table-view twin so no value is gated behind hover. */
function chartCard(parent, opts) {
  var c = card(parent, opts);
  var swap = el('div', 'viewswap');
  var bChart = el('button', null, 'Chart'); bChart.type = 'button'; bChart.setAttribute('aria-pressed', 'true');
  var bTable = el('button', null, 'Table'); bTable.type = 'button'; bTable.setAttribute('aria-pressed', 'false');
  swap.appendChild(bChart); swap.appendChild(bTable);
  c.head.appendChild(swap);

  var legendHost = el('div');
  var chartHost = el('div', 'chart');
  var tableHost = el('div', 'table-wrap');
  tableHost.hidden = true;
  c.body.appendChild(legendHost);
  c.body.appendChild(chartHost);
  c.body.appendChild(tableHost);

  var mode = 'chart';
  bChart.addEventListener('click', function () { mode = 'chart'; sync(); });
  bTable.addEventListener('click', function () { mode = 'table'; sync(); });
  function sync() {
    bChart.setAttribute('aria-pressed', mode === 'chart');
    bTable.setAttribute('aria-pressed', mode === 'table');
    chartHost.hidden = mode !== 'chart';
    legendHost.hidden = mode !== 'chart';
    tableHost.hidden = mode !== 'table';
  }
  return {
    card: c.card, head: c.head, legendHost: legendHost, chartHost: chartHost, tableHost: tableHost,
    isTable: function () { return mode === 'table'; }
  };
}

/* Table twin generated from the same labels + series a chart was given. */
function twinTable(host, labels, series, opts) {
  clear(host);
  opts = opts || {};
  var t = el('table');
  var thead = el('thead'), hr = el('tr');
  hr.appendChild(el('th', null, opts.dimName || 'Category'));
  series.forEach(function (se) { hr.appendChild(el('th', null, se.name)); });
  thead.appendChild(hr); t.appendChild(thead);
  var tb = el('tbody');
  labels.forEach(function (lab, i) {
    var tr = el('tr');
    tr.appendChild(el('td', null, lab));
    series.forEach(function (se) {
      var v = se.values[i];
      tr.appendChild(el('td', null, (v === null || v === undefined || isNaN(v)) ? '—' : (opts.valueFmt || nf)(v)));
    });
    tb.appendChild(tr);
  });
  t.appendChild(tb);
  host.appendChild(t);
}

function renderTwin(cc, cfg, kind) {
  clear(cc.chartHost); clear(cc.legendHost);
  var series = cfg.series || [];
  var marks = (cfg.thresholds || []).map(function (t) { return { name: t.name, dash: true, values: t.values }; });
  legend(cc.legendHost, series.concat(marks), function (i) {
    if (i >= series.length) return;
    series[i].off = !series[i].off;
    renderTwin(cc, cfg, kind);
  });
  if (kind === 'lines') lineChart(cc.chartHost, cfg);
  else columnChart(cc.chartHost, cfg);
  twinTable(cc.tableHost, cfg.labels, series.concat(marks), { dimName: cfg.dimName, valueFmt: cfg.valueFmt });
}

/* Sortable table builder used by every ranked table on the page. */
function dataTable(host, spec) {
  clear(host);
  var t = el('table');
  var thead = el('thead'), hr = el('tr');
  spec.columns.forEach(function (col, ci) {
    var th = el('th', null, col.label);
    th.tabIndex = 0;
    var apply = function () {
      var asc = spec.sortCol === ci ? !spec.sortAsc : !!col.ascFirst;
      spec.sortCol = ci; spec.sortAsc = asc;
      dataTable(host, spec);
    };
    th.addEventListener('click', apply);
    th.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); apply(); } });
    if (spec.sortCol === ci) th.className = 'sorted' + (spec.sortAsc ? ' sorted-asc' : '');
    hr.appendChild(th);
  });
  thead.appendChild(hr); t.appendChild(thead);

  var rows = spec.rows.slice();
  if (spec.sortCol !== undefined && spec.sortCol !== null) {
    var col = spec.columns[spec.sortCol], dir = spec.sortAsc ? 1 : -1;
    rows.sort(function (a, b) {
      var av = col.sort ? col.sort(a) : col.value(a), bv = col.sort ? col.sort(b) : col.value(b);
      if (av === null || av === undefined) av = -Infinity;
      if (bv === null || bv === undefined) bv = -Infinity;
      if (typeof av === 'string' || typeof bv === 'string') return dir * String(av).localeCompare(String(bv));
      return dir * (av - bv);
    });
  }
  var tb = el('tbody');
  rows.forEach(function (r) {
    var tr = el('tr');
    if (spec.isActive && spec.isActive(r)) tr.className = 'is-active';
    spec.columns.forEach(function (col) {
      var td = el('td');
      var out = col.render ? col.render(r) : col.value(r);
      if (out instanceof Node) td.appendChild(out);
      else td.textContent = (out === null || out === undefined) ? '—' : out;
      if (col.dim) td.className = 'dim';
      tr.appendChild(td);
    });
    if (spec.onRow) { tr.style.cursor = 'pointer'; tr.addEventListener('click', function () { spec.onRow(r); }); }
    tb.appendChild(tr);
  });
  t.appendChild(tb);
  host.appendChild(t);
}

function barCell(value, max, display, slot) {
  var wrap = el('div', 'bar-cell');
  wrap.appendChild(el('span', 'bar-value', display));
  var track = el('span', 'bar-track');
  var fill = el('span', 'bar-fill');
  fill.style.width = (max > 0 ? Math.max(0, Math.min(100, value / max * 100)) : 0) + '%';
  if (slot) fill.style.background = seriesColor(slot);
  track.appendChild(fill);
  wrap.appendChild(track);
  return wrap;
}

function meterRow(host, label, sub, value, max, status, display) {
  var row = el('div', 'meter-row');
  var lab = el('div', 'm-label', label);
  if (sub) lab.appendChild(el('small', null, sub));
  row.appendChild(lab);
  var m = el('div', 'meter' + (status ? ' is-' + status : ''));
  var i2 = el('i');
  i2.style.width = (max > 0 ? Math.max(0, Math.min(100, value / max * 100)) : 0) + '%';
  m.appendChild(i2);
  row.appendChild(m);
  row.appendChild(el('div', 'm-value', display));
  host.appendChild(row);
}

function statTile(host, opts) {
  var t = el('div', 'stat' + (opts.status ? ' status-' + opts.status : ''));
  t.appendChild(el('div', 'label', opts.label));
  var v = el('div', 'value');
  v.appendChild(document.createTextNode(opts.value));
  /* No unit on an em dash — "—d" reads as a broken value, not "no data". */
  if (opts.unit && opts.value !== '—') v.appendChild(el('span', 'unit', opts.unit));
  t.appendChild(v);
  if (opts.context) t.appendChild(el('div', 'context', opts.context));
  if (opts.spark && opts.spark.length > 1) {
    var sp = el('div', 'spark');
    t.appendChild(sp);
    host.appendChild(t);
    sparkline(sp, opts.spark, opts.sparkSlot || 1);
    return t;
  }
  host.appendChild(t);
  return t;
}

function sparkline(host, values, slot) {
  var W = Math.max(60, host.clientWidth || 140), H = 26, pad = 3;
  /* Baselined at zero: rescaling a near-flat count series onto its own min
     turns rounding noise into a dramatic-looking trend. */
  var max = Math.max.apply(null, values) || 1;
  var min = Math.min(0, Math.min.apply(null, values));
  var span = (max - min) || 1;
  var svg = s('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, 'aria-hidden': 'true' });
  var pts = values.map(function (v, i) {
    return (pad + (i / (values.length - 1)) * (W - pad * 2)) + ',' + (H - pad - ((v - min) / span) * (H - pad * 2));
  });
  svg.appendChild(s('path', { d: 'M' + pts.join('L') + 'L' + (W - pad) + ',' + (H - pad) + 'L' + pad + ',' + (H - pad) + 'Z',
    fill: seriesColor(slot), opacity: .1 }));
  svg.appendChild(s('polyline', { points: pts.join(' '), fill: 'none', stroke: seriesColor(slot), 'stroke-width': 2,
    'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
  var last = pts[pts.length - 1].split(',');
  svg.appendChild(s('circle', { cx: last[0], cy: last[1], r: 3.5, fill: seriesColor(slot), stroke: 'var(--surface)', 'stroke-width': 2 }));
  host.appendChild(svg);
}

/* ══ 7. Views ════════════════════════════════════════════════════════════ */

var views = {};
function defineView(id, title, eyebrow, sub, build) {
  views[id] = { id: id, title: title, eyebrow: eyebrow, sub: sub, build: build, root: null, update: null };
}

function viewRoot(id) {
  var v = views[id];
  if (v.root) return v.root;
  var root = el('section', 'view');
  root.id = 'view-' + id;
  root.setAttribute('role', 'tabpanel');
  var head = el('header', 'page-head');
  head.appendChild(el('p', 'eyebrow', v.eyebrow));
  head.appendChild(el('h1', null, v.title));
  if (v.sub) head.appendChild(el('p', 'sub', v.sub));
  root.appendChild(head);
  var body = el('div');
  root.appendChild(body);
  document.getElementById('views').appendChild(root);
  v.root = root;
  v.update = v.build(body);
  return root;
}

/* ── Overview ───────────────────────────────────────────────────────── */
defineView('overview', 'Operations overview', 'Triage & Ops',
  'Headline KPIs, sprint burn and the workflow mix for the current slice. Every tile, chart and table on this page recomputes from the filters above.',
  function (body) {
    var stats = el('div', 'stat-row'); body.appendChild(stats);

    var s1 = el('div', 'section'); body.appendChild(s1);
    var g1 = el('div', 'grid grid-2'); s1.appendChild(g1);
    var burn = chartCard(g1, { title: 'Sprint intake vs closure',
      note: 'Bugs opened and bugs closed per sprint — both counted in bugs, on one shared axis.' });
    var back = chartCard(g1, { title: 'Carried-over open backlog',
      note: 'Running total of opened minus closed. Its own chart, not a second y-axis on the one beside it.' });

    var s2 = el('div', 'section'); body.appendChild(s2);
    s2.appendChild(sectionTitle('Workflow mix', 'Click any bar to filter the whole dashboard by that value.'));
    var g2 = el('div', 'grid grid-3'); s2.appendChild(g2);
    var priC = chartCard(g2, { title: 'Priority', note: 'P1 is highest. Click a bar to filter.' });
    var sevC = chartCard(g2, { title: 'Severity', note: 'Reported impact level.' });
    var stC  = chartCard(g2, { title: 'Life cycle status', note: 'All workflow states, in life cycle order.' });

    var s3 = el('div', 'section'); body.appendChild(s3);
    s3.appendChild(sectionTitle('Needs attention now',
      'Computed once from the full dataset at build time — these do not follow the filters.'));
    var ins = el('div', 'insight-list'); s3.appendChild(ins);
    (D.insights || []).forEach(function (item, i) {
      var c = el('div', 'insight');
      c.appendChild(el('div', 'rank', String(i + 1)));
      var b = el('div');
      b.appendChild(el('h3', null, item.title));
      b.appendChild(el('p', null, item.detail));
      c.appendChild(b);
      ins.appendChild(c);
    });

    return function () {
      clear(stats);
      var openStatus = A.openLate > A.open * 0.5 ? 'critical' : (A.openLate > 0 ? 'warning' : 'good');
      var slaS = slaVerdict(A.slaPct);
      var reS = A.reopenPct > 5 ? 'critical' : A.reopenPct > 2 ? 'warning' : 'good';
      var lastBl = A.backlog[A.backlog.length - 1], prevBl = A.backlog.length > 1 ? A.backlog[A.backlog.length - 2] : null;
      var delta = prevBl === null ? null : lastBl - prevBl;

      statTile(stats, { label: 'Bugs in scope', value: nf(A.total),
        context: nf(A.closed) + ' closed · ' + pct(A.closeRate) + ' close rate',
        spark: Array.from(A.sprOpened), sparkSlot: 1 });
      statTile(stats, { label: 'Open bugs', value: nf(A.open), status: openStatus,
        context: nf(A.openLate) + ' already past their SLA target' });
      statTile(stats, { label: 'SLA compliance', value: pct(A.slaPct), status: slaS[0],
        context: 'of closed bugs met their priority target' });
      statTile(stats, { label: 'Avg resolution time', value: nf1(A.avgRes), unit: 'd',
        context: 'median ' + days(A.medRes) + ' · p90 ' + days(A.p90Res),
        spark: A.sprAvgRes.map(function (v) { return v === null ? 0 : v; }), sparkSlot: 3 });
      statTile(stats, { label: 'Defect density', value: nf1(A.density),
        context: 'bugs per KLOC across ' + nf1(A.kloc) + ' KLOC in scope' });
      statTile(stats, { label: 'Reopen rate', value: pct(A.reopenPct), status: reS,
        context: 'bugs that failed verification after a fix' });
      statTile(stats, { label: 'Open backlog', value: nf(lastBl),
        status: delta === null ? null : (delta <= 0 ? 'good' : delta > Math.max(lastBl, 1) * 0.05 ? 'critical' : 'warning'),
        context: delta === null ? 'latest sprint in scope'
          : (delta >= 0 ? '+' : '') + nf(delta) + ' vs the prior sprint',
        spark: Array.from(A.backlog), sparkSlot: 7 });
      statTile(stats, { label: 'Avg age of an open bug', value: nf1(A.avgAge), unit: 'd',
        context: 'measured to ' + D.meta.snapshot });

      var sprints = DIM.sprint;
      renderTwin(burn, { labels: sprints, dimName: 'Sprint', yTitle: 'Bugs',
        series: [{ name: 'Opened', values: Array.from(A.sprOpened), slot: 1 },
                 { name: 'Closed', values: Array.from(A.sprClosed), slot: 3 }] });
      renderTwin(back, { labels: sprints, dimName: 'Sprint', yTitle: 'Open bugs', labelEnds: true,
        series: [{ name: 'Open backlog', values: Array.from(A.backlog), slot: 7, area: true, kind: 'line' }] }, 'lines');

      distCard(priC, 'priority', 1, true);
      distCard(sevC, 'severity', 2, true);
      distCard(stC, 'status', 4, true);
    };
  });

function sectionTitle(text, sub) {
  var h = el('h2', 'section-title', text);
  if (sub) h.appendChild(el('span', 'sub', sub));
  return h;
}

/* Distribution card over one dimension, click-to-filter, keeping the
   dimension's own order (workflow states, P1..P5) rather than by rank. */
function distCard(cc, dim, slot, keepOrder) {
  var rows = (keepOrder ? ordered(dim) : ranked(dim)).filter(function (r) { return r.value > 0 || keepOrder; });
  clear(cc.chartHost); clear(cc.legendHost);
  rankedBars(cc.chartHost, {
    rows: rows, slot: slot, labelW: 118, selected: state.sel[dim],
    measure: 'Bugs',
    tipRows: function (r) {
      return [{ name: 'Bugs', value: nf(r.value), color: seriesColor(slot) },
              { name: 'Share', value: pct(A.total ? r.value / A.total * 100 : 0) },
              { name: 'Open', value: nf(openOf(dim)[r.k]) }];
    },
    onSelect: function (k) { toggleFilter(dim, k); }
  });
  var spec = {
    rows: rows, sortCol: 1, sortAsc: false,
    columns: [
      { label: DIM_LABEL[dim] || dim, value: function (r) { return r.label; } },
      { label: 'Bugs', value: function (r) { return r.value; }, render: function (r) { return nf(r.value); } },
      { label: 'Open', value: function (r) { return openOf(dim)[r.k]; }, render: function (r) { return nf(openOf(dim)[r.k]); } },
      { label: 'Avg days', value: function (r) { return avgDaysOf(dim, r.k); }, render: function (r) { return days(avgDaysOf(dim, r.k)); } },
      { label: 'Share', value: function (r) { return r.value; }, render: function (r) { return pct(A.total ? r.value / A.total * 100 : 0); } }
    ],
    isActive: function (r) { return state.sel[dim] && state.sel[dim].has(r.k); },
    onRow: function (r) { if (state.sel[dim]) toggleFilter(dim, r.k); }
  };
  dataTable(cc.tableHost, spec);
}

/* ── Trends & resolution time ────────────────────────────────────────── */
defineView('trends', 'Bug trends & resolution time', 'Trends',
  'How intake, closure and turnaround move over the reporting window, and how long bugs actually take to close.',
  function (body) {
    var s1 = el('div', 'section'); body.appendChild(s1);
    var g1 = el('div', 'grid grid-2'); s1.appendChild(g1);
    var partial = [];
    if (D.meta.monthFirstPartial) partial.push(D.meta.monthFirst);
    if (D.meta.monthLastPartial) partial.push(D.meta.monthLast);
    var trend = chartCard(g1, { title: 'Bug reporting trend',
      note: 'Bugs reported vs bugs closed, by calendar month.' +
        (partial.length ? ' The reporting window starts and ends mid-month, so ' +
          partial.join(' and ') + ' cover only part of a month — their dip is a window artifact, not a real decline.' : '') });
    var rtime = chartCard(g1, { title: 'Resolution time trend',
      note: 'Average days to close, by the sprint the bug closed in. Gaps are sprints with no closures in scope.' });

    var s2 = el('div', 'section'); body.appendChild(s2);
    var g2 = el('div', 'grid grid-2'); s2.appendChild(g2);
    var rdist = chartCard(g2, { title: 'How long closed bugs took',
      note: 'Distribution of days-to-close across every closed bug in scope.' });
    var adist = chartCard(g2, { title: 'Age profile of the open queue',
      note: 'How long the still-open bugs have been open, measured to the snapshot date.' });

    var s3 = el('div', 'section'); body.appendChild(s3);
    var g3 = el('div', 'grid grid-2'); s3.appendChild(g3);
    var slaC = chartCard(g3, { title: 'Resolution time vs SLA target',
      note: 'Average and median days to close per priority, against that priority’s target (dashed rule).' });
    var meters = card(g3, { title: 'SLA compliance by priority',
      note: 'Share of closed bugs that met their target. Status badge, not color alone.' });

    return function () {
      renderTwin(trend, { labels: DIM.month, dimName: 'Month', yTitle: 'Bugs', maxTicks: 10,
        series: [{ name: 'Reported', values: Array.from(A.monOpened), slot: 1, area: true, kind: 'line' },
                 { name: 'Closed', values: Array.from(A.monClosed), slot: 3, kind: 'line' }],
        tipFoot: function (i) {
          return partial.indexOf(DIM.month[i]) >= 0 ? 'Partial month — the reporting window cuts it short' : null;
        } }, 'lines');

      renderTwin(rtime, { labels: DIM.sprint, dimName: 'Sprint', yTitle: 'Days to close', valueFmt: nf1,
        series: [{ name: 'Avg days to close', values: A.sprAvgRes, slot: 2, kind: 'line', area: true }] }, 'lines');

      renderTwin(rdist, { labels: RES_BINS.map(binLabel), dimName: 'Days to close', yTitle: 'Closed bugs',
        series: [{ name: 'Closed bugs', values: Array.from(A.resHist), slot: 3 }] });

      renderTwin(adist, { labels: AGE_BINS.map(binLabel), dimName: 'Days open', yTitle: 'Open bugs',
        series: [{ name: 'Open bugs', values: Array.from(A.ageHist), slot: 8 }] });

      var pri = DIM.priority;
      var avg = pri.map(function (_, k) { return avgDaysOf('priority', k) || 0; });
      var med = pri.map(function (_, k) { return medianDaysFor(k); });
      renderTwin(slaC, { labels: pri, dimName: 'Priority', yTitle: 'Days', valueFmt: nf1, labelExtreme: false,
        series: [{ name: 'Average days', values: avg, slot: 1 },
                 { name: 'Median days', values: med, slot: 5 }],
        thresholds: [{ name: 'SLA target', values: Array.from(SLA) }],
        tipFoot: function (i) { return 'Target ' + SLA[i] + ' days · ' + nf(closedOf('priority')[i]) + ' closed'; } });

      clear(meters.body);
      var list = el('div', 'meter-list'); meters.body.appendChild(list);
      pri.forEach(function (p, k) {
        var v = slaPctOf('priority', k);
        meterRow(list, p, SLA[k] + '-day target', v === null ? 0 : v, 100, slaVerdict(v)[0], pct(v));
      });
      var overall = slaVerdict(A.slaPct);
      var foot = el('p', 'note');
      foot.appendChild(document.createTextNode(
        'Overall: ' + pct(A.slaPct) + ' of ' + nf(A.closed) + ' closed bugs met their target. '));
      foot.appendChild(badge(overall[0], overall[1]));
      meters.body.appendChild(foot);
    };
  });

/* Median days-to-close for one priority — needs a second pass over the slice,
   so it is computed on demand rather than in the main aggregate. */
function medianDaysFor(priorityK) {
  var vals = [];
  var pri = COL.priority, rd = COL.resDays;
  for (var i = 0; i < N; i++) if (mask[i] && pri[i] === priorityK && rd[i] >= 0) vals.push(rd[i]);
  if (!vals.length) return null;
  vals.sort(function (a, b) { return a - b; });
  return vals[Math.floor(vals.length / 2)];
}

/* ── Distribution ───────────────────────────────────────────────────── */
defineView('distribution', 'Sprint, module & release distribution', 'Distribution',
  'Where the defects sit across the delivery taxonomy. Every bar and cell is click-to-filter.',
  function (body) {
    var s1 = el('div', 'section'); body.appendChild(s1);
    var g1 = el('div', 'grid grid-2'); s1.appendChild(g1);
    var modC = chartCard(g1, { title: 'Bugs by module', note: 'Absolute volume. Size-normalised density is in the table below.' });
    var relC = chartCard(g1, { title: 'Release-wise quality', note: 'Closed vs still-open bugs per release version.' });

    var s2 = el('div', 'section'); body.appendChild(s2);
    var g2 = el('div', 'grid grid-2'); s2.appendChild(g2);
    var heatC = chartCard(g2, { title: 'Module × Priority', note: 'Where the urgent work concentrates. Click a cell to filter both dimensions.' });
    var densC = card(g2, { title: 'Defect density by module', note: 'Bugs per KLOC — volume normalised by module size, so raw counts don’t flatter large modules.' });

    var s3 = el('div', 'section'); body.appendChild(s3);
    var g3 = el('div', 'grid grid-2'); s3.appendChild(g3);
    var featC = chartCard(g3, { title: 'Bugs by feature', note: 'The user-facing capability the defect breaks.' });
    var compC = chartCard(g3, { title: 'Bugs by component', note: 'The deployable / code unit it lives in.' });

    var s4 = el('div', 'section'); body.appendChild(s4);
    var g4 = el('div', 'grid grid-3'); s4.appendChild(g4);
    var envC = chartCard(g4, { title: 'Environment', note: 'Where the bug was observed.' });
    var domC = chartCard(g4, { title: 'Domain', note: 'System domain that owns it.' });
    var resC = chartCard(g4, { title: 'Resolution outcome', note: 'How closed bugs were resolved.' });

    var densTable = el('div', 'table-wrap'); densC.body.appendChild(densTable);

    return function () {
      distCard(modC, 'module', 1, false);
      distCard(featC, 'feature', 3, false);
      distCard(compC, 'component', 4, false);
      distCard(envC, 'environment', 5, true);
      distCard(domC, 'domain', 7, false);
      distCard(resC, 'resolution', 6, true);

      var rel = DIM.release;
      renderTwin(relC, { labels: rel, dimName: 'Release', yTitle: 'Bugs', stacked: true, labelExtreme: false,
        series: [{ name: 'Closed', values: Array.from(closedOf('release')), slot: 3 },
                 { name: 'Still open', values: Array.from(openOf('release')), slot: 8 }],
        tipFoot: function (i) {
          var tot = counts('release')[i];
          return tot ? nf1(closedOf('release')[i] / tot * 100) + '% closed · ' + nf(urgentOf('release')[i]) + ' P1/P2' : null;
        },
        onSelect: function (k) { toggleFilter('release', k); } });

      var mods = DIM.module, pris = DIM.priority;
      clear(heatC.chartHost); clear(heatC.legendHost);
      heatmap(heatC.chartHost, {
        rowLabels: mods, colLabels: pris, values: Array.from(A.modPri), measure: 'bugs',
        onSelect: function (r, c) { state.sel.module.add(r); state.sel.priority.add(c); recompute(); }
      });
      var heatRows = [];
      mods.forEach(function (m, r) {
        pris.forEach(function (p, c) { heatRows.push({ m: m, p: p, v: A.modPri[r * pris.length + c] }); });
      });
      dataTable(heatC.tableHost, { rows: heatRows, sortCol: 2, sortAsc: false, columns: [
        { label: 'Module', value: function (r) { return r.m; } },
        { label: 'Priority', value: function (r) { return r.p; } },
        { label: 'Bugs', value: function (r) { return r.v; }, render: function (r) { return nf(r.v); } }
      ] });

      var dens = [];
      counts('module').forEach(function (v, k) {
        if (!v) return;
        dens.push({ k: k, module: mods[k], bugs: v, open: openOf('module')[k], kloc: KLOC[k],
          density: KLOC[k] ? v / KLOC[k] : 0, avg: avgDaysOf('module', k),
          closePct: v ? closedOf('module')[k] / v * 100 : 0 });
      });
      var maxD = Math.max.apply(null, dens.map(function (r) { return r.density; })) || 1;
      dataTable(densTable, { rows: dens, sortCol: 4, sortAsc: false,
        isActive: function (r) { return state.sel.module.has(r.k); },
        onRow: function (r) { toggleFilter('module', r.k); },
        columns: [
          { label: 'Module', value: function (r) { return r.module; } },
          { label: 'Bugs', value: function (r) { return r.bugs; }, render: function (r) { return nf(r.bugs); } },
          { label: 'Open', value: function (r) { return r.open; }, render: function (r) { return nf(r.open); } },
          { label: 'KLOC', value: function (r) { return r.kloc; }, render: function (r) { return nf1(r.kloc); } },
          { label: 'Bugs / KLOC', value: function (r) { return r.density; },
            render: function (r) { return barCell(r.density, maxD, nf1(r.density), 1); } },
          { label: 'Avg days', value: function (r) { return r.avg; }, render: function (r) { return days(r.avg); } },
          { label: 'Closed %', value: function (r) { return r.closePct; }, render: function (r) { return pct(r.closePct); } }
        ] });
    };
  });

/* ── Quality & team ─────────────────────────────────────────────────── */
defineView('quality', 'Root cause, team & release risk', 'Quality',
  'What keeps breaking, who carries the queue, and which release is riskiest to ship.',
  function (body) {
    var s1 = el('div', 'section'); body.appendChild(s1);
    var g1 = el('div', 'grid grid-2'); s1.appendChild(g1);
    var rcC = chartCard(g1, { title: 'Recurring root causes',
      note: 'The cheapest defects to prevent are the ones that keep repeating. Click to filter.' });
    var teamC = card(g1, { title: 'Team workload',
      note: 'Grouped by routing-policy owner (bug category → specialist). The dataset’s own developer_role column is uniformly random, so it is not used here.' });
    var teamTable = el('div', 'table-wrap'); teamC.body.appendChild(teamTable);

    var s2 = el('div', 'section'); body.appendChild(s2);
    var g2 = el('div', 'grid grid-2'); s2.appendChild(g2);
    var relRisk = card(g2, { title: 'Release risk',
      note: 'Ranked by unresolved P1/P2 count — closure rates sit within noise of each other across releases, so volume is the real differentiator.' });
    var relTable = el('div', 'table-wrap'); relRisk.body.appendChild(relTable);
    var catC = chartCard(g2, { title: 'Bug categories', note: 'What kind of defect it is. Click to filter.' });

    return function () {
      var rcRows = ranked('category').map(function (r) {
        return { k: r.k, label: LOOK.rootCauseByCategory[r.k], value: r.value };
      }).sort(function (a, b) { return b.value - a.value; }).slice(0, 8);
      clear(rcC.chartHost); clear(rcC.legendHost);
      rankedBars(rcC.chartHost, { rows: rcRows, slot: 2, labelW: 210, rowH: 28,
        selected: state.sel.category,
        tipRows: function (r) {
          return [{ name: 'Bugs', value: nf(r.value), color: seriesColor(2) },
                  { name: 'Share', value: pct(A.total ? r.value / A.total * 100 : 0) },
                  { name: 'Urgent (P1/P2)', value: nf(urgentOf('category')[r.k]) },
                  { name: 'Avg days', value: days(avgDaysOf('category', r.k)) }];
        },
        onSelect: function (k) { toggleFilter('category', k); } });
      dataTable(rcC.tableHost, { rows: rcRows, sortCol: 1, sortAsc: false,
        isActive: function (r) { return state.sel.category.has(r.k); },
        onRow: function (r) { toggleFilter('category', r.k); },
        columns: [
          { label: 'Root cause', value: function (r) { return r.label; } },
          { label: 'Bugs', value: function (r) { return r.value; }, render: function (r) { return nf(r.value); } },
          { label: 'Share', value: function (r) { return r.value; }, render: function (r) { return pct(A.total ? r.value / A.total * 100 : 0); } },
          { label: 'Urgent', value: function (r) { return urgentOf('category')[r.k]; }, render: function (r) { return nf(urgentOf('category')[r.k]); } },
          { label: 'Avg days', value: function (r) { return avgDaysOf('category', r.k); }, render: function (r) { return days(avgDaysOf('category', r.k)); } }
        ] });

      distCard(catC, 'category', 7, false);

      var teams = [];
      counts('team').forEach(function (v, k) {
        if (!v) return;
        teams.push({ k: k, team: DIM.team[k], assigned: v, open: openOf('team')[k],
          urgentOpen: urgentOpenFor('team', k), avg: avgDaysOf('team', k), sla: slaPctOf('team', k) });
      });
      var maxA = Math.max.apply(null, teams.map(function (r) { return r.assigned; })) || 1;
      dataTable(teamTable, { rows: teams, sortCol: 1, sortAsc: false,
        isActive: function (r) { return state.sel.team.has(r.k); },
        onRow: function (r) { toggleFilter('team', r.k); },
        columns: [
          { label: 'Routed owner', value: function (r) { return r.team; } },
          { label: 'Assigned', value: function (r) { return r.assigned; },
            render: function (r) { return barCell(r.assigned, maxA, nf(r.assigned), 1); } },
          { label: 'Open', value: function (r) { return r.open; }, render: function (r) { return nf(r.open); } },
          { label: 'Urgent open', value: function (r) { return r.urgentOpen; }, render: function (r) { return nf(r.urgentOpen); } },
          { label: 'Avg days', value: function (r) { return r.avg; }, render: function (r) { return days(r.avg); } },
          { label: 'SLA', value: function (r) { return r.sla; }, render: function (r) {
              var w = el('span'); w.appendChild(document.createTextNode(pct(r.sla) + ' '));
              var v = slaVerdict(r.sla); w.appendChild(badge(v[0], v[1])); return w; } }
        ] });

      var rels = [];
      counts('release').forEach(function (v, k) {
        if (!v) return;
        rels.push({ k: k, release: DIM.release[k], bugs: v, open: openOf('release')[k],
          urgentOpen: urgentOpenFor('release', k), avg: avgDaysOf('release', k),
          closePct: v ? closedOf('release')[k] / v * 100 : 0 });
      });
      var maxU = Math.max.apply(null, rels.map(function (r) { return r.urgentOpen; })) || 1;
      dataTable(relTable, { rows: rels, sortCol: 3, sortAsc: false,
        isActive: function (r) { return state.sel.release.has(r.k); },
        onRow: function (r) { toggleFilter('release', r.k); },
        columns: [
          { label: 'Release', value: function (r) { return r.release; } },
          { label: 'Bugs', value: function (r) { return r.bugs; }, render: function (r) { return nf(r.bugs); } },
          { label: 'Open', value: function (r) { return r.open; }, render: function (r) { return nf(r.open); } },
          { label: 'Urgent open (P1/P2)', value: function (r) { return r.urgentOpen; },
            render: function (r) { return barCell(r.urgentOpen, maxU, nf(r.urgentOpen), 8); } },
          { label: 'Avg days', value: function (r) { return r.avg; }, render: function (r) { return days(r.avg); } },
          { label: 'Closed %', value: function (r) { return r.closePct; }, render: function (r) { return pct(r.closePct); } }
        ] });
    };
  });

/* Urgent-and-still-open needs both conditions at once, which the per-dimension
   accumulators don't carry — one narrow extra pass, only for the tables using it. */
var urgentOpenCache = {};
function urgentOpenFor(dim, k) {
  if (urgentOpenCache[dim]) return urgentOpenCache[dim][k];
  var arr = new Int32Array(DIM[dim].length);
  var col = COL[dim], pri = COL.priority, rd = COL.resDays;
  for (var i = 0; i < N; i++) if (mask[i] && rd[i] < 0 && P_URGENT[pri[i]]) arr[col[i]]++;
  urgentOpenCache[dim] = arr;
  return arr[k];
}

/* ── Model & triage ─────────────────────────────────────────────────── */
defineView('model', 'Model performance & live triage', 'Models',
  'What the trained classifiers actually learned, how their predictions line up with the recorded data, and a console that runs them on a new bug.',
  function (body) {
    var s0 = el('div', 'section'); body.appendChild(s0);
    s0.appendChild(sectionTitle('Verdicts', 'Each accuracy figure is checked against the dataset’s known structure before it is trusted.'));
    var verdicts = el('div'); s0.appendChild(verdicts);
    buildVerdicts(verdicts);

    var s1 = el('div', 'section'); body.appendChild(s1);
    s1.appendChild(sectionTitle('Predictions against the recorded data',
      'The saved priority model re-scored every bug in the dataset at build time. Choose which split to read — held-out test is the only honest one.'));
    var splitBar = el('div', 'seg'); s1.appendChild(splitBar);
    [['all', 'All rows'], ['test', 'Held-out test'], ['train', 'Training rows'], ['unseen', 'Never sampled']].forEach(function (o) {
      var b = el('button', null, o[1]); b.type = 'button';
      b.dataset.split = o[0];
      b.setAttribute('aria-pressed', state.split === o[0]);
      b.addEventListener('click', function () { state.split = o[0]; recompute(); });
      splitBar.appendChild(b);
    });
    var g1 = el('div', 'grid grid-2'); g1.style.marginTop = '13px'; s1.appendChild(g1);
    var confC = chartCard(g1, { title: 'Recorded priority × predicted priority',
      note: 'The diagonal is agreement. Off-diagonal mass is where the model would re-triage the bug.' });
    var confHistC = chartCard(g1, { title: 'Prediction confidence',
      note: 'Highest class probability the model assigned. Low-confidence predictions are the ones worth a human look.' });

    var s2 = el('div', 'section'); body.appendChild(s2);
    var escC = card(s2, { title: 'Model-flagged escalations',
      note: 'Still-open bugs the model would rank MORE urgent than their recorded priority — sorted by confidence. This is the model feeding triage, not just scoring itself.' });
    var escTable = el('div', 'table-wrap table-scroll'); escC.body.appendChild(escTable);

    var s3 = el('div', 'section'); body.appendChild(s3);
    s3.appendChild(sectionTitle('Triage console', 'Report a bug and route it through the same models the pipeline trained.'));
    var cg = el('div', 'console-grid'); s3.appendChild(cg);
    buildConsole(cg);

    var s4 = el('div', 'section'); body.appendChild(s4);
    var cmpC = card(s4, { title: 'Full model comparison', note: 'Accuracy by model and target. Bold marks the best model per column.' });
    var cmpTable = el('div', 'table-wrap'); cmpC.body.appendChild(cmpTable);
    buildComparison(cmpTable);

    return function () {
      var pris = DIM.priority;
      Array.prototype.forEach.call(splitBar.children, function (b) {
        b.setAttribute('aria-pressed', b.dataset.split === state.split);
      });
      if (!HAS_MODEL) {
        var why = 'No model scores in this build. Run "python src/05_modeling.py" to train the ' +
                  'models, then rebuild with "python src/08_dashboard.py" (and drop --no-model).';
        [confC, confHistC].forEach(function (cc) {
          clear(cc.chartHost); clear(cc.legendHost); clear(cc.tableHost);
          cc.chartHost.appendChild(el('p', 'note', why));
        });
        clear(escTable);
        escTable.appendChild(el('p', 'note', why));
        splitBar.hidden = true;
        return;
      }
      splitBar.hidden = false;
      clear(confC.chartHost); clear(confC.legendHost);
      heatmap(confC.chartHost, { rowLabels: pris.map(function (p) { return 'Recorded ' + p; }),
        colLabels: pris.map(function (p) { return 'Pred ' + p; }),
        values: Array.from(A.confusion), measure: 'bugs' });
      var agree = 0, tot = 0;
      for (var r = 0; r < pris.length; r++) for (var c = 0; c < pris.length; c++) {
        var v = A.confusion[r * pris.length + c]; tot += v; if (r === c) agree += v;
      }
      var note = confC.card.querySelector('.note');
      note.textContent = 'Agreement on this split: ' + pct(tot ? agree / tot * 100 : 0) +
        ' over ' + nf(tot) + ' bugs. The diagonal is agreement; off-diagonal mass is where the model would re-triage.';
      var cRows = [];
      pris.forEach(function (p, ri) { pris.forEach(function (q2, ci) {
        cRows.push({ a: 'Recorded ' + p, b: 'Pred ' + q2, v: A.confusion[ri * pris.length + ci] }); }); });
      dataTable(confC.tableHost, { rows: cRows, sortCol: 2, sortAsc: false, columns: [
        { label: 'Recorded', value: function (x) { return x.a; } },
        { label: 'Predicted', value: function (x) { return x.b; } },
        { label: 'Bugs', value: function (x) { return x.v; }, render: function (x) { return nf(x.v); } } ] });

      renderTwin(confHistC, { labels: ['0–10%', '10–20%', '20–30%', '30–40%', '40–50%',
                                       '50–60%', '60–70%', '70–80%', '80–90%', '90–100%'],
        dimName: 'Confidence', yTitle: 'Bugs',
        series: [{ name: 'Predictions', values: Array.from(A.confHist), slot: 7 }] });

      var esc = A.escalate.slice(0, 400).map(function (i) {
        return { i: i, id: bugId(i), title: LOOK.titleByCategory[COL.category[i]],
          module: DIM.module[COL.module[i]], recorded: DIM.priority[COL.priority[i]],
          predicted: DIM.priority[COL.predPriority[i]], conf: COL.predConf[i],
          age: D.meta.snapshotDay - COL.createdDay[i], owner: DIM.team[COL.team[i]] };
      });
      dataTable(escTable, { rows: esc, sortCol: 5, sortAsc: false, columns: [
        { label: 'Bug', value: function (r) { return r.id; }, render: function (r) {
            var w = el('span'); w.appendChild(el('span', 'mono', r.id));
            w.appendChild(document.createTextNode(' ' + r.title)); return w; } },
        { label: 'Module', value: function (r) { return r.module; } },
        { label: 'Owner', value: function (r) { return r.owner; } },
        { label: 'Recorded', value: function (r) { return r.recorded; } },
        { label: 'Model says', value: function (r) { return r.predicted; }, render: function (r) {
            return badge(r.predicted === 'P1' ? 'critical' : 'serious', r.predicted); } },
        { label: 'Confidence', value: function (r) { return r.conf; }, render: function (r) { return r.conf + '%'; } },
        { label: 'Age', value: function (r) { return r.age; }, render: function (r) { return nf(r.age) + 'd'; } }
      ] });
      var eNote = escC.card.querySelector('.note');
      eNote.textContent = nf(A.escalate.length) + ' open bugs in scope would be ranked more urgent by the model' +
        (A.escalate.length > 400 ? ' (top 400 shown)' : '') +
        '. Priority is a derived field, so read this as "the scoring rule and the recorded value disagree", not as ground truth.';
    };
  });

function buildVerdicts(host) {
  var ev = D.modelEval || {};
  (D.verdicts || []).forEach(function (v) {
    var rows = ev[v.target];
    if (!rows) return;
    var c = el('div', 'card verdict');
    var head = el('div', 'verdict-head');
    head.appendChild(el('h2', null, v.target));
    head.appendChild(badge(v.status, v.label));
    c.appendChild(head);
    var p = el('p', 'lede');
    v.narrative.forEach(function (part) {
      if (typeof part === 'string') p.appendChild(document.createTextNode(part));
      else if (part.code) p.appendChild(el('code', null, part.code));
      else if (part.strong) p.appendChild(el('strong', null, part.strong));
    });
    c.appendChild(p);
    var items = D.modelOrder.filter(function (m) { return rows[m]; })
      .map(function (m) { return { name: m, acc: rows[m].Accuracy, slot: D.modelOrder.indexOf(m) + 1 }; })
      .sort(function (a, b) { return b.acc - a.acc; });
    var chart = el('div', 'hbars');
    items.forEach(function (it, i) {
      var row = el('div', 'hbar' + (i === 0 ? ' is-best' : ''));
      row.appendChild(el('div', 'h-label', it.name));
      var track = el('div', 'h-track');
      var fill = el('div', 'h-fill');
      fill.style.width = (it.acc * 100).toFixed(1) + '%';
      fill.style.background = seriesColor(it.slot);
      track.appendChild(fill);
      row.appendChild(track);
      row.appendChild(el('div', 'h-value', (it.acc * 100).toFixed(1) + '%'));
      chart.appendChild(row);
    });
    c.appendChild(chart);
    host.appendChild(c);
  });
}

function buildComparison(host) {
  var ev = D.modelEval || {};
  var targets = ['Severity', 'Priority', 'Bug Category'].filter(function (t) { return ev[t]; });
  var best = {};
  targets.forEach(function (t) {
    best[t] = Object.keys(ev[t]).reduce(function (a, b) { return ev[t][a].Accuracy >= ev[t][b].Accuracy ? a : b; });
  });
  var rows = D.modelOrder.filter(function (m) { return targets.some(function (t) { return ev[t][m]; }); })
    .map(function (m) { return { model: m }; });
  var cols = [{ label: 'Model', value: function (r) { return r.model; } }];
  targets.forEach(function (t) {
    cols.push({ label: t, value: function (r) { return ev[t][r.model] ? ev[t][r.model].Accuracy : null; },
      render: function (r) {
        var m = ev[t][r.model];
        if (!m) return '—';
        var txt = (m.Accuracy * 100).toFixed(1) + '%';
        return best[t] === r.model ? el('strong', null, txt) : txt;
      } });
  });
  dataTable(host, { rows: rows, columns: cols });
}

/* ── Triage console: live model API when served, documented rule offline ─ */
var LIVE = { on: false, checked: false, model: null, listeners: [] };

/* Probed once at boot so the header badge is honest before anyone opens the
   Models tab. From file:// the fetch simply rejects and we stay offline. */
function probeLive() {
  var settle = function () {
    LIVE.checked = true;
    var pill = document.getElementById('live-pill');
    if (pill) {
      pill.classList.toggle('is-live', LIVE.on);
      pill.querySelector('.pill-text').textContent = LIVE.on ? 'Live models' : 'Offline build';
      pill.title = LIVE.on
        ? 'Connected to the local model server — the triage console runs real inference'
        : 'Opened as a static file. Run "python src/09_serve.py" for live model inference.';
    }
    LIVE.listeners.forEach(function (fn) { fn(); });
  };
  fetch('api/health')
    .then(function (r) { return r.ok ? r.json() : null; })
    .then(function (j) {
      LIVE.on = !!(j && j.models_loaded);
      LIVE.model = j && j.priority_model;
      settle();
    })
    .catch(settle);
}

function buildConsole(host) {
  var formCard = card(host, { title: 'Report a bug', note: 'The same inputs 07_bug_triage.py takes on the command line.' });
  var form = el('form', 'form-grid');
  form.id = 'triage-form';
  formCard.body.appendChild(form);

  function field(name, label, node, full) {
    var f = el('div', 'field' + (full ? ' full' : ''));
    var l = el('label', null, label);
    l.setAttribute('for', 'f-' + name);
    node.id = 'f-' + name;
    node.name = name;
    f.appendChild(l); f.appendChild(node);
    form.appendChild(f);
    return node;
  }
  function select(options, def) {
    var sel = el('select');
    options.forEach(function (o) {
      var op = el('option', null, o);
      op.value = o;
      if (o === def) op.selected = true;
      sel.appendChild(op);
    });
    return sel;
  }

  var title = field('title', 'Title', el('input'), true);
  title.type = 'text';
  title.value = 'Checkout page crashes on payment submit';
  var desc = field('desc', 'Description', el('textarea'), true);
  desc.value = 'The payment page freezes and returns a server error after the user submits the order.';
  var cat = field('category', 'Bug category', select(DIM.category, DIM.category[0]));
  var envSel = field('environment', 'Environment', select(DIM.environment, 'Production'));
  var errSel = field('errorCode', 'Error code', select(DIM.errorCode.map(String), '500'));
  var domSel = field('domain', 'Domain', select(DIM.domain, DIM.domain[0]));
  var sevSel = field('severity', 'Severity (if known)', select(['— predict it —'].concat(DIM.severity), '— predict it —'));
  var techSel = field('tech', 'Tech stack', select(DIM.tech, DIM.tech[0]));

  var btns = el('div', 'btn-row');
  var go = el('button', 'btn', 'Run triage'); go.type = 'submit';
  var reset = el('button', 'btn secondary', 'Reset'); reset.type = 'reset';
  btns.appendChild(go); btns.appendChild(reset);
  formCard.body.appendChild(btns);

  var modeNote = el('p', 'note');
  modeNote.style.marginTop = '11px';
  formCard.body.appendChild(modeNote);

  var outCard = card(host, { title: 'Triage result', note: 'Prediction, routing, diagnosis and where the ticket enters the life cycle.' });
  var out = el('div'); outCard.body.appendChild(out);
  out.appendChild(el('p', 'result-empty', 'Fill the form and run triage to see a prediction.'));

  function setMode() {
    modeNote.textContent = LIVE.on
      ? ('Live mode: predictions come from models/best_priority_model.pkl (' + (LIVE.model || 'saved model') +
         ') and models/best_severity_model.pkl, over the local HTTP API.')
      : 'Offline mode: this page is open as a static file, so the console applies the documented priority scoring rule and the routing table instead of loading the pickled models. Run "python src/09_serve.py" for live model inference here.';
  }
  setMode();
  LIVE.listeners.push(setMode);

  form.addEventListener('submit', function (e) {
    e.preventDefault();
    var payload = {
      title: title.value, desc: desc.value, category: cat.value,
      environment: envSel.value, error_code: parseFloat(errSel.value),
      domain: domSel.value, tech_stack: techSel.value,
      severity: sevSel.value.indexOf('—') === 0 ? null : sevSel.value
    };
    go.disabled = true;
    go.textContent = 'Running…';
    var done = function (res) {
      go.disabled = false; go.textContent = 'Run triage';
      renderTriage(out, payload, res);
    };
    if (LIVE.on) {
      fetch('api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload) })
        .then(function (r) { return r.json(); })
        .then(function (j) { done(j.error ? localTriage(payload, j.error) : j); })
        .catch(function (err) { done(localTriage(payload, String(err))); });
    } else {
      setTimeout(function () { done(localTriage(payload, null)); }, 0);
    }
  });
}

/* The documented priority scoring rule from 01_data_collection.py, applied
   client-side so the console still works from a file:// page. */
function localTriage(p, fallbackReason) {
  var sevW = { Critical: 4, High: 3, Medium: 2, Low: 1 };
  var envW = { Production: 2, Staging: 1, Development: 0 };
  var severity = p.severity || 'High';
  var score = (sevW[severity] || 2) + (envW[p.environment] || 0) +
              ([500, 502, 503].indexOf(Math.round(p.error_code)) >= 0 ? 1 : 0);
  var pri = score >= 6 ? 'P1' : score === 5 ? 'P2' : score === 4 ? 'P3' : score === 3 ? 'P4' : 'P5';
  var probs = DIM.priority.map(function (q2) { return q2 === pri ? 1 : 0; });
  return {
    source: 'rule', severity: severity, severity_source: p.severity ? 'provided by reporter' : 'assumed (no model available offline)',
    priority: pri, priority_probs: probs, score: score,
    owner: routeOwner(p.category, p.domain),
    root_cause: LOOK.rootCauseByCategory[DIM.category.indexOf(p.category)] || '—',
    suggested_fix: LOOK.fixByCategory[DIM.category.indexOf(p.category)] || '—',
    note: fallbackReason ? ('Live model call failed (' + fallbackReason + ') — fell back to the documented rule.') : null
  };
}

function routeOwner(category, domain) {
  if (D.routing.domain_override && D.routing.domain_override[domain]) return D.routing.domain_override[domain];
  return D.routing.by_category[category] || 'Full-Stack Developer';
}

function renderTriage(out, input, res) {
  clear(out);
  if (res.note) {
    var warn = el('p', 'note');
    warn.appendChild(badge('warning', 'Fallback'));
    warn.appendChild(document.createTextNode(' ' + res.note));
    out.appendChild(warn);
  }

  var head = el('div', 'hero');
  head.appendChild(el('div', 'figure', res.priority));
  var cap = el('div', 'caption');
  cap.textContent = (D.priorityNote && D.priorityNote[res.priority]) || '';
  head.appendChild(cap);
  out.appendChild(head);

  var src = el('p', 'note');
  src.appendChild(badge(res.source === 'model' ? 'good' : 'neutral',
    res.source === 'model' ? ('Predicted by ' + (res.model || 'the saved model')) : 'Documented scoring rule'));
  out.appendChild(src);

  if (res.priority_probs && res.source === 'model') {
    var probs = el('div', 'prob-list');
    var maxP = Math.max.apply(null, res.priority_probs) || 1;
    DIM.priority.forEach(function (p, i) {
      var v = res.priority_probs[i] * 100;
      meterRow(probs, p, null, v, maxP * 100, p === res.priority ? 'good' : null, nf1(v) + '%');
    });
    out.appendChild(el('p', 'note', 'Class probabilities from the priority model:'));
    out.appendChild(probs);
  }

  var kv = el('dl', 'kv');
  function pair(k, v) {
    kv.appendChild(el('dt', null, k));
    var dd = el('dd');
    if (v instanceof Node) dd.appendChild(v); else dd.textContent = v;
    kv.appendChild(dd);
  }
  pair('Severity', res.severity + '  (' + (res.severity_source || '') + ')');
  var owner = el('span', 'route-line');
  owner.appendChild(document.createTextNode(res.owner));
  if (res.priority === 'P1' || res.priority === 'P2') {
    owner.appendChild(badge('critical', 'Escalated to the team lead'));
  }
  pair('Assigned to', owner);
  pair('Root cause', res.root_cause);
  pair('Suggested fix', res.suggested_fix);
  if (res.score !== undefined && res.score !== null) pair('Impact score', String(res.score) + '  (severity + environment + blocking error code)');
  out.appendChild(kv);

  var flow = el('div', 'stage-flow');
  ['New', 'Assigned', 'In Progress', 'Fixed', 'Pending Retest', 'Verified', 'Closed'].forEach(function (st, i) {
    if (i) flow.appendChild(el('span', 'arrow', '→'));
    flow.appendChild(el('span', 'stage' + (i < 2 ? ' on' : ''), st));
  });
  out.appendChild(el('p', 'note', 'Life cycle position for a newly triaged ticket:'));
  out.appendChild(flow);
}

/* ── Bug explorer ───────────────────────────────────────────────────── */
defineView('explorer', 'Bug explorer', 'Records',
  'Every bug record behind the charts, filtered by the same controls. Sort any column, search across ID, module, feature, component and category, and export the current slice.',
  function (body) {
    var c = card(body, { title: 'Bug records', note: '' });
    var head = el('div', 'explorer-head');
    var search = el('input');
    search.type = 'search';
    search.placeholder = 'Search id, module, feature, component, category…';
    search.style.minWidth = '260px';
    search.style.padding = '7px 10px';
    search.style.borderRadius = '8px';
    search.style.border = '1px solid var(--border)';
    search.style.background = 'var(--surface-2)';
    search.style.color = 'var(--ink)';
    search.style.font = 'inherit';
    head.appendChild(search);
    var exportBtn = el('button', 'btn secondary', 'Export CSV');
    exportBtn.type = 'button';
    head.appendChild(exportBtn);
    var pager = el('div', 'pager');
    head.appendChild(pager);
    c.body.insertBefore(head, c.body.firstChild);
    var tableHost = el('div', 'table-wrap'); c.body.appendChild(tableHost);

    var page = 0, PER = 40, matched = [];
    var sortCol = null, sortAsc = false;

    function collect() {
      var q = search.value.trim().toLowerCase();
      matched = [];
      for (var i = 0; i < N; i++) {
        if (!mask[i]) continue;
        if (q) {
          var hay = bugId(i).toLowerCase() + ' ' + DIM.module[COL.module[i]].toLowerCase() + ' ' +
            DIM.feature[COL.feature[i]].toLowerCase() + ' ' + DIM.component[COL.component[i]].toLowerCase() + ' ' +
            DIM.category[COL.category[i]].toLowerCase() + ' ' + DIM.status[COL.status[i]].toLowerCase();
          if (hay.indexOf(q) < 0) continue;
        }
        matched.push(i);
      }
      page = 0;
      draw();
    }

    function rowOf(i) {
      return {
        i: i, id: bugId(i), sprint: DIM.sprint[COL.sprint[i]], release: DIM.release[COL.release[i]],
        module: DIM.module[COL.module[i]], feature: DIM.feature[COL.feature[i]],
        component: DIM.component[COL.component[i]], priority: DIM.priority[COL.priority[i]],
        severity: DIM.severity[COL.severity[i]], status: DIM.status[COL.status[i]],
        resolution: DIM.resolution[COL.resolution[i]],
        rootCause: LOOK.rootCauseByCategory[COL.category[i]],
        opened: dayToDate(COL.createdDay[i]),
        closed: COL.resDays[i] >= 0 ? dayToDate(COL.createdDay[i] + COL.resDays[i]) : null,
        resDays: COL.resDays[i] >= 0 ? COL.resDays[i] : null,
        owner: DIM.team[COL.team[i]]
      };
    }

    var COLUMNS = [
      { label: 'Bug ID', value: function (r) { return r.id; }, render: function (r) { return el('span', 'mono', r.id); } },
      { label: 'Sprint', value: function (r) { return r.sprint; } },
      { label: 'Release', value: function (r) { return r.release; } },
      { label: 'Module', value: function (r) { return r.module; } },
      { label: 'Feature', value: function (r) { return r.feature; } },
      { label: 'Component', value: function (r) { return r.component; } },
      { label: 'Priority', value: function (r) { return r.priority; } },
      { label: 'Severity', value: function (r) { return r.severity; } },
      { label: 'Status', value: function (r) { return r.status; } },
      { label: 'Resolution', value: function (r) { return r.resolution; } },
      { label: 'Opened', value: function (r) { return r.opened; } },
      { label: 'Closed', value: function (r) { return r.closed; }, render: function (r) { return r.closed || '—'; } },
      { label: 'Days', value: function (r) { return r.resDays; }, render: function (r) { return r.resDays === null ? '—' : nf(r.resDays); } },
      { label: 'Routed owner', value: function (r) { return r.owner; } },
      { label: 'Root cause', value: function (r) { return r.rootCause; } }
    ];

    function draw() {
      var rows = matched.map(rowOf);
      if (sortCol !== null) {
        var col = COLUMNS[sortCol], dir = sortAsc ? 1 : -1;
        rows.sort(function (a, b) {
          var av = col.value(a), bv = col.value(b);
          if (av === null || av === undefined) av = sortAsc ? Infinity : -Infinity;
          if (bv === null || bv === undefined) bv = sortAsc ? Infinity : -Infinity;
          if (typeof av === 'string' || typeof bv === 'string') return dir * String(av).localeCompare(String(bv));
          return dir * (av - bv);
        });
      }
      var pages = Math.max(1, Math.ceil(rows.length / PER));
      page = Math.min(page, pages - 1);
      var slice = rows.slice(page * PER, page * PER + PER);

      var spec = { rows: slice, columns: COLUMNS, sortCol: sortCol, sortAsc: sortAsc };
      dataTable(tableHost, spec);
      /* dataTable re-sorts its own slice; take over sorting for the whole set */
      var ths = tableHost.querySelectorAll('thead th');
      Array.prototype.forEach.call(ths, function (th, ci) {
        var fresh = th.cloneNode(true);
        th.parentNode.replaceChild(fresh, th);
        fresh.tabIndex = 0;
        var apply = function () {
          sortAsc = sortCol === ci ? !sortAsc : true;
          sortCol = ci; draw();
        };
        fresh.addEventListener('click', apply);
        fresh.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); apply(); } });
      });

      clear(pager);
      var prev = el('button', null, '‹'); prev.type = 'button'; prev.disabled = page === 0;
      prev.setAttribute('aria-label', 'Previous page');
      prev.addEventListener('click', function () { page--; draw(); });
      var next = el('button', null, '›'); next.type = 'button'; next.disabled = page >= pages - 1;
      next.setAttribute('aria-label', 'Next page');
      next.addEventListener('click', function () { page++; draw(); });
      pager.appendChild(prev);
      pager.appendChild(el('span', null, nf(rows.length) + ' records · page ' + (page + 1) + ' of ' + nf(pages)));
      pager.appendChild(next);
      c.card.querySelector('.note').textContent =
        'Showing ' + nf(slice.length) + ' of ' + nf(rows.length) + ' records matching the current filters.';
    }

    search.addEventListener('input', debounce(collect, 180));
    exportBtn.addEventListener('click', function () {
      var headers = COLUMNS.map(function (c2) { return c2.label; });
      var lines = [headers.join(',')];
      matched.forEach(function (i) {
        var r = rowOf(i);
        lines.push(COLUMNS.map(function (c2) {
          var v = c2.value(r);
          v = (v === null || v === undefined) ? '' : String(v);
          return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
        }).join(','));
      });
      var blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' });
      var a = el('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'bug_records_filtered.csv';
      document.body.appendChild(a); a.click();
      setTimeout(function () { URL.revokeObjectURL(a.href); document.body.removeChild(a); }, 0);
    });

    return collect;
  });

function debounce(fn, ms) {
  var t = null;
  return function () { var a = arguments, self = this; clearTimeout(t); t = setTimeout(function () { fn.apply(self, a); }, ms); };
}

/* ══ 8. Chrome: tabs, filters, chips, theme ═════════════════════════════ */

var VIEW_ORDER = [
  ['overview', 'Overview'],
  ['trends', 'Trends & resolution'],
  ['distribution', 'Distribution'],
  ['quality', 'Quality & team'],
  ['model', 'Models & triage'],
  ['explorer', 'Bug explorer']
];

function buildTabs() {
  var bar = document.getElementById('tabs');
  VIEW_ORDER.forEach(function (v) {
    var b = el('button', 'tab', v[1]);
    b.type = 'button';
    b.id = 'tab-' + v[0];
    b.setAttribute('role', 'tab');
    b.setAttribute('aria-selected', state.view === v[0]);
    b.setAttribute('aria-controls', 'view-' + v[0]);
    b.addEventListener('click', function () { showView(v[0]); });
    bar.appendChild(b);
  });
}

function showView(id) {
  state.view = id;
  VIEW_ORDER.forEach(function (v) {
    var tab = document.getElementById('tab-' + v[0]);
    if (tab) tab.setAttribute('aria-selected', v[0] === id);
    if (views[v[0]].root) views[v[0]].root.hidden = v[0] !== id;
  });
  var root = viewRoot(id);
  root.hidden = false;
  if (views[id].update) views[id].update();
  if (location.hash !== '#' + id) history.replaceState(null, '', '#' + id);
  window.scrollTo({ top: 0, behavior: 'auto' });
}

function buildFilters() {
  var bar = document.getElementById('filters');

  function multi(dim) {
    var g = el('div', 'fgroup');
    var lab = el('label', null, DIM_LABEL[dim]);
    lab.setAttribute('for', 'flt-' + dim);
    var sel = el('select');
    sel.id = 'flt-' + dim;
    sel.multiple = false;
    sel.size = 1;
    var any = el('option', null, 'All');
    any.value = '';
    sel.appendChild(any);
    DIM[dim].forEach(function (v, k) {
      var o = el('option', null, v);
      o.value = String(k);
      sel.appendChild(o);
    });
    sel.addEventListener('change', function () {
      state.sel[dim].clear();
      if (sel.value !== '') state.sel[dim].add(parseInt(sel.value, 10));
      recompute();
    });
    g.appendChild(lab); g.appendChild(sel);
    bar.appendChild(g);
    return sel;
  }

  ['module', 'priority', 'severity', 'status', 'release', 'team'].forEach(multi);

  var gState = el('div', 'fgroup');
  gState.appendChild(el('label', null, 'Bug state'));
  var seg = el('div', 'seg');
  [['all', 'All'], ['open', 'Open'], ['closed', 'Closed']].forEach(function (o) {
    var b = el('button', null, o[1]);
    b.type = 'button';
    b.dataset.life = o[0];
    b.setAttribute('aria-pressed', state.life === o[0]);
    b.addEventListener('click', function () { state.life = o[0]; syncControls(); recompute(); });
    seg.appendChild(b);
  });
  gState.appendChild(seg);
  bar.appendChild(gState);

  var gRange = el('div', 'fgroup');
  gRange.appendChild(el('label', null, 'Sprint range'));
  var rw = el('div', 'range-wrap');
  var lo = el('input'), hi = el('input');
  [lo, hi].forEach(function (r, i) {
    r.type = 'range'; r.min = 0; r.max = DIM.sprint.length - 1;
    r.value = i === 0 ? 0 : DIM.sprint.length - 1;
    r.id = 'flt-sprint-' + (i ? 'hi' : 'lo');
    r.setAttribute('aria-label', (i ? 'Last' : 'First') + ' sprint in range');
  });
  var read = el('div', 'range-read');
  function syncRange() {
    var a = parseInt(lo.value, 10), b = parseInt(hi.value, 10);
    if (a > b) { if (this === lo) { b = a; hi.value = a; } else { a = b; lo.value = b; } }
    state.sprint = [a, b];
    read.textContent = DIM.sprint[a] + ' → ' + DIM.sprint[b];
    recompute();
  }
  lo.addEventListener('input', syncRange);
  hi.addEventListener('input', syncRange);
  read.textContent = DIM.sprint[0] + ' → ' + DIM.sprint[DIM.sprint.length - 1];
  rw.appendChild(lo); rw.appendChild(hi);
  gRange.appendChild(rw); gRange.appendChild(read);
  bar.appendChild(gRange);

  var gReset = el('div', 'fgroup');
  gReset.appendChild(el('label', null, ' '));
  var rb = el('button', 'icon-btn', 'Reset all filters');
  rb.type = 'button';
  rb.addEventListener('click', resetFilters);
  gReset.appendChild(rb);
  bar.appendChild(gReset);
}

function syncControls() {
  FILTERABLE.forEach(function (d) {
    var sel = document.getElementById('flt-' + d);
    if (!sel) return;
    var s2 = state.sel[d];
    sel.value = s2.size === 1 ? String(Array.from(s2)[0]) : '';
  });
  var lo = document.getElementById('flt-sprint-lo'), hi = document.getElementById('flt-sprint-hi');
  if (lo) { lo.value = state.sprint[0]; hi.value = state.sprint[1]; }
  Array.prototype.forEach.call(document.querySelectorAll('[data-life]'), function (b) {
    b.setAttribute('aria-pressed', b.dataset.life === state.life);
  });
}

function renderChips() {
  var bar = document.getElementById('chips');
  clear(bar);
  var items = [];
  FILTERABLE.forEach(function (d) {
    state.sel[d].forEach(function (k) {
      items.push({ label: DIM_LABEL[d] + ': ' + DIM[d][k], clear: function () { state.sel[d].delete(k); syncControls(); recompute(); } });
    });
  });
  if (state.life !== 'all') {
    items.push({ label: 'State: ' + state.life, clear: function () { state.life = 'all'; syncControls(); recompute(); } });
  }
  if (state.sprint[0] !== 0 || state.sprint[1] !== DIM.sprint.length - 1) {
    items.push({ label: 'Sprints ' + DIM.sprint[state.sprint[0]] + '–' + DIM.sprint[state.sprint[1]],
      clear: function () { state.sprint = [0, DIM.sprint.length - 1]; syncControls(); recompute(); } });
  }
  if (!items.length) return;
  bar.appendChild(el('span', 'chips-label', 'Filtered by'));
  items.forEach(function (it) {
    var c = el('button', 'chip');
    c.type = 'button';
    c.appendChild(el('span', null, it.label));
    c.appendChild(el('span', 'x', '×'));
    c.setAttribute('aria-label', 'Remove filter ' + it.label);
    c.addEventListener('click', it.clear);
    bar.appendChild(c);
  });
  var all = el('button', 'chip chip-clear', 'Clear all');
  all.type = 'button';
  all.addEventListener('click', resetFilters);
  bar.appendChild(all);
}

function renderScope() {
  var note = document.getElementById('scope');
  clear(note);
  if (anyFilter()) {
    note.appendChild(document.createTextNode('Showing '));
    note.appendChild(el('strong', null, nf(A.total)));
    note.appendChild(document.createTextNode(' of ' + nf(N) + ' bug records (' +
      nf1(N ? A.total / N * 100 : 0) + '% of the dataset) · ' +
      nf(A.open) + ' open · ' + nf(A.closed) + ' closed'));
  } else {
    note.appendChild(document.createTextNode('Showing all '));
    note.appendChild(el('strong', null, nf(N)));
    note.appendChild(document.createTextNode(' bug records · ' + DIM.sprint.length + ' sprints · ' +
      DIM.release.length + ' releases · ' + DIM.module.length + ' modules · snapshot ' + D.meta.snapshot));
  }
}

function initTheme() {
  var saved = null;
  try { saved = localStorage.getItem('bugdash-theme'); } catch (e) { /* file:// with storage blocked */ }
  if (saved === 'light') document.documentElement.setAttribute('data-theme', 'light');
  var btn = document.getElementById('theme-btn');
  function label() {
    var light = document.documentElement.getAttribute('data-theme') === 'light';
    btn.textContent = light ? '◑ Dark' : '☀ Light';
    btn.setAttribute('aria-label', light ? 'Switch to dark theme' : 'Switch to light theme');
  }
  label();
  btn.addEventListener('click', function () {
    var light = document.documentElement.getAttribute('data-theme') === 'light';
    if (light) document.documentElement.removeAttribute('data-theme');
    else document.documentElement.setAttribute('data-theme', 'light');
    try { localStorage.setItem('bugdash-theme', light ? 'dark' : 'light'); } catch (e) { /* ignore */ }
    label();
    if (views[state.view] && views[state.view].update) views[state.view].update();
  });
}

/* ══ 9. Recompute cycle ══════════════════════════════════════════════════ */

/* Coalesces a burst of filter changes into one pass. Deliberately a timer and
   not requestAnimationFrame: rAF is suspended in a background tab, so a filter
   changed just before switching away would sit un-applied. */
var pending = null;
function recompute() {
  if (pending) return;
  document.body.classList.add('recomputing');
  pending = setTimeout(function () {
    pending = null;
    buildMask();
    urgentOpenCache = {};
    aggregate();
    renderChips();
    renderScope();
    if (views[state.view] && views[state.view].update) views[state.view].update();
    document.body.classList.remove('recomputing');
  }, 0);
}

window.addEventListener('resize', debounce(function () {
  if (views[state.view] && views[state.view].update) views[state.view].update();
}, 180));

/* ══ 10. Boot ════════════════════════════════════════════════════════════ */

function boot() {
  document.getElementById('meta-generated').textContent = D.meta.generated;
  document.getElementById('meta-rows').textContent = nf(N) + ' records';
  buildTabs();
  buildFilters();
  initTheme();
  probeLive();

  var initial = (location.hash || '').replace('#', '');
  if (!views[initial]) initial = 'overview';
  state.view = initial;
  VIEW_ORDER.forEach(function (v) { if (v[0] !== initial) { /* built lazily */ } });

  buildMask();
  aggregate();
  renderChips();
  renderScope();
  showView(initial);
  document.body.classList.add('ready');
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
else boot();

})();
