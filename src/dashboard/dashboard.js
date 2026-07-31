/* Bug Analytics Dashboard — client side.
 *
 * The generator embeds every bug record as compact character-encoded columns.
 * Selecting a filter recomputes the matching row set and redraws every chart,
 * so the whole page always describes one consistent selection.
 *
 * No libraries: charts are SVG built as markup strings.
 */
(function () {
'use strict';

var P = JSON.parse(document.getElementById('payload').textContent);

/* ── Decoding ─────────────────────────────────────────────────────────────
 * Categorical columns are one character per row, counts are two characters
 * (base 64, most significant first). 4095 is the "no value" sentinel.
 */
var INDEX = new Int16Array(128).fill(-1);
for (var a = 0; a < P.alphabet.length; a++) INDEX[P.alphabet.charCodeAt(a)] = a;

function decode1(s) {
  var out = new Uint8Array(s.length);
  for (var i = 0; i < s.length; i++) out[i] = INDEX[s.charCodeAt(i)];
  return out;
}

function decode2(s) {
  var n = s.length >> 1, out = new Uint16Array(n);
  for (var i = 0; i < n; i++) {
    out[i] = (INDEX[s.charCodeAt(2 * i)] << 6) | INDEX[s.charCodeAt(2 * i + 1)];
  }
  return out;
}

var N        = P.n;
var NULL2    = P.null2;
var created  = decode2(P.cols.created);
var days     = decode2(P.cols.days);

/* Sprint and release are a pure function of the report date, so they are
 * rebuilt here instead of being shipped as their own columns. P.maxSprint is
 * the cap stage 01 applied: the data ends mid-sprint, and that trailing stub is
 * folded into the last full sprint rather than plotted as its own short bar. */
function releaseLabel(sprintIdx) {
  var r = Math.floor(sprintIdx / P.sprintsPerRelease);
  return 'v' + (1 + Math.floor(r / P.minorsPerMajor)) + '.' + (r % P.minorsPerMajor);
}

var sprintCodes = new Uint8Array(N), releaseCodes = new Uint8Array(N);
var sprintLabels = [], releaseLabels = [], releaseSeen = {};
var maxSprint = 0;
for (var i = 0; i < N; i++) {
  var s = Math.min(Math.floor(created[i] / P.sprintDays), P.maxSprint);
  sprintCodes[i] = s;
  if (s > maxSprint) maxSprint = s;
}
for (var s2 = 0; s2 <= maxSprint; s2++) {
  sprintLabels.push('SPR-' + String(s2 + 1).padStart(2, '0'));
  var rl = releaseLabel(s2);
  if (!(rl in releaseSeen)) { releaseSeen[rl] = releaseLabels.length; releaseLabels.push(rl); }
}
for (var j = 0; j < N; j++) releaseCodes[j] = releaseSeen[releaseLabel(sprintCodes[j])];

/* Day index of the snapshot, used to age the bugs that are still open. */
var snapshotDay = 0;
var maxCreatedDay = 0;
for (var k = 0; k < N; k++) {
  var end = created[k] + (days[k] === NULL2 ? 0 : days[k]);
  if (end > snapshotDay) snapshotDay = end;
  if (created[k] > maxCreatedDay) maxCreatedDay = created[k];
}

/* ── Dimensions ──────────────────────────────────────────────────────────── */
var DIM = {
  release:     { label: 'Release version', codes: releaseCodes, labels: releaseLabels },
  sprint:      { label: 'Sprint',          codes: sprintCodes,  labels: sprintLabels },
  module:      { label: 'Module' },
  feature:     { label: 'Feature' },
  component:   { label: 'Component' },
  priority:    { label: 'Priority' },
  severity:    { label: 'Severity' },
  status:      { label: 'Status' },
  resolution:  { label: 'Resolution' },
  rootcause:   { label: 'Root cause' },
  team:        { label: 'Team' },
  environment: { label: 'Environment' }
};

Object.keys(DIM).forEach(function (key) {
  if (DIM[key].codes) return;
  DIM[key].codes = decode1(P.cols[key]);
  DIM[key].labels = P.dims[key];
});

var FILTER_KEYS = ['release', 'sprint', 'module', 'feature', 'component', 'priority',
                   'severity', 'status', 'resolution', 'rootcause', 'team', 'environment'];

var state = { filters: {}, rows: null };
FILTER_KEYS.forEach(function (key) { state.filters[key] = -1; });   /* -1 = All */

/* ── Formatting ──────────────────────────────────────────────────────────── */
function fmt(n)  { return Math.round(n).toLocaleString('en-US'); }
function fmt1(n) { return (Math.round(n * 10) / 10).toFixed(1); }
function pct(n)  { return fmt1(n) + '%'; }
function esc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
                  .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
/* Trims the middle rather than the tail: the root causes in this dataset all
 * share the same opening words and differ only at the end. */
function clip(s, max) {
  s = String(s);
  if (s.length <= max) return s;
  var head = Math.floor((max - 1) * 0.4);
  return s.slice(0, head) + '…' + s.slice(s.length - (max - 1 - head));
}

/* ── Filtering ───────────────────────────────────────────────────────────── */
function applyFilters() {
  var active = [];
  FILTER_KEYS.forEach(function (key) {
    if (state.filters[key] >= 0) active.push([DIM[key].codes, state.filters[key]]);
  });

  var rows = [];
  outer:
  for (var i = 0; i < N; i++) {
    for (var f = 0; f < active.length; f++) {
      if (active[f][0][i] !== active[f][1]) continue outer;
    }
    rows.push(i);
  }
  state.rows = rows;
}

/* ── Aggregation ─────────────────────────────────────────────────────────── */
function tally(rows, key) {
  var dim = DIM[key], counts = new Float64Array(dim.labels.length);
  for (var i = 0; i < rows.length; i++) counts[dim.codes[rows[i]]]++;
  return counts;
}

/* Mean days-to-close per level of a dimension, over closed bugs only. */
function meanDays(rows, key) {
  var dim = DIM[key], n = dim.labels.length;
  var sum = new Float64Array(n), cnt = new Float64Array(n);
  for (var i = 0; i < rows.length; i++) {
    var r = rows[i];
    if (days[r] === NULL2) continue;
    sum[dim.codes[r]] += days[r];
    cnt[dim.codes[r]]++;
  }
  var out = new Array(n);
  for (var j = 0; j < n; j++) out[j] = cnt[j] ? sum[j] / cnt[j] : null;
  return out;
}

function topLevels(counts, labels, limit) {
  var items = [];
  for (var i = 0; i < counts.length; i++) {
    if (counts[i] > 0) items.push({ code: i, label: labels[i], value: counts[i] });
  }
  items.sort(function (a, b) { return b.value - a.value; });
  return limit ? items.slice(0, limit) : items;
}

function slaTarget(priorityCode) {
  return P.sla[DIM.priority.labels[priorityCode]] || 30;
}

/* ── SVG primitives ──────────────────────────────────────────────────────── */
var PALETTE = ['#1565c0', '#00897b', '#7b1fa2', '#ef6c00', '#c62828', '#2e7d32',
               '#5d4037', '#455a64', '#ad1457', '#0277bd', '#6a1b9a', '#f9a825',
               '#4527a0', '#00838f', '#d84315', '#558b2f'];
var PRIORITY_COLOR = { P1: '#b71c1c', P2: '#e53935', P3: '#fb8c00', P4: '#f9a825', P5: '#43a047' };
var RESOLUTION_COLOR = {
  Fixed: '#2e7d32', Unresolved: '#c62828', Duplicate: '#f9a825',
  Invalid: '#7b1fa2', "Won't Fix": '#455a64'
};

/* Colour carries meaning or it carries nothing.
 *   priority / resolution : fixed semantic colours, identical in every chart
 *   small dimensions      : a stable colour per level, so a module keeps its
 *                           colour across the bar chart, the density chart and
 *                           anywhere else it appears
 *   large dimensions      : a single-hue ramp ordered by rank, because sixteen
 *                           unrelated hues would imply differences that the
 *                           categories do not have
 */
function ramp(t) {
  var from = [21, 101, 192], to = [125, 189, 240];
  return 'rgb(' + from.map(function (c, i) {
    return Math.round(c + (to[i] - c) * t);
  }).join(',') + ')';
}

function colorFor(key, label, code, rank, count) {
  if (key === 'priority' && PRIORITY_COLOR[label]) return PRIORITY_COLOR[label];
  if (key === 'resolution' && RESOLUTION_COLOR[label]) return RESOLUTION_COLOR[label];
  if (key && DIM[key] && DIM[key].labels.length <= 8) {
    return PALETTE[code % PALETTE.length];
  }
  return ramp(count > 1 ? rank / (count - 1) : 0);
}

function svgOpen(w, h) {
  return '<svg viewBox="0 0 ' + w + ' ' + h + '" role="img" ' +
         'preserveAspectRatio="xMidYMid meet">';
}

function empty(el) {
  el.innerHTML = '<p class="empty">No bugs match the current filters.</p>';
}

/* Round tick values whose top tick is always at or above the largest datum —
 * the last tick doubles as the axis maximum, so stopping below it would draw
 * bars taller than the plot area. */
function niceTicks(max, count) {
  if (max <= 0) return [0, 1];
  var raw = max / count;
  var mag = Math.pow(10, Math.floor(Math.log10(raw)));
  var step = [1, 2, 2.5, 5, 10].map(function (m) { return m * mag; })
                               .find(function (v) { return v >= raw; }) || 10 * mag;
  var ticks = [];
  for (var v = 0; v < max; v += step) ticks.push(v);
  ticks.push(ticks.length ? ticks[ticks.length - 1] + step : step);
  return ticks;
}

/* ── Horizontal bars ─────────────────────────────────────────────────────── */
function drawBarsH(el, items, opts) {
  opts = opts || {};
  if (!items.length) { empty(el); return; }

  var W = opts.width || 520;
  var labelW = opts.labelWidth || 148;
  var rowH = 25, pad = 12;
  var H = items.length * rowH + pad * 2;
  var chartW = W - labelW - 92;
  var max = Math.max.apply(null, items.map(function (d) { return d.value; })) || 1;
  var total = items.reduce(function (a, d) { return a + d.value; }, 0);

  var out = [svgOpen(W, H)];
  items.forEach(function (d, i) {
    var y = pad + i * rowH;
    var w = Math.max(1, (d.value / max) * chartW);
    var color = d.color || colorFor(opts.dim, d.label, d.code, i, items.length);
    /* Values are bug counts unless the caller says otherwise — the density
     * chart plots a ratio, where a "share of total" reading would be nonsense. */
    var share = total ? (d.value / total * 100) : 0;
    var main = d.tipMain || (fmt(d.value) + ' bugs (' + fmt1(share) + '%)');
    var tip = d.label + ' — ' + main + (d.tip ? '\n' + d.tip : '');
    out.push(
      '<text class="label-text" x="' + (labelW - 8) + '" y="' + (y + 15) +
        '" text-anchor="end">' + esc(clip(d.label, opts.labelChars || 22)) + '</text>',
      '<rect class="bar" x="' + labelW + '" y="' + (y + 4) + '" width="' + w +
        '" height="' + (rowH - 9) + '" rx="3" fill="' + color + '"' +
        (opts.dim ? ' data-dim="' + opts.dim + '" data-code="' + d.code + '"' : '') +
        '><title>' + esc(tip) + '</title></rect>',
      '<text class="value-text" x="' + (labelW + w + 7) + '" y="' + (y + 15) + '">' +
        esc(d.display || fmt(d.value)) + '</text>'
    );
  });
  out.push('</svg>');
  el.innerHTML = out.join('');
}

/* ── Vertical bars, optionally with a target marker per column ───────────── */
function drawBarsV(el, items, opts) {
  opts = opts || {};
  if (!items.length) { empty(el); return; }

  var W = opts.width || 520, H = opts.height || 268;
  var left = 46, right = 12, top = 22, bottom = opts.rotate ? 62 : 34;
  var chartW = W - left - right, chartH = H - top - bottom;
  var values = items.map(function (d) { return d.value; })
                    .concat(items.map(function (d) { return d.target || 0; }));
  var max = Math.max.apply(null, values) || 1;
  var ticks = niceTicks(max, 4);
  var scaleMax = ticks[ticks.length - 1] || max;
  var slot = chartW / items.length;
  var barW = Math.min(slot * 0.62, 68);

  var out = [svgOpen(W, H)];
  ticks.forEach(function (t) {
    var y = top + chartH - (t / scaleMax) * chartH;
    out.push('<line class="grid-line" x1="' + left + '" y1="' + y + '" x2="' + (W - right) +
             '" y2="' + y + '"/>',
             '<text class="tick-text" x="' + (left - 7) + '" y="' + (y + 4) +
             '" text-anchor="end">' + esc(opts.tickFormat ? opts.tickFormat(t) : fmt(t)) +
             '</text>');
  });

  items.forEach(function (d, i) {
    var h = Math.max(1, (d.value / scaleMax) * chartH);
    var x = left + i * slot + (slot - barW) / 2;
    var y = top + chartH - h;
    var color = d.color || colorFor(opts.dim, d.label, d.code, i, items.length);
    out.push(
      '<rect class="bar" x="' + x + '" y="' + y + '" width="' + barW + '" height="' + h +
        '" rx="3" fill="' + color + '"' +
        (opts.dim ? ' data-dim="' + opts.dim + '" data-code="' + d.code + '"' : '') +
        '><title>' + esc(d.tip || (d.label + ' — ' + fmt(d.value))) + '</title></rect>'
    );

    /* The target marker can sit above a short bar, so the value label clears
     * whichever of the two is higher. */
    var labelY = y;
    if (d.target != null) {
      var ty = top + chartH - (d.target / scaleMax) * chartH;
      out.push('<line x1="' + (x - 5) + '" y1="' + ty + '" x2="' + (x + barW + 5) +
               '" y2="' + ty + '" stroke="#1565c0" stroke-width="2" ' +
               'stroke-dasharray="5 3"><title>SLA target ' + d.target + ' days</title></line>');
      labelY = Math.min(labelY, ty);
    }
    out.push('<text class="value-text" x="' + (x + barW / 2) + '" y="' + (labelY - 6) +
             '" text-anchor="middle">' + esc(d.display || fmt(d.value)) + '</text>');

    var lx = left + i * slot + slot / 2;
    if (opts.rotate) {
      out.push('<text class="tick-text" transform="translate(' + lx + ',' +
               (top + chartH + 12) + ') rotate(-38)" text-anchor="end">' +
               esc(clip(d.label, 16)) + '</text>');
    } else {
      out.push('<text class="tick-text" x="' + lx + '" y="' + (top + chartH + 16) +
               '" text-anchor="middle">' + esc(clip(d.label, 12)) + '</text>');
    }
  });

  out.push('<line class="axis-line" x1="' + left + '" y1="' + (top + chartH) +
           '" x2="' + (W - right) + '" y2="' + (top + chartH) + '"/>');
  if (opts.legend) out.push(legendMarkup(opts.legend, left, H - 12));
  out.push('</svg>');
  el.innerHTML = out.join('');
}

function legendMarkup(entries, x, y) {
  var out = [], cursor = x;
  entries.forEach(function (e) {
    out.push('<rect x="' + cursor + '" y="' + (y - 8) + '" width="10" height="10" rx="2" fill="' +
             e.color + '"/>',
             '<text class="legend-text" x="' + (cursor + 14) + '" y="' + y + '">' +
             esc(e.label) + '</text>');
    cursor += 22 + e.label.length * 6.2;
  });
  return out.join('');
}

/* ── Donut ───────────────────────────────────────────────────────────────── */
function drawDonut(el, items, opts) {
  opts = opts || {};
  if (!items.length) { empty(el); return; }

  var W = opts.width || 520, H = 250;
  var cx = 118, cy = H / 2, outer = 92, inner = 55;
  var total = items.reduce(function (a, d) { return a + d.value; }, 0) || 1;
  var out = [svgOpen(W, H)], angle = -Math.PI / 2;

  items.forEach(function (d, i) {
    var sweep = (d.value / total) * Math.PI * 2;
    var end = angle + sweep;
    var color = d.color || colorFor(opts.dim, d.label, d.code, i, items.length);
    var large = sweep > Math.PI ? 1 : 0;
    var path = [
      'M', cx + outer * Math.cos(angle), cy + outer * Math.sin(angle),
      'A', outer, outer, 0, large, 1, cx + outer * Math.cos(end), cy + outer * Math.sin(end),
      'L', cx + inner * Math.cos(end), cy + inner * Math.sin(end),
      'A', inner, inner, 0, large, 0, cx + inner * Math.cos(angle), cy + inner * Math.sin(angle),
      'Z'
    ].join(' ');
    out.push('<path class="bar" d="' + path + '" fill="' + color + '" stroke="var(--card)" ' +
             'stroke-width="2"' +
             (opts.dim ? ' data-dim="' + opts.dim + '" data-code="' + d.code + '"' : '') +
             '><title>' + esc(d.label + ' — ' + fmt(d.value) + ' (' +
             fmt1(d.value / total * 100) + '%)') + '</title></path>');
    angle = end;
  });

  out.push('<text class="label-text" x="' + cx + '" y="' + (cy - 2) +
           '" text-anchor="middle" style="font-size:19px;font-weight:650">' +
           esc(fmt(total)) + '</text>',
           '<text class="tick-text" x="' + cx + '" y="' + (cy + 15) +
           '" text-anchor="middle">bugs</text>');

  var lx = 236, ly = cy - (items.length * 21) / 2 + 12;
  items.forEach(function (d, i) {
    var color = d.color || colorFor(opts.dim, d.label, d.code, i, items.length);
    out.push('<rect class="bar" x="' + lx + '" y="' + (ly + i * 21 - 9) +
             '" width="11" height="11" rx="2" fill="' + color + '"' +
             (opts.dim ? ' data-dim="' + opts.dim + '" data-code="' + d.code + '"' : '') + '/>',
             '<text class="label-text" x="' + (lx + 18) + '" y="' + (ly + i * 21) + '">' +
             esc(clip(d.label, 18)) + '</text>',
             '<text class="value-text" x="' + (W - 14) + '" y="' + (ly + i * 21) +
             '" text-anchor="end">' + esc(fmt(d.value)) + '  (' +
             esc(fmt1(d.value / total * 100)) + '%)</text>');
  });

  out.push('</svg>');
  el.innerHTML = out.join('');
}

/* ── Combo: opened / closed bars plus a backlog line ─────────────────────── */
function drawSprintChart(el, rows) {
  var buckets = sprintLabels.map(function (label, i) {
    return { label: label, code: i, opened: 0, closed: 0 };
  });

  for (var i = 0; i < rows.length; i++) {
    var r = rows[i];
    buckets[sprintCodes[r]].opened++;
    if (days[r] !== NULL2) {
      var cs = Math.floor((created[r] + days[r]) / P.sprintDays);
      if (cs < buckets.length) buckets[cs].closed++;
    }
  }

  var running = 0;
  buckets.forEach(function (b) { running += b.opened - b.closed; b.backlog = running; });

  if (!rows.length) { empty(el); return; }

  var W = 1100, H = 320;
  var left = 58, right = 58, top = 18, bottom = 58;
  var chartW = W - left - right, chartH = H - top - bottom;
  var maxBar = Math.max(1, Math.max.apply(null, buckets.map(function (b) {
    return Math.max(b.opened, b.closed);
  })));
  var maxLine = Math.max(1, Math.max.apply(null, buckets.map(function (b) {
    return Math.abs(b.backlog);
  })));
  var ticks = niceTicks(maxBar, 4), scaleMax = ticks[ticks.length - 1] || maxBar;
  var slot = chartW / buckets.length, barW = Math.min(slot * 0.36, 22);

  var out = [svgOpen(W, H)];
  ticks.forEach(function (t) {
    var y = top + chartH - (t / scaleMax) * chartH;
    out.push('<line class="grid-line" x1="' + left + '" y1="' + y + '" x2="' + (W - right) +
             '" y2="' + y + '"/>',
             '<text class="tick-text" x="' + (left - 7) + '" y="' + (y + 4) +
             '" text-anchor="end">' + esc(fmt(t)) + '</text>');
  });

  buckets.forEach(function (b, i) {
    var cx = left + i * slot + slot / 2;
    [['opened', '#e53935', -1], ['closed', '#43a047', 1]].forEach(function (spec) {
      var h = Math.max(0.5, (b[spec[0]] / scaleMax) * chartH);
      var x = cx + (spec[2] < 0 ? -barW - 1 : 1);
      out.push('<rect class="bar" x="' + x + '" y="' + (top + chartH - h) + '" width="' + barW +
               '" height="' + h + '" rx="2" fill="' + spec[1] + '" data-dim="sprint" data-code="' +
               i + '"><title>' + esc(b.label + ' — ' + fmt(b[spec[0]]) + ' ' + spec[0] +
               '\nbacklog after sprint: ' + fmt(b.backlog)) + '</title></rect>');
    });
    if (i % 2 === 0 || buckets.length <= 16) {
      out.push('<text class="tick-text" transform="translate(' + cx + ',' +
               (top + chartH + 13) + ') rotate(-42)" text-anchor="end">' +
               esc(b.label) + '</text>');
    }
  });

  var points = buckets.map(function (b, i) {
    var x = left + i * slot + slot / 2;
    var y = top + chartH - (b.backlog / maxLine) * chartH;
    return x + ',' + y;
  }).join(' ');
  out.push('<polyline points="' + points + '" fill="none" stroke="#1565c0" stroke-width="2.4" ' +
           'stroke-linejoin="round"/>');
  buckets.forEach(function (b, i) {
    var x = left + i * slot + slot / 2;
    var y = top + chartH - (b.backlog / maxLine) * chartH;
    out.push('<circle cx="' + x + '" cy="' + y + '" r="3.2" fill="#1565c0"><title>' +
             esc(b.label + ' — open backlog ' + fmt(b.backlog)) + '</title></circle>');
  });

  [0, maxLine / 2, maxLine].forEach(function (v) {
    var y = top + chartH - (v / maxLine) * chartH;
    out.push('<text class="tick-text" x="' + (W - right + 7) + '" y="' + (y + 4) +
             '" fill="#1565c0">' + esc(fmt(v)) + '</text>');
  });

  out.push('<line class="axis-line" x1="' + left + '" y1="' + (top + chartH) + '" x2="' +
           (W - right) + '" y2="' + (top + chartH) + '"/>');
  out.push(legendMarkup([
    { label: 'Opened in sprint', color: '#e53935' },
    { label: 'Closed in sprint', color: '#43a047' },
    { label: 'Open backlog (right axis)', color: '#1565c0' }
  ], left, H - 10));
  out.push('</svg>');
  el.innerHTML = out.join('');
}

/* ── Heatmap ─────────────────────────────────────────────────────────────── */
function drawHeatmap(el, rows) {
  var mods = DIM.module.labels, pris = DIM.priority.labels;
  var matrix = mods.map(function () { return new Array(pris.length).fill(0); });
  for (var i = 0; i < rows.length; i++) {
    matrix[DIM.module.codes[rows[i]]][DIM.priority.codes[rows[i]]]++;
  }

  var order = mods.map(function (m, i) {
    return { i: i, total: matrix[i].reduce(function (a, b) { return a + b; }, 0) };
  }).filter(function (r) { return r.total > 0; })
    .sort(function (a, b) { return b.total - a.total; });

  if (!order.length) { empty(el); return; }

  var W = 1100, labelW = 190, right = 26, top = 30;
  var cellW = (W - labelW - right) / pris.length, cellH = 40;
  var H = top + order.length * cellH + 16;
  /* Scale the ramp across the range the cells actually occupy. Anchoring it at
   * zero would leave every cell in the same shade whenever the counts are all
   * of a similar size, which is exactly when the differences matter. */
  var flat = [];
  order.forEach(function (row) { flat = flat.concat(matrix[row.i]); });
  var max = Math.max.apply(null, flat) || 1;
  var min = Math.min.apply(null, flat);
  var span = (max - min) || 1;

  var out = [svgOpen(W, H)];
  pris.forEach(function (p, c) {
    out.push('<text class="tick-text" x="' + (labelW + c * cellW + cellW / 2) + '" y="' +
             (top - 10) + '" text-anchor="middle" style="font-weight:600">' + esc(p) + '</text>');
  });

  order.forEach(function (row, r) {
    var m = row.i, y = top + r * cellH;
    out.push('<text class="label-text" x="' + (labelW - 10) + '" y="' + (y + cellH / 2 + 4) +
             '" text-anchor="end">' + esc(clip(mods[m], 24)) + '</text>');
    pris.forEach(function (p, c) {
      var v = matrix[m][c];
      var t = (v - min) / span;
      /* pale blue where the load is lightest, deep red where it concentrates */
      var color = 'rgb(' + Math.round(228 - 110 * t) + ',' + Math.round(238 - 190 * t) +
                  ',' + Math.round(248 - 190 * t) + ')';
      out.push('<rect class="cell" x="' + (labelW + c * cellW + 1.5) + '" y="' + (y + 1.5) +
               '" width="' + (cellW - 3) + '" height="' + (cellH - 3) + '" rx="4" fill="' +
               color + '" data-dim="priority" data-code="' + c + '"><title>' +
               esc(mods[m] + ' / ' + p + ' — ' + fmt(v) + ' bugs') + '</title></rect>',
               '<text x="' + (labelW + c * cellW + cellW / 2) + '" y="' + (y + cellH / 2 + 4) +
               '" text-anchor="middle" style="font-size:12px;font-weight:640;fill:' +
               (t > 0.55 ? '#ffffff' : '#1b2230') + '">' + esc(fmt(v)) + '</text>');
    });
  });

  out.push('</svg>');
  el.innerHTML = out.join('');
}

/* ── Line / trend charts ─────────────────────────────────────────────────── *
 * Shared by the weekly bug-reporting trend and the sprint-over-sprint
 * resolution-time trend. Series may contain nulls (a sprint with nothing
 * closed yet) — those are rendered as gaps rather than dragging the line to
 * zero. */
var MONTH_ABBR = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

function weekLabel(originParts, weekIndex) {
  var d = new Date(originParts[0], originParts[1] - 1, originParts[2]);
  d.setDate(d.getDate() + weekIndex * 7);
  return MONTH_ABBR[d.getMonth()] + ' ' + d.getDate();
}

function rollingAvg(arr, window) {
  var out = new Array(arr.length);
  for (var i = 0; i < arr.length; i++) {
    var lo = Math.max(0, i - window + 1), sum = 0, cnt = 0;
    for (var j = lo; j <= i; j++) { sum += arr[j]; cnt++; }
    out[i] = cnt ? sum / cnt : null;
  }
  return out;
}

function drawLineChart(el, categories, series, opts) {
  opts = opts || {};
  if (!categories.length) { empty(el); return; }

  var W = opts.width || 1100;
  var doRotate = opts.rotate != null ? opts.rotate : categories.length > 10;
  var H = opts.height || 280;
  var left = 54, right = 20, top = 18, bottom = doRotate ? 56 : 34;
  var chartW = W - left - right, chartH = H - top - bottom;

  var allVals = [];
  series.forEach(function (s) {
    s.values.forEach(function (v) { if (v != null && isFinite(v)) allVals.push(v); });
  });
  var max = allVals.length ? Math.max.apply(null, allVals) : 1;
  var ticks = niceTicks(max, 4);
  var scaleMax = ticks[ticks.length - 1] || max || 1;

  var n = categories.length;
  var stepX = n > 1 ? chartW / (n - 1) : 0;
  function xAt(i) { return left + (n > 1 ? i * stepX : chartW / 2); }
  function yAt(v) { return top + chartH - (v / scaleMax) * chartH; }

  var out = [svgOpen(W, H)];
  ticks.forEach(function (t) {
    var y = yAt(t);
    out.push('<line class="grid-line" x1="' + left + '" y1="' + y + '" x2="' + (W - right) +
             '" y2="' + y + '"/>',
             '<text class="tick-text" x="' + (left - 7) + '" y="' + (y + 4) +
             '" text-anchor="end">' + esc(opts.yTickFormat ? opts.yTickFormat(t) : fmt(t)) +
             '</text>');
  });

  var everyLabel = opts.labelEvery || Math.max(1, Math.ceil(n / (doRotate ? 18 : 10)));
  categories.forEach(function (label, i) {
    if (i % everyLabel !== 0 && i !== n - 1) return;
    var x = xAt(i);
    if (doRotate) {
      out.push('<text class="tick-text" transform="translate(' + x + ',' + (top + chartH + 12) +
               ') rotate(-42)" text-anchor="end">' + esc(label) + '</text>');
    } else {
      out.push('<text class="tick-text" x="' + x + '" y="' + (top + chartH + 16) +
               '" text-anchor="middle">' + esc(label) + '</text>');
    }
  });

  series.forEach(function (s) {
    var segments = [], current = [];
    for (var i = 0; i < n; i++) {
      var v = s.values[i];
      if (v == null || !isFinite(v)) {
        if (current.length) { segments.push(current); current = []; }
        continue;
      }
      current.push([xAt(i), yAt(v), i, v]);
    }
    if (current.length) segments.push(current);

    segments.forEach(function (seg) {
      var linePts = seg.map(function (p) { return p[0] + ',' + p[1]; }).join(' ');
      if (s.area) {
        var baseline = yAt(0);
        var areaPts = linePts + ' ' + seg[seg.length - 1][0] + ',' + baseline +
                      ' ' + seg[0][0] + ',' + baseline;
        out.push('<polygon points="' + areaPts + '" fill="' + s.color + '" opacity="0.16"/>');
      }
      out.push('<polyline points="' + linePts + '" fill="none" stroke="' + s.color +
               '" stroke-width="' + (s.thick ? 2.6 : 2) + '" stroke-linejoin="round" ' +
               'stroke-linecap="round"' + (s.dashed ? ' stroke-dasharray="6 4"' : '') + '/>');
      if (!s.dashed) {
        seg.forEach(function (p) {
          out.push('<circle cx="' + p[0] + '" cy="' + p[1] + '" r="2.6" fill="' + s.color +
                   '"><title>' + esc(categories[p[2]] + ' — ' + s.label + ': ' + fmt1(p[3]) +
                   (opts.tipUnit ? ' ' + opts.tipUnit : '')) + '</title></circle>');
        });
      }
    });
  });

  out.push('<line class="axis-line" x1="' + left + '" y1="' + (top + chartH) + '" x2="' +
           (W - right) + '" y2="' + (top + chartH) + '"/>');
  if (series.length > 1) {
    out.push(legendMarkup(series.map(function (s) {
      return { label: s.label, color: s.color };
    }), left, H - 8));
  }
  out.push('</svg>');
  el.innerHTML = out.join('');
}

function renderBugTrend(rows) {
  var el = document.getElementById('chart-bugtrend');
  if (!rows.length) { empty(el); return; }
  var maxWeek = Math.floor(maxCreatedDay / 7);
  var counts = new Array(maxWeek + 1).fill(0);
  for (var i = 0; i < rows.length; i++) counts[Math.floor(created[rows[i]] / 7)]++;

  var originParts = P.origin.split('-').map(Number);
  var categories = counts.map(function (_, w) { return weekLabel(originParts, w); });
  var avg = rollingAvg(counts, 4);

  drawLineChart(el, categories, [
    { label: 'New bugs / week', color: '#5a7fb8', values: counts, area: true },
    { label: '4-week average', color: '#1565c0', values: avg, thick: true }
  ], { height: 260, yTickFormat: fmt, tipUnit: 'bugs' });
}

function renderResolutionTrend(rows) {
  var el = document.getElementById('chart-restime-trend');
  if (!rows.length) { empty(el); return; }
  var sums = new Array(sprintLabels.length).fill(0);
  var cnts = new Array(sprintLabels.length).fill(0);
  var overallSum = 0, overallCnt = 0;

  for (var i = 0; i < rows.length; i++) {
    var r = rows[i];
    if (days[r] === NULL2) continue;
    var s = sprintCodes[r];
    sums[s] += days[r];
    cnts[s]++;
    overallSum += days[r];
    overallCnt++;
  }

  var values = sums.map(function (sum, idx) { return cnts[idx] ? sum / cnts[idx] : null; });
  var series = [{ label: 'Avg resolution days', color: '#7b1fa2', values: values, thick: true }];
  if (overallCnt) {
    var overallAvg = overallSum / overallCnt;
    series.push({
      label: 'Overall average (' + fmt1(overallAvg) + 'd)', color: '#8b93a5',
      values: sprintLabels.map(function () { return overallAvg; }), dashed: true
    });
  }

  drawLineChart(el, sprintLabels, series, {
    height: 260, rotate: true, yTickFormat: function (t) { return fmt(t) + 'd'; }, tipUnit: 'days'
  });
}

/* ── Tables ──────────────────────────────────────────────────────────────── */
function renderTable(el, headers, rows) {
  if (!rows.length) { el.innerHTML = '<tbody><tr><td class="empty">No bugs match the ' +
                                     'current filters.</td></tr></tbody>'; return; }
  var head = '<thead><tr>' + headers.map(function (h) {
    return '<th>' + esc(h) + '</th>';
  }).join('') + '</tr></thead>';
  var body = '<tbody>' + rows.map(function (r) {
    return '<tr>' + r.map(function (cell) {
      if (cell && typeof cell === 'object') {
        return '<td class="' + cell.cls + '">' + esc(cell.text) + '</td>';
      }
      return '<td>' + esc(cell) + '</td>';
    }).join('') + '</tr>';
  }).join('') + '</tbody>';
  el.innerHTML = head + body;
}

function rateClass(value, goodAt, badAt) {
  if (value >= goodAt) return 'good';
  if (value <= badAt) return 'bad';
  return 'warn';
}

/* ── KPI strip ───────────────────────────────────────────────────────────── */
function renderKpis(rows) {
  var total = rows.length, closed = 0, sumDays = 0, openLate = 0, openAge = 0, reopened = 0;
  var closedDays = [], metSla = 0;
  var reopenedCode = DIM.status.labels.indexOf('Reopened');
  var modulesSeen = {};

  for (var i = 0; i < total; i++) {
    var r = rows[i];
    modulesSeen[DIM.module.labels[DIM.module.codes[r]]] = true;
    if (DIM.status.codes[r] === reopenedCode) reopened++;
    if (days[r] === NULL2) {
      var age = snapshotDay - created[r];
      openAge += age;
      if (age > slaTarget(DIM.priority.codes[r])) openLate++;
    } else {
      closed++;
      sumDays += days[r];
      closedDays.push(days[r]);
      if (days[r] <= slaTarget(DIM.priority.codes[r])) metSla++;
    }
  }

  closedDays.sort(function (a, b) { return a - b; });
  var median = closedDays.length
    ? closedDays[Math.floor(closedDays.length / 2)] : 0;
  var p90 = closedDays.length
    ? closedDays[Math.min(closedDays.length - 1, Math.floor(closedDays.length * 0.9))] : 0;

  var kloc = 0;
  Object.keys(modulesSeen).forEach(function (m) { kloc += (P.kloc[m] || 0); });

  var open = total - closed;
  var cards = [
    { value: fmt(total), label: 'Bugs in scope',
      foot: pct(total / N * 100) + ' of all records' },
    { value: fmt(closed), label: 'Closed',
      foot: total ? pct(closed / total * 100) + ' close rate' : '—' },
    { value: fmt(open), label: 'Still open',
      foot: open ? fmt1(openAge / open) + ' days average age' : '—' },
    { value: closed ? fmt1(sumDays / closed) : '—', label: 'Avg days to close',
      foot: closed ? 'median ' + fmt(median) + ' d, p90 ' + fmt(p90) + ' d' : 'no closed bugs' },
    { value: closed ? pct(metSla / closed * 100) : '—', label: 'SLA compliance',
      foot: 'of closed bugs met their target' },
    { value: fmt(openLate), label: 'Open past SLA',
      foot: open ? pct(openLate / open * 100) + ' of the open queue' : '—' },
    { value: kloc ? fmt1(total / kloc) : '—', label: 'Defect density',
      foot: kloc ? 'bugs per KLOC over ' + fmt1(kloc) + ' KLOC' : 'no size baseline' },
    { value: total ? pct(reopened / total * 100) : '—', label: 'Reopen rate',
      foot: fmt(reopened) + ' bugs reopened' }
  ];

  document.getElementById('kpi-strip').innerHTML = cards.map(function (c) {
    return '<div class="kpi"><div class="value">' + esc(c.value) + '</div>' +
           '<div class="label">' + esc(c.label) + '</div>' +
           '<div class="foot">' + esc(c.foot) + '</div></div>';
  }).join('');

  document.getElementById('scope-count').textContent = fmt(total);
}

/* ── Card renderers ──────────────────────────────────────────────────────── */
function renderStatus(rows) {
  var counts = tally(rows, 'status');
  var items = DIM.status.labels.map(function (label, i) {
    return { code: i, label: label, value: counts[i] };
  }).filter(function (d) { return d.value > 0; });
  drawBarsH(document.getElementById('chart-status'), items,
            { dim: 'status', labelWidth: 118, labelChars: 16 });
}

function renderResolution(rows) {
  var counts = tally(rows, 'resolution');
  var items = topLevels(counts, DIM.resolution.labels);
  drawDonut(document.getElementById('chart-resolution'), items, { dim: 'resolution' });
}

function renderModule(rows) {
  var counts = tally(rows, 'module');
  var items = topLevels(counts, DIM.module.labels);
  drawBarsH(document.getElementById('chart-module'), items,
            { dim: 'module', labelWidth: 152, labelChars: 22 });
}

function renderPriority(rows) {
  var counts = tally(rows, 'priority');
  var items = DIM.priority.labels.map(function (label, i) {
    return { code: i, label: label, value: counts[i] };
  }).filter(function (d) { return d.value > 0; });
  drawBarsV(document.getElementById('chart-priority'), items, { dim: 'priority' });
}

function renderDensity(rows) {
  var counts = tally(rows, 'module');
  var items = [];
  DIM.module.labels.forEach(function (label, i) {
    if (!counts[i]) return;
    var size = P.kloc[label] || 0;
    if (!size) return;
    items.push({
      code: i, label: label, value: counts[i] / size,
      display: fmt1(counts[i] / size),
      tipMain: fmt1(counts[i] / size) + ' bugs per KLOC',
      tip: fmt(counts[i]) + ' bugs over ' + fmt1(size) + ' KLOC'
    });
  });
  items.sort(function (a, b) { return b.value - a.value; });
  drawBarsH(document.getElementById('chart-density'), items,
            { dim: 'module', labelWidth: 152, labelChars: 22 });
}

function renderResolutionTime(rows) {
  var avg = meanDays(rows, 'priority');
  var items = [];
  DIM.priority.labels.forEach(function (label, i) {
    if (avg[i] == null) return;
    items.push({
      code: i, label: label, value: avg[i], display: fmt1(avg[i]) + ' d',
      target: P.sla[label] || null,
      tip: label + ' — average ' + fmt1(avg[i]) + ' days to close, SLA target ' +
           (P.sla[label] || '—') + ' days'
    });
  });
  drawBarsV(document.getElementById('chart-restime'), items, {
    dim: 'priority',
    tickFormat: function (t) { return fmt(t) + 'd'; },
    height: 288,
    legend: [{ label: 'SLA target', color: '#1565c0' }]
  });
}

function renderRootCause(rows) {
  var counts = tally(rows, 'rootcause');
  var avg = meanDays(rows, 'rootcause');
  var items = topLevels(counts, DIM.rootcause.labels, 10).map(function (d) {
    d.tip = avg[d.code] == null ? 'no closed bugs yet'
                                : 'average ' + fmt1(avg[d.code]) + ' days to close';
    return d;
  });
  drawBarsH(document.getElementById('chart-rootcause'), items,
            { dim: 'rootcause', labelWidth: 210, labelChars: 34 });
}

function renderFeature(rows) {
  var counts = tally(rows, 'feature');
  var items = topLevels(counts, DIM.feature.labels, 10);
  drawBarsH(document.getElementById('chart-feature'), items,
            { dim: 'feature', labelWidth: 168, labelChars: 24 });
}

function renderReleaseTable(rows) {
  var stats = releaseLabels.map(function (label) {
    return { label: label, bugs: 0, closed: 0, urgent: 0, sum: 0 };
  });
  for (var i = 0; i < rows.length; i++) {
    var r = rows[i], row = stats[releaseCodes[r]];
    row.bugs++;
    if (DIM.priority.codes[r] <= 1) row.urgent++;
    if (days[r] !== NULL2) { row.closed++; row.sum += days[r]; }
  }

  var body = stats.filter(function (s) { return s.bugs > 0; }).map(function (s) {
    var rate = s.bugs ? s.closed / s.bugs * 100 : 0;
    return [
      s.label, fmt(s.bugs), fmt(s.closed), fmt(s.bugs - s.closed), fmt(s.urgent),
      s.closed ? fmt1(s.sum / s.closed) : '—',
      { text: pct(rate), cls: rateClass(rate, 32, 24) }
    ];
  });

  renderTable(document.getElementById('table-release'),
    ['Release', 'Bugs', 'Closed', 'Open', 'P1/P2', 'Avg days to close', 'Close rate'], body);
}

function renderTeamTable(rows) {
  var stats = DIM.team.labels.map(function (label) {
    return { label: label, assigned: 0, closed: 0, urgentOpen: 0, sum: 0, met: 0 };
  });
  for (var i = 0; i < rows.length; i++) {
    var r = rows[i], row = stats[DIM.team.codes[r]];
    row.assigned++;
    if (days[r] === NULL2) {
      if (DIM.priority.codes[r] <= 1) row.urgentOpen++;
    } else {
      row.closed++;
      row.sum += days[r];
      if (days[r] <= slaTarget(DIM.priority.codes[r])) row.met++;
    }
  }

  var body = stats.filter(function (s) { return s.assigned > 0; })
    .sort(function (a, b) { return b.assigned - a.assigned; })
    .map(function (s) {
      var sla = s.closed ? s.met / s.closed * 100 : 0;
      var rate = s.assigned ? s.closed / s.assigned * 100 : 0;
      return [
        s.label, fmt(s.assigned), fmt(s.closed), fmt(s.assigned - s.closed),
        fmt(s.urgentOpen),
        s.closed ? fmt1(s.sum / s.closed) : '—',
        { text: pct(rate), cls: rateClass(rate, 32, 24) },
        { text: s.closed ? pct(sla) : '—', cls: rateClass(sla, 60, 40) }
      ];
    });

  renderTable(document.getElementById('table-team'),
    ['Team', 'Assigned', 'Closed', 'Open', 'Open P1/P2', 'Avg days to close',
     'Close rate', 'SLA met'], body);
}

/* ── Persistence: filters auto-remembered, views saved by name ──────────────
 * Both live only in this browser's localStorage — nothing is sent anywhere.
 * Reads/writes are wrapped in try/catch since storage can throw (private
 * browsing, disabled cookies, a full quota) and none of that should break
 * the dashboard itself. */
var FILTERS_KEY = 'bugDashboard.filters.v1';
var VIEWS_KEY   = 'bugDashboard.savedViews.v1';

function loadStoredFilters() {
  try {
    var raw = localStorage.getItem(FILTERS_KEY);
    if (!raw) return null;
    var obj = JSON.parse(raw);
    var out = {};
    FILTER_KEYS.forEach(function (key) {
      var v = obj[key];
      out[key] = (typeof v === 'number' && v >= -1 && v < DIM[key].labels.length) ? v : -1;
    });
    return out;
  } catch (e) { return null; }
}

function persistFilters() {
  try { localStorage.setItem(FILTERS_KEY, JSON.stringify(state.filters)); } catch (e) {}
}

function loadSavedViews() {
  try {
    var arr = JSON.parse(localStorage.getItem(VIEWS_KEY) || '[]');
    return Array.isArray(arr) ? arr : [];
  } catch (e) { return []; }
}

function persistSavedViews(views) {
  try { localStorage.setItem(VIEWS_KEY, JSON.stringify(views)); } catch (e) {}
}

function saveCurrentView(name) {
  name = (name || '').trim();
  if (!name) return;
  var views = loadSavedViews().filter(function (v) { return v.name !== name; });
  views.push({ name: name, filters: state.filters });
  persistSavedViews(views);
  renderSavedViews();
}

function deleteSavedView(name) {
  persistSavedViews(loadSavedViews().filter(function (v) { return v.name !== name; }));
  renderSavedViews();
}

function applyFilterSet(filters) {
  FILTER_KEYS.forEach(function (key) {
    var v = filters[key];
    state.filters[key] = (typeof v === 'number' && v >= -1 && v < DIM[key].labels.length) ? v : -1;
  });
  persistFilters();
  syncFilterUI();
  renderActiveFilters();
  refresh();
}

function renderSavedViews() {
  var el = document.getElementById('saved-views-list');
  var views = loadSavedViews();
  if (!views.length) { el.innerHTML = ''; return; }
  el.innerHTML = views.map(function (v, i) {
    return '<span class="view-chip">' +
      '<button type="button" class="view-chip-apply" data-view-index="' + i + '">' +
        esc(clip(v.name, 28)) + '</button>' +
      '<button type="button" class="view-chip-del" data-view-name="' + esc(v.name) +
        '" aria-label="Delete saved view ' + esc(v.name) + '">&times;</button></span>';
  }).join('');
}

function renderActiveFilters() {
  var el = document.getElementById('active-filters');
  var active = FILTER_KEYS.filter(function (key) { return state.filters[key] >= 0; });
  if (!active.length) { el.innerHTML = ''; return; }
  el.innerHTML = active.map(function (key) {
    var dim = DIM[key], label = dim.labels[state.filters[key]];
    return '<span class="filter-chip">' + esc(dim.label) + ': ' + esc(clip(label, 24)) +
      '<button type="button" class="chip-remove" data-key="' + key +
      '" aria-label="Remove ' + esc(dim.label) + ' filter">&times;</button></span>';
  }).join('');
}

/* ── Filter UI ───────────────────────────────────────────────────────────── */
function buildFilterUI() {
  var grid = document.getElementById('filter-grid');
  grid.innerHTML = FILTER_KEYS.map(function (key) {
    var dim = DIM[key];
    var options = ['<option value="-1">All</option>'].concat(
      dim.labels.map(function (label, i) {
        return '<option value="' + i + '">' + esc(clip(label, 46)) + '</option>';
      })
    ).join('');
    return '<div class="filter"><label for="f-' + key + '">' + esc(dim.label) + '</label>' +
           '<select id="f-' + key + '" data-key="' + key + '">' + options + '</select></div>';
  }).join('');

  grid.addEventListener('change', function (event) {
    var select = event.target.closest('select');
    if (!select) return;
    state.filters[select.dataset.key] = parseInt(select.value, 10);
    persistFilters();
    syncFilterUI();
    renderActiveFilters();
    refresh();
  });

  document.getElementById('reset-filters').addEventListener('click', function () {
    FILTER_KEYS.forEach(function (key) { state.filters[key] = -1; });
    persistFilters();
    syncFilterUI();
    renderActiveFilters();
    refresh();
  });

  document.getElementById('active-filters').addEventListener('click', function (event) {
    var btn = event.target.closest('.chip-remove');
    if (!btn) return;
    state.filters[btn.dataset.key] = -1;
    persistFilters();
    syncFilterUI();
    renderActiveFilters();
    refresh();
  });

  document.getElementById('save-view-btn').addEventListener('click', function () {
    var input = document.getElementById('save-view-name');
    saveCurrentView(input.value);
    input.value = '';
  });
  document.getElementById('save-view-name').addEventListener('keydown', function (event) {
    if (event.key !== 'Enter') return;
    saveCurrentView(this.value);
    this.value = '';
  });
  document.getElementById('saved-views-list').addEventListener('click', function (event) {
    var apply = event.target.closest('.view-chip-apply');
    if (apply) {
      var views = loadSavedViews();
      var view = views[parseInt(apply.dataset.viewIndex, 10)];
      if (view) applyFilterSet(view.filters);
      return;
    }
    var del = event.target.closest('.view-chip-del');
    if (del) deleteSavedView(del.dataset.viewName);
  });

  /* Clicking a bar, wedge or cell drills into that value. */
  document.querySelector('main.dashboard-main').addEventListener('click', function (event) {
    var target = event.target.closest('[data-dim]');
    if (!target) return;
    var key = target.dataset.dim, code = parseInt(target.dataset.code, 10);
    state.filters[key] = (state.filters[key] === code) ? -1 : code;
    persistFilters();
    syncFilterUI();
    renderActiveFilters();
    refresh();
  });
}

function syncFilterUI() {
  FILTER_KEYS.forEach(function (key) {
    var select = document.getElementById('f-' + key);
    select.value = String(state.filters[key]);
    select.classList.toggle('active', state.filters[key] >= 0);
  });
}

function renderInsights() {
  document.getElementById('insight-list').innerHTML = P.insights.map(function (item) {
    return '<div class="insight"><h3>' + esc(item.title) + '</h3><p>' +
           esc(item.detail) + '</p></div>';
  }).join('');
}

/* ── Refresh ─────────────────────────────────────────────────────────────── */
function refresh() {
  applyFilters();
  var rows = state.rows;
  renderKpis(rows);
  drawSprintChart(document.getElementById('chart-sprint'), rows);
  renderBugTrend(rows);
  renderResolutionTrend(rows);
  renderStatus(rows);
  renderResolution(rows);
  renderModule(rows);
  renderPriority(rows);
  renderDensity(rows);
  renderResolutionTime(rows);
  drawHeatmap(document.getElementById('chart-heatmap'), rows);
  renderRootCause(rows);
  renderFeature(rows);
  renderReleaseTable(rows);
  renderTeamTable(rows);
}

var restoredFilters = loadStoredFilters();
if (restoredFilters) state.filters = restoredFilters;

buildFilterUI();
syncFilterUI();
renderActiveFilters();
renderSavedViews();
renderInsights();
refresh();

})();
