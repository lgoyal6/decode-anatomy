// Draws results/01_roofline.json, 04_overhead.json, 03_sweep.json and 04_matrix.json.
//
// The two figures deliberately come from different measurements. Figure 1 is a
// comparison between two real executions, neither of them profiled. Figure 2 is
// a profiled decomposition, and profiling inflates the launch overhead it is
// being used to measure by 1.83x. Mixing them would produce a number that is
// wrong in a way nobody could see, so they are kept apart and labelled.

const el = (id) => document.getElementById(id);
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

const VERDICTS = [
  { key: 'non-kernel', label: 'not kernel work', colour: () => css('--bad'), hatch: true },
  { key: 'latency-bound', label: 'latency-bound', colour: () => css('--warn'), hatch: false },
  { key: 'memory-bound', label: 'memory-bound', colour: () => css('--ox'), hatch: false },
  { key: 'compute-bound', label: 'compute-bound', colour: () => css('--ok'), hatch: false },
];

const state = { roof: null, matrix: null, overhead: [], sweep: [], gBatch: 1, batch: 1, ctxStep: 1, ctxs: [] };

// `size` is a fixed pixel height where the content does not scale with width,
// which is the case for a bar chart of two bars: making it taller on a wider
// screen only adds empty box.
function fitCanvas(canvas, size) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w0 = canvas.clientWidth || 1200;
  const h0 = Math.round(size <= 1 ? w0 * size : size);
  canvas.width = Math.round(w0 * dpr);
  canvas.height = Math.round(h0 * dpr);
  canvas.style.height = h0 + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w0, h0);
  return { ctx, w: w0, h: h0 };
}

// ---------------------------------------------------------- figure 1

function drawGraph() {
  const p = state.overhead.find((o) => o.batch === state.gBatch);
  if (!p) return;
  const g = p.cuda_graph;
  // Right padding leaves room for the duration printed past the bar's end, and
  // the height is only what two bars plus a caption need.
  const { ctx, w, h } = fitCanvas(el('plot-graph'), 132);
  const pad = { l: 92, r: 86, t: 22, b: 30 };
  const barH = 30;
  const span = w - pad.l - pad.r;
  const scale = span / Math.max(g.eager_ms, 0.001);

  const row = (y, label, ms, parts) => {
    ctx.textAlign = 'right';
    ctx.font = "12px 'Courier New', monospace";
    ctx.fillStyle = css('--sub');
    ctx.fillText(label, pad.l - 10, y + barH / 2 + 4);
    let x = pad.l;
    parts.forEach(([value, colour, hatch]) => {
      const bw = value * scale;
      ctx.fillStyle = colour;
      ctx.fillRect(x, y, bw, barH);
      if (hatch) {
        // Hatching, so the removable slice is distinguishable without colour.
        ctx.save();
        ctx.beginPath(); ctx.rect(x, y, bw, barH); ctx.clip();
        ctx.strokeStyle = 'rgba(255,255,255,.55)';
        ctx.lineWidth = 1.5;
        for (let i = -barH; i < bw + barH; i += 7) {
          ctx.beginPath(); ctx.moveTo(x + i, y + barH); ctx.lineTo(x + i + barH, y); ctx.stroke();
        }
        ctx.restore();
      }
      x += bw;
    });
    ctx.textAlign = 'left';
    ctx.font = "13px 'Times New Roman', serif";
    ctx.fillStyle = css('--sub');
    ctx.fillText(`${ms.toFixed(2)} ms`, x + 8, y + barH / 2 + 4);
  };

  row(pad.t, 'eager', g.eager_ms, [
    [g.graph_ms, css('--ox'), false],
    [g.removable_ms, css('--bad'), true],
  ]);
  row(pad.t + barH + 18, 'as a graph', g.graph_ms, [[g.graph_ms, css('--ox'), false]]);

  ctx.textAlign = 'left';
  ctx.font = "13px 'Times New Roman', serif";
  ctx.fillStyle = css('--sub');
  ctx.fillText(
    `hatched: ${g.removable_pct.toFixed(1)}% removed by capturing the same work as a graph`,
    pad.l,
    h - 8,
  );
}

function renderGraph() {
  const p = state.overhead.find((o) => o.batch === state.gBatch);
  if (!p) return;
  const g = p.cuda_graph;
  el('g-eager').textContent = `${g.eager_ms.toFixed(2)} ms`;
  el('g-graph').textContent = `${g.graph_ms.toFixed(2)} ms`;
  el('g-removable').textContent = `${g.removable_pct.toFixed(1)}%`;
  el('g-speedup').textContent = `${g.speedup.toFixed(2)}x`;
  el('g-kernels').textContent = Math.round(p.n_kernels_per_step).toLocaleString('en-US');
  drawGraph();

  const b = el('g-banner');
  if (g.removable_pct > 20) {
    b.className = 'banner alarm';
    b.textContent =
      `At batch ${p.batch}, ${g.removable_pct.toFixed(1)}% of the step is launch overhead rather than ` +
      `arithmetic or memory traffic. Neither ceiling explains it.`;
  } else if (g.removable_pct > 8) {
    b.className = 'banner';
    b.textContent = `At batch ${p.batch} the overhead is ${g.removable_pct.toFixed(1)}%, shrinking as the work per launch grows.`;
  } else {
    b.className = 'banner calm';
    b.textContent =
      `At batch ${p.batch} it is ${g.removable_pct.toFixed(1)}%. The same kernels now have enough work ` +
      `in them that launching costs almost nothing.`;
  }
}

// ---------------------------------------------------------- figure 2

function point() {
  const ctxLen = state.ctxs[state.ctxStep];
  return state.sweep.find((p) => p.batch === state.batch && p.cache_len === ctxLen);
}

function drawSplit() {
  const p = point();
  if (!p) return;
  const { ctx, w, h } = fitCanvas(el('plot-split'), 128);
  const pad = { l: 24, r: 24, t: 34, b: 56 };
  const barH = 46;
  const span = w - pad.l - pad.r;
  const s = p.split_pct_of_span;

  let x = pad.l;
  VERDICTS.forEach((v) => {
    const value = s[v.key] || 0;
    const bw = (value / 100) * span;
    if (bw <= 0) return;
    ctx.fillStyle = v.colour();
    ctx.fillRect(x, pad.t, bw, barH);
    if (v.hatch) {
      ctx.save();
      ctx.beginPath(); ctx.rect(x, pad.t, bw, barH); ctx.clip();
      ctx.strokeStyle = 'rgba(255,255,255,.5)';
      ctx.lineWidth = 1.5;
      for (let i = -barH; i < bw + barH; i += 8) {
        ctx.beginPath(); ctx.moveTo(x + i, pad.t + barH); ctx.lineTo(x + i + barH, pad.t); ctx.stroke();
      }
      ctx.restore();
    }
    if (bw > 46) {
      ctx.fillStyle = css('--paper');
      ctx.font = "13px 'Times New Roman', serif";
      ctx.textAlign = 'center';
      ctx.fillText(`${value.toFixed(1)}%`, x + bw / 2, pad.t + barH / 2 + 5);
    }
    x += bw;
  });

  // legend below, with the hatch repeated so it reads without colour
  let lx = pad.l;
  ctx.textAlign = 'left';
  ctx.font = "13px 'Times New Roman', serif";
  VERDICTS.forEach((v) => {
    const value = s[v.key] || 0;
    ctx.fillStyle = v.colour();
    ctx.fillRect(lx, pad.t + barH + 18, 14, 12);
    if (v.hatch) {
      ctx.strokeStyle = 'rgba(255,255,255,.6)';
      ctx.lineWidth = 1.4;
      for (let i = -12; i < 26; i += 6) {
        ctx.beginPath(); ctx.moveTo(lx + i, pad.t + barH + 30); ctx.lineTo(lx + i + 12, pad.t + barH + 18); ctx.stroke();
      }
    }
    ctx.fillStyle = css('--sub');
    const text = `${v.label} ${value.toFixed(1)}%`;
    ctx.fillText(text, lx + 20, pad.t + barH + 29);
    lx += 20 + ctx.measureText(text).width + 26;
  });
}

function renderSplit() {
  const p = point();
  if (!p) return;
  const s = p.split_pct_of_span;
  el('s-mem').textContent = `${s['memory-bound'].toFixed(1)}%`;
  el('s-com').textContent = `${s['compute-bound'].toFixed(1)}%`;
  el('s-lat').textContent = `${s['latency-bound'].toFixed(1)}%`;
  el('s-non').textContent = `${s['non-kernel'].toFixed(1)}%`;
  el('s-tok').textContent = `${p.tokens_per_s.toFixed(0)} tok/s`;
  el('cap-point').textContent = `batch ${p.batch}, ${p.cache_len} tokens of context`;
  drawSplit();

  const b = el('s-banner');
  const dominant = [...VERDICTS].sort((a, c) => (s[c.key] || 0) - (s[a.key] || 0))[0];
  b.className = dominant.key === 'non-kernel' ? 'banner alarm' : 'banner';
  b.textContent =
    dominant.key === 'non-kernel'
      ? `The largest share of this step is ${dominant.label}, at ${s[dominant.key].toFixed(1)}%. ` +
        `"Memory-bound" describes ${s['memory-bound'].toFixed(1)}% of it.`
      : `Largest share here is ${dominant.label}, at ${s[dominant.key].toFixed(1)}%, with ` +
        `${s['non-kernel'].toFixed(1)}% still outside the kernels.`;
}

// ---------------------------------------------------------- figure 1

// One colour per precision. The roofs are bf16, so the fp32 and tf32 points sit
// far under the compute roof by construction and are drawn dimmer: they are
// context for the shape, not candidates for touching it.
const DTYPES = [
  { key: 'bf16', colour: () => css('--ox') },
  { key: 'fp16', colour: () => css('--ok') },
  { key: 'tf32', colour: () => css('--warn') },
  { key: 'fp32', colour: () => css('--faint') },
];

function drawRoofline() {
  const r = state.roof;
  if (!r) return;
  const { ctx, w, h } = fitCanvas(el('plot-roof'), 420);
  const pad = { l: 62, r: 20, t: 20, b: 64 };
  const iw = w - pad.l - pad.r;
  const ih = h - pad.t - pad.b;

  // Both axes are log. A roofline on linear axes is unreadable: the memory roof
  // is a straight line only in log-log, which is the whole point of the chart.
  const X0 = 32, X1 = 8192, Y0 = 4, Y1 = 256;
  const lgx = Math.log10(X1) - Math.log10(X0);
  const lgy = Math.log10(Y1) - Math.log10(Y0);
  const px = (v) => pad.l + ((Math.log10(v) - Math.log10(X0)) / lgx) * iw;
  const py = (v) => pad.t + ih - ((Math.log10(v) - Math.log10(Y0)) / lgy) * ih;

  const bwTfPerFb = r.summary.peak_hbm_gbs / 1000; // GB/s x FLOP/byte -> GFLOP/s -> TFLOP/s
  const roofTf = r.summary.peak_bf16_tflops;
  const ridge = r.summary.ridge_point_flop_per_byte;

  // grid and ticks
  ctx.strokeStyle = css('--grid');
  ctx.lineWidth = 1;
  ctx.fillStyle = css('--faint');
  ctx.font = "11px 'JetBrains Mono', ui-monospace, monospace";
  for (let d = X0; d <= X1; d *= 2) {
    const x = px(d);
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + ih); ctx.stroke();
    ctx.textAlign = 'center';
    ctx.fillText(String(d), x, pad.t + ih + 16);
  }
  for (let d = Y0; d <= Y1; d *= 2) {
    const y = py(d);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + iw, y); ctx.stroke();
    ctx.textAlign = 'right';
    ctx.fillText(String(d), pad.l - 8, y + 4);
  }

  ctx.textAlign = 'center';
  ctx.fillStyle = css('--sub');
  ctx.font = "12px 'Inter', system-ui, sans-serif";
  ctx.fillText('arithmetic intensity (FLOP / byte)', pad.l + iw / 2, pad.t + ih + 40);
  ctx.save();
  ctx.translate(16, pad.t + ih / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('achieved TFLOP/s', 0, 0);
  ctx.restore();

  // the two roofs
  ctx.strokeStyle = css('--ink');
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(px(X0), py(bwTfPerFb * X0));
  ctx.lineTo(px(ridge), py(roofTf));
  ctx.lineTo(px(X1), py(roofTf));
  ctx.stroke();

  // ridge point, where one roof stops binding and the other starts
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = css('--rule');
  ctx.beginPath();
  ctx.moveTo(px(ridge), py(roofTf));
  ctx.lineTo(px(ridge), pad.t + ih);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = css('--sub');
  ctx.font = "11px 'JetBrains Mono', ui-monospace, monospace";
  ctx.textAlign = 'left';
  ctx.fillText(`ridge ${ridge.toFixed(0)}`, px(ridge) + 6, pad.t + 14);

  ctx.textAlign = 'right';
  ctx.fillText(`${roofTf.toFixed(0)} TFLOP/s bf16`, pad.l + iw - 6, py(roofTf) - 8);
  ctx.save();
  const ax = px(X0) + 30, ay = py(bwTfPerFb * X0 * 1.9);
  ctx.translate(ax, ay);
  ctx.rotate(-Math.atan2(ih / lgy, iw / lgx));
  ctx.textAlign = 'left';
  ctx.fillText(`${r.summary.peak_hbm_gbs.toFixed(0)} GB/s`, 0, -6);
  ctx.restore();

  // measured GEMMs
  r.gemm.forEach((g) => {
    const d = DTYPES.find((t) => t.key === g.dtype);
    if (!d) return;
    ctx.fillStyle = d.colour();
    ctx.beginPath();
    ctx.arc(px(g.arith_intensity), py(g.tflops), 3.5, 0, Math.PI * 2);
    ctx.fill();
  });

  // legend
  let lx = pad.l;
  ctx.textAlign = 'left';
  ctx.font = "12px 'Inter', system-ui, sans-serif";
  DTYPES.forEach((d) => {
    ctx.fillStyle = d.colour();
    ctx.beginPath();
    ctx.arc(lx + 5, pad.t + ih + 56, 3.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = css('--sub');
    ctx.fillText(d.key, lx + 14, pad.t + ih + 60);
    lx += 14 + ctx.measureText(d.key).width + 22;
  });
}

function renderRoofline() {
  const r = state.roof;
  if (!r) return;
  const s = r.summary;
  el('cap-roof').textContent = `${r.spec.name}, ${r.spec.arch}`;
  el('r-bw').textContent = `${s.peak_hbm_gbs.toFixed(0)} GB/s`;
  el('r-bwpct').textContent = `${s.peak_hbm_pct_of_spec.toFixed(1)}%`;
  el('r-tf').textContent = `${s.peak_bf16_tflops.toFixed(1)} TFLOP/s`;
  el('r-tfpct').textContent = `${s.peak_bf16_pct_of_spec.toFixed(1)}%`;
  el('r-ridge').textContent = `${s.ridge_point_flop_per_byte.toFixed(1)}`;
  drawRoofline();

  el('r-banner').className = 'banner';
  el('r-banner').textContent =
    `Neither roof is the quoted one. Bandwidth tops out at ` +
    `${s.peak_hbm_gbs.toFixed(0)} GB/s against a quoted ${r.spec.mem_bw_gbs.toFixed(0)}, and bf16 at ` +
    `${s.peak_bf16_tflops.toFixed(1)} against a quoted ${r.spec.bf16_tensor_tflops.toFixed(1)}. ` +
    `Drawing the roofline from the datasheet would move the ridge point and every verdict with it.`;
}

// ---------------------------------------------------------- figure 4

function drawMatrix() {
  const m = state.matrix;
  if (!m) return;
  const rows = m.rows;
  const { ctx, w, h } = fitCanvas(el('plot-matrix'), 420);
  const pad = { l: 128, r: 74, t: 18, b: 52 };
  const iw = w - pad.l - pad.r;
  const band = (h - pad.t - pad.b) / rows.length;
  const barH = Math.min(9, band / 3);

  // Counts, not percentages. The percentages all sit between 92 and 100, where a
  // truthful axis shows nothing and a truncated one exaggerates; "15 of 200
  // completions changed" is the same fact and it is readable.
  const answersMoved = (r) => Math.round(((100 - r.answer_identical_pct) / 100) * m.n_prompts);
  const MAX = 16;
  const sx = iw / MAX;

  ctx.strokeStyle = css('--grid');
  ctx.lineWidth = 1;
  ctx.fillStyle = css('--faint');
  ctx.font = "11px 'JetBrains Mono', ui-monospace, monospace";
  ctx.textAlign = 'center';
  for (let n = 0; n <= MAX; n += 4) {
    const x = pad.l + n * sx;
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + band * rows.length); ctx.stroke();
    ctx.fillText(String(n), x, pad.t + band * rows.length + 16);
  }
  ctx.fillStyle = css('--sub');
  ctx.font = "12px 'Inter', system-ui, sans-serif";
  ctx.fillText(`completions that changed, of ${m.n_prompts}`, pad.l + iw / 2, pad.t + band * rows.length + 38);

  rows.forEach((r, i) => {
    const y = pad.t + i * band;
    ctx.textAlign = 'right';
    ctx.font = "12px 'JetBrains Mono', ui-monospace, monospace";
    ctx.fillStyle = r.condition === 'reference' ? css('--faint') : css('--sub');
    ctx.fillText(r.condition, pad.l - 10, y + band / 2 + 4);

    const bytes = r.n_diverged;
    const ans = answersMoved(r);
    ctx.fillStyle = css('--warn');
    ctx.fillRect(pad.l, y + band / 2 - barH - 1, bytes * sx, barH);
    ctx.fillStyle = css('--bad');
    ctx.fillRect(pad.l, y + band / 2 + 1, ans * sx, barH);

    ctx.textAlign = 'left';
    ctx.font = "11px 'JetBrains Mono', ui-monospace, monospace";
    ctx.fillStyle = css('--faint');
    ctx.fillText(`${bytes} / ${ans}`, pad.l + Math.max(bytes, ans) * sx + 8, y + band / 2 + 4);
  });

  // legend
  let lx = pad.l;
  ctx.textAlign = 'left';
  ctx.font = "12px 'Inter', system-ui, sans-serif";
  [['--warn', 'text changed'], ['--bad', 'answer changed']].forEach(([v, label]) => {
    ctx.fillStyle = css(v);
    ctx.fillRect(lx, pad.t + band * rows.length + 44, 13, 8);
    ctx.fillStyle = css('--sub');
    ctx.fillText(label, lx + 19, pad.t + band * rows.length + 52);
    lx += 19 + ctx.measureText(label).width + 24;
  });
}

function renderMatrix() {
  const m = state.matrix;
  if (!m) return;
  const rows = m.rows;
  const worst = rows.reduce((a, b) => (b.n_diverged > a.n_diverged ? b : a));
  const answers = rows.map((r) => Math.round(((100 - r.answer_identical_pct) / 100) * m.n_prompts));
  const accs = rows.map((r) => r.accuracy_pct);
  const divs = rows.map((r) => r.median_first_divergence).filter((v) => v !== null);

  el('d-worst').textContent = `${worst.n_diverged} / ${m.n_prompts}`;
  el('d-ans').textContent = `${Math.max(...answers)} / ${m.n_prompts}`;
  const spread = Math.max(...accs) - Math.min(...accs);
  el('d-acc').textContent = spread === 0 ? `0.0 pts` : `${spread.toFixed(1)} pts`;
  el('d-self').textContent = state.determinism && state.determinism.within_engine_all_identical
    ? `${state.determinism.repeats}/${state.determinism.repeats} identical`
    : 'not identical';
  el('d-div').textContent = divs.length ? `token ${Math.min(...divs)}` : 'n/a';

  drawMatrix();

  el('d-banner').className = spread === 0 ? 'banner calm' : 'banner alarm';
  el('d-banner').textContent =
    spread === 0
      ? `Every condition scores ${accs[0].toFixed(1)}%. The worst of them, ${worst.condition}, ` +
        `returns different text for ${worst.n_diverged} of ${m.n_prompts} prompts and the same score for all of them.`
      : `Accuracy moves by ${spread.toFixed(1)} points across the nine conditions.`;
}

// ---------------------------------------------------------- wiring

function picker(node, items, current, onPick) {
  node.innerHTML = '';
  items.forEach(({ key, label }) => {
    const b = document.createElement('button');
    b.textContent = label;
    b.setAttribute('aria-pressed', String(key === current()));
    b.addEventListener('click', () => {
      onPick(key);
      [...node.children].forEach((c) => c.setAttribute('aria-pressed', String(c === b)));
    });
    node.appendChild(b);
  });
}

async function main() {
  const [oRes, sRes, rRes, mRes, dRes] = await Promise.all([
    fetch('./data/04_overhead.json'),
    fetch('./data/03_sweep.json'),
    fetch('./data/01_roofline.json'),
    fetch('./data/04_matrix.json'),
    fetch('./data/04_determinism.json'),
  ]);
  if (!oRes.ok || !sRes.ok) {
    el('g-banner').textContent = 'Could not load the measurements.';
    return;
  }
  // The roofline and the matrix are independent of the two figures above, so a
  // failure in either leaves the rest of the page working rather than blank.
  state.roof = rRes.ok ? await rRes.json() : null;
  state.matrix = mRes.ok ? await mRes.json() : null;
  state.determinism = dRes.ok ? await dRes.json() : null;
  if (!state.roof) el('r-banner').textContent = 'Could not load the roofline.';
  if (!state.matrix) el('d-banner').textContent = 'Could not load the matrix.';
  state.overhead = (await oRes.json()).points;
  const sweep = await sRes.json();
  state.sweep = sweep.points;
  state.ctxs = [...new Set(state.sweep.map((p) => p.cache_len))].sort((a, b) => a - b);

  el('cap-what').textContent = `${sweep.model} on an ${sweep.env?.gpu_name || 'RTX A6000'}`;

  picker(
    el('graphbatch'),
    state.overhead.map((o) => ({ key: o.batch, label: `batch ${o.batch}` })),
    () => state.gBatch,
    (k) => { state.gBatch = k; renderGraph(); },
  );
  picker(
    el('batch'),
    [...new Set(state.sweep.map((p) => p.batch))].sort((a, b) => a - b).map((k) => ({ key: k, label: String(k) })),
    () => state.batch,
    (k) => { state.batch = k; renderSplit(); },
  );

  const scrub = el('scrub');
  scrub.max = String(state.ctxs.length - 1);
  scrub.value = String(state.ctxStep);
  scrub.addEventListener('input', (e) => { state.ctxStep = Number(e.target.value); renderSplit(); });
  window.addEventListener('resize', () => {
    renderRoofline(); renderGraph(); renderSplit(); renderMatrix();
  });

  renderRoofline();
  renderGraph();
  renderSplit();
  renderMatrix();
}

main();
