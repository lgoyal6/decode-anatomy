// Draws results/04_overhead.json and results/03_sweep.json.
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

const state = { overhead: [], sweep: [], gBatch: 1, batch: 1, ctxStep: 1, ctxs: [] };

function fitCanvas(canvas, ratio) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w0 = canvas.clientWidth || 1200;
  const h0 = Math.round(w0 * ratio);
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
  const { ctx, w, h } = fitCanvas(el('plot-graph'), 0.13);
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
  const { ctx, w, h } = fitCanvas(el('plot-split'), 0.105);
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
  const [oRes, sRes] = await Promise.all([
    fetch('./data/04_overhead.json'),
    fetch('./data/03_sweep.json'),
  ]);
  if (!oRes.ok || !sRes.ok) {
    el('g-banner').textContent = 'Could not load the measurements.';
    return;
  }
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
  window.addEventListener('resize', () => { renderGraph(); renderSplit(); });

  renderGraph();
  renderSplit();
}

main();
