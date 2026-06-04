/* ═══════════════════════════════════════════════════════════
   GAN Architecture - Interactive Illustration
   ═══════════════════════════════════════════════════════════ */

// ─── PARTICLE BACKGROUND ───
(function initParticles() {
  const canvas = document.getElementById('particleCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let w, h, particles = [];

  function resize() {
    w = canvas.width = canvas.offsetWidth;
    h = canvas.height = canvas.offsetHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  class Particle {
    constructor() { this.reset(); }
    reset() {
      this.x = Math.random() * w;
      this.y = Math.random() * h;
      this.vx = (Math.random() - 0.5) * 0.5;
      this.vy = (Math.random() - 0.5) * 0.5;
      this.r = Math.random() * 2 + 0.5;
      this.color = ['#00d4ff', '#ff6b9d', '#a855f7'][Math.floor(Math.random() * 3)];
      this.alpha = Math.random() * 0.5 + 0.2;
    }
    update() {
      this.x += this.vx; this.y += this.vy;
      if (this.x < 0 || this.x > w) this.vx *= -1;
      if (this.y < 0 || this.y > h) this.vy *= -1;
    }
    draw() {
      ctx.beginPath(); ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
      ctx.fillStyle = this.color; ctx.globalAlpha = this.alpha;
      ctx.fill(); ctx.globalAlpha = 1;
    }
  }

  for (let i = 0; i < 80; i++) particles.push(new Particle());

  function drawLines() {
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 120) {
          ctx.beginPath(); ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = particles[i].color;
          ctx.globalAlpha = (1 - dist / 120) * 0.15;
          ctx.stroke(); ctx.globalAlpha = 1;
        }
      }
    }
  }

  function animate() {
    ctx.clearRect(0, 0, w, h);
    particles.forEach(p => { p.update(); p.draw(); });
    drawLines();
    requestAnimationFrame(animate);
  }
  animate();
})();

// ─── NOISE VISUALIZATION ───
function initNoiseViz() {
  const noiseViz = document.getElementById('noiseViz');
  if (!noiseViz) return;
  noiseViz.innerHTML = '';
  for (let i = 0; i < 25; i++) {
    const dot = document.createElement('div');
    dot.className = 'noise-dot';
    const v = Math.random();
    const hue = Math.floor(v * 60 + 180);
    dot.style.background = `hsl(${hue},80%,${40 + v * 30}%)`;
    noiseViz.appendChild(dot);
  }
}
initNoiseViz();

// ─── LAYER BAR WIDTHS ───
function initLayerBars() {
  const genWidths = [30, 50, 75, 100]; // growing
  const discWidths = [100, 75, 50, 25]; // shrinking
  document.querySelectorAll('.generator-layers .layer-bar').forEach((bar, i) => {
    bar.style.width = genWidths[i] + '%';
  });
  document.querySelectorAll('.discriminator-layers .layer-bar').forEach((bar, i) => {
    bar.style.width = discWidths[i] + '%';
  });
}
initLayerBars();

// ─── ARCHITECTURE FLOW ANIMATION ───
let flowAnimating = false;
function animateFlow() {
  if (flowAnimating) return;
  flowAnimating = true;
  const blocks = ['noiseBlock','generatorBlock','mergeBlock','discriminatorBlock','outputBlock'];
  const arrows = ['arrow1','arrow2','arrow3','arrow4'];
  const delay = 500;

  blocks.forEach((id) => document.getElementById(id)?.classList.remove('active'));
  arrows.forEach((id) => document.getElementById(id)?.classList.remove('flowing'));

  let step = 0;
  function next() {
    if (step < blocks.length) {
      const el = document.getElementById(blocks[step]);
      if (el) el.classList.add('active');
      if (step > 0) {
        const ar = document.getElementById(arrows[step - 1]);
        if (ar) ar.classList.add('flowing');
      }
      // Animate gauge at the end
      if (step === blocks.length - 1) {
        const val = Math.random();
        animateGauge(val);
      }
      step++;
      setTimeout(next, delay);
    } else {
      flowAnimating = false;
    }
  }
  next();
}

function resetFlow() {
  ['noiseBlock','generatorBlock','mergeBlock','discriminatorBlock','outputBlock']
    .forEach(id => document.getElementById(id)?.classList.remove('active'));
  ['arrow1','arrow2','arrow3','arrow4']
    .forEach(id => document.getElementById(id)?.classList.remove('flowing'));
  animateGauge(0.5);
  flowAnimating = false;
}

function animateGauge(val) {
  const fill = document.getElementById('gaugeFill');
  const vText = document.getElementById('gaugeValue');
  if (fill) fill.style.width = (val * 100) + '%';
  if (vText) vText.textContent = val.toFixed(2);
}

// ─── GAN DEMO ───
let epoch = 0;
let autoTrain = false;
let autoInterval = null;
let ganQuality = 0; // 0 to 1, improves with epochs

// "Real" pattern: a simple smiley face in pixel art
const REAL_PATTERN = [
  [0,0,0,0,0,0,0,0,0,0],
  [0,0,0,1,1,1,1,0,0,0],
  [0,0,1,0,0,0,0,1,0,0],
  [0,1,0,1,0,0,1,0,1,0],
  [0,1,0,0,0,0,0,0,1,0],
  [0,1,0,1,0,0,1,0,1,0],
  [0,1,0,0,1,1,0,0,1,0],
  [0,0,1,0,0,0,0,1,0,0],
  [0,0,0,1,1,1,1,0,0,0],
  [0,0,0,0,0,0,0,0,0,0],
];

const GRID = 10;
let noiseData = [];

function generateNoiseArray() {
  noiseData = [];
  for (let y = 0; y < GRID; y++) {
    const row = [];
    for (let x = 0; x < GRID; x++) row.push(Math.random());
    noiseData.push(row);
  }
}
generateNoiseArray();

function drawGrid(canvasId, data, colorFn, cellSize) {
  const c = document.getElementById(canvasId);
  if (!c) return;
  const ctx = c.getContext('2d');
  const size = cellSize || Math.floor(c.width / GRID);
  ctx.clearRect(0, 0, c.width, c.height);
  const offsetX = (c.width - size * GRID) / 2;
  const offsetY = (c.height - size * GRID) / 2;
  for (let y = 0; y < GRID; y++) {
    for (let x = 0; x < GRID; x++) {
      const v = data[y][x];
      ctx.fillStyle = colorFn(v);
      ctx.fillRect(offsetX + x * size, offsetY + y * size, size - 1, size - 1);
    }
  }
}

function noiseColor(v) {
  const h = Math.floor(v * 60 + 200);
  return `hsl(${h},70%,${30 + v * 40}%)`;
}

function realColor(v) {
  return v > 0.5 ? '#00d4ff' : '#111122';
}

function fakeColor(v) {
  return v > 0.5 ? `hsl(${190 + Math.random()*10},90%,${50+Math.random()*10}%)` : `hsl(240,20%,${8+Math.random()*8}%)`;
}

function generateFakeData(quality) {
  const fake = [];
  for (let y = 0; y < GRID; y++) {
    const row = [];
    for (let x = 0; x < GRID; x++) {
      const real = REAL_PATTERN[y][x];
      // Mix: as quality increases, output approaches real pattern
      if (Math.random() < quality) {
        row.push(real);
      } else {
        row.push(Math.random() > 0.5 ? 1 : 0);
      }
    }
    fake.push(row);
  }
  return fake;
}

function generateNewNoise() {
  generateNoiseArray();
  drawGrid('noiseCanvas', noiseData, noiseColor, 18);
  initNoiseViz();
}

function renderDemo() {
  drawGrid('noiseCanvas', noiseData, noiseColor, 18);
  drawGrid('realSampleCanvas', REAL_PATTERN, realColor, 8);
}

function runGANDemo() {
  epoch++;
  ganQuality = Math.min(1, ganQuality + 0.04 + Math.random() * 0.03);

  generateNoiseArray();
  drawGrid('noiseCanvas', noiseData, noiseColor, 18);

  // Show Generator processing
  const gIcon = document.getElementById('step-g');
  if (gIcon) { gIcon.classList.add('active'); setTimeout(() => gIcon.classList.remove('active'), 500); }

  const fakeData = generateFakeData(ganQuality);
  drawGrid('fakeCanvas', fakeData, fakeColor, 18);
  drawGrid('fakeSampleCanvas', fakeData, fakeColor, 8);
  drawGrid('realSampleCanvas', REAL_PATTERN, realColor, 8);

  // Discriminator scores
  const realConf = 0.55 + Math.random() * 0.4 * (1 - ganQuality * 0.7);
  const fakeConf = 0.15 + ganQuality * 0.6 + Math.random() * 0.15;

  document.getElementById('realScore').textContent = (realConf * 100).toFixed(1) + '%';
  document.getElementById('fakeScore').textContent = (fakeConf * 100).toFixed(1) + '%';
  document.getElementById('realFill').style.width = (realConf * 100) + '%';
  document.getElementById('fakeFill').style.width = (fakeConf * 100) + '%';

  const verdict = document.getElementById('verdict');
  if (ganQuality < 0.3) {
    verdict.textContent = '❌ Discriminator mudah membedakan!';
    verdict.style.background = 'rgba(248,113,113,.1)';
    verdict.style.color = '#f87171';
  } else if (ganQuality < 0.7) {
    verdict.textContent = '⚠️ Generator semakin baik...';
    verdict.style.background = 'rgba(251,191,36,.1)';
    verdict.style.color = '#fbbf24';
  } else {
    verdict.textContent = '✅ Generator hampir sempurna! D(G(z)) ≈ 0.5';
    verdict.style.background = 'rgba(52,211,153,.1)';
    verdict.style.color = '#34d399';
  }

  document.getElementById('epochCount').textContent = epoch;
}

function toggleAutoTrain() {
  autoTrain = !autoTrain;
  const btn = document.getElementById('autoTrainBtn');
  if (autoTrain) {
    btn.textContent = '⏸ Stop Training';
    btn.classList.add('btn-primary');
    btn.classList.remove('btn-outline');
    autoInterval = setInterval(runGANDemo, 600);
  } else {
    btn.textContent = '🔄 Auto Training';
    btn.classList.remove('btn-primary');
    btn.classList.add('btn-outline');
    clearInterval(autoInterval);
  }
}

// ─── LOSS CHART ───
let lossData = { gen: [], disc: [] };
let lossAnimFrame = null;

function simulateLoss() {
  lossData = { gen: [], disc: [] };
  let step = 0;
  const total = 100;

  function addPoint() {
    if (step >= total) return;
    const t = step / total;
    // G loss starts high, decreases
    const gLoss = 2.5 * Math.exp(-2 * t) + 0.7 + (Math.random() - 0.5) * 0.3;
    // D loss starts low, then fluctuates
    const dLoss = 0.3 + 0.4 * Math.sin(t * 6) * Math.exp(-t) + 0.5 + (Math.random() - 0.5) * 0.2;
    lossData.gen.push(gLoss);
    lossData.disc.push(dLoss);
    drawLossChart();
    step++;
    lossAnimFrame = setTimeout(addPoint, 40);
  }
  addPoint();
}

function clearLossChart() {
  if (lossAnimFrame) clearTimeout(lossAnimFrame);
  lossData = { gen: [], disc: [] };
  drawLossChart();
}

function drawLossChart() {
  const c = document.getElementById('lossChart');
  if (!c) return;
  const ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  const pad = { top: 20, right: 20, bottom: 30, left: 50 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  ctx.clearRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,.06)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = pad.top + (plotH / 5) * i;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,.3)';
    ctx.font = '11px Inter';
    ctx.textAlign = 'right';
    ctx.fillText((3 - (3 / 5) * i).toFixed(1), pad.left - 8, y + 4);
  }

  // Axis labels
  ctx.fillStyle = 'rgba(255,255,255,.3)';
  ctx.font = '11px Inter';
  ctx.textAlign = 'center';
  ctx.fillText('Epoch →', W / 2, H - 4);

  if (lossData.gen.length < 2) return;
  const maxVal = 3;
  const len = lossData.gen.length;

  function drawLine(data, color) {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    for (let i = 0; i < data.length; i++) {
      const x = pad.left + (i / (len - 1)) * plotW;
      const y = pad.top + (1 - data[i] / maxVal) * plotH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Glow
    ctx.save();
    ctx.globalAlpha = 0.15;
    ctx.lineWidth = 8;
    ctx.stroke();
    ctx.restore();
  }

  drawLine(lossData.gen, '#00d4ff');
  drawLine(lossData.disc, '#ff6b9d');
}

// ─── APP CANVASES (abstract generative art) ───
function initAppCanvases() {
  const canvases = [
    { id: 'appCanvas1', type: 'faces' },
    { id: 'appCanvas2', type: 'style' },
    { id: 'appCanvas3', type: 'medical' },
    { id: 'appCanvas4', type: 'superres' },
  ];

  canvases.forEach(({ id, type }) => {
    const c = document.getElementById(id);
    if (!c) return;
    const ctx = c.getContext('2d');
    const W = c.width, H = c.height;

    // Dark background
    ctx.fillStyle = '#0d0d18';
    ctx.fillRect(0, 0, W, H);

    if (type === 'faces') {
      // Abstract face grid
      for (let i = 0; i < 12; i++) {
        const x = (i % 4) * 65 + 15;
        const y = Math.floor(i / 4) * 60 + 15;
        const grad = ctx.createRadialGradient(x + 25, y + 25, 5, x + 25, y + 25, 25);
        grad.addColorStop(0, `hsl(${200 + i * 10},70%,60%)`);
        grad.addColorStop(1, 'transparent');
        ctx.fillStyle = grad;
        ctx.fillRect(x, y, 50, 50);
        ctx.strokeStyle = 'rgba(0,212,255,.2)';
        ctx.strokeRect(x, y, 50, 50);
      }
    } else if (type === 'style') {
      // Abstract brush strokes
      for (let i = 0; i < 30; i++) {
        ctx.beginPath();
        ctx.moveTo(Math.random() * W, Math.random() * H);
        ctx.bezierCurveTo(Math.random()*W, Math.random()*H, Math.random()*W, Math.random()*H, Math.random()*W, Math.random()*H);
        ctx.strokeStyle = `hsla(${Math.random()*360},80%,60%,.3)`;
        ctx.lineWidth = Math.random() * 4 + 1;
        ctx.stroke();
      }
    } else if (type === 'medical') {
      // Abstract scan pattern
      for (let y = 0; y < H; y += 4) {
        for (let x = 0; x < W; x += 4) {
          const dist = Math.sqrt((x - W/2)**2 + (y - H/2)**2);
          const v = Math.max(0, 1 - dist / (W/2));
          if (Math.random() < v * 0.8) {
            ctx.fillStyle = `rgba(0,212,255,${v * 0.6})`;
            ctx.fillRect(x, y, 3, 3);
          }
        }
      }
    } else {
      // Super res blocks
      const sizes = [16, 8, 4];
      sizes.forEach((s, idx) => {
        const ox = idx * 90;
        for (let y = 10; y < H - 10; y += s) {
          for (let x = ox + 5; x < ox + 85; x += s) {
            const bright = 20 + Math.random() * 40;
            ctx.fillStyle = `hsl(${270+Math.random()*40},60%,${bright}%)`;
            ctx.fillRect(x, y, s - 1, s - 1);
          }
        }
        if (idx < 2) {
          ctx.fillStyle = '#00d4ff';
          ctx.font = 'bold 16px Inter';
          ctx.fillText('→', ox + 85, H / 2 + 5);
        }
      });
    }
  });
}

// ─── SCROLL REVEAL ───
function initScrollReveal() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
      }
    });
  }, { threshold: 0.2 });

  document.querySelectorAll('.training-step, .timeline-item').forEach(el => {
    observer.observe(el);
  });
}

// ─── SMOOTH SCROLL NAV ───
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function(e) {
    e.preventDefault();
    const target = document.querySelector(this.getAttribute('href'));
    if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
});

// ─── NAVBAR SCROLL EFFECT ───
window.addEventListener('scroll', () => {
  const nav = document.querySelector('.navbar');
  if (nav) {
    if (window.scrollY > 50) nav.style.background = 'rgba(10,10,15,.92)';
    else nav.style.background = 'rgba(10,10,15,.7)';
  }
});

// ─── INIT ───
document.addEventListener('DOMContentLoaded', () => {
  renderDemo();
  initAppCanvases();
  initScrollReveal();
  drawLossChart();
});
