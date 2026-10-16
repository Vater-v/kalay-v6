/* СПЛАТ · движок графики. Каждая сцена — функция, которая строит DOM и заполняет GSAP-таймлайн.
   Рендер детерминированный: render.mjs вызывает window.__setFrame(f) и снимает кадр. */
window.SCENES = {};
const FPS = 30;
const W = 1920, H = 1080;

function el(tag, cls, html, parent) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html != null) e.innerHTML = html;
  (parent || document.getElementById('root')).appendChild(e);
  return e;
}
function css(e, styles) { Object.assign(e.style, styles); return e; }

// Детерминированный шум для зерна (одна текстура, сдвигается по кадрам).
function makeGrain(parent) {
  const c = document.createElement('canvas');
  c.width = W + 128; c.height = H + 128; c.className = 'grain';
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(c.width, c.height);
  let s = 1337;
  for (let i = 0; i < img.data.length; i += 4) {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    const v = (s >> 16) & 255;
    img.data[i] = img.data[i + 1] = img.data[i + 2] = v; img.data[i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  parent.appendChild(c);
  return c;
}

// Фон «сукно». opts.light — позиция мягкого света, двигается медленно.
function feltBg(tl, dur) {
  const bg = el('div', 'bg');
  el('div', 'weave', null, bg);
  const sw = el('div', 'sweep', null, bg);
  el('div', 'vignette', null, bg);
  tl.fromTo(sw, { x: -120, y: -20 }, { x: 160, y: 30, duration: dur, ease: 'none' }, 0);
  return bg;
}

// Счётчик: форматирует число с пробелами-разделителями.
function fmt(n, dec = 0) {
  const s = Number(n).toFixed(dec);
  const [i, d] = s.split('.');
  const ii = i.replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
  return d ? ii + ',' + d : ii;
}
function countUp(tl, node, from, to, at, dur, opts = {}) {
  const o = { v: from };
  const pre = opts.pre || '', post = opts.post || '', dec = opts.dec || 0;
  node.textContent = pre + fmt(from, dec) + post;
  tl.to(o, { v: to, duration: dur, ease: opts.ease || 'power3.out', onUpdate: () => { node.textContent = pre + fmt(o.v, dec) + post; } }, at);
}

// Слова по одному (для титров): возвращает массив span.
function splitWords(node) {
  const words = node.textContent.split(' ');
  node.innerHTML = words.map(w => `<span class="w" style="display:inline-block">${w}</span>`).join(' ');
  return [...node.querySelectorAll('.w')];
}

function params() {
  const p = Object.fromEntries(new URLSearchParams(location.search));
  return p;
}

function boot() {
  const P = params();
  const id = P.scene;
  const sc = window.SCENES[id];
  if (!sc) { document.getElementById('root').innerHTML = '<div style="padding:40px;font:32px Onest;color:#fff">Сцены: ' + Object.keys(window.SCENES).join(', ') + '</div>'; return; }
  const tl = gsap.timeline({ paused: true });
  const dur = Number(P.dur || sc.dur);
  if (!sc.alpha) {
    feltBg(tl, dur);
  }
  // Зерно отключено по умолчанию (минимализм и скорость рендера); включить: ?grain=1
  const grain = (!sc.alpha && P.grain === '1') ? makeGrain(document.getElementById('stage')) : null;
  sc.build(document.getElementById('root'), tl, P, dur);
  tl.set({}, {}, dur); // фиксируем длину
  window.__scene = { id, dur, alpha: !!sc.alpha, fps: FPS, frames: Math.round(dur * FPS) };
  window.__setFrame = (f) => {
    tl.time(f / FPS, false);
    if (grain) { const k = (f * 7919) % 64; grain.style.transform = `translate(${-k}px, ${-((f * 104729) % 64)}px)`; }
  };
  window.__setFrame(0);
  if (P.play) {
    let f = 0; const n = window.__scene.frames;
    setInterval(() => { window.__setFrame(f); f = (f + 1) % n; }, 1000 / FPS);
  }
  if (P.at) window.__setFrame(Math.round(Number(P.at) * FPS));
  document.fonts.ready.then(() => { window.__ready = true; });
}
