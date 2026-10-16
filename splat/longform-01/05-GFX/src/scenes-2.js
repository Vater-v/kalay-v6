/* Сцены пачки A: G04, G05(+full), G06, G07, G08, G09, G11, G12, G13, G14a, G14b, G15.
   Всё в IIFE, чтобы не конфликтовать с глобальными именами других файлов сцен.
   Движение — только через tl (включая «живые» столы: onUpdate-прокси, чистая функция от t). */
(function () {
'use strict';

/* ───────────── общие помощники ───────────── */
const SVGNS = 'http://www.w3.org/2000/svg';
function sv(tag, attrs, parent) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function svgBox(parent, w, h, extra) {
  const s = sv('svg', { width: w, height: h, viewBox: '0 0 ' + w + ' ' + h }, parent);
  Object.assign(s.style, { position: 'absolute', left: '0', top: '0', overflow: 'visible' }, extra || {});
  return s;
}
const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const lerp = (a, b, t) => a + (b - a) * t;
const EZ = {};
const ez = (n) => EZ[n] || (EZ[n] = gsap.parseEase(n));
// прогресс сегмента [a, a+d] с easing
function seg(t, a, d, e) { return ez(e || 'power2.out')(clamp((t - a) / d)); }
// детерминированный генератор (mulberry32)
function rng(a) {
  a >>>= 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
// обратная функция easing: x∈[0,1], при котором ease(x) = p
function timeAt(easeName, p) {
  const f = ez(easeName); let lo = 0, hi = 1;
  for (let i = 0; i < 32; i++) { const m = (lo + hi) / 2; if (f(m) < p) lo = m; else hi = m; }
  return (lo + hi) / 2;
}
const NB = ' ';           // неразрывный пробел
const MINUS = '−';
function pad2(n) { return (n < 10 ? '0' : '') + n; }
function money2(v) { return '$' + v.toFixed(2); }
function useStyle(root, text) { return el('style', null, text, root); }

/* ───────────── G04 · карточка главы ───────────── */
SCENES['G04-chapter'] = { dur: 2.5, build(root, tl, P) {
  const n = P.n || '01', t = P.t || 'Название главы', d = P.d || '';
  const wrap = el('div', 'abs col', null, root);
  css(wrap, { left: '96px', right: '96px', top: '0', bottom: '0', alignItems: 'center', justifyContent: 'center', textAlign: 'center' });
  const num = el('div', 'display', n, wrap);
  css(num, { fontSize: '520px', lineHeight: '.98', color: 'transparent', webkitTextFillColor: 'transparent',
    backgroundImage: 'linear-gradient(180deg, rgba(242,193,78,.95) 0%, rgba(242,193,78,.5) 55%, rgba(242,193,78,.12) 100%)', webkitBackgroundClip: 'text', backgroundClip: 'text' });
  const title = el('div', 'title', t, wrap);
  css(title, { fontSize: '96px', lineHeight: '1.08', maxWidth: '1500px', textWrap: 'balance', marginTop: '20px' });
  const words = splitWords(title);
  let chip = null;
  if (d) {
    chip = el('div', 'chip', d, wrap);
    css(chip, { fontSize: '40px', padding: '16px 38px', marginTop: '44px' });
  }
  tl.from(num, { opacity: 0, y: 90, scale: .94, duration: .65, ease: 'expo.out' }, 0)
    .from(words, { opacity: 0, y: 40, duration: .45, stagger: .08, ease: 'power3.out' }, .3);
  if (chip) tl.from(chip, { opacity: 0, y: 28, duration: .4, ease: 'power3.out' }, .7);
}};

/* ───────────── G05 · таймлайн ───────────── */
const TLD = [
  ['9 апр', 'Проект жизни'], ['23 апр', 'Курс и марафон'], ['4 мая', '«Потому что хочу»'], ['17 мая', 'Ноу-лайф'],
  ['19 июн', 'NL100'], ['22 июн', '18 столов'], ['14 июл', 'Итог марафона'], ['18 июл', 'Пари'],
  ['24 июл', 'Брови'], ['10 сен', 'Новый марафон'], ['12 сен', '40 БИ'], ['1 окт', 'Стопнул'],
];
const TL_HOT = { 8: 'var(--red)', 11: 'var(--mint)' };
const tlColor = (i) => TL_HOT[i] || 'var(--gold)';

// ?at=N — компактная полоска (alpha)
SCENES['G05-timeline'] = { dur: 3, alpha: true, build(root, tl, P) {
  const N = clamp(parseInt(P.at, 10) || 0, 0, TLD.length - 1);
  const W = 820, H = 112;
  const plate = el('div', 'abs glass', null, root);
  css(plate, { left: '64px', top: '48px', width: W + 'px', height: H + 'px', borderRadius: '24px' });
  const x0 = 46, x1 = W - 46, step = (x1 - x0) / (TLD.length - 1);
  const X = (i) => x0 + i * step;
  const ay = 84;
  const track = el('div', null, null, plate);
  css(track, { position: 'absolute', left: x0 + 'px', top: (ay - 3) + 'px', width: (x1 - x0) + 'px', height: '6px', borderRadius: '3px', background: 'rgba(244,241,232,.18)' });
  const fill = el('div', null, null, plate);
  const fx0 = N > 0 ? X(N - 1) : x0;
  css(fill, { position: 'absolute', left: x0 + 'px', top: (ay - 3) + 'px', width: (fx0 - x0) + 'px', height: '6px', borderRadius: '3px', background: 'var(--gold)' });
  const dotEls = TLD.map((_, i) => {
    const dEl = el('div', null, null, plate);
    css(dEl, { position: 'absolute', left: (X(i) - 7) + 'px', top: (ay - 7) + 'px', width: '14px', height: '14px', borderRadius: '50%', background: i < N ? tlColor(i) : 'rgba(244,241,232,.34)' });
    return dEl;
  });
  const marker = el('div', null, null, plate);
  css(marker, { position: 'absolute', left: (fx0 - 16) + 'px', top: (ay - 16) + 'px', width: '32px', height: '32px', borderRadius: '50%', border: '6px solid ' + tlColor(N), background: '#0a1612' });
  const lab = el('div', 'row', null, plate);
  css(lab, { position: 'absolute', left: '34px', right: '34px', top: '12px', height: '46px', gap: '18px', alignItems: 'baseline' });
  const dt = el('div', 'title', TLD[N][0], lab); css(dt, { fontSize: '32px', color: tlColor(N), whiteSpace: 'nowrap' });
  const cp = el('div', 'body', TLD[N][1], lab); css(cp, { fontSize: '36px', fontWeight: 800, whiteSpace: 'nowrap' });
  tl.from(plate, { opacity: 0, x: -40, duration: .35, ease: 'expo.out' }, 0);
  if (N > 0) {
    tl.fromTo(marker, { left: X(N - 1) - 16 }, { left: X(N) - 16, duration: .7, ease: 'expo.out' }, .2)
      .fromTo(fill, { width: X(N - 1) - x0 }, { width: X(N) - x0, duration: .7, ease: 'expo.out' }, .2)
      .fromTo(dotEls[N], { background: 'rgba(244,241,232,.34)' }, { background: tlColor(N), duration: .3 }, .5);
  } else {
    tl.from(marker, { scale: 0, duration: .4, ease: 'expo.out' }, .2);
  }
  tl.from(lab, { opacity: 0, y: 12, duration: .35, ease: 'power3.out' }, .4);
}};

// полноэкранная версия
SCENES['G05-timeline-full'] = { dur: 6, build(root, tl) {
  const n = TLD.length, axY = 600, ax0 = 190, ax1 = 1730, step = (ax1 - ax0) / (n - 1);
  const X = (i) => ax0 + i * step;
  const head = el('div', 'title', 'Апрель → октябрь 2026', root); css(head, { position: 'absolute', left: '96px', top: '96px', fontSize: '64px' });
  // ось
  const track = el('div', null, null, root);
  css(track, { position: 'absolute', left: (ax0 - 20) + 'px', top: (axY - 5) + 'px', width: (ax1 - ax0 + 40) + 'px', height: '10px', borderRadius: '5px', background: 'rgba(244,241,232,.16)' });
  const fill = el('div', null, null, root);
  css(fill, { position: 'absolute', left: (ax0 - 20) + 'px', top: (axY - 5) + 'px', width: '20px', height: '10px', borderRadius: '5px', background: 'linear-gradient(90deg,#f7d77e,var(--gold))' });
  tl.from(head, { opacity: 0, y: 20, duration: .4, ease: 'power3.out' }, 0)
    .from(track, { opacity: 0, scaleX: 0, transformOrigin: 'left center', duration: .5, ease: 'power3.out' }, 0)
    .from(fill, { opacity: 0, duration: .2 }, 0);
  const t0 = .4, dt = .4;
  TLD.forEach((p, i) => {
    const up = i % 2 === 0;
    const ti = t0 + i * dt;
    const col = tlColor(i);
    const big = i === 7 || i === 8;
    const D = big ? 46 : 36;
    // линия-«ножка»
    const stemH = 74;
    const stem = el('div', null, null, root);
    css(stem, { position: 'absolute', left: (X(i) - 1.5) + 'px', width: '3px', height: stemH + 'px', background: 'rgba(244,241,232,.38)', top: (up ? axY - D / 2 - stemH : axY + D / 2) + 'px', transformOrigin: up ? 'center bottom' : 'center top' });
    // точка
    const dot = el('div', null, null, root);
    css(dot, { position: 'absolute', left: (X(i) - D / 2) + 'px', top: (axY - D / 2) + 'px', width: D + 'px', height: D + 'px', borderRadius: '50%', background: col, boxShadow: '0 0 0 7px #0b1f18' });
    // подпись
    const lab = el('div', 'col', null, root);
    css(lab, { position: 'absolute', left: (X(i) - 140) + 'px', width: '280px', alignItems: 'center', textAlign: 'center', gap: '8px' });
    if (up) lab.style.bottom = (1080 - (axY - D / 2 - stemH) + 14) + 'px'; else lab.style.top = (axY + D / 2 + stemH + 14) + 'px';
    const dEl = el('div', 'title', p[0], lab); css(dEl, { fontSize: '38px', color: col, whiteSpace: 'nowrap' });
    const cEl = el('div', 'body', p[1], lab); css(cEl, { fontSize: '36px', fontWeight: 800, lineHeight: '1.12' });
    if (up) lab.insertBefore(cEl, dEl);
    // анимация
    tl.from(dot, { scale: 0, duration: .35, ease: 'expo.out' }, ti)
      .from(stem, { scaleY: 0, duration: .25, ease: 'power3.out' }, ti + .05)
      .from(lab, { opacity: 0, y: up ? 18 : -18, duration: .35, ease: 'power3.out' }, ti + .1)
      .fromTo(fill, { width: (i === 0 ? 20 : X(i - 1) - ax0 + 20) + 'px' }, { width: (X(i) - ax0 + 20) + 'px', duration: .38, ease: 'power2.out', immediateRender: false }, ti - (i === 0 ? .05 : .1));
  });
}};

/* ───────────── G06 · Dota ───────────── */
SCENES['G06-dota'] = { dur: 4, build(root, tl) {
  const perDay = [5, 7, 6, 8, 4, 9, 6, 7, 5, 8, 6, 7, 5, 6]; // сумма 89
  const total = perDay.reduce((a, b) => a + b, 0);
  const cd = 1.4, cs = .2, ease = 'power2.out';
  const left = el('div', 'abs col', null, root); css(left, { left: '96px', top: '0', bottom: '0', justifyContent: 'center' });
  const num = el('div', 'num', '0', left); css(num, { fontSize: '500px', lineHeight: '.9', letterSpacing: '-.04em', color: 'var(--paper)' });
  const lab = el('div', 'body', 'матчей за 14 дней', left); css(lab, { fontSize: '68px', fontWeight: 800, marginTop: '30px' });
  const chip = el('div', 'chip pill-gold', '6+ в день', left); css(chip, { fontSize: '46px', padding: '16px 40px', marginTop: '40px', alignSelf: 'flex-start', fontWeight: 800 });
  // календарь: 14 клеток без подложки
  const cell = 104, gp = 14;
  const cw = 7 * cell + 6 * gp, chh = 2 * cell + gp;
  const cal = el('div', 'abs', null, root);
  css(cal, { left: (1824 - cw) + 'px', top: ((1080 - chh) / 2) + 'px', width: cw + 'px', height: chh + 'px' });
  const cells = perDay.map((v, i) => {
    const c = el('div', null, null, cal);
    css(c, { position: 'absolute', left: ((i % 7) * (cell + gp)) + 'px', top: (Math.floor(i / 7) * (cell + gp)) + 'px', width: cell + 'px', height: cell + 'px', borderRadius: '22px', background: 'rgba(244,241,232,.07)', border: '1.5px solid rgba(244,241,232,.16)', overflow: 'hidden' });
    const f = el('div', null, null, c); css(f, { position: 'absolute', inset: '0', background: 'linear-gradient(160deg,#f7d77e,var(--gold))' });
    return f;
  });
  tl.from(left, { opacity: 0, x: -40, duration: .45, ease: 'expo.out' }, 0)
    .from(cal, { opacity: 0, x: 50, duration: .45, ease: 'expo.out' }, .05);
  countUp(tl, num, 0, total, cs, cd, { ease });
  let cum = 0;
  cells.forEach((f, i) => {
    cum += perDay[i];
    const tt = cs + cd * timeAt(ease, cum / total);
    tl.fromTo(f, { opacity: 0, scale: .55 }, { opacity: 1, scale: 1, duration: .28, ease: 'expo.out' }, Math.max(.15, tt - .2));
  });
  tl.from(lab, { opacity: 0, y: 20, duration: .4, ease: 'power3.out' }, .5)
    .from(chip, { opacity: 0, y: 24, scale: .92, duration: .4, ease: 'expo.out' }, 1.1);
}};

/* ───────────── G07 · сон ───────────── */
SCENES['G07-sleep'] = { dur: 4, build(root, tl) {
  const q = el('div', 'body muted', '«Вопрос анкеты №12: режим сна»', root);
  css(q, { position: 'absolute', left: '0', right: '0', top: '84px', textAlign: 'center', fontSize: '34px', fontWeight: 600 });
  // время — главный акцент
  const CY = 500;
  const digital = el('div', 'num', '07:00', root);
  css(digital, { position: 'absolute', left: '96px', top: (CY - 150) + 'px', fontSize: '300px', lineHeight: '1', color: 'var(--paper)', whiteSpace: 'nowrap' });
  const S = { m: 420 };
  const draw = () => {
    const mm = Math.round(S.m) % 1440;
    digital.textContent = pad2(Math.floor(mm / 60) % 24) + ':' + pad2(mm % 60);
  };
  draw();
  // правая часть: иконка + подпись
  const RX = 1450;
  const slot = el('div', 'abs', null, root); css(slot, { left: (RX - 190) + 'px', top: (CY - 250) + 'px', width: '380px', height: '380px' });
  const phone = el('div', 'abs', null, slot); css(phone, { left: '90px', top: '0', width: '200px', height: '380px' });
  const shaker = el('div', 'abs', null, phone); css(shaker, { left: '0', top: '0', width: '200px', height: '380px' });
  shaker.innerHTML = `<svg width="200" height="380" viewBox="0 0 200 380" style="overflow:visible">
    <rect x="25" y="10" width="150" height="360" rx="36" fill="#0c1914" stroke="#f4f1e8" stroke-width="7"/>
    <circle cx="100" cy="190" r="48" fill="#3ddc97"/>
    <path d="M84 168c-4 0-8 3-8 8 0 22 18 40 40 40 5 0 8-4 8-8v-8l-14-5-6 6c-8-4-14-10-18-18l6-6-5-14z" fill="#0f1412"/>
    <path d="M10 150c-8 14-8 32 0 46M-8 136c-14 22-14 64 0 86" fill="none" stroke="#f2c14e" stroke-width="7" stroke-linecap="round"/>
    <path d="M190 150c8 14 8 32 0 46M208 136c14 22 14 64 0 86" fill="none" stroke="#f2c14e" stroke-width="7" stroke-linecap="round"/>
  </svg>`;
  const arcs = shaker.querySelectorAll('path[stroke="#f2c14e"]');
  const head = el('div', 'abs', null, slot); css(head, { left: '10px', top: '10px', width: '360px', height: '360px' });
  head.innerHTML = `<svg width="360" height="360" viewBox="0 0 300 300" style="overflow:visible">
    <rect x="12" y="12" width="276" height="276" rx="70" fill="#efe6d2"/>
    <circle cx="100" cy="126" r="17" fill="#0f1412"/><circle cx="200" cy="126" r="17" fill="#0f1412"/>
    <path d="M108 208h84" stroke="#0f1412" stroke-width="11" stroke-linecap="round"/>
  </svg>`;
  const mkCap = (txt, cls) => {
    const c = el('div', 'body ' + cls, txt, root);
    css(c, { position: 'absolute', left: (RX - 400) + 'px', width: '800px', textAlign: 'center', top: (CY + 170) + 'px', fontSize: '68px', fontWeight: 800, whiteSpace: 'nowrap' });
    return c;
  };
  const cap1 = mkCap('звонок сестры', ''), cap2 = mkCap('квадратная голова', 'gold');
  // анимация
  tl.from(q, { opacity: 0, y: -16, duration: .4, ease: 'power3.out' }, 0)
    .from(digital, { opacity: 0, y: 30, duration: .5, ease: 'expo.out' }, .1);
  tl.fromTo(S, { m: 420 }, { m: 780, duration: .95, ease: 'power2.inOut', onUpdate: draw, immediateRender: false }, .55);
  tl.fromTo(phone, { opacity: 0, x: 60 }, { opacity: 1, x: 0, duration: .4, ease: 'expo.out' }, 1.45)
    .from(cap1, { opacity: 0, y: 24, duration: .4, ease: 'power3.out' }, 1.55);
  const V = { v: 0 };
  tl.fromTo(V, { v: 0 }, { v: 1, duration: .8, ease: 'none', immediateRender: false, onUpdate: () => {
    const k = Math.floor(V.v * 16), on = V.v > 0 && V.v < 1;
    shaker.style.transform = on ? 'translate(' + ((k % 2 ? 3 : -3)) + 'px,' + ((k % 4 < 2 ? -1 : 1)) + 'px) rotate(' + ((k % 2 ? .6 : -.6)) + 'deg)' : 'none';
  } }, 1.75);
  tl.fromTo(arcs, { opacity: 0 }, { opacity: 1, duration: .2, stagger: .05, ease: 'power2.out' }, 1.7);
  tl.fromTo(S, { m: 780 }, { m: 900, duration: .65, ease: 'power2.inOut', onUpdate: draw, immediateRender: false }, 2.6)
    .to(phone, { opacity: 0, x: -50, duration: .3, ease: 'power3.in' }, 2.6)
    .to(cap1, { opacity: 0, y: -20, duration: .25, ease: 'power3.in' }, 2.6)
    .fromTo(head, { opacity: 0, y: 40, scale: .85 }, { opacity: 1, y: 0, scale: 1, duration: .5, ease: 'expo.out' }, 2.95)
    .fromTo(cap2, { opacity: 0, y: 24 }, { opacity: 1, y: 0, duration: .4, ease: 'power3.out' }, 3.05)
    .to(digital, { color: '#f2c14e', duration: .3 }, 3.1);
}};

/* ───────────── G08 · страницы А4 ───────────── */
// ?dip=1 — продолжение для опции «Причём одиннадцатым шрифтом…»: стопка уже собрана, появляются плашка «шрифт 11» и «дипломы — шрифт 14».
SCENES['G08-pages'] = { dur: 5, build(root, tl, P) {
  const dipMode = P.dip === '1';
  const NP = 48, cs = .2, cd = 2.3, ease = 'power2.out', TH = 6;
  useStyle(root, `.sheet{position:absolute;left:0;top:0;width:380px;height:538px;border-radius:4px;background:linear-gradient(180deg,#faf7ee,#ebe6d6);border:1px solid rgba(15,20,18,.3)}
.sheet:before{content:'';position:absolute;left:46px;right:46px;top:56px;bottom:64px;background:repeating-linear-gradient(180deg,rgba(15,20,18,.34) 0 5px,transparent 5px 25px)}
`);
  const holder = el('div', 'abs', null, root);
  css(holder, { left: '270px', top: '320px', width: '380px', height: '538px', perspective: '2400px', perspectiveOrigin: '50% 30%' });
  const stack = el('div', 'abs', null, holder);
  css(stack, { left: '0', top: '0', width: '380px', height: '538px', transformStyle: 'preserve-3d', transform: 'rotateX(57deg) rotateZ(-32deg)' });
  const shadow = el('div', 'abs', null, stack);
  css(shadow, { left: '-30px', top: '-30px', width: '440px', height: '598px', background: 'rgba(0,0,0,.6)', filter: 'blur(26px)', borderRadius: '30px', transform: 'translateZ(-3px)' });
  const sheets = [];
  for (let i = 0; i < NP; i++) {
    const s = el('div', 'sheet', null, stack);
    s.style.transform = 'translateZ(' + (i * TH) + 'px)';
    sheets.push(s);
  }
  // правая колонка
  const col = el('div', 'abs col', null, root); css(col, { left: '1040px', top: '0', bottom: '0', justifyContent: 'center', gap: '0' });
  const num = el('div', 'num', '1', col); css(num, { fontSize: '360px', lineHeight: '.92', letterSpacing: '-.04em' });
  const lab = el('div', 'body', 'страниц А4', col); css(lab, { fontSize: '72px', fontWeight: 800, marginTop: '14px' });
  const chip = el('div', 'chip', 'шрифт <b class="f11" style="color:#f4f1e8;font-family:Unb;font-weight:900;margin-left:2px">11</b>', col);
  css(chip, { fontSize: '46px', padding: '16px 38px', marginTop: '40px', alignSelf: 'flex-start', borderWidth: '2px', fontWeight: 800 });
  const f11 = chip.querySelector('.f11');
  // плашка шрифта и строка про дипломы есть в обеих версиях (в основной — невидимые), чтобы колонка не прыгала на стыке
  const dip = el('div', 'body muted', 'дипломы — шрифт 14', col);
  css(dip, { fontSize: '44px', fontWeight: 600, marginTop: '30px', alignSelf: 'flex-start' });
  if (dipMode) {
    num.textContent = String(NP);
    css(f11, { color: '#f2c14e' }); css(chip, { borderColor: 'rgba(242,193,78,.95)', backgroundColor: 'rgba(242,193,78,.16)' });
    tl.fromTo(chip, { opacity: 0, y: 20 }, { opacity: 1, y: 0, duration: .4, ease: 'power3.out' }, .2)
      .fromTo(dip, { opacity: 0, y: 20 }, { opacity: 1, y: 0, duration: .4, ease: 'power3.out' }, .9);
    return;
  }
  css(chip, { visibility: 'hidden' }); css(dip, { visibility: 'hidden' });
  // анимация
  tl.from(col, { opacity: 0, x: 40, duration: .45, ease: 'expo.out' }, 0);
  countUp(tl, num, 1, NP, cs, cd, { ease });
  sheets.forEach((s, i) => {
    const ta = cs + cd * timeAt(ease, i / (NP - 1));
    tl.fromTo(s, { z: i * TH + 110, opacity: 0 }, { z: i * TH, opacity: 1, duration: .24, ease: 'power2.out' }, ta);
  });
  tl.from(lab, { opacity: 0, y: 20, duration: .4, ease: 'power3.out' }, .45);
}};

/* ───────────── G09 · лестница лимитов ───────────── */
SCENES['G09-limits'] = { dur: 4, build(root, tl, P) {
  const hi = (P.hi || '').toUpperCase(), arrow = P.arrow === '1';
  const data = [['NL2', 2], ['NL5', 5], ['NL10', 10], ['NL25', 25], ['NL100', 100]];
  const SW = 316, GP = 12, x0 = 146, base = 990, H = (i) => 200 + i * 135;
  const cap = el('div', 'title', 'Лимит = <span class="gold">бай-ин</span> за столом', root);
  css(cap, { position: 'absolute', left: '96px', top: '92px', fontSize: '60px', lineHeight: '1.1' });
  const blocks = data.map(([name, v], i) => {
    const b = el('div', 'abs', null, root);
    css(b, { left: (x0 + i * (SW + GP)) + 'px', top: (base - H(i)) + 'px', width: SW + 'px', height: H(i) + 'px' });
    const bg = el('div', 'abs', null, b);
    css(bg, { left: '0', top: '0', right: '0', bottom: '0', borderRadius: '26px 26px 10px 10px', background: 'rgba(244,241,232,' + (.07 + i * .028) + ')', border: '1.5px solid rgba(244,241,232,.22)', transformOrigin: '50% 100%' });
    const gold = el('div', 'abs', null, b);
    css(gold, { left: '0', top: '0', right: '0', bottom: '0', borderRadius: '26px 26px 10px 10px', background: 'linear-gradient(180deg,#f9dd8c,#f2c14e)', opacity: 0, boxShadow: '0 0 70px rgba(242,193,78,.45)' });
    const txt = el('div', 'abs col', null, b); css(txt, { left: '30px', top: '26px', gap: '8px' });
    const nm = el('div', 'display', name, txt); css(nm, { fontSize: '62px', color: 'var(--paper)' });
    const val = el('div', 'num', '$' + v, txt); css(val, { fontSize: '56px', color: 'var(--gold)' });
    tl.from(bg, { scaleY: 0, duration: .55, ease: 'expo.out' }, .1 + i * .1)
      .from(txt, { opacity: 0, y: 26, duration: .4, ease: 'power3.out' }, .28 + i * .1);
    return { b, bg, gold, txt, nm, val, x: x0 + i * (SW + GP), top: base - H(i) };
  });
  tl.from(cap, { opacity: 0, y: -20, duration: .4, ease: 'power3.out' }, 0);
  const hi_i = data.findIndex(d => d[0] === hi);
  if (hi_i >= 0) {
    blocks.forEach((k, i) => {
      if (i === hi_i) {
        tl.to(k.gold, { opacity: 1, duration: .35, ease: 'power2.out' }, 1.0)
          .to([k.nm, k.val], { color: '#0f1412', duration: .3 }, 1.0);
      } else tl.to(k.b, { opacity: .42, duration: .4 }, 1.0);
    });
  }
  if (arrow) {
    const o = blocks[2], d = blocks[4];
    const p0 = { x: o.x + SW / 2, y: o.top - 30 }, p1 = { x: 1480, y: 110 }, p2 = { x: d.x + SW / 2, y: d.top - 40 };
    const svg = svgBox(root, 1920, 1080);
    const path = sv('path', { d: 'M' + p0.x + ' ' + p0.y + ' Q' + p1.x + ' ' + p1.y + ' ' + p2.x + ' ' + p2.y, fill: 'none', stroke: '#ff5a4e', 'stroke-width': 16, 'stroke-linecap': 'round' }, svg);
    const L = path.getTotalLength();
    path.style.strokeDasharray = L + ' ' + (L + 10); path.style.strokeDashoffset = L;
    const head = sv('polygon', { points: '-30,-26 30,0 -30,26', fill: '#ff5a4e' }, svg); head.style.opacity = 0;
    const lab = el('div', 'body red', 'играть дороже', root);
    css(lab, { position: 'absolute', left: (p0.x - 520) + 'px', width: '470px', top: (p0.y - 120) + 'px', textAlign: 'right', fontSize: '62px', fontWeight: 800, whiteSpace: 'nowrap' });
    const A = { p: 0 };
    const upd = () => {
      const p = A.p, len = L * p;
      path.style.strokeDashoffset = L - len;
      if (p > 0.02) {
        const a = path.getPointAtLength(len), b = path.getPointAtLength(Math.max(0, len - 4));
        const ang = Math.atan2(a.y - b.y, a.x - b.x) * 180 / Math.PI;
        head.setAttribute('transform', 'translate(' + a.x + ' ' + a.y + ') rotate(' + ang + ')');
        head.style.opacity = 1;
      } else head.style.opacity = 0;
    };
    tl.fromTo(lab, { opacity: 0, x: -30 }, { opacity: 1, x: 0, duration: .4, ease: 'power3.out' }, 1.5)
      .fromTo(A, { p: 0 }, { p: 1, duration: .5, ease: 'power3.inOut', onUpdate: upd, immediateRender: false }, 1.8);
  }
}};

/* ───────────── общая «оболочка» стола (вид сверху) ───────────── */
function tableShell(parent, w, h, o) {
  o = o || {};
  const box = el('div', 'abs', null, parent); css(box, { width: w + 'px', height: h + 'px' });
  const ow = w * (o.ow || .84), oh = h * (o.oh || .6), ox = (w - ow) / 2, oy = (h - oh) / 2;
  const oval = el('div', 'abs', null, box);
  css(oval, { left: ox + 'px', top: oy + 'px', width: ow + 'px', height: oh + 'px', borderRadius: '50%',
    background: 'radial-gradient(ellipse at 50% 42%, #218259 0%, #166647 58%, #0f4c36 100%)',
    border: (o.bw || Math.max(6, Math.round(w * .022))) + 'px solid #2b1d14',
    boxShadow: 'inset 0 0 0 ' + Math.max(2, Math.round(w * .004)) + 'px rgba(242,193,78,.3), inset 0 14px 34px rgba(0,0,0,.35), 0 22px 44px rgba(0,0,0,.5)' });
  const rx = ow / 2, ry = oh / 2, cx = w / 2, cy = h / 2;
  const seat = (deg, gap) => { const a = deg * Math.PI / 180, g = gap == null ? 0 : gap; return { x: cx + (rx + g) * Math.cos(a), y: cy + (ry + g) * Math.sin(a) }; };
  return { box, oval, cx, cy, rx, ry, seat };
}
function seatDisc(parent, x, y, d, o) {
  o = o || {};
  const e = el('div', 'abs', null, parent);
  css(e, { left: (x - d / 2) + 'px', top: (y - d / 2) + 'px', width: d + 'px', height: d + 'px', borderRadius: '50%',
    background: o.bg || 'linear-gradient(160deg,#3a4a43,#1b2823)', border: (o.bw || 4) + 'px solid ' + (o.bc || 'rgba(244,241,232,.35)'), boxShadow: '0 6px 14px rgba(0,0,0,.4)' });
  return e;
}
function miniCard(parent, rank, suit, w, h, x, y, back) {
  const red = suit === '♥' || suit === '♦';
  const c = el('div', 'pcard' + (red ? ' r' : '') + (back ? ' back' : ''), back ? '' : '<div class="rk">' + rank + '</div><div class="st">' + suit + '</div>', parent);
  css(c, { position: 'absolute', left: x + 'px', top: y + 'px', width: w + 'px', height: h + 'px', borderRadius: Math.round(w * .11) + 'px' });
  if (!back) {
    const rk = c.querySelector('.rk'), st = c.querySelector('.st');
    css(rk, { left: Math.round(w * .1) + 'px', top: Math.round(h * .04) + 'px', fontSize: Math.round(w * .44) + 'px' });
    css(st, { right: Math.round(w * .09) + 'px', bottom: Math.round(h * .03) + 'px', fontSize: Math.round(w * .56) + 'px' });
  } else c.style.borderWidth = Math.max(3, Math.round(w * .06)) + 'px';
  return c;
}
const RC_CSS = `.rc{position:absolute;left:0;top:0;width:46px;height:46px;border-radius:50%;background:radial-gradient(circle,#f6f1e3 0 26%,var(--red) 28% 100%);border:4px dashed rgba(255,255,255,.72);box-shadow:0 5px 12px rgba(0,0,0,.4);opacity:0}
.rc.m{background:radial-gradient(circle,#f6f1e3 0 26%,var(--mint) 28% 100%)}
.rc.g{background:radial-gradient(circle,#fff6dc 0 26%,var(--gold) 28% 100%)}`;

/* ───────────── G11 · дисперсия ───────────── */
SCENES['G11-variance'] = { dur: 5, build(root, tl) {
  const NS = 10;
  // фаза 1: 10 стопок гаснут, счётчик просадки
  const ph1 = el('div', 'abs', null, root); css(ph1, { left: '0', top: '0', width: '1920px', height: '1080px' });
  const hn = el('div', 'num', MINUS + '0' + NB + 'БИ', ph1);
  css(hn, { position: 'absolute', left: '0', width: '1920px', textAlign: 'center', top: '210px', fontSize: '210px', color: 'var(--paper)' });
  const row = el('div', 'abs', null, ph1); css(row, { left: '0', top: '560px', width: '1920px', height: '120px' });
  const stacks = [];
  const sx0 = (1920 - (NS * 120 + (NS - 1) * 44)) / 2;
  for (let i = 0; i < NS; i++) {
    const st = el('div', 'chipstack', null, row);
    css(st, { position: 'absolute', left: (sx0 + i * 164) + 'px', top: '0', height: '120px' });
    for (let j = 0; j < 6; j++) {
      const c = el('div', 'pchip', null, st);
      css(c, { top: (82 - j * 14) + 'px', background: 'radial-gradient(ellipse at center, #fff6dc 0 28%, var(--gold) 30% 100%)' });
    }
    stacks.push(st);
  }
  // фаза 2: график банкролла
  const CW = 1620, CH = 640, CL = 150, CT = 170;
  const chart = el('div', 'abs', null, root); css(chart, { left: CL + 'px', top: CT + 'px', width: CW + 'px', height: CH + 'px' });
  const svg = svgBox(chart, CW, CH);
  const Y = (v) => (CH - 34) - (v - 28) / 36 * (CH - 90), Xp = (u) => 20 + u * (CW - 60);
  sv('line', { x1: 0, x2: 0, y1: 0, y2: CH - 14, stroke: 'rgba(244,241,232,.3)', 'stroke-width': 3 }, svg);
  sv('line', { x1: 0, x2: CW, y1: CH - 14, y2: CH - 14, stroke: 'rgba(244,241,232,.3)', 'stroke-width': 3 }, svg);
  const s1 = [[0, 34], [.04, 35.5], [.08, 34.5], [.12, 37.5], [.16, 36.5], [.20, 40], [.24, 38.8], [.28, 42], [.32, 41.2], [.36, 44.5], [.40, 47.5], [.43, 46.5], [.46, 52]];
  const s2 = [[.46, 52], [.49, 49.2], [.51, 50], [.55, 46.4], [.57, 47.2], [.62, 42]];
  const s3 = [[.62, 42], [.66, 43.5], [.70, 42.8], [.74, 46], [.78, 45], [.82, 49.5], [.86, 48.6], [.90, 53], [.94, 56.5], [.97, 55.8], [1, 60]];
  const mkPath = (pts, col, w) => {
    const p = sv('polyline', { points: pts.map(q => Xp(q[0]).toFixed(1) + ',' + Y(q[1]).toFixed(1)).join(' '), fill: 'none', stroke: col, 'stroke-width': w, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, svg);
    const L = p.getTotalLength(); p.style.strokeDasharray = L + ' ' + (L + 10); p.style.strokeDashoffset = L;
    return { p, L };
  };
  const g1 = mkPath(s1, '#f2c14e', 11), g2 = mkPath(s2, '#ff5a4e', 13), g3 = mkPath(s3, '#f2c14e', 11);
  const px = Xp(.46), tx = Xp(.62), py = Y(52), ty = Y(42);
  const dash1 = sv('line', { x1: px, y1: py, x2: tx + 70, y2: py, stroke: 'rgba(255,90,78,.7)', 'stroke-width': 3, 'stroke-dasharray': '8 10' }, svg);
  const dash2 = sv('line', { x1: tx, y1: py, x2: tx, y2: ty, stroke: 'rgba(255,90,78,.9)', 'stroke-width': 4, 'stroke-dasharray': '8 8' }, svg);
  dash1.style.opacity = 0; dash2.style.opacity = 0;
  const dl = el('div', 'num red', MINUS + '10' + NB + 'БИ', chart);
  css(dl, { position: 'absolute', left: (tx - 220) + 'px', top: (ty + 30) + 'px', width: '440px', textAlign: 'center', fontSize: '92px', whiteSpace: 'nowrap' });
  const ylab = el('div', 'note', 'банкролл', chart); css(ylab, { position: 'absolute', left: '22px', top: '-2px' });
  const note = el('div', 'note', 'условный пример', root); css(note, { position: 'absolute', left: '0', right: '0', bottom: '40px', textAlign: 'center' });
  // анимация
  tl.from(hn, { opacity: 0, y: -20, duration: .4, ease: 'power3.out' }, 0)
    .from(stacks, { opacity: 0, y: 30, duration: .4, stagger: .05, ease: 'power3.out' }, .1)
    .from(note, { opacity: 0, duration: .4 }, .5);
  const f0 = .7, fs = .11;
  stacks.forEach((st, i) => {
    tl.fromTo(st, { filter: 'grayscale(0)', opacity: 1 }, { filter: 'grayscale(1)', opacity: .16, y: 14, duration: .3, ease: 'power2.out', immediateRender: false }, f0 + i * fs);
  });
  const S = { k: 0 };
  tl.fromTo(S, { k: 0 }, { k: 1, duration: 5, ease: 'none', immediateRender: false, onUpdate: () => {
    const t = S.k * 5, n = clamp(Math.floor((t - f0) / fs) + 1, 0, NS);
    hn.textContent = MINUS + n + NB + 'БИ';
    hn.style.color = n === 0 ? '' : '#ff5a4e';
  } }, 0);
  tl.to(ph1, { opacity: 0, duration: .3, ease: 'power2.in' }, 2.2)
    .from(chart, { opacity: 0, duration: .35 }, 2.35)
    .fromTo(g1.p, { strokeDashoffset: g1.L }, { strokeDashoffset: 0, duration: .8, ease: 'power1.inOut', immediateRender: false }, 2.5)
    .fromTo(g2.p, { strokeDashoffset: g2.L }, { strokeDashoffset: 0, duration: .4, ease: 'power2.in', immediateRender: false }, 3.3)
    .fromTo([dash1, dash2], { opacity: 0 }, { opacity: 1, duration: .3 }, 3.65)
    .from(dl, { opacity: 0, y: -14, duration: .35, ease: 'power3.out' }, 3.7)
    .fromTo(g3.p, { strokeDashoffset: g3.L }, { strokeDashoffset: 0, duration: .8, ease: 'power1.inOut', immediateRender: false }, 3.85);
}};

/* ───────────── G12 · ×10 ───────────── */
SCENES['G12-x10'] = { dur: 4, build(root, tl) {
  const TW = 720, TH = 460, ty = 230;
  const mk = (left, name, loss, delay, dir) => {
    const t = tableShell(root, TW, TH, { ow: .94, oh: .66 }); css(t.box, { left: left + 'px', top: ty + 'px' });
    const cards = [['K', '♠'], ['9', '♥'], ['4', '♣'], ['A', '♦']].map((c, i) => miniCard(t.box, c[0], c[1], 74, 106, t.cx - 164 + i * 88, t.cy - 82));
    const pot = el('div', 'chipstack', null, t.box); css(pot, { position: 'absolute', left: (t.cx - 60) + 'px', top: (t.cy + 56) + 'px', height: '60px' });
    for (let j = 0; j < 4; j++) { const c = el('div', 'pchip', null, pot); css(c, { top: (30 - j * 9) + 'px', width: '120px' }); }
    const lab = el('div', 'row', null, root);
    css(lab, { position: 'absolute', left: left + 'px', width: TW + 'px', top: '725px', justifyContent: 'center', gap: '26px', alignItems: 'baseline' });
    const nm = el('div', 'title', name, lab); css(nm, { fontSize: '60px' });
    const ls = el('div', 'num red', MINUS + '$0', lab); css(ls, { fontSize: '104px' });
    tl.from(t.box, { opacity: 0, x: dir * 70, duration: .55, ease: 'expo.out' }, 0)
      .from(cards, { opacity: 0, y: -30, duration: .35, stagger: .07, ease: 'power3.out' }, .45 + delay)
      .from(pot, { opacity: 0, y: -26, duration: .35, ease: 'power3.out' }, .85 + delay)
      .from(lab, { opacity: 0, y: 20, duration: .4, ease: 'power3.out' }, .6 + delay);
    countUp(tl, ls, 0, loss, .85 + delay, .7, { pre: MINUS + '$' });
  };
  mk(60, 'NL10', 10, 0, -1);
  mk(1920 - 60 - TW, 'NL100', 100, .15, 1);
  const x = el('div', 'num gold', '×10', root);
  css(x, { position: 'absolute', left: '780px', width: '360px', top: '390px', textAlign: 'center', fontSize: '150px', lineHeight: '1', textShadow: '0 10px 40px rgba(0,0,0,.5)' });
  const note = el('div', 'note', 'условный пример', root); css(note, { position: 'absolute', left: '0', right: '0', bottom: '44px', textAlign: 'center' });
  tl.from(x, { opacity: 0, scale: 1.6, duration: .45, ease: 'expo.out' }, 1.7)
    .from(note, { opacity: 0, duration: .4 }, .6);
}};

/* ───────────── G13 · рейкбек ───────────── */
SCENES['G13-rakeback'] = { dur: 5, build(root, tl) {
  useStyle(root, RC_CSS);
  // стол
  const t = tableShell(root, 600, 360); css(t.box, { left: '130px', top: '180px' });
  const pot = el('div', 'chipstack', null, t.box); css(pot, { position: 'absolute', left: (t.cx - 60) + 'px', top: (t.cy - 30) + 'px', height: '60px' });
  for (let j = 0; j < 4; j++) { const c = el('div', 'pchip', null, pot); css(c, { top: (30 - j * 9) + 'px' }); }
  // сейф
  const SX = 1400, SY = 360;
  const safe = el('div', 'abs', null, root); css(safe, { left: (SX - 150) + 'px', top: (SY - 150) + 'px', width: '300px', height: '300px' });
  safe.innerHTML = `<svg width="300" height="300" viewBox="0 0 300 300" style="overflow:visible;filter:drop-shadow(0 24px 30px rgba(0,0,0,.5))">
    <rect x="22" y="26" width="256" height="248" rx="30" fill="#1b2a24" stroke="#f2c14e" stroke-width="7"/>
    <rect x="46" y="50" width="208" height="200" rx="18" fill="#22352d" stroke="rgba(244,241,232,.25)" stroke-width="3"/>
    <circle cx="150" cy="150" r="62" fill="#0d1713" stroke="#f2c14e" stroke-width="7"/>
    <text x="150" y="172" text-anchor="middle" font-family="Unb" font-weight="900" font-size="60" fill="#f2c14e">$</text>
  </svg>`;
  const sl = el('div', 'body gold', 'комиссия рума', root);
  css(sl, { position: 'absolute', left: (SX - 300) + 'px', width: '600px', textAlign: 'center', top: '92px', fontSize: '56px', fontWeight: 800 });
  // игрок
  const PX = 430, PY = 730;
  const pl = el('div', 'abs', null, root); css(pl, { left: (PX - 70) + 'px', top: (PY - 70) + 'px', width: '140px', height: '140px' });
  pl.innerHTML = `<svg width="140" height="140" viewBox="0 0 140 140"><circle cx="70" cy="70" r="66" fill="#1b2823" stroke="#f4f1e8" stroke-width="6"/><circle cx="70" cy="54" r="22" fill="#f4f1e8"/><path d="M26 112c5-28 24-36 44-36s39 8 44 36c-12 12-27 18-44 18s-32-6-44-18z" fill="#f4f1e8"/></svg>`;
  const plab = el('div', 'body', 'игрок', root); css(plab, { position: 'absolute', left: (PX - 200) + 'px', width: '400px', textAlign: 'center', top: (PY + 84) + 'px', fontSize: '44px', fontWeight: 800 });
  // стрелки (svg)
  const svg = svgBox(root, 1920, 1080);
  const topA = { x0: 760, x1: 1200, y: SY };
  const topLine = sv('line', { x1: topA.x0, y1: topA.y, x2: topA.x1, y2: topA.y, stroke: 'rgba(244,241,232,.55)', 'stroke-width': 8, 'stroke-linecap': 'round', 'stroke-dasharray': '2 18' }, svg);
  const rp = 'M' + (SX) + ' ' + (SY + 175) + ' C' + SX + ' 730, 1000 770, ' + (PX + 100) + ' ' + (PY + 4);
  const ret = sv('path', { d: rp, fill: 'none', stroke: '#3ddc97', 'stroke-width': 8, 'stroke-linecap': 'round' }, svg);
  const RL = ret.getTotalLength();
  ret.style.strokeDasharray = RL + ' ' + (RL + 10); ret.style.strokeDashoffset = RL;
  const re = ret.getPointAtLength(RL), rb = ret.getPointAtLength(RL - 6);
  const rAng = Math.atan2(re.y - rb.y, re.x - rb.x) * 180 / Math.PI;
  const retHead = el('div', 'abs', '<svg width="50" height="50" viewBox="0 0 50 50"><polygon points="0,2 50,25 0,48" fill="#3ddc97"/></svg>', root);
  css(retHead, { left: (re.x - 50) + 'px', top: (re.y - 25) + 'px', transformOrigin: '100% 50%' });
  gsap.set(retHead, { rotation: rAng });
  const rl = el('div', 'title mint', 'рейкбек', root);
  css(rl, { position: 'absolute', left: '840px', width: '420px', textAlign: 'center', top: '600px', fontSize: '60px' });
  const cap = el('div', 'body', 'чем больше играешь — <span class="gold">тем больше возвращают</span>', root);
  css(cap, { position: 'absolute', left: '0', right: '0', textAlign: 'center', bottom: '84px', fontSize: '60px', fontWeight: 800 });
  // потоки фишек
  const flows = [];
  const mkChip = (cls) => el('div', 'rc ' + cls, null, root);
  for (let k = 0; k < 11; k++) flows.push({ el: mkChip(''), at: .7 + k * .24, dur: .85, path: 'top' });
  for (let k = 0; k < 3; k++) flows.push({ el: mkChip('m'), at: 1.95 + k * .26, dur: 1.0, path: 'ret' });
  for (let k = 0; k < 9; k++) flows.push({ el: mkChip('m'), at: 2.75 + k * .2, dur: 1.0, path: 'ret' });
  const TK = { v: 0 };
  tl.fromTo(TK, { v: 0 }, { v: 1, duration: .45, ease: 'power3.inOut', immediateRender: false }, 2.5);
  flows.forEach((f) => {
    const o = { p: 0 };
    tl.fromTo(o, { p: 0 }, { p: 1, duration: f.dur, ease: 'none', immediateRender: false, onUpdate: () => {
      const p = o.p;
      let x, y;
      if (f.path === 'top') { x = lerp(topA.x0 - 60, topA.x1 - 30, p); y = topA.y; }
      else { const q = ret.getPointAtLength(RL * p); x = q.x; y = q.y; }
      const op = p <= 0 || p >= 1 ? 0 : Math.min(1, p * 8, (1 - p) * 8);
      f.el.style.transform = 'translate(' + (x - 23) + 'px,' + (y - 23) + 'px) scale(' + (f.path === 'top' ? 1 : lerp(.8, 1, TK.v)) + ')';
      f.el.style.opacity = op;
    } }, f.at);
  });
  // появление
  tl.from(t.box, { opacity: 0, x: -50, duration: .5, ease: 'expo.out' }, 0)
    .from([safe, sl], { opacity: 0, x: 50, duration: .5, ease: 'expo.out', stagger: .05 }, .1)
    .from([pl, plab], { opacity: 0, y: 30, duration: .45, ease: 'power3.out', stagger: .05 }, .25)
    .from(topLine, { opacity: 0, duration: .3 }, .5)
    .fromTo(ret, { strokeDashoffset: RL }, { strokeDashoffset: 0, duration: .7, ease: 'power2.inOut', immediateRender: false }, 1.4)
    .from(retHead, { opacity: 0, duration: .2 }, 2.0)
    .from(rl, { opacity: 0, y: 16, duration: .4, ease: 'power3.out' }, 1.8)
    // 2,5 с — утолщение и подпись
    .fromTo(ret, { strokeWidth: 8 }, { strokeWidth: 24, duration: .45, ease: 'power3.inOut', immediateRender: false }, 2.5)
    .to(retHead, { scale: 1.7, duration: .45, ease: 'power3.inOut' }, 2.5)
    .to(rl, { scale: 1.15, duration: .45, ease: 'power3.inOut', transformOrigin: '50% 50%' }, 2.5)
    .from(cap, { opacity: 0, y: 30, duration: .5, ease: 'power3.out' }, 2.65);
}};

/* ───────────── G15 · живая игра ───────────── */
SCENES['G15-live'] = { dur: 4, build(root, tl) {
  const row = el('div', 'abs row', null, root);
  css(row, { left: '0', right: '0', top: '150px', justifyContent: 'center', alignItems: 'flex-start', gap: '34px' });
  const term = (txt, label, color) => {
    const b = el('div', 'col', null, row); css(b, { alignItems: 'center', gap: '14px', textAlign: 'center' });
    const n = el('div', 'num', txt, b); css(n, { fontSize: '120px', color, whiteSpace: 'nowrap' });
    const l = el('div', 'body', label, b); css(l, { fontSize: '46px', fontWeight: 800, color: color === 'var(--gold)' ? 'var(--gold)' : 'var(--paper)' });
    return { b, n, l };
  };
  const op = (txt) => { const o = el('div', 'num muted', txt, row); css(o, { fontSize: '96px', paddingTop: '14px' }); return o; };
  const a = term('0', 'раздач', 'var(--paper)');
  const o1 = op('÷');
  const b = term('~30', 'раздач в час вживую', 'var(--paper)');
  const o2 = op('=');
  const c = term('≈' + NB + '0', 'суток', 'var(--gold)');
  // календарь: 28 клеток (4 недели)
  const cell = 96, gp = 14, cols = 7, rows = 4;
  const calW = cols * cell + (cols - 1) * gp, calH = rows * cell + (rows - 1) * gp;
  const cal = el('div', 'abs', null, root); css(cal, { left: ((1920 - calW) / 2) + 'px', top: '500px', width: calW + 'px', height: calH + 'px' });
  const cells = [];
  for (let i = 0; i < 28; i++) {
    const e = el('div', null, null, cal);
    css(e, { position: 'absolute', left: ((i % cols) * (cell + gp)) + 'px', top: (Math.floor(i / cols) * (cell + gp)) + 'px', width: cell + 'px', height: cell + 'px', borderRadius: '22px', background: 'rgba(244,241,232,.07)', border: '1.5px solid rgba(244,241,232,.16)', overflow: 'hidden' });
    const f = el('div', null, null, e); css(f, { position: 'absolute', inset: '0', background: 'linear-gradient(160deg,#f7d77e,var(--gold))' });
    cells.push(f);
  }
  const note = el('div', 'note', 'оценка для живой игры', root); css(note, { position: 'absolute', left: '0', right: '0', bottom: '44px', textAlign: 'center' });
  tl.from([a.b], { opacity: 0, y: 40, duration: .45, ease: 'expo.out' }, 0)
    .from([o1, b.b], { opacity: 0, y: 40, duration: .45, ease: 'expo.out', stagger: .06 }, .4)
    .from([o2, c.b], { opacity: 0, y: 40, duration: .45, ease: 'expo.out', stagger: .06 }, .8)
    .from(cal, { opacity: 0, duration: .4 }, .8)
    .from(note, { opacity: 0, duration: .4 }, .6);
  countUp(tl, a.n, 0, 20000, .1, .9, { ease: 'power3.out' });
  const S = { v: 0 };
  tl.fromTo(S, { v: 0 }, { v: 30, duration: .6, ease: 'power3.out', immediateRender: false, onUpdate: () => { b.n.textContent = '~' + Math.round(S.v); } }, .5);
  countUp(tl, c.n, 0, 28, .9, 1.0, { pre: '≈' + NB, ease: 'power2.out' });
  cells.forEach((f, i) => {
    const tt = .9 + 1.0 * timeAt('power2.out', (i + 1) / 28);
    tl.fromTo(f, { opacity: 0, scale: .55 }, { opacity: 1, scale: 1, duration: .26, ease: 'expo.out' }, Math.max(.8, tt - .2));
  });
}};

/* ───────────── живые столы (вид сверху) ───────────── */
const SUITS = ['♠', '♥', '♦', '♣'], RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'];
const TABLE_CSS = `.tc{position:absolute;left:0;top:0;will-change:transform}
.tc .cb,.tc .cf{position:absolute;left:0;top:0;right:0;bottom:0;border-radius:7px;box-shadow:0 4px 10px rgba(0,0,0,.45)}
.tc .cb{background:repeating-linear-gradient(45deg,#8e1d2c 0 6px,#a52637 6px 12px);border:3px solid #fbfaf6}
.tc .cf{background:#fbfaf6;color:#111;font-family:Unb;font-weight:900;display:none}
.tc .cf.r{color:#d6263a}
.tc .cf b{position:absolute;font-weight:900}
.tstack{position:absolute;width:150px;text-align:center;font-family:Onest;font-weight:700;font-size:27px;line-height:30px;color:var(--paper);white-space:nowrap}
.tpot{position:absolute;width:200px;text-align:center;font-family:Unb;font-weight:700;font-size:32px;line-height:34px;color:var(--gold);white-space:nowrap}`;

function faceHtml(c, w, h) {
  const red = c.s === '♥' || c.s === '♦';
  return { cls: 'cf' + (red ? ' r' : ''),
    html: '<b style="left:' + Math.round(w * .1) + 'px;top:' + Math.round(h * .03) + 'px;font-size:' + Math.round(w * .44) + 'px;line-height:1.1">' + c.r + '</b><b style="right:' + Math.round(w * .09) + 'px;bottom:' + Math.round(h * .02) + 'px;font-size:' + Math.round(w * .52) + 'px;line-height:1.1">' + c.s + '</b>' };
}
function makeTc(parent, w, h, withBack) {
  const c = el('div', 'tc', null, parent); css(c, { width: w + 'px', height: h + 'px', opacity: 0 });
  const b = withBack ? el('div', 'cb', null, c) : null;
  const f = el('div', 'cf', null, c);
  f.style.fontSize = '10px';
  return { c, b, f, key: '' };
}
function setFace(o, card, w, h) {
  const k = card.r + card.s;
  if (o.key === k) return;
  o.key = k;
  const fh = faceHtml(card, w, h);
  o.f.className = fh.cls; o.f.innerHTML = fh.html;
}

function genHand(tb, h) {
  if (tb.hands[h]) return tb.hands[h];
  const prev = h > 0 ? genHand(tb, h - 1) : null;
  const r = rng(tb.seed * 7919 + h * 104729 + 11);
  let stacks0;
  if (prev) stacks0 = prev.stacksEnd.slice();
  else { const r0 = rng(tb.seed * 31 + 7); stacks0 = []; for (let i = 0; i < 6; i++) stacks0.push(Math.round((7 + r0() * 9) * 20) / 20); }
  const d = (h + tb.seed) % 6, order = [];
  for (let j = 0; j < 6; j++) order.push((d + 3 + j) % 6);
  const ra = 1 + Math.floor(r() * 3), ca = ra + 1 + Math.floor(r() * (5 - ra));
  const A = order[ra], B = order[ca];
  const q = (x) => Math.round(x * 20) / 20;
  const bets = [q(.3 + r() * .5), q(.6 + r() * 1.2)];
  const actions = [], dealT = [], foldT = [];
  for (let j = 0; j < 6; j++) dealT[order[j]] = .04 + j * .06;
  let c = .45;
  for (let p = 0; p < 6; p++) {
    if (p === ra) { actions.push({ seat: A, type: 'bet', amt: bets[0], t0: c, dur: .26 }); c += .26; }
    else if (p === ca) { actions.push({ seat: B, type: 'call', amt: bets[0], t0: c, dur: .26 }); c += .26; }
    else { actions.push({ seat: order[p], type: 'fold', amt: 0, t0: c, dur: .16 }); foldT[order[p]] = c + .16; c += .16; }
  }
  const tFlop = c + .04;
  actions.push({ seat: A, type: 'bet', amt: bets[1], t0: tFlop + .3, dur: .24 });
  actions.push({ seat: B, type: 'call', amt: bets[1], t0: tFlop + .54, dur: .24 });
  const boardT = [tFlop, tFlop + .07, tFlop + .14, tFlop + .84, tFlop + 1.14];
  const tS = tFlop + 1.55;
  const pot = Math.round((bets[0] + bets[1]) * 2 * 100) / 100;
  const winner = r() < .5 ? A : B;
  // карты: 9 уникальных
  const used = {}, draw = () => { let k; do { k = Math.floor(r() * 52); } while (used[k]); used[k] = 1; return { r: RANKS[k % 13], s: SUITS[Math.floor(k / 13)] }; };
  const board = [draw(), draw(), draw(), draw(), draw()];
  const holes = {}; holes[A] = [draw(), draw()]; holes[B] = [draw(), draw()];
  for (let s = 0; s < 6; s++) if (!holes[s]) holes[s] = [draw(), draw()];
  const pay = [pot / 3, pot / 3, pot / 3].map(v => Math.floor(v * 100) / 100);
  pay[2] = Math.round((pot - pay[0] - pay[1]) * 100) / 100;
  const contrib = (bets[0] + bets[1]);
  const stacksEnd = stacks0.slice();
  stacksEnd[A] -= contrib; stacksEnd[B] -= contrib; stacksEnd[winner] += pot;
  for (let i = 0; i < 6; i++) stacksEnd[i] = Math.round(stacksEnd[i] * 100) / 100;
  const tC = tS + .9;
  const H = { stacks0, stacksEnd, order, A, B, actions, dealT, foldT, boardT, tFlop, tS, tC, T0: tC + .22, pot, winner, board, holes, pay };
  tb.hands[h] = H;
  return H;
}

function buildLiveTable(parent, cfg) {
  const W = 900, Hh = 500, cx = 450, cy = 250, rx = 310, ry = 140;
  const cell = el('div', 'abs', null, parent); css(cell, { left: cfg.x + 'px', top: cfg.y + 'px', width: W + 'px', height: Hh + 'px' });
  const sh = tableShell(cell, W, Hh, { ow: 620 / W, oh: 280 / Hh, bw: 16 });
  const box = sh.box;
  const tb = { seed: cfg.seed, hands: [], cell };
  const k = cfg.k, phase = cfg.phase;
  const T = genHand(tb, 0).T0 * k;
  const angs = [0, 60, 120, 180, 240, 300];
  const seatXY = angs.map(a => { const r = a * Math.PI / 180; return { x: cx + (rx + 34) * Math.cos(r), y: cy + (ry + 34) * Math.sin(r) }; });
  const holeXY = seatXY.map(p => ({ x: p.x + (cx - p.x) * .40, y: p.y + (cy - p.y) * .40 }));
  const deck = { x: cx, y: cy - 118 };
  // места
  const seats = seatXY.map((p) => {
    const disc = seatDisc(box, p.x, p.y, 64, { bw: 4, bc: 'rgba(244,241,232,.45)' });
    const tx = el('div', 'tstack', '', box); css(tx, { left: (p.x - 75) + 'px', top: (p.y < cy - 20 ? p.y - 66 : p.y + 36) + 'px' });
    return { disc, tx };
  });
  // банк
  const potTx = el('div', 'tpot', '', box); css(potTx, { left: (cx - 100) + 'px', top: (cy - 88 - 34) + 'px' });
  const potChips = [[-20, 8], [0, 14], [20, 8]].map(([dx, dy], i) => {
    const c = el('div', 'rc', null, box); css(c, { width: '34px', height: '34px', borderWidth: '3px', opacity: 0, transform: 'translate(' + (cx + dx - 17) + 'px,' + (cy + 60 + dy - 17) + 'px)' });
    return c;
  });
  // карты
  const hole = [];
  for (let s = 0; s < 6; s++) hole.push([makeTc(box, 40, 56, true), makeTc(box, 40, 56, true)]);
  const board = [];
  for (let i = 0; i < 5; i++) board.push(makeTc(box, 58, 82, false));
  const boardXY = (i) => ({ x: cx + (i - 2) * 66, y: cy + 2 });
  // летящие фишки
  const chips = [];
  for (let i = 0; i < 12; i++) { const c = el('div', 'rc', null, box); css(c, { width: '34px', height: '34px', borderWidth: '3px' }); if (i >= 6) c.className = 'rc g'; chips.push(c); }
  // кольцо-таймер
  const svg = svgBox(box, W, Hh);
  const RR = 43, CIRC = 2 * Math.PI * RR;
  const ring = sv('circle', { cx: 0, cy: 0, r: RR, fill: 'none', 'stroke-width': 7, 'stroke-linecap': 'butt' }, svg);
  ring.style.strokeDasharray = CIRC + ' ' + CIRC; ring.style.opacity = 0;
  const gold = [242, 193, 78], red = [255, 90, 78];
  const mix = (p) => { const m = clamp((p - .5) / .35); return 'rgb(' + Math.round(lerp(gold[0], red[0], m)) + ',' + Math.round(lerp(gold[1], red[1], m)) + ',' + Math.round(lerp(gold[2], red[2], m)) + ')'; };
  let lastHand = -1;
  tb.render = (t) => {
    const tt = t + phase, h = Math.floor(tt / T), u = (tt - h * T) / k;
    const H = genHand(tb, h);
    if (h !== lastHand) {
      lastHand = h;
      for (let s = 0; s < 6; s++) { setFace(hole[s][0], H.holes[s][0], 40, 56); setFace(hole[s][1], H.holes[s][1], 40, 56); }
      for (let i = 0; i < 5; i++) setFace(board[i], H.board[i], 58, 82);
    }
    const cleanup = 1 - clamp((u - H.tC) / .2);
    // стеки и банк
    const stk = H.stacks0.slice();
    let pot = 0;
    const chipEv = [];
    H.actions.forEach((a) => {
      const td = a.t0 + a.dur;
      if (a.amt > 0) {
        if (u >= td) stk[a.seat] -= a.amt;
        const land = td + .26;
        if (u >= land) pot += a.amt;
        chipEv.push({ t0: td, dur: .26, from: seatXY[a.seat], to: { x: cx, y: cy + 62 }, g: 0 });
      }
    });
    // выплата
    const pay0 = H.tS + .32;
    for (let j = 0; j < 3; j++) {
      const ts = pay0 + j * .07;
      if (u >= ts) pot -= H.pay[j];
      if (u >= ts + .3) stk[H.winner] += H.pay[j];
      chipEv.push({ t0: ts, dur: .3, from: { x: cx, y: cy + 62 }, to: seatXY[H.winner], g: 1 });
    }
    pot = Math.max(0, Math.round(pot * 100) / 100);
    for (let s = 0; s < 6; s++) {
      const folded = H.foldT[s] != null && u >= H.foldT[s];
      const op = (folded ? .4 : 1) * (u < .2 ? lerp(.4, 1, u / .2) : 1) * (h > 0 && u < .2 && !folded ? 1 : 1);
      seats[s].disc.style.opacity = folded ? .4 : 1;
      seats[s].tx.style.opacity = folded ? .4 : 1;
      seats[s].tx.textContent = money2(Math.max(0, stk[s]));
    }
    potTx.textContent = money2(pot);
    potTx.style.opacity = pot > 0 ? 1 : 0;
    potChips.forEach((c, i) => { c.style.opacity = pot > [0, 1.2, 2.6][i] ? 1 : 0; });
    // карты на руках
    const flipA = clamp((u - H.tS) / .3);
    for (let s = 0; s < 6; s++) {
      const td = H.dealT[s], p = (u - td) / .3, ft = H.foldT[s] == null ? 1e9 : H.foldT[s];
      const contender = s === H.A || s === H.B;
      for (let c = 0; c < 2; c++) {
        const o = hole[s][c];
        if (p <= 0) { o.c.style.opacity = 0; continue; }
        const qf = ez('power3.out')(clamp(p));
        const hp = holeXY[s], ox = (c ? 13 : -13), rot = (c ? 7 : -7);
        let x = lerp(deck.x, hp.x + ox, qf), y = lerp(deck.y, hp.y, qf), op = clamp(p * 5), sc = 1;
        const fo = clamp((u - ft) / .22);
        if (fo > 0) { x += (cx - hp.x) * .3 * fo; y += (cy - hp.y) * .3 * fo; op *= 1 - fo; sc = 1 - .15 * fo; }
        op *= cleanup;
        let sx = 1, face = false;
        if (contender && flipA > 0) { sx = Math.abs(1 - 2 * flipA); face = flipA > .5; }
        o.c.style.opacity = op;
        o.c.style.transform = 'translate(' + (x - 20) + 'px,' + (y - 28) + 'px) rotate(' + lerp(0, rot, qf) + 'deg) scale(' + (sx * sc) + ',' + sc + ')';
        o.b.style.display = face ? 'none' : 'block';
        o.f.style.display = face ? 'block' : 'none';
      }
    }
    // общие карты
    for (let i = 0; i < 5; i++) {
      const o = board[i], p = (u - H.boardT[i]) / .3;
      if (p <= 0) { o.c.style.opacity = 0; continue; }
      const qf = ez('power3.out')(clamp(p)), bp = boardXY(i);
      o.c.style.opacity = clamp(p * 6) * cleanup;
      o.f.style.display = 'block';
      o.c.style.transform = 'translate(' + (lerp(deck.x, bp.x, qf) - 29) + 'px,' + (lerp(deck.y, bp.y, qf) - 41) + 'px) scale(' + clamp(ez('power2.out')(clamp((p - .1) / .7))) + ',1)';
    }
    // летящие фишки
    for (let i = 0; i < chips.length; i++) {
      const e = chipEv[i], c = chips[i];
      if (!e || u < e.t0 || u > e.t0 + e.dur) { c.style.opacity = 0; continue; }
      const p = (u - e.t0) / e.dur, qf = ez('power2.inOut')(p);
      c.style.opacity = Math.min(1, p * 8, (1 - p) * 6 + .2 > 1 ? 1 : (1 - p) * 6 + .2);
      c.style.transform = 'translate(' + (lerp(e.from.x, e.to.x, qf) - 17) + 'px,' + (lerp(e.from.y, e.to.y, qf) - 17 - Math.sin(p * Math.PI) * 18) + 'px)';
    }
    // кольцо-таймер
    let act = null;
    for (let i = 0; i < H.actions.length; i++) { const a = H.actions[i]; if (u >= a.t0 && u < a.t0 + a.dur) { act = a; break; } }
    if (act) {
      const p = (u - act.t0) / act.dur, sp = seatXY[act.seat];
      ring.style.opacity = 1;
      ring.setAttribute('transform', 'translate(' + sp.x + ' ' + sp.y + ') rotate(-90)');
      ring.style.strokeDashoffset = CIRC * p;
      ring.style.stroke = mix(p);
    } else ring.style.opacity = 0;
  };
  tb.render(0);
  return tb;
}

/* ───────────── G14a · 4 стола ───────────── */
SCENES['G14a-4tables'] = { dur: 8, build(root, tl, P, dur) {
  useStyle(root, RC_CSS + TABLE_CSS);
  const cfgs = [
    { x: 48, y: 28, seed: 3, k: 1.0, phase: 1.7 },
    { x: 972, y: 28, seed: 7, k: .93, phase: .35 },
    { x: 48, y: 552, seed: 11, k: 1.06, phase: 2.9 },
    { x: 972, y: 552, seed: 19, k: .98, phase: .95 },
  ];
  const tables = cfgs.map(c => buildLiveTable(root, c));
  const frame = el('div', 'abs', null, root);
  css(frame, { left: '36px', top: '16px', width: '924px', height: '524px', border: '7px solid var(--gold)', borderRadius: '36px', boxShadow: '0 0 44px rgba(242,193,78,.35)' });
  tl.from(tables.map(t => t.cell), { opacity: .25, scale: 1.02, duration: .4, ease: 'power2.out' }, 0)
    .from(frame, { opacity: 0, scale: 1.04, duration: .35, ease: 'expo.out' }, 1.0);
  const D = { t: 0 };
  tl.fromTo(D, { t: 0 }, { t: dur, duration: dur, ease: 'none', immediateRender: false, onUpdate: () => tables.forEach(tb => tb.render(D.t)) }, 0);
}};

/* ───────────── G14b · 18 столов ───────────── */
SCENES['G14b-18tables'] = { dur: 3, build(root, tl) {
  const COLS = 6, ROWS = 3, CW = 280, CH = 280, GX = 10, GY = 12;
  const gx0 = (1920 - (COLS * CW + (COLS - 1) * GX)) / 2, gy0 = 80;
  const r = rng(18);
  const order = []; for (let i = 0; i < 18; i++) order.push(i);
  for (let i = 17; i > 0; i--) { const j = Math.floor(r() * (i + 1)); const t = order[i]; order[i] = order[j]; order[j] = t; }
  const cells = [];
  for (let i = 0; i < 18; i++) {
    const t = tableShell(root, CW, CH, { ow: .72, oh: .58, bw: 8 });
    css(t.box, { left: (gx0 + (i % COLS) * (CW + GX)) + 'px', top: (gy0 + Math.floor(i / COLS) * (CH + GY)) + 'px' });
    [0, 60, 120, 180, 240, 300].forEach(a => { const p = t.seat(a, 14); seatDisc(t.box, p.x, p.y, 30, { bw: 3 }); });
    for (let c = 0; c < 5; c++) {
      const su = SUITS[(i + c * 3) % 4];
      const cd = el('div', null, su, t.box);
      css(cd, { position: 'absolute', left: (t.cx - 66 + c * 29 - 12) + 'px', top: (t.cy - 18) + 'px', width: '25px', height: '36px', borderRadius: '4px', background: '#fbfaf6', boxShadow: '0 2px 6px rgba(0,0,0,.4)', color: (su === '♥' || su === '♦') ? '#d6263a' : '#111', fontSize: '20px', lineHeight: '36px', textAlign: 'center', fontFamily: 'Unb', fontWeight: 900 });
    }
    cells.push(t.box);
  }
  order.forEach((idx, rank) => {
    tl.from(cells[idx], { opacity: 0, y: -70, scale: .85, duration: .32, ease: 'expo.out' }, .02 + rank * .066);
  });
  const scrim = el('div', 'abs', null, root); css(scrim, { left: '0', top: '0', width: '1920px', height: '1080px', background: 'rgba(7,16,13,.62)' });
  const x = el('div', 'num gold', '×18', root);
  css(x, { position: 'absolute', left: '0', width: '1920px', top: '270px', textAlign: 'center', fontSize: '480px', lineHeight: '1', textShadow: '0 16px 60px rgba(0,0,0,.6)' });
  const note = el('div', 'note', 'иллюстрация масштаба', root); css(note, { position: 'absolute', left: '0', right: '0', bottom: '30px', textAlign: 'center' });
  tl.from(scrim, { opacity: 0, duration: .4, ease: 'power2.out' }, 1.4)
    .from(x, { opacity: 0, scale: 1.4, duration: .5, ease: 'expo.out' }, 1.45)
    .from(note, { opacity: 0, duration: .4 }, .6);
}};


})();
