/* scenes-3.js — сцены для ролика про брови на кону (инфографика, исполнитель B).
   G00 · G16–G22 · G24 · G26 · G29 · G30 · G32–G39. Всё в IIFE, чтобы не конфликтовать с другими файлами сцен. */
(function () {
'use strict';

const PAPER = '#f4f1e8', GOLD = '#f2c14e', RED = '#ff5a4e', MINT = '#3ddc97', MUTED = '#8ea398', INK = '#0f1412';
const NB = ' ';

// Блок с инлайн-стилями.
const mk = (parent, cls, html, st) => { const e = el('div', cls, html, parent); if (st) css(e, st); return e; };
// Контейнер на весь кадр, центрирует содержимое по вертикали и горизонтали.
const fillBox = (root, st) => mk(root, null, null, Object.assign({ position: 'absolute', left: '0px', top: '0px', width: '1920px', height: '1080px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }, st || {}));
const sty = (root, text) => el('style', null, text, root);

// Общие локальные стили (фишки сверху, стол).
const SHARED_CSS = `
.b-chip { position: absolute; width: 60px; height: 60px; border-radius: 50%; background: repeating-conic-gradient(#f6f1e3 0 14deg, var(--red) 14deg 45deg); box-shadow: 0 5px 8px rgba(0,0,0,.45); }
.b-chip::after { content: ''; position: absolute; inset: 9px; border-radius: 50%; background: var(--red); border: 3px solid rgba(255,255,255,.78); box-shadow: inset 0 0 0 4px rgba(0,0,0,.08); }
.b-rail { position: absolute; border-radius: 50%; background: linear-gradient(180deg, #2a211a 0%, #120d0a 100%); box-shadow: 0 40px 90px rgba(0,0,0,.6), 0 0 0 2px rgba(255,255,255,.06) inset; }
.b-felt { position: absolute; border-radius: 50%; background: radial-gradient(ellipse at 50% 42%, #1f8061 0%, #146048 48%, #0d4535 100%); box-shadow: inset 0 0 0 3px rgba(0,0,0,.35), inset 0 18px 80px rgba(0,0,0,.5); }
`;

// Стол сверху: рельса + сукно + линия ставок. Возвращает контейнер (для анимации появления).
function tableOval(parent, cx, cy, w, h) {
  const t = mk(parent, null, null, { position: 'absolute', left: (cx - w / 2) + 'px', top: (cy - h / 2) + 'px', width: w + 'px', height: h + 'px' });
  const rail = mk(t, 'b-rail', null, { inset: '0px' });
  const felt = mk(t, 'b-felt', null, { inset: '28px' });
  return t;
}
// Фишка сверху (top-down).
const chipEl = (parent, x, y, size, kind) => {
  const c = mk(parent, 'b-chip' + (kind ? ' ' + kind : ''), null, { left: (x - size / 2) + 'px', top: (y - size / 2) + 'px', width: size + 'px', height: size + 'px' });
  return c;
};

/* ───────────────────────── G16 · марафон: три счётчика ───────────────────────── */
SCENES['G16-marathon'] = { dur: 6, build(root, tl) {
  mk(root, 'cap', 'Марафон с <span class="gold">$50</span> · 23 апреля – 14 июля', { position: 'absolute', left: '0px', width: '1920px', top: '104px', textAlign: 'center', fontSize: '32px' });
  const wrap = fillBox(root, { paddingTop: '70px' });
  const col = mk(wrap, null, null, { width: '1500px', display: 'flex', flexDirection: 'column' });
  const rows = [
    { n: 11, tail: 'недель', at: .2, d: .9 },
    { n: 336, tail: 'часов', at: 1.6, d: 1.0 },
    { n: 316000, tail: 'раздач', at: 3.0, d: 1.25 },
  ];
  rows.forEach((r, i) => {
    const row = mk(col, null, null, { display: 'grid', gridTemplateColumns: '900px 1fr', columnGap: '52px', alignItems: 'baseline', padding: '16px 0 26px' });
    const num = mk(row, 'num', '', { fontSize: '176px', textAlign: 'right', color: GOLD });
    mk(row, 'title', r.tail, { fontSize: '88px', color: PAPER });
    tl.from(row, { opacity: 0, y: 44, duration: .45, ease: 'expo.out' }, r.at);
    countUp(tl, num, 0, r.n, r.at + .05, r.d);
    // Активный счётчик золотой, предыдущие «остывают» до бумажного цвета.
    if (i < 2) tl.to(num, { color: PAPER, duration: .6, ease: 'power2.inOut' }, r.at + r.d + .35);
  });
}};

/* ───────────────────────── G17 · прибыль марафона. ?stage=2 — плюс чип про EV ───────────────────────── */
SCENES['G17-profit'] = { dur: 6, build(root, tl, P) {
  const st2 = P.stage === '2';
  const col = mk(root, null, null, { position: 'absolute', left: '210px', top: '270px', width: '1500px', display: 'flex', flexDirection: 'column', gap: '96px' });
  const head = mk(col, null, null, { display: 'flex', alignItems: 'baseline', gap: '38px' });
  const total = mk(head, 'num', '', { fontSize: '196px' });
  mk(head, 'body', 'за марафон', { fontSize: '68px', color: MUTED });
  const bar = mk(col, null, null, { display: 'flex', gap: '10px', height: '176px' });
  const gseg = mk(bar, null, null, { width: '760px', borderRadius: '30px 10px 10px 30px', background: 'linear-gradient(90deg, #f7d77e, #f2c14e)', display: 'flex', flexDirection: 'column', justifyContent: 'center', padding: '0 0 0 48px', gap: '2px', color: INK, boxShadow: '0 0 0 rgba(242,193,78,0), 0 10px 20px rgba(0,0,0,0)' });
  mk(gseg, 'num', '$' + fmt(1800), { fontSize: '88px', color: INK });
  mk(gseg, 'body', 'джекпоты', { fontSize: '42px', fontWeight: 800, color: 'rgba(15,20,18,.78)' });
  const rseg = mk(bar, null, null, { width: '730px', borderRadius: '10px 30px 30px 10px', background: 'rgba(244,241,232,.10)', border: '1.5px solid rgba(244,241,232,.16)', display: 'flex', alignItems: 'center', justifyContent: 'center' });
  mk(rseg, 'body', 'рейкбек, гонки и игра', { fontSize: '48px', color: PAPER });

  const LIFT = { y: -46, boxShadow: '0 0 80px rgba(242,193,78,.55), 0 38px 60px rgba(0,0,0,.5)' };
  if (!st2) {
    tl.from(head, { opacity: 0, y: 40, duration: .45, ease: 'expo.out' }, 0);
    countUp(tl, total, 0, 3500, .1, .9, { pre: '$' });
    tl.fromTo(gseg, { clipPath: 'inset(0% 100% 0% 0%)' }, { clipPath: 'inset(0% 0% 0% 0%)', duration: .6, ease: 'power3.out' }, .35)
      .fromTo(rseg, { clipPath: 'inset(0% 100% 0% 0%)' }, { clipPath: 'inset(0% 0% 0% 0%)', duration: .6, ease: 'power3.out' }, .5)
      .set([gseg, rseg], { clipPath: 'none' }, 1.15)
      .to(gseg, Object.assign({ duration: .5, ease: 'expo.out' }, LIFT), 1.2)
      .to(rseg, { opacity: .5, duration: .5, ease: 'power2.out' }, 1.2)
      .to(gseg, { boxShadow: '0 0 110px rgba(242,193,78,.7), 0 38px 60px rgba(0,0,0,.5)', duration: 1.8, ease: 'sine.inOut', yoyo: true, repeat: 1 }, 1.8);
  } else {
    // Старт с финального кадра stage 1 — для стыка монтажом.
    total.textContent = '$' + fmt(3500);
    css(gseg, { transform: 'translateY(-46px)', boxShadow: LIFT.boxShadow }); css(rseg, { opacity: .5 });
    tl.to(gseg, { boxShadow: '0 0 110px rgba(242,193,78,.7), 0 38px 60px rgba(0,0,0,.5)', duration: 1.8, ease: 'sine.inOut', yoyo: true, repeat: 1 }, 0);
    const chip = mk(root, 'chip', 'за столами по EV: <span class="num red" style="font-size:64px;margin-left:6px">−$2' + NB + '381</span>', { position: 'absolute', left: '960px', top: '820px', marginLeft: '-430px', width: '860px', justifyContent: 'center', padding: '20px 40px', fontSize: '46px', background: 'rgba(255,90,78,.12)', border: '2.5px solid var(--red)' });
    const note = mk(root, 'body muted', 'по его же статистике', { position: 'absolute', left: '0px', width: '1920px', textAlign: 'center', top: '935px', fontSize: '42px' });
    tl.from(chip, { opacity: 0, y: 30, scale: .96, duration: .45, ease: 'expo.out' }, 1.0)
      .from(note, { opacity: 0, y: 16, duration: .4, ease: 'power3.out' }, 1.55);
  }
}};

/* ───────────────────────── G18 · $3 500 ÷ 336 ч ≈ $10 / час ───────────────────────── */
SCENES['G18-hour'] = { dur: 4, build(root, tl) {
  const wrap = fillBox(root, { gap: '50px', paddingBottom: '40px' });
  const row = mk(wrap, null, null, { display: 'flex', alignItems: 'center', gap: '64px' });
  const t1 = mk(row, 'num', '$' + fmt(3500), { fontSize: '160px' });
  const op = mk(row, 'num gold', '÷', { fontSize: '140px' });
  const t2 = mk(row, 'num', '336' + NB + 'ч', { fontSize: '160px' });
  const res = mk(wrap, null, null, { display: 'flex', alignItems: 'baseline', gap: '34px' });
  mk(res, 'num gold', '≈', { fontSize: '150px' });
  mk(res, 'num gold', '$10', { fontSize: '250px' });
  mk(res, 'num gold', '/' + NB + 'час', { fontSize: '120px' });
  const note = mk(root, 'note', 'так округляет сам Максим', { position: 'absolute', left: '0px', width: '1920px', textAlign: 'center', bottom: '110px', fontSize: '32px' });
  tl.from(t1, { opacity: 0, y: 44, duration: .45, ease: 'expo.out' }, .1)
    .from(t2, { opacity: 0, y: 44, duration: .45, ease: 'expo.out' }, .5)
    .from(op, { opacity: 0, scale: .4, duration: .4, ease: 'expo.out' }, 1.0)
    .from(res, { opacity: 0, y: 40, scale: .94, duration: .5, ease: 'expo.out' }, 1.5)
    .from(note, { opacity: 0, y: 14, duration: .4, ease: 'power3.out' }, 2.3);
}};

/* ───────────────────────── G19 · темп: нужно vs марафон, ×4,4 (1386 / 318 = 4,36) ───────────────────────── */
SCENES['G19-pace'] = { dur: 5, build(root, tl) {
  const X0 = 210, WMAX = 1500, BH = 120;
  const wShort = Math.round(WMAX * 318 / 1386);
  const col = mk(root, null, null, { position: 'absolute', left: X0 + 'px', top: '285px', width: WMAX + 'px', display: 'flex', flexDirection: 'column', gap: '96px' });
  const mkRow = (who, val, color, grad) => {
    const r = mk(col, null, null, { display: 'flex', flexDirection: 'column', gap: '20px', position: 'relative' });
    const lab = mk(r, null, null, { display: 'flex', alignItems: 'baseline', gap: '18px' });
    mk(lab, 'body muted', who, { fontSize: '48px' });
    mk(lab, 'title', '≈ ' + val, { fontSize: '64px', color });
    mk(lab, 'body', 'в неделю', { fontSize: '48px', color: PAPER });
    const bar = mk(r, 'bar', null, { height: BH + 'px', borderRadius: '22px' });
    const fill = mk(bar, 'fill', null, { borderRadius: '20px', background: grad });
    return { r, lab, bar, fill };
  };
  const a = mkRow('нужно:', '$' + fmt(1386), RED, 'linear-gradient(90deg, #ff8a7e, #ff5a4e)');
  const b = mkRow('марафон:', '$318', GOLD, 'linear-gradient(90deg, #f7d77e, #f2c14e)');
  // Зазор: пунктирная рамка от конца короткой полосы до длинной + ×4,4
  const gap = mk(b.bar, null, null, { position: 'absolute', left: (wShort + 14) + 'px', right: '10px', top: '0px', bottom: '0px', display: 'flex', alignItems: 'center', justifyContent: 'center' });
  const mult = mk(gap, 'num', '×4,4', { fontSize: '124px', letterSpacing: '-.02em' });
  tl.from(a.r, { opacity: 0, y: 30, duration: .4, ease: 'expo.out' }, .1)
    .fromTo(a.fill, { width: '0px' }, { width: WMAX + 'px', duration: .9, ease: 'power3.out' }, .2)
    .from(b.r, { opacity: 0, y: 30, duration: .4, ease: 'expo.out' }, .75)
    .fromTo(b.fill, { width: '0px' }, { width: wShort + 'px', duration: .7, ease: 'power3.out' }, .85)
    .from(mult, { opacity: 0, scale: .7, duration: .5, ease: 'expo.out' }, 1.8);
}};

/* ───────────────────────── G20 · правило банкролла. ?v=40 — «100 БИ» сминается в «40 БИ» ───────────────────────── */
SCENES['G20-bi'] = { dur: 5, build(root, tl, P) {
  const v40 = P.v === '40';
  const cap = mk(root, 'cap', 'Правило банкролла', { position: 'absolute', left: '0px', width: '1920px', top: '104px', textAlign: 'center', fontSize: '32px' });
  tl.from(cap, { opacity: 0, y: 14, duration: .35, ease: 'power3.out' }, 0);

  // Большое число + «на лимит». Две копии числа в одной ячейке (для замены без скачков).
  const big = mk(root, null, null, { position: 'absolute', left: '0px', width: '1920px', top: v40 ? '270px' : '170px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px' });
  const cell = mk(big, null, null, { display: 'grid', justifyItems: 'center', alignItems: 'center' });
  const A = mk(cell, 'num gold', '100' + NB + 'БИ', { fontSize: '236px', gridArea: '1 / 1', transformOrigin: '50% 100%' });
  mk(big, 'body', 'на лимит', { fontSize: '72px', color: MUTED });

  if (!v40) {
    const panel = mk(root, null, null, { position: 'absolute', left: '0px', top: '560px', width: '1920px', display: 'flex', flexDirection: 'column', gap: '18px', alignItems: 'center' });
    const l1 = mk(panel, 'title', 'NL25 × 100' + NB + 'БИ = <span class="gold">$' + fmt(2500) + '</span>', { fontSize: '56px' });
    const l2 = mk(panel, 'title', '<span class="muted" style="font-family:Onest;font-weight:600">у Максима:</span> <span class="red">$200</span>', { fontSize: '56px' });
    const arrow = mk(root, null, `<svg width="80" height="96" viewBox="0 0 80 96"><path d="M40 4V76" stroke="#f2c14e" stroke-width="10" stroke-linecap="round" fill="none" class="sh"/><path d="M14 54L40 82L66 54" stroke="#f2c14e" stroke-width="10" stroke-linecap="round" stroke-linejoin="round" fill="none" class="hd"/></svg>`, { position: 'absolute', left: '920px', top: '770px' });
    const pill = mk(root, 'title gold', 'старт с NL2', { position: 'absolute', left: '0px', width: '1920px', textAlign: 'center', top: '880px', fontSize: '76px', whiteSpace: 'nowrap' });
    tl.from(A, { opacity: 0, y: 50, scale: .94, duration: .5, ease: 'expo.out' }, .1)
      .from(big.lastChild, { opacity: 0, y: 20, duration: .4, ease: 'power3.out' }, .35)
      .from(panel, { opacity: 0, y: 30, duration: .4, ease: 'expo.out' }, .8)
      .from(l1, { opacity: 0, y: 14, duration: .35, ease: 'power3.out' }, .85)
      .from(l2, { opacity: 0, y: 14, duration: .35, ease: 'power3.out' }, 1.5)
      .from(arrow.querySelector('.sh'), { opacity: 0, duration: .01 }, 2.2)
      .fromTo(arrow, { clipPath: 'inset(0% 0% 100% 0%)' }, { clipPath: 'inset(0% 0% 0% 0%)', duration: .4, ease: 'power3.out' }, 2.2)
      .from(pill, { opacity: 0, y: 24, scale: .94, duration: .45, ease: 'expo.out' }, 2.65);
  } else {
    const B = mk(cell, 'num red', '40' + NB + 'БИ', { fontSize: '236px', gridArea: '1 / 1', opacity: 0, transformOrigin: '50% 100%' });
    const sub = mk(root, 'title', 'через <span class="gold">27' + NB + 'часов</span>', { position: 'absolute', left: '0px', width: '1920px', textAlign: 'center', top: '650px', fontSize: '76px' });
    const hint = mk(root, 'body muted', 'после 10' + NB + 'БИ на лимит <span style="font-family:\'Noto Color Emoji\'">😉</span>', { position: 'absolute', left: '0px', width: '1920px', textAlign: 'center', top: '860px', fontSize: '38px' });
    tl.from(A, { opacity: 0, y: 50, scale: .94, duration: .5, ease: 'expo.out' }, .1)
      .from(big.lastChild, { opacity: 0, y: 20, duration: .4, ease: 'power3.out' }, .35)
      // быстрое сжатие → замена числа → восстановление
      .to(A, { scaleY: .5, scaleX: 1.07, duration: .17, ease: 'power3.in' }, 1.3)
      .set(A, { opacity: 0 }, 1.47)
      .set(B, { opacity: 1 }, 1.47)
      .fromTo(B, { scaleY: .55, scaleX: 1.1 }, { scaleY: 1, scaleX: 1, duration: .4, ease: 'expo.out' }, 1.47)
      .from(sub, { opacity: 0, y: 24, duration: .45, ease: 'expo.out' }, 1.75)
      .from(hint, { opacity: 0, y: 12, duration: .4, ease: 'power3.out' }, 3.3)
      // мягкое мигание подсказки (две плавные пульсации)
      .to(hint, { opacity: .35, duration: .45, ease: 'sine.inOut', yoyo: true, repeat: 3 }, 3.9);
  }
}};

/* ───────────────────────── G21 · $100 распадается на $80 и $20 ───────────────────────── */
SCENES['G21-split'] = { dur: 4, build(root, tl) {
  const W = 1500, H = 170, X0 = 210, TOP = 420;
  const w80 = W * .8, w20 = W * .2;
  const R = 34;
  const base = { position: 'absolute', top: TOP + 'px', height: H + 'px', display: 'flex', alignItems: 'center', justifyContent: 'center' };
  const L = mk(root, null, null, Object.assign({}, base, { left: X0 + 'px', width: w80 + 'px', background: PAPER, borderRadius: R + 'px 0 0 ' + R + 'px' }));
  const Rr = mk(root, null, null, Object.assign({}, base, { left: (X0 + w80) + 'px', width: w20 + 'px', background: PAPER, borderRadius: '0 ' + R + 'px ' + R + 'px 0' }));
  const whole = mk(root, 'num', '$100', { position: 'absolute', left: '0px', width: '1920px', top: TOP + 'px', height: H + 'px', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '112px', color: INK });
  const v80 = mk(L, 'num', '$80', { fontSize: '96px', color: PAPER, opacity: 0 });
  const v20 = mk(Rr, 'num', '$20', { fontSize: '96px', color: INK, opacity: 0 });
  const labBase = { position: 'absolute', top: (TOP + H + 44) + 'px', width: '600px', textAlign: 'center', whiteSpace: 'nowrap', fontSize: '64px' };
  const lab80 = mk(root, 'body', 'на жизнь', Object.assign({}, labBase, { left: (X0 + w80 / 2 - 26 - 300) + 'px', color: MUTED }));
  const lab20 = mk(root, 'body gold', 'в банкролл', Object.assign({}, labBase, { left: (X0 + w80 + w20 / 2 + 26 - 300) + 'px', fontWeight: 800 }));
  const note = mk(root, 'note', 'условный пример', { position: 'absolute', left: '0px', width: '1920px', textAlign: 'center', bottom: '110px', fontSize: '30px' });

  tl.from([L, Rr, whole], { opacity: 0, scale: .94, duration: .5, ease: 'expo.out' }, 0);
  // распад
  const T = 1.0;
  tl.to(whole, { opacity: 0, duration: .2, ease: 'power2.in' }, T)
    .to(L, { x: -26, backgroundColor: 'rgba(244,241,232,.10)', borderTopLeftRadius: R + 'px', borderBottomLeftRadius: R + 'px', borderTopRightRadius: R + 'px', borderBottomRightRadius: R + 'px', duration: .55, ease: 'expo.out' }, T)
    .to(Rr, { x: 26, backgroundColor: GOLD, borderTopLeftRadius: R + 'px', borderBottomLeftRadius: R + 'px', borderTopRightRadius: R + 'px', borderBottomRightRadius: R + 'px', duration: .55, ease: 'expo.out' }, T)
    .to(v80, { opacity: 1, duration: .3 }, T + .25)
    .to(v20, { opacity: 1, duration: .3 }, T + .25)
    .from(lab80, { opacity: 0, y: 18, duration: .4, ease: 'power3.out' }, T + .45)
    .from(lab20, { opacity: 0, y: 18, duration: .4, ease: 'power3.out' }, T + .6)
    .from(note, { opacity: 0, duration: .4 }, T + 1.0);
}};

/* ───────────────────────── G22 · Астана → Дананг: две точки и дуга ───────────────────────── */
SCENES['G22-map'] = { dur: 5, build(root, tl) {
  const A = { x: 620, y: 220 }, B = { x: 1320, y: 800 };
  const mid = { x: (A.x + B.x) / 2, y: (A.y + B.y) / 2 };
  const dx = B.x - A.x, dy = B.y - A.y, len = Math.hypot(dx, dy);
  const ctrl = { x: mid.x + (dy / len) * 170, y: mid.y - (dx / len) * 170 };
  const d = `M ${A.x} ${A.y} Q ${ctrl.x} ${ctrl.y} ${B.x} ${B.y}`;
  const holder = mk(root, null, `<svg width="1920" height="1080" viewBox="0 0 1920 1080" style="position:absolute;left:0;top:0;overflow:visible">
    <defs><mask id="b22m" maskUnits="userSpaceOnUse" x="0" y="0" width="1920" height="1080"><path id="b22p" d="${d}" fill="none" stroke="#fff" stroke-width="16" stroke-linecap="round"/></mask></defs>
    <path d="${d}" fill="none" stroke="#f2c14e" stroke-width="7" stroke-linecap="round" stroke-dasharray="1 20" mask="url(#b22m)"/>
    <circle id="b22da" cx="${A.x}" cy="${A.y}" r="18" fill="#f4f1e8"/>
    <circle id="b22db" cx="${B.x}" cy="${B.y}" r="18" fill="#f2c14e"/>
    <g id="b22pl"><path d="M26,0 L22,-3 L14,-3.6 L4,-3.8 L-8,-24 L-14,-24 L-11,-3.6 L-19,-3 L-24,-10 L-29,-10 L-27,-2.4 L-30,-1.6 L-30,1.6 L-27,2.4 L-29,10 L-24,10 L-19,3 L-11,3.6 L-14,24 L-8,24 L4,3.8 L14,3.6 L22,3 Z" fill="#f4f1e8" stroke="#f4f1e8" stroke-width="2" stroke-linejoin="round" transform="scale(1.9)"/></g>
  </svg>`, { position: 'absolute', left: '0px', top: '0px', width: '1920px', height: '1080px' });
  const q = (id) => holder.querySelector('#' + id);
  const pathEl = q('b22p'), plane = q('b22pl');
  const L = pathEl.getTotalLength();
  pathEl.setAttribute('stroke-dasharray', L); pathEl.setAttribute('stroke-dashoffset', L);
  const place = (s) => {
    const p = pathEl.getPointAtLength(s * L), p2 = pathEl.getPointAtLength(Math.min(L, s * L + 2)), p1 = pathEl.getPointAtLength(Math.max(0, s * L - 2));
    plane.setAttribute('transform', `translate(${p.x} ${p.y}) rotate(${Math.atan2(p2.y - p1.y, p2.x - p1.x) * 180 / Math.PI})`);
    pathEl.setAttribute('stroke-dashoffset', L * (1 - s));
  };
  place(0);
  const o = { s: 0 };
  tl.to(o, { s: 1, duration: 2.5, ease: 'power1.inOut', onUpdate: () => place(o.s) }, .8)
    .fromTo(plane, { opacity: 0 }, { opacity: 1, duration: .25 }, .8)
    .to(plane, { opacity: 0, duration: .3, ease: 'power2.in' }, 3.15);
  const lab = (txt, st) => mk(root, 'title', txt, Object.assign({ position: 'absolute', fontSize: '64px', whiteSpace: 'nowrap' }, st));
  const la = lab('Астана', { left: (A.x - 44 - 420) + 'px', width: '420px', textAlign: 'right', top: (A.y - 34) + 'px' });
  const lb = lab('Дананг', { left: (B.x + 48) + 'px', top: (B.y - 34) + 'px', color: GOLD });
  tl.from(q('b22da'), { scale: 0, transformOrigin: A.x + 'px ' + A.y + 'px', duration: .4, ease: 'expo.out' }, .2)
    .from(la, { opacity: 0, x: 16, duration: .4, ease: 'power3.out' }, .3)
    .from(q('b22db'), { scale: 0, transformOrigin: B.x + 'px ' + B.y + 'px', duration: .4, ease: 'expo.out' }, 3.0)
    .from(lb, { opacity: 0, x: -16, duration: .4, ease: 'power3.out' }, 3.05);
  const cap = mk(root, 'body', 'билеты за <span class="num gold" style="font-size:60px;margin-left:10px">' + fmt(8500) + NB + '₽</span>', { position: 'absolute', left: '96px', bottom: '96px', display: 'flex', alignItems: 'baseline', fontWeight: 700, fontSize: '56px', whiteSpace: 'nowrap' });
  tl.from(cap, { opacity: 0, y: 30, duration: .45, ease: 'expo.out' }, .5);
}};

/* ───────────────────────── G24 · фон конечной заставки (1920×1080): тезис и пустые рамки ───────────────────────── */
SCENES['G24-end'] = { dur: 18, build(root, tl) {
  const frame = (x, y, w, h, r) => mk(root, null, null, { position: 'absolute', left: x + 'px', top: y + 'px', width: w + 'px', height: h + 'px', borderRadius: r + 'px', border: '2px solid rgba(244,241,232,.24)', background: 'rgba(0,0,0,.16)' });
  const f1 = frame(1064, 92, 760, 428, 22);     // место под «видео» 1
  const f2 = frame(1064, 560, 760, 428, 22);    // место под «видео» 2
  const circ = frame(430, 640, 260, 260, 130);  // место под кнопку подписки
  const txt = mk(root, null, null, { position: 'absolute', left: '96px', top: '112px', display: 'flex', flexDirection: 'column', gap: '16px' });
  const t1 = mk(txt, 'display', 'Брови отрастают.', { fontSize: '62px' });
  const t2 = mk(txt, 'display gold', 'Банкролл — нет.', { fontSize: '62px' });
  tl.from(t1, { opacity: 0, y: 24, duration: .6, ease: 'power3.out' }, .1)
    .from(t2, { opacity: 0, y: 24, duration: .6, ease: 'power3.out' }, .3)
    .from(f1, { opacity: 0, duration: .6, ease: 'power3.out' }, .3)
    .from(f2, { opacity: 0, duration: .6, ease: 'power3.out' }, .45)
    .from(circ, { opacity: 0, duration: .6, ease: 'power3.out' }, .6);
}};

/* ───────────────────────── G26 · плашка-подпись слева снизу (альфа) ───────────────────────── */
SCENES['G26-lower'] = { dur: 3, alpha: true, build(root, tl, P) {
  const plate = mk(root, null, null, { position: 'absolute', left: '96px', top: '820px', borderRadius: '22px', background: 'rgba(10,22,18,.94)', padding: '26px 52px 30px 44px', display: 'flex', flexDirection: 'column', gap: '8px' });
  const n = mk(plate, 'title', 'Максим · Shegol111', { fontSize: '58px', whiteSpace: 'nowrap' });
  const s = mk(plate, 'body', 'покерист, стример, тренер', { fontSize: '36px', color: MUTED, fontWeight: 500 });
  tl.fromTo(plate, { clipPath: 'inset(0px 100% 0px 0px round 22px)' }, { clipPath: 'inset(0px 0% 0px 0px round 22px)', duration: .55, ease: 'expo.out' }, 0)
    .set(plate, { clipPath: 'none' }, .6)
    .from(n, { opacity: 0, x: -34, duration: .5, ease: 'expo.out' }, .12)
    .from(s, { opacity: 0, x: -34, duration: .5, ease: 'expo.out' }, .24);
}};

/* ───────────────────────── G29 · перерисовка раздачи ─────────────────────────
   Параметры (монтажёр подставляет реальные карты):
     hero   — карты Максима, «6s3d» или «6s,3d» (ранг + масть: s ♠, h ♥, d ♦, c ♣; 10 или T; «?» — масть неизвестна, по умолчанию «6?3?»)
     board  — 5 карт борда через запятую, «??» = рубашка (по умолчанию «??,??,??,??,??»)
     vil    — карты соперника, «??,??» по умолчанию
     reveal — 1: карты соперника переворачиваются, над ними чип (tag, по умолчанию «флеш»)
     bettor — hero | vil (кто ставит на ривере, по умолчанию hero — Максим); bet — подпись ставки
     hn, vn — имена (по умолчанию «Максим», «Соперник»); tag — текст чипа */
SCENES['G29-hand'] = { dur: 5, build(root, tl, P) {
  sty(root, SHARED_CSS + `.b-flip { position: absolute; left: 0; top: 0; width: 150px; height: 210px; transform-style: preserve-3d; }
    .b-flip > .pcard { position: absolute; left: 0; top: 0; backface-visibility: hidden; -webkit-backface-visibility: hidden; }
    .b-flip > .face { transform: rotateY(180deg); }`);
  const SUITS = { s: '♠', h: '♥', d: '♦', c: '♣' };
  const parse = (str, n) => {
    const out = [], re = /(10|[2-9TJQKAtjqka]|\?)([shdcSHDC]|\?)/g; let m;
    while ((m = re.exec(str || '')) && out.length < n) out.push({ r: m[1].toUpperCase(), s: m[2].toLowerCase() });
    while (out.length < n) out.push({ r: '?', s: '?' });
    return out;
  };
  const reveal = P.reveal === '1';
  const hero = parse(P.hero || '6?3?', 2), board = parse(P.board || '??,??,??,??,??', 5), vil = parse(P.vil || '??,??', 2);
  const bettor = P.bettor === 'vil' ? 'vil' : 'hero';
  const unknown = (c) => c.r === '?' && c.s === '?';
  const faceEl = (parent, c) => {
    const red = c.s === 'h' || c.s === 'd';
    const rk = c.r === 'T' ? '10' : c.r;
    const d = el('div', 'pcard' + (red ? ' r' : ''), `<div class="rk">${rk}</div><div class="st">${SUITS[c.s] || ''}</div>`, parent);
    if (unknown(c)) { d.style.color = '#9aa39e'; d.querySelector('.st').textContent = ''; }
    return d;
  };
  const backEl = (parent) => el('div', 'pcard back', null, parent);
  // слот карты: размер с учётом масштаба, внутри — «настоящая» карта 150×210
  const slot = (x, y, s) => {
    const sl = mk(root, null, null, { position: 'absolute', left: x + 'px', top: y + 'px', width: (150 * s) + 'px', height: (210 * s) + 'px' });
    const inner = mk(sl, null, null, { position: 'absolute', left: '0px', top: '0px', width: '150px', height: '210px', transformOrigin: '0 0', transform: `scale(${s})`, perspective: '1000px' });
    return { sl, inner };
  };

  // стол
  const table = tableOval(root, 960, 540, 1720, 960);
  tl.from(table, { opacity: 0, scale: .97, duration: .5, ease: 'expo.out' }, 0);

  // подписи игроков
  const pill = (txt, dot, yc, extra) => {
    const p = mk(root, 'glass', txt, Object.assign({ position: 'absolute', right: (1920 - 790) + 'px', top: (yc - 38) + 'px', padding: '12px 32px', borderRadius: '999px', fontFamily: 'Onest', fontWeight: 700, fontSize: '40px', whiteSpace: 'nowrap', display: 'flex', alignItems: 'center' }, extra || {}));
    return p;
  };
  const SC = .9, HY = 701, VY = 190, BY = 435;
  const hx = 960 - (2 * 150 * SC + 18) / 2;
  const pH = pill(P.hn || 'Максим', GOLD, HY + 105 * SC);
  const pV = pill(P.vn || 'Соперник', MUTED, VY + 105 * SC);
  tl.from(pH, { opacity: 0, x: -24, duration: .4, ease: 'power3.out' }, .35).from(pV, { opacity: 0, x: -24, duration: .4, ease: 'power3.out' }, .5);

  // карты Максима (лицом) и соперника (рубашкой; при reveal переворачиваются)
  const heroSlots = hero.map((c, i) => {
    const s = slot(hx + i * (150 * SC + 18), HY, SC);
    unknown(c) ? backEl(s.inner) : faceEl(s.inner, c);
    return s.sl;
  });
  tl.from(heroSlots, { opacity: 0, y: 50, duration: .45, stagger: .1, ease: 'expo.out' }, .25);

  const flips = [];
  const vilSlots = vil.map((c, i) => {
    const s = slot(hx + i * (150 * SC + 18), VY, SC);
    const fl = mk(s.inner, 'b-flip');
    backEl(fl); const f = faceEl(fl, c); f.classList.add('face');
    flips.push(fl);
    return s.sl;
  });
  tl.from(vilSlots, { opacity: 0, y: -50, duration: .45, stagger: .1, ease: 'expo.out' }, .45);

  // борд: флоп, тёрн, ривер
  const bw = 150, bg = 20, bx0 = 960 - (5 * bw + 4 * bg) / 2;
  const bSlots = board.map((c, i) => {
    const s = slot(bx0 + i * (bw + bg), BY, 1);
    unknown(c) ? backEl(s.inner) : faceEl(s.inner, c);
    return s.sl;
  });
  [[0, .75], [1, .85], [2, .95], [3, 1.3], [4, 1.65]].forEach(([i, at]) => tl.from(bSlots[i], { opacity: 0, y: 36, scale: .94, duration: .4, ease: 'expo.out' }, at));

  // ставка на ривере
  const by = bettor === 'vil' ? VY + 105 * SC + 8 : HY + 105 * SC + 8;
  const bx = 1156;
  const chips = [];
  for (let i = 0; i < 5; i++) chips.push(chipEl(root, bx, by - i * 7, 60));
  const betLab = mk(root, 'glass', P.bet || 'ставит на ривере', { position: 'absolute', left: (bx + 50) + 'px', top: (by - 40) + 'px', padding: '12px 28px', borderRadius: '999px', fontFamily: 'Onest', fontWeight: 700, fontSize: '38px', whiteSpace: 'nowrap' });
  tl.from(chips, { opacity: 0, scale: .4, y: -30, duration: .3, stagger: .05, ease: 'expo.out' }, 2.0)
    .from(betLab, { opacity: 0, x: -20, duration: .4, ease: 'power3.out' }, 2.25);

  // вскрытие
  if (reveal) {
    flips.forEach((fl, i) => tl.to(fl, { rotationY: 180, duration: .6, ease: 'power3.inOut' }, 3.1 + i * .12));
    const tag = mk(root, 'chip', P.tag || 'флеш', { position: 'absolute', left: '50%', top: '104px', transform: 'translateX(-50%)', background: RED, color: '#fff', border: '0', fontFamily: 'Unb', fontWeight: 700, fontSize: '46px', padding: '10px 40px', whiteSpace: 'nowrap' });
    tl.from(tag, { opacity: 0, y: 20, scale: .9, duration: .4, ease: 'expo.out' }, 3.8);
  }
}};

/* ───────────────────────── G30 · сноска-глоссарий (альфа). Параметры: term, def ───────────────────────── */
SCENES['G30-chip'] = { dur: 2, alpha: true, build(root, tl, P) {
  const box = mk(root, null, null, { position: 'absolute', left: '96px', bottom: '96px', width: 'max-content', maxWidth: '1100px', borderRadius: '22px', background: 'rgba(10,22,18,.94)', display: 'flex' });
  const t = mk(box, 'body', null, { padding: '22px 40px 24px 36px', fontSize: '42px', fontWeight: 500, lineHeight: 1.22 });
  const term = document.createElement('b'); term.className = 'gold'; term.style.fontWeight = '800'; term.textContent = P.term || 'ГСЧ';
  const sep = document.createElement('span'); sep.style.color = MUTED; sep.textContent = ' — ';
  const def = document.createElement('span'); def.textContent = P.def || 'генератор случайных чисел';
  t.append(term, sep, def);
  tl.fromTo(box, { opacity: 0, x: -36 }, { opacity: 1, x: 0, duration: .35, ease: 'expo.out' }, 0);
}};

/* ───────────────────────── G32 · вопрос зрителю ───────────────────────── */
SCENES['G32-cta'] = { dur: 6, build(root, tl) {
  const cap = mk(root, 'cap', 'А вы бы что сделали?', { position: 'absolute', left: '0px', width: '1920px', top: '340px', textAlign: 'center', fontSize: '40px', letterSpacing: '.12em' });
  const row = mk(root, null, null, { position: 'absolute', left: '0px', top: '450px', width: '1920px', display: 'flex', justifyContent: 'center', gap: '50px' });
  const mkBtn = (html, color) => mk(row, 'title', html, { width: '835px', height: '260px', borderRadius: '36px', border: '3px solid ' + color, background: 'rgba(244,241,232,.06)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '24px', fontSize: '58px', whiteSpace: 'nowrap' });
  const b1 = mkBtn('Сбрил бы брови', RED);
  const b2 = mkBtn('Пошёл бы на NL100', GOLD);
  tl.from(cap, { opacity: 0, y: 16, duration: .4, ease: 'power3.out' }, 0)
    .from(b1, { opacity: 0, y: 40, duration: .5, ease: 'expo.out' }, .15)
    .from(b2, { opacity: 0, y: 40, duration: .5, ease: 'expo.out' }, .3);
  tl.to([b1, b2], { scale: 1.02, duration: 2.5, ease: 'sine.inOut', yoyo: true, repeat: 1 }, .9);
}};

/* ───────────────────────── G33 · календарь. ?v=6w | ?v=flip ───────────────────────── */
SCENES['G33-calendar'] = { dur: 4, build(root, tl, P) {
  if (P.v === 'flip') {
    // отрывной календарь: листки 18 → 24 июля, красная обводка на 24
    const W = 640, H = 720;
    const stack = mk(root, null, null, { position: 'absolute', left: ((1920 - W) / 2) + 'px', top: ((1080 - H) / 2) + 'px', width: W + 'px', height: H + 'px', perspective: '2400px' });
    const days = [18, 19, 20, 21, 22, 23, 24];
    const pages = days.map((d, i) => {
      const pg = mk(stack, null, null, { position: 'absolute', left: '0px', top: '0px', width: W + 'px', height: H + 'px', borderRadius: '30px', background: PAPER, overflow: 'hidden', transformOrigin: '50% 0%', zIndex: 20 - i, backfaceVisibility: 'hidden' });
      mk(pg, 'title', 'ИЮЛЬ', { position: 'absolute', left: '0px', width: W + 'px', top: '50px', textAlign: 'center', fontSize: '56px', letterSpacing: '.22em', paddingLeft: '.22em', color: '#6b7a73' });
      mk(pg, 'num', String(d), { position: 'absolute', left: '0px', width: W + 'px', top: '170px', textAlign: 'center', fontSize: '420px', color: INK, letterSpacing: '-.04em' });
      const shade = mk(pg, null, null, { position: 'absolute', inset: '0px', background: 'linear-gradient(180deg, rgba(0,0,0,.0), rgba(0,0,0,.65))', opacity: 0 });
      return { pg, shade };
    });
    tl.from(stack, { opacity: 0, y: 40, scale: .96, duration: .45, ease: 'expo.out' }, 0);
    const T0 = .55, STEP = .3, FD = .27;
    pages.slice(0, 6).forEach((o, i) => {
      const t = T0 + i * STEP;
      tl.fromTo(o.pg, { rotationX: 0 }, { rotationX: 92, duration: FD, ease: 'power2.in' }, t)
        .fromTo(o.shade, { opacity: 0 }, { opacity: 1, duration: FD, ease: 'power2.in' }, t)
        .set(o.pg, { opacity: 0 }, t + FD);
    });
    const tEnd = T0 + 5 * STEP + FD;       // 24-е открыто
    const ring = mk(stack, null, null, { position: 'absolute', left: '-16px', top: '-16px', width: (W + 32) + 'px', height: (H + 32) + 'px', borderRadius: '44px', border: '10px solid ' + RED, zIndex: 50 });
    tl.fromTo(ring, { opacity: 0, scale: 1.07 }, { opacity: 1, scale: 1, duration: .4, ease: 'expo.out' }, tEnd + .15);
  } else {
    // 6 недель: 18 июля → 31 августа, ряд = неделя (без дней недели — только даты)
    const COLS = 7, CW = 128, CH = 92, GP = 10, GX = 130;
    const total = 45, rowsN = 7;
    const GY = 236;
    mk(root, 'cap', '18 июля → 31 августа', { position: 'absolute', left: GX + 'px', top: '128px', fontSize: '34px' });
    const cells = [];
    for (let i = 0; i < total; i++) {
      const r = Math.floor(i / COLS), c = i % COLS;
      const july = i <= 13, day = july ? 18 + i : i - 13;
      const marked = i < 42;
      const cell = mk(root, null, null, { position: 'absolute', left: (GX + c * (CW + GP)) + 'px', top: (GY + r * (CH + GP)) + 'px', width: CW + 'px', height: CH + 'px', borderRadius: '16px', background: 'rgba(244,241,232,.07)', border: '1.5px solid rgba(244,241,232,.14)', display: 'flex', alignItems: 'center', justifyContent: 'center' });
      mk(cell, 'num', String(day), { fontSize: '44px', color: marked ? PAPER : MUTED });
      cells.push({ cell, r });
    }
    tl.from(cells.map(o => o.cell), { opacity: 0, scale: .82, duration: .3, stagger: .012, ease: 'power3.out' }, .1);
    for (let r = 0; r < 6; r++) {
      tl.to(cells.filter(o => o.r === r).map(o => o.cell), { backgroundColor: 'rgba(242,193,78,.22)', borderColor: 'rgba(242,193,78,.6)', duration: .3, stagger: .025, ease: 'power2.out' }, .9 + r * .14);
    }
    const bx = GX + COLS * (CW + GP) + 80, bh = 6 * CH + 5 * GP;
    const big = mk(root, null, null, { position: 'absolute', left: bx + 'px', top: (GY + bh / 2 - 250) + 'px', display: 'flex', flexDirection: 'column', gap: '4px' });
    mk(big, 'num gold', '6', { fontSize: '380px' });
    mk(big, 'title', 'недель', { fontSize: '112px' });
    tl.from(big, { opacity: 0, x: -30, duration: .5, ease: 'expo.out' }, 1.5);
  }
}};

/* ───────────────────────── G34 · «заработать $10 000» → «иметь на счёте» ───────────────────────── */
SCENES['G34-strike'] = { dur: 4, build(root, tl) {
  const wrap = fillBox(root, { paddingTop: '90px' });
  const line = mk(wrap, 'display', null, { fontSize: '118px', position: 'relative', whiteSpace: 'nowrap', display: 'flex', alignItems: 'baseline', gap: '40px' });
  const word = mk(line, null, null, { position: 'relative', display: 'inline-block' });
  const wt = mk(word, null, 'заработать', { display: 'inline-block' });
  const sum = mk(line, null, '$' + fmt(10000), { color: GOLD });
  const k = mk(word, 'strike', null, { top: '55%', height: '13px', left: '-3%', width: '106%' });
  const above = mk(word, 'title gold', 'иметь на счёте', { position: 'absolute', left: '0px', width: '100%', textAlign: 'center', top: '-104px', fontSize: '62px', whiteSpace: 'nowrap', letterSpacing: '-.01em' });
  tl.from(line, { opacity: 0, y: 36, duration: .45, ease: 'expo.out' }, .1)
    .to(k, { scaleX: 1, duration: .3, ease: 'power4.in' }, 1.1)
    .to(wt, { opacity: .45, duration: .35, ease: 'power2.out' }, 1.35)
    .from(above, { opacity: 0, y: 26, duration: .5, ease: 'expo.out' }, 1.8);
}};

/* ───────────────────────── G35 · условная таблица гонки ───────────────────────── */
SCENES['G35-race'] = { dur: 4, build(root, tl) {
  const col = mk(root, null, null, { position: 'absolute', left: '280px', top: '190px', width: '1360px', display: 'flex', flexDirection: 'column', gap: '22px' });
  const head = mk(col, 'cap', 'Гонка · очки', { fontSize: '30px', marginBottom: '8px' });
  const data = [100, 88, 76, 54, 40];
  const rows = data.map((v, i) => {
    const top = i < 3;
    const r = mk(col, null, null, { display: 'flex', alignItems: 'center', gap: '36px', height: '88px' });
    mk(r, 'title', 'Игрок ' + (i + 1), { fontSize: '42px', width: '270px', color: top ? PAPER : MUTED });
    const bar = mk(r, 'bar', null, { flex: '1', height: '56px', borderRadius: '14px', background: 'transparent', border: '0' });
    const fill = mk(bar, 'fill', null, { borderRadius: '12px' });
    if (!top) fill.style.background = 'rgba(244,241,232,.28)';
    return { r, fill, v };
  });
  const cap = mk(root, 'body', 'рум платит призы тем, кто <span class="gold">сыграл больше всех</span>', { position: 'absolute', left: '0px', width: '1920px', textAlign: 'center', top: '860px', fontSize: '48px', fontWeight: 600 });
  tl.from(head, { opacity: 0, y: 14, duration: .35, ease: 'power3.out' }, 0);
  rows.forEach((o, i) => {
    tl.from(o.r, { opacity: 0, x: -30, duration: .35, ease: 'power3.out' }, .15 + i * .08)
      .fromTo(o.fill, { width: '0%' }, { width: o.v + '%', duration: 1.1, ease: 'power3.out' }, .4 + i * .08);
  });
  tl.from(cap, { opacity: 0, y: 16, duration: .45, ease: 'power3.out' }, 1.0);
}};

/* ───────────────────────── G36 · «правило 100 БИ прожило» + таймер ───────────────────────── */
SCENES['G36-timer'] = { dur: 4, build(root, tl) {
  const wrap = fillBox(root, { gap: '46px' });
  const h = mk(wrap, 'title', 'Правило <span class="gold">100' + NB + 'БИ</span> прожило', { fontSize: '84px', textAlign: 'center' });
  const panel = mk(wrap, null, null, { display: 'flex', alignItems: 'baseline', gap: '26px' });
  const hh = mk(panel, 'num', '0', { fontSize: '230px', minWidth: '300px', textAlign: 'right' });
  mk(panel, 'title muted', 'ч', { fontSize: '86px', marginRight: '34px' });
  const mm = mk(panel, 'num', '00', { fontSize: '230px', minWidth: '300px', textAlign: 'right' });
  mk(panel, 'title muted', 'мин', { fontSize: '86px' });
  const since = mk(wrap, 'body muted', 'с первой раздачи 11 сентября', { fontSize: '46px', marginTop: '-10px' });
  tl.from(since, { opacity: 0, y: 16, duration: .4, ease: 'power3.out' }, .3);
  const TOTAL = 27 * 60 + 42;
  const o = { v: 0 };
  const upd = () => { const t = Math.round(o.v); hh.textContent = String(Math.floor(t / 60)); mm.textContent = String(t % 60).padStart(2, '0'); };
  upd();
  tl.from(h, { opacity: 0, y: 30, duration: .45, ease: 'expo.out' }, 0)
    .from(panel, { opacity: 0, y: 30, duration: .45, ease: 'expo.out' }, .15)
    .to(o, { v: TOTAL, duration: 2.1, ease: 'power3.out', onUpdate: upd }, .5)
    .to([hh, mm], { color: GOLD, duration: .4, ease: 'power2.out' }, 2.5);
}};

/* ───────────────────────── G37 · анте: фишки от шести мест в центр ───────────────────────── */
SCENES['G37-ante'] = { dur: 4, build(root, tl) {
  sty(root, SHARED_CSS);
  const CX = 960, CY = 450, TW = 1240, TH = 600;
  const table = tableOval(root, CX, CY, TW, TH);
  tl.from(table, { opacity: 0, scale: .96, duration: .5, ease: 'expo.out' }, 0);
  const RX = 650, RY = 330;
  const angs = [240, 300, 0, 60, 120, 180];
  const seats = angs.map((a, i) => {
    const r = a * Math.PI / 180, x = CX + RX * Math.cos(r), y = CY + RY * Math.sin(r);
    const s = mk(root, null, null, { position: 'absolute', left: (x - 48) + 'px', top: (y - 48) + 'px', width: '96px', height: '96px', borderRadius: '50%', background: 'rgba(10,22,18,.9)', border: '2.5px solid rgba(244,241,232,.35)' });
    return { s, x, y };
  });
  tl.from(seats.map(o => o.s), { opacity: 0, scale: .6, duration: .4, stagger: .07, ease: 'expo.out' }, .15);
  // фишки
  const chips = seats.map((o, i) => {
    const sx = CX + (o.x - CX) * .76, sy = CY + (o.y - CY) * .76;
    const a = (i / 6) * Math.PI * 2 + .4;
    const tx = CX + Math.cos(a) * 38, ty = CY + Math.sin(a) * 24;
    const c = chipEl(root, sx, sy, 50);
    return { c, dx: tx - sx, dy: ty - sy };
  });
  tl.from(chips.map(o => o.c), { opacity: 0, scale: .3, duration: .3, stagger: .05, ease: 'expo.out' }, .6);
  chips.forEach(o => tl.to(o.c, { x: o.dx, y: o.dy, duration: .9, ease: 'power3.inOut' }, 1.0));
  // пульс пота в момент прихода фишек
  const cap = mk(root, 'body', '<span class="gold">анте</span> — небольшой взнос каждого в каждой раздаче', { position: 'absolute', left: '0px', width: '1920px', textAlign: 'center', top: '868px', fontSize: '48px', fontWeight: 600 });
  tl.from(cap, { opacity: 0, y: 16, duration: .45, ease: 'power3.out' }, .5);
}};

/* ───────────────────────── G38 · ЛУДКА ───────────────────────── */
SCENES['G38-ludka'] = { dur: 4, build(root, tl) {
  const word = mk(root, 'display', 'ЛУДКА', { position: 'absolute', left: '0px', width: '1920px', top: '230px', textAlign: 'center', fontSize: '300px' });
  const X0 = 260, W = 1400, Y = 690;
  const l1 = mk(root, 'title mint', 'по расчёту', { position: 'absolute', left: X0 + 'px', top: (Y - 100) + 'px', fontSize: '62px' });
  const l2 = mk(root, 'title red', 'на азарте', { position: 'absolute', right: (1920 - X0 - W) + 'px', top: (Y - 100) + 'px', fontSize: '62px', whiteSpace: 'nowrap', textAlign: 'right' });
  const line = mk(root, null, null, { position: 'absolute', left: X0 + 'px', top: (Y - 6) + 'px', width: (W - 40) + 'px', height: '12px', borderRadius: '6px', background: 'linear-gradient(90deg, #3ddc97, #ffb155 55%, #ff5a4e)', transformOrigin: 'left center' });
  const head = mk(root, null, `<svg width="60" height="76" viewBox="0 0 60 76"><path d="M4 8L52 38L4 68Z" fill="#ff5a4e" stroke="#ff5a4e" stroke-width="8" stroke-linejoin="round"/></svg>`, { position: 'absolute', left: (X0 + W - 64) + 'px', top: (Y - 38) + 'px' });
  tl.from(word, { opacity: 0, y: 44, duration: .5, ease: 'expo.out' }, 0)
    .from(l1, { opacity: 0, y: 16, duration: .4, ease: 'power3.out' }, .6)
    .fromTo(line, { scaleX: 0 }, { scaleX: 1, duration: .8, ease: 'power3.inOut' }, .7)
    .from(head, { opacity: 0, x: -20, duration: .25, ease: 'power2.out' }, 1.35)
    .from(l2, { opacity: 0, y: 16, duration: .45, ease: 'power3.out' }, 1.45);
}};

/* ───────────────────────── G39 · заработал ≠ на счёте ───────────────────────── */
SCENES['G39-earned'] = { dur: 4, build(root, tl) {
  const row = fillBox(root, { flexDirection: 'row', gap: '40px' });
  const mkCol = (lab, color) => {
    const c = mk(row, null, null, { width: '700px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '28px' });
    mk(c, 'body', lab, { fontSize: '50px', color: MUTED, textAlign: 'center', lineHeight: 1.15 });
    const n = mk(c, 'num', '$0', { fontSize: '150px', color });
    return { c, n };
  };
  const a = mkCol('заработал<br>за 11 недель', PAPER);
  const neq = mk(row, 'num', '≠', { fontSize: '210px', color: PAPER, width: '170px', textAlign: 'center' });
  const b = mkCol('на счёте<br>18 июля', GOLD);
  tl.from(a.c, { opacity: 0, y: 40, duration: .45, ease: 'expo.out' }, .1);
  countUp(tl, a.n, 0, 3500, .25, 1.0, { pre: '$' });
  tl.from(b.c, { opacity: 0, y: 40, duration: .45, ease: 'expo.out' }, .7);
  countUp(tl, b.n, 0, 1686, .85, 1.0, { pre: '$' });
  tl.from(neq, { opacity: 0, scale: .5, duration: .5, ease: 'expo.out' }, 1.5);
}};

/* ───────────────────────── G00 · чистый фон «сукно» ───────────────────────── */
SCENES['G00-bg'] = { dur: 10, build() {} };

})();
