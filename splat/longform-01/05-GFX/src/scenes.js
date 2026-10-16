/* Эталонные сцены. Правила стиля: быстрый вход (0,25–0,45 с, expo/power3.out), долгое удержание,
   без «мультяшных» отскоков, один акцент на кадр. Выход не анимируем — режет монтажёр. */

// G03 — титр ролика «БРОВИ НА КОНУ».
SCENES['G03-title'] = { dur: 3, build(root, tl) {
  const wrap = el('div', 'center col', null, root); css(wrap, { alignItems: 'center', gap: '28px' });
  const t1 = el('div', 'display', 'БРОВИ', wrap); css(t1, { fontSize: '230px' });
  const t2 = el('div', 'display gold', 'НА КОНУ', wrap); css(t2, { fontSize: '150px' });
  const cut = el('div', 'abs', null, wrap); css(cut, { left: '-40px', right: '-40px', top: '46%', height: '14px', background: 'var(--red)', borderRadius: '8px', transformOrigin: 'left center' });
  tl.from(t1, { y: 60, opacity: 0, duration: .4, ease: 'expo.out' }, 0)
    .from(t2, { y: 60, opacity: 0, duration: .4, ease: 'expo.out' }, .08)
    .fromTo(cut, { scaleX: 0, rotate: -4 }, { scaleX: 1, rotate: -4, duration: .22, ease: 'power4.in' }, .55)
    .to(t1, { x: -6, duration: .05, yoyo: true, repeat: 3 }, .77);
}};

// G01 — карточка условий пари. ?v=2 — двусторонний вариант (глава 07).
SCENES['G01-bet'] = { dur: 6, build(root, tl, P) {
  const two = P.v === '2';
  const card = el('div', 'card center', null, root); css(card, { width: '1240px', padding: '64px 80px 70px', display: 'flex', flexDirection: 'column', gap: '30px' });
  const cap = el('div', 'cap', 'Условия пари', card); css(cap, { color: '#6b7a73' });
  const sumRow = el('div', 'row', null, card); css(sumRow, { gap: '28px', alignItems: 'baseline' });
  const sum = el('div', 'num', '$0', sumRow); css(sum, { fontSize: '170px', color: 'var(--ink)' });
  const sub = el('div', 'body', two ? 'на счёте<br>к концу лета' : 'к концу<br>лета', sumRow); css(sub, { fontSize: '44px', color: '#334039', lineHeight: '1.1', whiteSpace: 'nowrap' });
  const out = el('div', 'row', null, card); css(out, { gap: '40px' });
  if (!two) {
    const a = el('div', 'body', 'иначе —', out); css(a, { fontSize: '56px', color: '#334039' });
    const b = el('div', 'title red', 'бреет брови', out); css(b, { fontSize: '76px' });
    tl.from([a, b], { opacity: 0, y: 24, duration: .35, stagger: .12, ease: 'power3.out' }, 1.1);
  } else {
    const mk = (who, cond, res, color) => {
      const box = el('div', 'col', null, out); css(box, { flex: 1, gap: '10px', padding: '28px 32px', borderRadius: '20px', background: '#ebe6d8' });
      el('div', 'cap', cond, box).style.color = '#6b7a73';
      const r = el('div', 'title', res, box); css(r, { fontSize: '50px', color });
      const w = el('div', 'body', who, box); css(w, { fontSize: '36px', color: '#334039' });
      return box;
    };
    const b1 = mk('бреет Боссбайден', 'есть $10 000', 'Максим победил', '#16804f');
    const b2 = mk('бреет Максим', 'меньше', 'Боссбайден победил', '#c7362c');
    tl.from([b1, b2], { opacity: 0, y: 30, duration: .4, stagger: .15, ease: 'power3.out' }, 1.1);
  }
  tl.from(card, { scale: .92, opacity: 0, duration: .45, ease: 'expo.out' }, 0);
  countUp(tl, sum, 0, 10000, .2, .8, { pre: '$' });
}};

// G02 — баланс против цели. ?v=585 — осенняя версия ($585 / $1 000).
SCENES['G02-balance'] = { dur: 6, build(root, tl, P) {
  const autumn = P.v === '585';
  const have = autumn ? 585 : 1686, goal = autumn ? 1000 : 10000;
  const wrap = el('div', 'center col', null, root); css(wrap, { width: '1400px', gap: '34px' });
  const cap = el('div', 'cap', autumn ? 'Банкролл · 17 сентября' : 'Игровой счёт · 18 июля', wrap);
  const row = el('div', 'row', null, wrap); css(row, { justifyContent: 'space-between', alignItems: 'flex-end' });
  const a = el('div', 'num gold', '$0', row); css(a, { fontSize: '180px' });
  const b = el('div', 'num muted', '/ $' + fmt(goal), row); css(b, { fontSize: '86px', paddingBottom: '18px' });
  const bar = el('div', 'bar', null, wrap); css(bar, { height: '70px' });
  const fill = el('div', 'fill', null, bar);
  const pct = Math.round(have / goal * 100);
  const foot = el('div', 'row', null, wrap); css(foot, { justifyContent: 'space-between' });
  const l = el('div', 'body muted', '', foot);
  const r = el('div', 'body', autumn ? 'до NL25 с 40 БИ' : 'осталось <span class="gold">6 недель</span>', foot); css(r, { fontSize: '44px' });
  tl.from(wrap, { opacity: 0, y: 40, duration: .4, ease: 'expo.out' }, 0);
  countUp(tl, a, 0, have, .15, 1.1, { pre: '$' });
  tl.fromTo(fill, { width: '0%' }, { width: pct + '%', duration: 1.1, ease: 'power3.out' }, .15)
    .to(fill, { width: (pct + 1.5) + '%', duration: .12, yoyo: true, repeat: 3, ease: 'power1.inOut' }, 1.35)
    .from(r, { opacity: 0, y: 16, duration: .3 }, .9);
}};

// G10 — лестница бай-инов на NL10. ?hi=40|100|10 — какую ступень подсветить.
SCENES['G10-ladder'] = { dur: 6, build(root, tl, P) {
  const hi = P.hi || '40';
  const data = [['10 БИ', 100], ['40 БИ', 400], ['100 БИ', 1000]];
  const wrap = el('div', 'center col', null, root); css(wrap, { width: '1500px', gap: '26px' });
  const cap = el('div', 'cap', 'Банкролл для NL10 · бай-ин $10', wrap);
  const rows = data.map(([k, v]) => {
    const r = el('div', 'row', null, wrap); css(r, { gap: '30px' });
    const lab = el('div', 'title', k, r); css(lab, { width: '260px', fontSize: '54px', textAlign: 'right' });
    const bar = el('div', 'bar', null, r); css(bar, { flex: 1, height: '84px' });
    const f = el('div', 'fill', null, bar);
    const val = el('div', 'num', '$' + fmt(v), r); css(val, { width: '260px', fontSize: '64px' });
    const on = k.startsWith(hi + ' ');
    if (!on) { f.style.background = 'rgba(244,241,232,.28)'; lab.style.color = 'var(--muted)'; val.style.color = 'var(--muted)'; }
    else { val.classList.add('gold'); }
    return { r, f, v, on, lab };
  });
  tl.from(cap, { opacity: 0, y: 20, duration: .3 }, 0);
  rows.forEach((o, i) => {
    tl.from(o.r, { opacity: 0, x: -40, duration: .35, ease: 'power3.out' }, .1 + i * .12)
      .fromTo(o.f, { width: '0%' }, { width: (o.v / 1000 * 100) + '%', duration: .7, ease: 'power3.out' }, .2 + i * .12);
  });
  const hiRow = rows.find(o => o.on);
  if (hiRow) {
    const glow = el('div', 'abs', null, hiRow.r); css(glow, { inset: '-14px -20px', border: '4px solid var(--gold)', borderRadius: '22px' });
    tl.from(glow, { opacity: 0, scale: 1.05, duration: .3, ease: 'power3.out' }, 1.1);
  }
}};

// G23 — тезис. Используется дважды (главы 08 и 11).
SCENES['G23-thesis'] = { dur: 4, build(root, tl) {
  const wrap = el('div', 'center col', null, root); css(wrap, { alignItems: 'center', gap: '30px', width: '1600px', textAlign: 'center' });
  const a = el('div', 'display', 'Брови отрастают.', wrap); css(a, { fontSize: '128px' });
  const b = el('div', 'display gold', 'Банкролл — нет.', wrap); css(b, { fontSize: '128px' });
  tl.from(a, { opacity: 0, y: 30, duration: .5, ease: 'power3.out' }, .1)
    .from(b, { opacity: 0, y: 30, duration: .5, ease: 'power3.out' }, 1.0);
}};

// G25 — карточка цитаты (прозрачный фон). Параметры: who, date, text, kind=max|forum, tag=1.
SCENES['G25-quote'] = { dur: 2, alpha: true, build(root, tl, P) {
  const isMax = (P.kind || 'max') === 'max';
  // Полноширинная полоса снизу, карточка по центру: ширина по тексту, но не уже 760 и не шире 1500.
  const lane = el('div', 'abs', null, root); css(lane, { left: 0, right: 0, bottom: '110px', display: 'flex', justifyContent: 'center' });
  const card = el('div', null, null, lane);
  css(card, { minWidth: '760px', maxWidth: '1400px', padding: '34px 52px 40px', borderRadius: '26px', background: isMax ? 'rgba(244,241,232,.97)' : 'rgba(16,30,25,.94)',
    color: isMax ? 'var(--ink)' : 'var(--paper)', boxShadow: 'var(--shadow)', border: isMax ? '0' : '1.5px solid rgba(244,241,232,.18)' });
  const head = el('div', 'row', null, card); css(head, { gap: '16px', marginBottom: '14px' });
  const who = el('div', 'body', P.who || 'Максим', head); css(who, { fontSize: '30px', fontWeight: 800 });
  const date = el('div', 'body', P.date || '', head); css(date, { fontSize: '28px', fontWeight: 500, opacity: .6 });
  if (P.tag) { const t = el('div', 'chip', 'Озвучка сообщения', head); css(t, { marginLeft: 'auto', fontSize: '22px', padding: '8px 18px', background: isMax ? '#e4ddca' : 'rgba(244,241,232,.08)', color: isMax ? '#3b463f' : 'var(--paper)' }); }
  const txt = el('div', 'body', P.text || '', card);
  const len = (P.text || '').length;
  css(txt, { fontSize: (len < 40 ? 64 : len < 90 ? 54 : 46) + 'px', fontWeight: 700, lineHeight: 1.2 });
  tl.fromTo(card, { opacity: 0, y: 30, scale: .97 }, { opacity: 1, y: 0, scale: 1, duration: .3, ease: 'expo.out' }, 0);
}};

// G28 — счётчик «НО» (прозрачный). ?n=1..12 — угол; ?center=1 — крупно по центру (первый показ, красный); ?final=1 — по центру, зелёный (кульминация).
SCENES['G28-no'] = { dur: 1.2, alpha: true, build(root, tl, P) {
  const n = Number(P.n || 0);
  const big = P.final === '1' || P.center === '1';
  const green = P.final === '1';
  const box = el('div', 'abs row', null, root);
  css(box, big ? { left: '50%', top: '50%', transform: 'translate(-50%,-50%)', gap: '28px' } : { right: '56px', top: '44px', gap: '14px' });
  const st = el('div', 'stamp', 'НО', box); css(st, { fontSize: big ? '150px' : '44px', borderWidth: big ? '12px' : '5px', padding: big ? '14px 40px' : '2px 14px', borderRadius: big ? '26px' : '12px' });
  const c = el('div', 'num', '×' + n, box); css(c, { fontSize: big ? '170px' : '52px', color: green ? 'var(--mint)' : 'var(--paper)' });
  if (green) { st.style.color = 'var(--mint)'; st.style.borderColor = 'var(--mint)'; }
  if (big) tl.fromTo(box, { scale: .6, opacity: 0 }, { scale: 1, opacity: 1, duration: .35, ease: 'expo.out' }, 0);
}};
