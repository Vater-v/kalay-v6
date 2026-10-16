/* Обложки 16:9 (рендер 1920×1080 → уменьшение до 1280×720). Портрет 24 июля — реальное фото из поста T822. */
function thumbBase(root, side) {
  const base = el('div', 'abs', null, root); css(base, { inset: 0, background: '#07100d' });
  const ph = el('div', 'abs', null, root);
  css(ph, { top: 0, bottom: 0, width: '1180px', [side]: 0, backgroundImage: 'url(assets/portrait-0724.jpg)', backgroundSize: '1240px auto', backgroundPosition: '50% 26%', filter: 'saturate(.92) contrast(1.05)' });
  // Фокус на лице: затемняем стену по краям фото и плавно уводим фото в фон слева.
  const vig = el('div', 'abs', null, root);
  css(vig, { top: 0, bottom: 0, width: '1180px', [side]: 0, background: 'radial-gradient(760px 900px at 56% 44%, rgba(7,16,13,0) 0%, rgba(7,16,13,0) 50%, rgba(7,16,13,.55) 100%)' });
  const fade = el('div', 'abs', null, root);
  const dir = side === 'right' ? 'to right' : 'to left';
  css(fade, { top: 0, bottom: 0, width: '1180px', [side]: 0, background: `linear-gradient(${dir}, #07100d 0%, rgba(7,16,13,.6) 16%, rgba(7,16,13,0) 36%)` });
  return ph;
}
SCENES['T01-thumb'] = { dur: 1, build(root, tl) {
  thumbBase(root, 'right');
  const box = el('div', 'abs col', null, root); css(box, { left: '90px', top: '170px', gap: '6px' });
  const a = el('div', 'num gold', '$10 000', box); css(a, { fontSize: '186px' });
  const b = el('div', 'display', 'ИЛИ', box); css(b, { fontSize: '120px', color: 'var(--paper)', marginTop: '18px' });
  const c = el('div', 'display', 'БРОВИ', box); css(c, { fontSize: '196px', color: 'var(--red)' });
}};
SCENES['T02-thumb'] = { dur: 1, build(root, tl) {
  thumbBase(root, 'right');
  const box = el('div', 'abs col', null, root); css(box, { left: '90px', top: '250px', gap: '24px' });
  const a = el('div', 'display', 'СБРИЛ', box); css(a, { fontSize: '200px' });
  const b = el('div', 'display gold', 'САМ', box); css(b, { fontSize: '200px' });
  const c = el('div', 'title', 'за месяц до дедлайна', box); css(c, { fontSize: '58px', color: 'var(--paper)', opacity: .9 });
}};
SCENES['T03-thumb'] = { dur: 1, build(root, tl) {
  thumbBase(root, 'right');
  const box = el('div', 'abs col', null, root); css(box, { left: '90px', top: '200px', gap: '10px' });
  const a = el('div', 'display', 'ПОСПОРИЛ', box); css(a, { fontSize: '150px' });
  const b = el('div', 'display', 'НА БРОВИ', box); css(b, { fontSize: '150px', color: 'var(--red)' });
  const c = el('div', 'num gold', '$10 000', box); css(c, { fontSize: '150px', marginTop: '30px' });
}};
