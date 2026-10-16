// Сборка таймлайна по реальной озвучке. Нужен FFmpeg (ffmpeg и ffprobe в PATH).
//   node tools/timeline.mjs                    — базовая версия (ядро, тег core)
//   node tools/timeline.mjs --tags core,cut    — ядро + все опции
//   node tools/timeline.mjs --tags core,cut,res — со всем резервом
//   node tools/timeline.mjs --add 01-09,01-10 --drop 05-04  — точечно добавить/убрать реплики
//   node tools/timeline.mjs --trim             — срезать тишину по краям каждого клипа (порог −45 дБ)
//   node tools/timeline.mjs --edl-start 3600   — маркеры EDL с 01:00:00:00 (если таймлайн в Resolve начинается с часа)
// Результат в build/: vo-spine.wav (голос целиком, с паузами и «окнами» без голоса), timeline.csv,
// subtitles.srt, chapters.txt (главы для описания YouTube), markers-resolve.edl (маркеры DaVinci Resolve), report.txt.
// Реплики без mp3 подставляются оценкой по длине текста и помечаются в отчёте.
import { readFile, writeFile, mkdir, access, rm } from 'node:fs/promises';
import { resolve, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { spawnSync } from 'node:child_process';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const { values } = parseArgs({ options: {
  tags: { type: 'string', default: 'core' }, add: { type: 'string' }, drop: { type: 'string' },
  vo: { type: 'string', default: 'vo' }, out: { type: 'string', default: 'build' },
  lines: { type: 'string', default: '02-VO/vo-lines.json' }, trim: { type: 'boolean', default: false },
  target: { type: 'string', default: '960' }, cps: { type: 'string', default: '13' }, fps: { type: 'string', default: '30' },
  'no-wav': { type: 'boolean', default: false }, 'edl-start': { type: 'string', default: '0' },
} });
const EDL0 = Number(values['edl-start']) || 0;
const FPS = Number(values.fps);
const exists = async (p) => { try { await access(p); return true; } catch { return false; } };
const run = (cmd, args) => {
  const r = spawnSync(cmd, args, { encoding: 'utf8', maxBuffer: 64 << 20 });
  if (r.error) throw new Error(r.error.code === 'ENOENT' ? `Не найден «${cmd}» в PATH. Установите FFmpeg (нужны ffmpeg и ffprobe) и откройте новый терминал.` : `${cmd}: ${r.error.message}`);
  if (r.status !== 0) throw new Error(`${cmd} ${args.join(' ')}\n${r.stderr}`);
  return r.stdout;
};
const tc = (s) => { const ms = Math.round(s * 1000), h = Math.floor(ms / 3600000), m = Math.floor(ms % 3600000 / 60000), sec = Math.floor(ms % 60000 / 1000); return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')},${String(ms % 1000).padStart(3, '0')}`; };
const mmss = (s) => { s = Math.floor(s); const m = Math.floor(s / 60); return `${m}:${String(s % 60).padStart(2, '0')}`; };
const smpte = (s) => { const f = Math.round(s * FPS); const fr = f % FPS, t = Math.floor(f / FPS); return [Math.floor(t / 3600), Math.floor(t % 3600 / 60), t % 60, fr].map(n => String(n).padStart(2, '0')).join(':'); };

function splitSub(text, maxLen = 84) {
  // Делим длинную реплику на куски ≤ maxLen по границам предложений/запятых.
  if (text.length <= maxLen) return [text];
  const parts = text.split(/(?<=[.!?…:;])\s+|(?<=,)\s+|(?<=—)\s+/);
  const out = []; let cur = '';
  for (const p of parts) { if ((cur + ' ' + p).trim().length > maxLen && cur) { out.push(cur.trim()); cur = p; } else cur = (cur + ' ' + p).trim(); }
  if (cur) out.push(cur);
  return out;
}
function twoLines(s, max = 42) {
  if (s.length <= max) return s;
  const words = s.split(' '); let best = s, bestD = 1e9;
  for (let i = 1; i < words.length; i++) { const a = words.slice(0, i).join(' '), b = words.slice(i).join(' '); const d = Math.abs(a.length - b.length); if (d < bestD) { bestD = d; best = a + '\n' + b; } }
  return best;
}

async function main() {
  const all = JSON.parse((await readFile(resolve(ROOT, values.lines), 'utf8')).replace(/^\uFEFF/, ''));
  const list = (v) => (v || '').split(',').map(x => x.trim()).filter(Boolean);
  const tags = new Set(list(values.tags)), add = new Set(list(values.add)), drop = new Set(list(values.drop));
  const known = new Set(all.map(l => l.id));
  const unknown = [...add, ...drop].filter(id => !known.has(id));
  if (unknown.length) throw new Error('Нет реплик: ' + unknown.join(', '));
  const lines = all.filter(l => (tags.has(l.tag) || add.has(l.id)) && !drop.has(l.id));
  const outDir = resolve(ROOT, values.out); await mkdir(outDir, { recursive: true });
  const tmp = join(outDir, '_tmp'); await mkdir(tmp, { recursive: true });

  let t = 0; const rows = []; const missing = []; const concat = [];
  const silence = async (sec, name) => {
    const f = join(tmp, name); run('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i', 'anullsrc=r=48000:cl=mono', '-t', sec.toFixed(3), '-c:a', 'pcm_s24le', f]); return f;
  };
  for (const l of lines) {
    const mp3 = resolve(ROOT, values.vo, `${l.id}.mp3`);
    let dur, wav = null;
    if (await exists(mp3)) {
      wav = join(tmp, `${l.id}.wav`);
      const af = values.trim ? ['-af', 'silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.05,areverse,silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.08,areverse'] : [];
      if (!values['no-wav']) run('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-i', mp3, ...af, '-ar', '48000', '-ac', '1', '-c:a', 'pcm_s24le', wav]);
      dur = Number(run('ffprobe', ['-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', values['no-wav'] ? mp3 : wav]).trim());
    } else {
      dur = [...l.tts].length / Number(values.cps) + 0.2; missing.push(l.id);
      if (!values['no-wav']) wav = await silence(dur, `${l.id}.missing.wav`);
    }
    if (!Number.isFinite(dur) || dur < 0.05) throw new Error(`Реплика ${l.id}: длительность не определилась (${dur})${values.trim ? ' — после --trim остался пустой клип (тише −45 дБ), проверьте файл или запустите без --trim' : ''}.`);
    const start = t, end = t + dur;
    rows.push({ ...l, start, end, dur });
    if (wav) concat.push(wav);
    const gap = (l.pause || 0) + (l.win || 0);
    if (gap > 0 && !values['no-wav']) concat.push(await silence(gap, `${l.id}.gap.wav`));
    t = end + gap;
  }
  const total = t;

  // CSV
  const esc = (s) => '"' + String(s ?? '').replace(/"/g, '""') + '"';
  const num = (x, d = 2) => Number(x || 0).toFixed(d).replace('.', ',');  // запятая — чтобы русский Excel не превращал числа в даты
  const csv = ['id;chapter;role;start;end;start_tc;dur;pause;window;tag;text;gfx;vis;sfx;mus'].concat(rows.map(r =>
    [r.id, r.chapter, r.role_name, num(r.start), num(r.end), smpte(r.start), num(r.dur), num(r.pause, 1), num(r.win, 1), r.tag, esc(r.text), esc(r.gfx), esc(r.vis), esc(r.sfx), esc(r.mus)].join(';')));
  await writeFile(join(outDir, 'timeline.csv'), '\ufeff' + csv.join('\n'));

  // SRT — время куска пропорционально числу символов; мат — звёздочками, как в кадре (в голосе остаётся)
  const maskSwear = (t) => t.replace(/Пиздец/g, 'П****ц').replace(/Пиздюли/g, 'П****ли');
  let n = 0; const srt = [];
  for (const r of rows) {
    const chunks = splitSub(maskSwear(r.text)); const totalLen = chunks.reduce((s, c) => s + c.length, 0); let s0 = r.start;
    for (const c of chunks) { const d = r.dur * c.length / totalLen; srt.push(`${++n}\n${tc(s0)} --> ${tc(s0 + d)}\n${twoLines(c)}\n`); s0 += d; }
  }
  await writeFile(join(outDir, 'subtitles.srt'), srt.join('\n'));

  // Главы YouTube
  const chapters = []; const seen = new Set();
  for (const r of rows) if (!seen.has(r.chapter)) { seen.add(r.chapter); chapters.push(`${mmss(chapters.length ? r.start : 0)} ${r.chapter_title}`); }
  await writeFile(join(outDir, 'chapters.txt'), chapters.join('\n') + '\n');

  // Маркеры DaVinci Resolve (Timeline → Import → Timeline Markers from EDL)
  const edl = ['TITLE: SPLAT longform markers', 'FCM: NON-DROP FRAME', ''];
  rows.forEach((r, i) => {
    const a = smpte(EDL0 + r.start), b = smpte(EDL0 + r.start + 1 / FPS);
    const color = r.role === 'N' ? 'ResolveColorBlue' : r.role === 'M' ? 'ResolveColorYellow' : 'ResolveColorPink';
    const name = `${r.id} ${r.gfx ? '[' + r.gfx.split(' ')[0] + '] ' : ''}${r.text}`.replace(/\|/g, '/').slice(0, 120);
    edl.push(`${String(i + 1).padStart(3, '0')}  001      V     C        ${a} ${b} ${a} ${b}  `, ` |C:${color} |M:${name} |D:1`, '');
  });
  await writeFile(join(outDir, 'markers-resolve.edl'), edl.join('\n'));

  // WAV-«хребет» голоса
  if (!values['no-wav']) {
    const list = join(tmp, 'list.txt');
    await writeFile(list, concat.map(f => `file '${f.replace(/'/g, "'\\''")}'`).join('\n'));
    run('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', list, '-c:a', 'pcm_s24le', join(outDir, 'vo-spine.wav')]);
  }
  await rm(tmp, { recursive: true, force: true });

  // Отчёт и подсказки по хронометражу
  const target = Number(values.target);
  const est = (l) => [...l.tts].length / Number(values.cps) + 0.2 + (l.pause || 0) + (l.win || 0);
  const notIn = (l) => !rows.find(r => r.id === l.id);
  const opt = all.filter(l => l.tag === 'cut' && notIn(l));
  const res = all.filter(l => l.tag === 'res' && notIn(l));
  const dropC = [...rows.filter(r => r.tag === 'cut' && !add.has(r.id)), ...rows.filter(r => r.drop_candidate && r.tag !== 'cut')];
  const rep = [];
  rep.push(`Итоговая длина: ${mmss(total)} (${total.toFixed(1)} с) при цели ${mmss(target)}. Реплик: ${rows.length}.`);
  if (missing.length) rep.push(`Без озвучки (подставлена оценка по длине текста): ${missing.length} — ${missing.join(', ')}`);
  const diff = total - target;
  if (diff > 20) {
    rep.push(`Длиннее цели на ${diff.toFixed(0)} с. Кандидаты на сокращение (сначала опции, если они в выборке, затем отмеченные реплики ядра):`);
    let acc = 0; for (const r of dropC) { const g = r.dur + (r.pause || 0) + (r.win || 0); acc += g; rep.push(`  --drop ${r.id}  (−${g.toFixed(1)} с, итого −${acc.toFixed(0)} с): ${r.text}`); }
    const pz = rows.reduce((s, r) => s + (r.pause || 0), 0);
    rep.push(`  Ещё вариант: паузы между репликами сейчас ${pz.toFixed(0)} с суммарно — укоротить их на треть (поле pause в tools/script.src.txt, затем пересборка пакета) даст ~${(pz / 3).toFixed(0)} с. Окна без голоса (win) не трогать.`);
  } else if (diff < -20) {
    rep.push(`Короче цели на ${(-diff).toFixed(0)} с. Добавлять в таком порядке:`);
    let acc = 0;
    for (const l of [...opt, ...res]) { const has = l.tag === 'cut' ? 'уже озвучена' : 'озвучить: --go --only ' + l.id; acc += est(l); rep.push(`  --add ${l.id}  (+~${est(l).toFixed(1)} с, итого +${acc.toFixed(0)} с; ${l.tag === 'cut' ? 'опция' : 'резерв'}, ${has}): ${l.text}`); }
  } else rep.push('В пределах ±20 с от цели — можно монтировать.');
  rep.push('', 'Главы для описания:', ...chapters);
  await writeFile(join(outDir, 'report.txt'), rep.join('\n') + '\n');
  console.log(rep.join('\n'));
}
main().catch(async e => {
  console.error(e.message);
  await rm(join(resolve(ROOT, values.out), '_tmp'), { recursive: true, force: true }).catch(() => {});
  process.exitCode = 1;
});
