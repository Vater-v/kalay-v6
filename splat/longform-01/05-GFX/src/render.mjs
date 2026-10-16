// Рендер сцен графики в видео/картинки. Нужны Node 18+, Playwright (Chromium) и FFmpeg в PATH.
//   node render.mjs --job jobs.json                     — все задания из списка
//   node render.mjs --scene G01-bet --q "v=2" --out out/G01b.mp4
//   node render.mjs --scene G25-quote --q "who=Максим&text=..." --out out/q.png   (PNG = последний кадр, прозрачный)
// Форматы: .mp4 (H.264, без альфы), .mov (QuickTime/PNG с альфой), .png (один кадр, прозрачный для alpha-сцен).
import { createRequire } from 'node:module';
import { mkdtemp, rm, readFile, mkdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve, dirname, extname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { spawn } from 'node:child_process';
import { parseArgs } from 'node:util';

const HERE = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
let chromium;
try { ({ chromium } = require('playwright')); }
catch {
  try { ({ chromium } = require(process.env.PLAYWRIGHT_MODULE || '/opt/node22/lib/node_modules/playwright')); }
  catch { console.error('Не найден Playwright. В папке 05-GFX/src выполните: npm i playwright && npx playwright install chromium'); process.exit(1); }
}

const { values } = parseArgs({ options: {
  job: { type: 'string' }, scene: { type: 'string' }, q: { type: 'string', default: '' },
  out: { type: 'string' }, dur: { type: 'string' }, only: { type: 'string' },
} });

function ffmpeg(args) {
  return new Promise((ok, fail) => {
    const p = spawn('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', ...args], { stdio: 'inherit' });
    p.on('exit', (c) => c === 0 ? ok() : fail(new Error('ffmpeg exit ' + c)));
  });
}

async function renderOne(browser, job) {
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  const q = new URLSearchParams(job.q || '');
  q.set('scene', job.scene);
  if (job.dur) q.set('dur', String(job.dur));
  const url = pathToFileURL(join(HERE, 'index.html')).href + '?' + q.toString();
  await page.goto(url);
  await page.waitForFunction(() => window.__ready === true && window.__scene, null, { timeout: 20000 });
  const sc = await page.evaluate(() => window.__scene);
  const out = resolve(job.base || '.', job.out);
  await mkdir(dirname(out), { recursive: true });
  const ext = extname(out).toLowerCase();
  const shot = async (f, path) => {
    await page.evaluate((fr) => window.__setFrame(fr), f);
    // Непрозрачные кадры — JPEG (в разы быстрее PNG), прозрачные — PNG с альфой.
    if (path.endsWith('.jpg')) await page.screenshot({ path, type: 'jpeg', quality: 95 });
    else await page.screenshot({ path, omitBackground: sc.alpha, type: 'png' });
  };
  if (ext === '.png') {
    const f = job.frame != null ? job.frame : sc.frames - 1;
    await shot(f, out);
  } else {
    const dir = await mkdtemp(join(tmpdir(), 'splat-frames-'));
    const fx = sc.alpha || ext === '.mov' ? '.png' : '.jpg';
    for (let f = 0; f < sc.frames; f++) await shot(f, join(dir, String(f).padStart(5, '0') + fx));
    const input = ['-framerate', String(sc.fps), '-i', join(dir, '%05d' + fx)];
    // .mov — QuickTime с кодеком PNG: прозрачность, без потерь, компактно для графики; открывают Premiere, Resolve, After Effects, Final Cut.
    if (ext === '.mov') await ffmpeg([...input, '-c:v', 'png', '-pix_fmt', 'rgba', out]);
    else await ffmpeg([...input, '-c:v', 'libx264', '-preset', 'slow', '-crf', '17', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out]);
    await rm(dir, { recursive: true, force: true });
  }
  await page.close();
  return { out, frames: sc.frames, alpha: sc.alpha };
}

// Пути out в файле заданий считаются от папки, где лежит сам файл заданий.
const jobs = values.job
  ? JSON.parse(await readFile(values.job, 'utf8')).map(j => ({ ...j, base: dirname(resolve(values.job)) }))
  : [{ scene: values.scene, q: values.q, out: values.out, dur: values.dur }];
const filtered = values.only ? jobs.filter(j => new RegExp(values.only).test(j.out) || new RegExp(values.only).test(j.scene)) : jobs;
const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH || undefined, args: ['--font-render-hinting=none', '--disable-lcd-text'] });
for (const j of filtered) {
  const t0 = Date.now();
  const r = await renderOne(browser, j);
  console.log(`✓ ${j.scene} → ${r.out} (${r.frames} кадров, ${((Date.now() - t0) / 1000).toFixed(1)} с)`);
}
await browser.close();
