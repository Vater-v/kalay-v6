// Пакетная озвучка лонга по репликам: один запрос = одна реплика (одно-два предложения).
// По умолчанию — сухой прогон без запросов. Реальная генерация только с флагом --go.
//
//   node tools/tts-batch.mjs                         — сухой прогон: что и сколько уйдёт в синтез
//   node tools/tts-batch.mjs --go --only 00-01,00-02  — сгенерировать две реплики (проба голоса/ремарок)
//   node tools/tts-batch.mjs --go                    — ядро и опции (теги core,cut; в базовый монтаж входит только core)
//   node tools/tts-batch.mjs --go --only 03-07,08-01 --voices 02-VO/voices.candA.json --out vo-test-A — проба голосов
//   node tools/tts-batch.mjs --go --from 05-01        — продолжить с реплики 05-01
//
// Ключ: OPENROUTER_API_KEY в .env рядом с папкой проекта или в окружении. Ключ никогда не печатается.
// Готовые файлы не перезаписываются: чтобы переозвучить реплику, удалите vo/<id>.mp3 и vo/<id>.mp3.json.
// Автоповторов нет: при ошибке скрипт останавливается (флаг --keep-going — пропустить и идти дальше).
import { readFile, writeFile, mkdir, access } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { createHash } from 'node:crypto';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const ENDPOINT = 'https://openrouter.ai/api/v1/audio/speech';

const { values } = parseArgs({ options: {
  go: { type: 'boolean', default: false },
  tags: { type: 'string', default: 'core,cut' },
  only: { type: 'string' },
  from: { type: 'string' },
  limit: { type: 'string' },
  out: { type: 'string', default: 'vo' },
  lines: { type: 'string', default: '02-VO/vo-lines.json' },
  voices: { type: 'string', default: '02-VO/voices.json' },
  'keep-going': { type: 'boolean', default: false },
} });

function hideKey(s) {
  const k = process.env.OPENROUTER_API_KEY?.trim();
  return String(s).replaceAll(k || '\0', '[ключ скрыт]').replace(/sk-or-v1-[\w-]+/g, '[ключ скрыт]').slice(0, 600);
}
async function exists(p) { try { await access(p); return true; } catch { return false; } }

async function main() {
  // .env: OPENROUTER_API_KEY=... (process.loadEnvFile есть только в Node 20.12+, поэтому разбираем сами)
  // Понимает BOM, CRLF и UTF-16 (так сохраняет PowerShell 5.1), кавычки и комментарии после значения.
  for (const envPath of [resolve(ROOT, '.env'), resolve(ROOT, '..', '.env')]) {
    let buf; try { buf = await readFile(envPath); } catch { continue; }
    const txt = (buf[0] === 0xff && buf[1] === 0xfe ? buf.toString('utf16le') : buf.toString('utf8')).replace(/^\uFEFF/, '');
    for (const raw of txt.split(/\r?\n/)) {
      const m = raw.match(/^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$/);
      if (!m) continue;
      let v = m[2];
      if (v.length > 1 && (v[0] === '"' || v[0] === "'") && v.endsWith(v[0])) v = v.slice(1, -1); else v = v.replace(/\s+#.*$/, '');
      if (!(m[1] in process.env)) process.env[m[1]] = v;
    }
    break;
  }
  const lines = JSON.parse((await readFile(resolve(ROOT, values.lines), 'utf8')).replace(/^\uFEFF/, ''));
  let voicesPath = resolve(ROOT, values.voices);
  if (!(await exists(voicesPath))) {
    if (values.go) throw new Error(`Нет ${values.voices}. Скопируйте 02-VO/voices.example.json в voices.json и впишите голоса ролей.`);
    console.log(`Нет ${values.voices} — сухой прогон по 02-VO/voices.example.json.`);
    voicesPath = resolve(ROOT, '02-VO/voices.example.json');
  }
  const voices = JSON.parse((await readFile(voicesPath, 'utf8')).replace(/^\uFEFF/, ''));
  const tags = new Set(values.tags.split(',').map(s => s.trim()));
  let todo = lines.filter(l => tags.has(l.tag));
  if (values.only) {
    const ids = new Set(values.only.split(',').map(s => s.trim()).filter(Boolean));
    const unknown = [...ids].filter(id => !lines.some(l => l.id === id));
    if (unknown.length) throw new Error('Нет реплик: ' + unknown.join(', '));
    todo = lines.filter(l => ids.has(l.id));
  }
  if (values.from) { const i = todo.findIndex(l => l.id === values.from); if (i < 0) throw new Error('Нет реплики ' + values.from); todo = todo.slice(i); }
  if (values.limit) todo = todo.slice(0, Number(values.limit));
  const outDir = resolve(ROOT, values.out);

  const plan = [];
  for (const l of todo) {
    const role = voices.roles[l.voice] || {};
    const unset = !role.voice || /ВПИШИТЕ|PUT_/i.test(role.voice);
    if (unset && values.go) throw new Error(`В ${values.voices} не задан голос для роли «${l.voice}» (реплика ${l.id}).`);
    const input = `${l.direction}\n${l.tts}`;
    const file = resolve(outDir, `${l.id}.mp3`);
    plan.push({ l, role, input, file, done: await exists(file) });
  }
  const pending = plan.filter(p => !p.done);
  const chars = pending.reduce((s, p) => s + [...p.input].length, 0);
  console.log(`Реплик в выборке: ${plan.length}. Уже готово: ${plan.length - pending.length}. К синтезу: ${pending.length} (${chars} символов вместе с ремарками).`);
  if (!values.go) {
    for (const p of pending.slice(0, 12)) console.log(`  ${p.l.id} [${p.l.voice} → ${/ВПИШИТЕ|PUT_/i.test(p.role.voice || 'ВПИШИТЕ') ? 'голос не задан' : p.role.voice}] ${p.input.replace(/\n/g, ' ⏎ ')}`);
    if (pending.length > 12) console.log(`  … и ещё ${pending.length - 12}`);
    console.log('Это сухой прогон. Для генерации добавьте --go.');
    return;
  }
  await mkdir(outDir, { recursive: true });
  const key = process.env.OPENROUTER_API_KEY?.trim();
  if (!key || /^<|^your|^replace|^paste/i.test(key)) throw new Error('OPENROUTER_API_KEY не найден.');
  const model = voices.model;
  const cat = await fetch('https://openrouter.ai/api/v1/models?output_modalities=speech', { signal: AbortSignal.timeout(20000) });
  if (!cat.ok) throw new Error('Не удалось проверить модель: HTTP ' + cat.status);
  if (!(await cat.json()).data.find(m => m.id === model)) throw new Error(`Модели ${model} нет в каталоге OpenRouter.`);

  let ok = 0;
  for (const p of pending) {
    const body = { model: p.role.model || model, voice: p.role.voice, input: p.input, response_format: 'mp3' };
    if (voices.provider) body.provider = voices.provider;
    process.stdout.write(`→ ${p.l.id} (${p.l.voice}) … `);
    try {
      const r = await fetch(ENDPOINT, { method: 'POST', redirect: 'error', signal: AbortSignal.timeout(240000),
        headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      if (!r.ok) { const j = await r.json().catch(() => ({})); throw new Error(`HTTP ${r.status}: ${hideKey(j.error?.message || 'ошибка синтеза')}`); }
      const mime = r.headers.get('content-type') || '';
      if (!/^audio\/mpeg/i.test(mime)) throw new Error('Ожидался MP3, пришло ' + hideKey(mime));
      const buf = Buffer.from(await r.arrayBuffer());
      const sig = buf.subarray(0, 3).toString() === 'ID3' || (buf[0] === 0xff && (buf[1] & 0xe0) === 0xe0);
      if (buf.length < 1000 || !sig) throw new Error('Ответ не похож на MP3, файл не сохранён.');
      await writeFile(p.file, buf, { flag: 'wx' });
      await writeFile(p.file + '.json', JSON.stringify({ id: p.l.id, role: p.l.voice, model: body.model, voice: body.voice,
        input: p.input, inputSha256: createHash('sha256').update(p.input).digest('hex'), bytes: buf.length,
        generationId: r.headers.get('x-generation-id'), generatedAt: new Date().toISOString(), status: 'не прослушано' }, null, 2), { flag: 'wx' });
      ok++; console.log(`ok (${buf.length} байт)`);
    } catch (e) {
      console.log('ОШИБКА'); console.error('  ' + hideKey(e.message));
      if (!values['keep-going']) { console.error('Остановлено. Автоповтора нет — проверьте историю OpenRouter перед новым запуском.'); process.exitCode = 1; break; }
    }
  }
  console.log(`Готово новых файлов: ${ok}. Дальше: node tools/timeline.mjs`);
}
main().catch(e => { console.error(hideKey(e.message + (e.cause?.code ? ` (${e.cause.code})` : ''))); process.exitCode = 1; });
