"""Собирает пакет лонга из script.src.txt: сценарий, реплики для TTS, монтажный план, задания для графики.
Запуск: python3 build_package.py <script.src.txt> <папка_пакета>"""
import json, re, sys, os, csv, io
sys.dont_write_bytecode = True  # без __pycache__ в пакете
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')  # Windows: вывод в cp1251 не падает на «≈»
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parse

if len(sys.argv) != 3:
    raise SystemExit('Использование: python3 tools/build_package.py tools/script.src.txt .   (Windows: py -3 tools\\build_package.py tools\\script.src.txt .)')
SRC, OUT = sys.argv[1], sys.argv[2]
CPS = 13.0
chs = parse.parse(SRC)
lines = [l for c in chs for l in c['lines']]
for l in lines:
    l['tts'] = l['tts'].replace('Боссбайден', 'Босс Байден')
for c in chs:
    for l in c['lines']:
        l['chapter'] = c['num']; l['chapter_title'] = c['title']; l['chapter_when'] = c['when']; l['chapter_music'] = c['music']

def est(l):
    return len(l['tts']) / CPS + 0.2 + l['pause'] + l['win']

def mmss(s):
    s = int(round(s)); return f'{s // 60}:{s % 60:02d}'

# Оценочное время (базовая версия = только core; cut и res в таймлайн не входят)
t = 0.0
for l in lines:
    l['est_start'] = t if l['tag'] == 'core' else None
    if l['tag'] == 'core':
        t += est(l)
TOTAL = t
bad = [l['id'] for l in lines if l['sentences'] > 2]
if bad:
    raise SystemExit('больше двух предложений в репликах: ' + ', '.join(bad))

def src_url(tok):
    tok = tok.strip()
    m = re.match(r'P(\d{7})$', tok)
    if m: return f'https://forum.gipsyteam.ru/index.php?viewtopic=172597&view=findpost&p={m.group(1)}'
    m = re.match(r'T(\d+)$', tok)
    if m: return f'https://t.me/logovo_shegla/{m.group(1)}'
    if tok.startswith('attach'):
        return 'https://forum.gipsyteam.ru/index.php?act=attach&type=post&id=' + tok.split()[-1]
    return ''

ROLE_LABEL = {'N': 'Рассказчик', 'M': 'Максим (чтение поста)'}
def role_label(l):
    return ROLE_LABEL.get(l['role'], 'Форум · ' + l['role'][2:])
TAG_LABEL = {'core': '', 'cut': '＋ опция (озвучивается сразу; в монтаж — если ролик короче 16:00)', 'res': '＋＋ резерв (не озвучивается по умолчанию)'}

def w(path, text):
    full = os.path.join(OUT, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)

# ---------- 02-VO/vo-lines.json ----------
vo = []
for l in lines:
    m = l['meta']
    vo.append(dict(id=l['id'], chapter=l['chapter'], chapter_title=l['chapter_title'], tag=l['tag'],
                   role=l['role'] if l['role'] in ('N', 'M') else 'F', role_name=role_label(l), voice=l['voice'],
                   dir=l['dir'], direction=parse.DIRS[l['dir']][0], text=l['text'], tts=l['tts'],
                   pause=l['pause'], win=l['win'], win_desc=l['win_desc'], no_counter=l['no_n'], drop_candidate=bool(l['meta'].get('drop')),
                   gfx=m.get('gfx', ''), vis=m.get('vis', ''), mot=m.get('mot', ''), txt=m.get('txt', ''),
                   sfx=m.get('sfx', ''), mus=m.get('mus', ''), src=m.get('src', ''), note=m.get('note', '')))
w('02-VO/vo-lines.json', json.dumps(vo, ensure_ascii=False, indent=1))
buf = io.StringIO(); cw = csv.writer(buf, delimiter='\t', lineterminator='\n')
cw.writerow(['id', 'тег', 'роль', 'ремарка', 'текст для синтеза', 'текст на экране/субтитры', 'пауза после, с', 'окно без голоса, с'])
for v in vo:
    cw.writerow([v['id'], v['tag'], v['role_name'], v['direction'], v['tts'], v['text'], v['pause'], v['win']])
w('02-VO/vo-lines.tsv', '﻿' + buf.getvalue())

# ---------- 01-SCRIPT.md ----------
out = []
core_words = sum(parse.words(l['text']) for l in lines if l['tag'] == 'core')
out.append('# Брови на кону\n')
out.append('*Лонг СПЛАТ · история Максима (Shegol111), апрель — октябрь 2026 · редакция v3, 16 октября 2026*\n')
def plural(n, one, few, many):
    n = abs(n) % 100
    return many if 11 <= n <= 14 else one if n % 10 == 1 else few if 2 <= n % 10 <= 4 else many
out.append(f'Базовая версия: **{core_words} {plural(core_words, "слово", "слова", "слов")}**, оценка **≈ {mmss(TOTAL)}** при темпе голоса 13 символов/с '
           f'(реальный темп измерит `tools/timeline.mjs` после озвучки). Реплики «＋ опция» озвучиваются вместе с основой и добавляются первыми, если ролик выйдет короче 16:00; '
           '«＋＋ резерв» — второй эшелон. Роли, ремарки и подписи не озвучиваются.\n')
out.append('Голоса: **Рассказчик** — основной; **Максим** — чтение его постов (другой голос, не имитация); **Форум** — ответы читателей. '
           'Счётчик «НО» в углу кадра прибавляет единицу на каждом «но» Максима (⓪ → ⑩; ⑪ — только если в монтаж входит опция 10-14).\n')
for c in chs:
    st = next((l['est_start'] for l in c['lines'] if l['est_start'] is not None), 0)
    out.append(f"\n## {c['num']} · {c['title']}\n")
    out.append(f"*{c['when']} · ≈ {mmss(st)} · музыка: {c['music']}*\n")
    for l in c['lines']:
        tag = TAG_LABEL[l['tag']]
        tagtxt = f' · {tag}' if tag else ''
        no = f' · **НО ×{l["no_n"]}**' if l['no_n'] else ''
        if l['role'] == 'N':
            out.append(f"`{l['id']}`{tagtxt}{no}  \n{l['text']}\n")
        else:
            out.append(f"`{l['id']}` **{role_label(l)}**{tagtxt}{no}  \n> {l['text']}\n")
        if l['win']:
            out.append(f"*[окно без голоса {l['win']:g} с — {l['win_desc']}]*\n")
# Источники
toks = []
for l in lines:
    for tok in re.split(r',\s*', l['meta'].get('src', '')):
        tok = tok.strip()
        if tok and tok not in toks: toks.append(tok)
out.append('\n---\n\n## Первоисточники (для редактора, не озвучивать)\n')
for tok in toks:
    u = src_url(tok)
    out.append(f'- {tok}: {u}' if u else f'- {tok}')
out.append('\nДословные тексты всех постов — в `07-RESEARCH/forum-posts/` и `07-RESEARCH/forum-posts/telegram-posts.json`.\n')
w('01-SCRIPT.md', '\n'.join(out))

# ---------- 03-EDIT/EDIT-PLAN.md + edit-plan.csv ----------
ep = ['# Монтажный план по репликам\n',
      'Время — оценка (13 симв/с) для базовой версии; точное даст `tools/timeline.mjs` по реальной озвучке. '
      'Обозначения: **Кадр** — что на экране; **Графика** — готовый файл из `05-GFX/render/` (см. `05-GFX/GFX-INDEX.md`); '
      '**Движение** — камера/анимация; **Звук** — эффекты из `04-SOUND/sfx/` (см. `04-SOUND/SOUND-PLAN.md`); **Музыка** — смена подложки.\n',
      'Общие правила кадра — в `03-EDIT/EDIT-RULES.md`. Пост на экране — всегда крупный кроп нужной строки на фоне `G00-bg`, никогда не весь форум мелко.\n']
for c in chs:
    ep.append(f"\n## {c['num']} · {c['title']} — {c['when']}\n\n**Музыка главы:** {c['music']}\n")
    for l in c['lines']:
        m = l['meta']
        tm = f"≈ {mmss(l['est_start'])}" if l['est_start'] is not None else ('опция' if l['tag'] == 'cut' else 'резерв')
        tag = TAG_LABEL[l['tag']]
        head = f"**{l['id']}** · {tm} · {role_label(l)}" + (f' · {tag}' if tag else '') + (' · ⤵ кандидат на сокращение' if l['meta'].get('drop') else '')
        ep.append(f"{head}  \n«{l['text']}»")
        rows = [('Кадр', m.get('vis')), ('Графика', m.get('gfx')), ('Текст в кадре', m.get('txt')), ('Движение', m.get('mot')),
                ('Звук', m.get('sfx')), ('Музыка', m.get('mus')), ('Источник', ', '.join(f'{t} {src_url(t)}'.strip() for t in re.split(r',\s*', m.get('src', '')) if t.strip()) or None),
                ('Окно без голоса', f"{l['win']:g} с — {l['win_desc']}" if l['win'] else None), ('Внимание', m.get('note')),
                ('Счётчик НО', f"тик → ×{l['no_n']}" if l['no_n'] else None)]
        for k, v in rows:
            if v: ep.append(f'- {k}: {v}')
        if l['role'] != 'N' and not m.get('gfx', '').startswith('G25'):
            ep.append('- Карточка цитаты: файл из `05-GFX/render/quotes/` с этим id.')
        ep.append('')
w('03-EDIT/EDIT-PLAN.md', '\n'.join(ep))
buf = io.StringIO(); cw = csv.writer(buf, delimiter=';', lineterminator='\n')
cw.writerow(['id', 'тег', 'старт_оценка', 'роль', 'текст', 'кадр', 'графика', 'текст_в_кадре', 'движение', 'звук', 'музыка', 'окно_с', 'источник', 'внимание'])
for l in lines:
    m = l['meta']
    cw.writerow([l['id'], l['tag'], mmss(l['est_start']) if l['est_start'] is not None else '', role_label(l), l['text'], m.get('vis', ''), m.get('gfx', ''), m.get('txt', ''),
                 m.get('mot', ''), m.get('sfx', ''), m.get('mus', ''), str(l['win']).replace('.', ',') if l['win'] else '', m.get('src', ''), m.get('note', '')])
w('03-EDIT/edit-plan.csv', '﻿' + buf.getvalue())

# ---------- задания на карточки цитат ----------
DATE_BY_SRC = {}
jobs = []; first = True
last_card = {}
for l in lines:
    if l['role'] == 'N':
        continue
    g = l['meta'].get('gfx', '')
    mm = re.search(r'«([^»]+?) · ([^»]+?)»', g) if g.startswith('G25') or 'карточка' in g else None
    spk = 'Максим' if l['role'] == 'M' else l['role'][2:]
    if mm:
        who, date = mm.group(1), mm.group(2)
        last_card[(l['chapter'], spk)] = (who, date)
    else:
        who, date = last_card.get((l['chapter'], spk), (spk, l['chapter_when']))
    if l['role'] != 'M':
        who = 'Читатель форума'
    shown = l['text'].replace('Пиздец', 'П****ц').replace('Пиздюли', 'П****ли')
    q = dict(who=who, date=date, text=shown, kind='max' if l['role'] == 'M' else 'forum')
    if first:
        q['tag'] = '1'; first = False
    from urllib.parse import urlencode
    jobs.append(dict(scene='G25-quote', q=urlencode(q), out=f'render/quotes/{l["id"]}-quote.png'))
for n in range(0, max(l['no_running'] for l in lines) + 1):
    jobs.append(dict(scene='G28-no', q=f'n={n}', out=f'render/no-counter/no-{n:02d}.png'))
w('05-GFX/jobs-quotes.json', json.dumps(jobs, ensure_ascii=False, indent=1))

# ---------- 03-EDIT/SOURCES-SHOTLIST.md ----------
NOTES = {
 'P8212904': 'Пост 9 апреля «Определение векторов и выход в новую жизнь». Кропы: абзац про видео; список «1. Дота 2 / 2. Проект своей жизни / 3. Поддержание своей жизнедеятельности»; «Аж 89 игр за 2 недели…»; «Получилось 48 страниц А4, 11-м шрифтом»; «за последние два месяца прилично так БРа потратил». Блюр: ничего особенного.',
 'attach 1510936': 'Картинка-анкета из поста 9 апреля (20 вопросов). Кропы только: шапка с правилом «без НЕ и НО», вопрос №9 (спальни, санузлы, площадь кухни), вопрос №12 (режим сна). Не показывать целиком.',
 'P8220266': 'Пост 23 апреля «Новый рывок». Кропы: «Я ЗАПУСКАЮ БЕСПЛАТНЫЙ КУРС ДЛЯ МИКРОЛИМИТОВ»; «Но вселенная найдет способ как меня вознаградить за это»; «После чего мне предложили тренировать группу в бекинге и стать тренером»; «если этот курс пройдет плохо, то я точно уйду с покера» (резерв). Обложку книги и графики профита крупно не показывать.',
 'P8225923': 'Читатель, 3 мая: «В чем прикол лудомарафона с шотами при БР в 10БИ?» и «…играя с 40 би на лимит и по 50 часов в неделю». Ник и аватар закрыть.',
 'P8225969': 'Максим, 4 мая: «устроит ответ "Потому что хочу"?» и «У меня жизнь тоже есть».',
 'P8226092': 'Читатель, 4 мая: «Да устроит. Твой же блог.» и «для многих это будет явно отрицательный опыт». Ник закрыть.',
 'P8226097': 'Максим, 4 мая: «Для меня 30 часов в неделю - это очень редкий случай за всю жизнь».',
 'P8233891': 'Отчёт 17 мая. Кропы: список причин «-Лень. -Устал. -Много дел. -Аниме давно не смотрел.»; «сейчас нахожусь в режиме ноу-лайф… я конечно гуляю, могу полежать на диване»; «толи рамка не та, толи музыка». Блюр: «ПОКЕРОК» в абзаце про 18+.',
 'P8237601': 'Отчёт 24 мая. Кропы: «записано еще 4-6 видео, которые я могу просто выложить и даже не монтировать. Но под последним видосом я получил 4 реакции)»; «люди бесплатное не ценят) Справедливо, я такой же)»; график 4/3 (резерв).',
 'P8237784': 'Читатель, 24 мая: «Нам тиктокерам тяжело смотреть видео по 2 часа… Или добавить фоном как мыло режешь.» Ник закрыть.',
 'P8237900': 'Ответ Максима: «я что-то так посмеялся с твоего.» + строка из 29 скобок — показать целиком, это шутка.',
 'P8252365': 'Пост 19 июня (разбор). Кропы: «Каждый раз, как только я выкладывал свои резы, у меня сразу шли минуса»; «Исключений из правил не было пока у меня)»; «на нл100 мы инвестировали уже 1200$»; «ГСЧ посчитает, что мы уже достаточно рыб прикормили». Скриншоты раздач из клиента рума НЕ показывать.',
 'P8252714': 'Пост 20 июня «День 34-й», спойлер «минусовые». Кроп подписи «Боссбайден на стриме просил на стек блефануть». Первый скриншот после подписи — источник карт для перерисовки G29 (карты Максима 6-3, борд, флеш соперника). Сам скриншот клиента в ролик не ставить.',
 'P8265329': 'Итоги марафона 14 июля. Кропы текста: «ЕВ винрейт составил 0 ЕВ ББ/100», «итого общий $час составил 10$/час», «сыгранно было 316к рук», прогноз «2,24$/час… 2,64$/час». Блюр: «покерок», «коин», «ПокерОК». Графики трекера с названиями румов не показывать.',
 'P8265347': 'Читатель, 14 июля: «Сложно, сложно, много букв, расчетов, но лайк поставил». Ник закрыть.',
 'P8267284': 'Пост 18 июля о пари. Кропы: «Как-то раз на стриме мы с Боссбайденом зарубились…», «Боссбайдену показалось, что брови забыл сбрить)», «я совсем забыл про столь значимый челенж))», «Сейчас на балансе 1686$. Чуть усложню себе…», «мне сложно разговаривать и думать». Фото из поста (налысо, с бровями) — можно. Блюр: «коин».',
 'P8267542': 'Комментарий 19 июля «Пиздец бровям». Ник закрыть; на экране слово частично звёздочками.',
 'P8267565': 'Ответ Максима «оп, хэйтеры подъехали, а я заждался».',
 'P8267908': 'Читатель, 19 июля: «Выложи фото с бровями, может мы их пследний раз видим😁😁😁». Ник закрыть.',
 'P8268131': 'Фото-ответ «вот ж они)» — цитирует фото с бровями из поста 18 июля.',
 'P8294413': 'Пост 10 сентября. Заголовок — первое слово закрыть плашкой. Кропы: «каждое мое лето знаменуется значимыми изменениями… и какой-то дичью» (резерв), «нашли билеты во Вьетнам по 8500 рублей, решили нажать кол», «данный пост может прочитать моя мама», «влез долги на 4500$», «Под 7000$)», «Но я не отчаиваюсь, ибо у меня есть слабоумие и отвага…», «БРМ 100БИ на лимит». Селфи на пляже — можно. Блюр: «ПокерОК», «ГГ».',
 'P8297903': 'Отчёт 17 сентября: «Первые три гонки», «38 часов», «ТОП-1 на нл10 взять) Первый раз в жизни», «БР составил 585$», «мисклики, которые стоили пару стеков».',
 'P8304987': 'Отчёт 1 октября. Кропы: «Хату надо будет платить + визаран», «чисто хожу и питаюсь в местные харчевни», «если бы я жил у себя дома и готовил бы сам себе еду…», «я сам себе в этот момент вырыл яму и пошел играть нл25», «Каждая раздача на стек сопровождалась внутренним тильтом… минус 5,5 стеков, закрыл столы, даже гонку доигрывать не стал», «Но благо хоть стопнул)», «Хз, стоит ли писать об этом)) Но я прошел обучение целительству))». Фото: кресло на скутере («Это я стул на мотике везу)»). Блюр: «старзы», «покерок», название платёжной системы; чужие лица не крупно.',
 'P8305057': 'Читатель, 1 октября: «лечение -отличная идея иногда хочется приложить подорожник к БРу! )». Ник закрыть.',
 'T772': 'Telegram 19 июня — заголовок «Пиздюли и закономерности)». На экране частично звёздочками.',
 'T777': 'Telegram 22 июня «День 35. Го руму рейка сгенерим…», «Буду играть 18 столов без селекта».',
 'T782': 'Telegram 24 июня «А так 20к рук за день наиграть, у меня такого опыта еще не было)».',
 'T791': 'Telegram 29 июня «Любимый покер, чел с отрицательным винрейтом стрим подрубил)» (можно вырезать).',
 'T794': 'Telegram 30 июня — «Финальный урок курса уже на канале!».',
 'T817': 'Telegram 14 июля «Марафон окончен». Блюр: «покерок», «коин».',
 'T822': 'Telegram 24 июля — текст признания + портрет без бровей (главное фото ролика). Блюр в P.s.: «коин». Портрет уже лежит в пакете: 05-GFX/src/assets/portrait-0724.jpg (1500×2000) — нужен в хуке (00-01, 00-02, 00-08), в главе 08 и в финале.',
 'T825': 'Telegram 27 июля «Чел без бровей стрим подрубил» (резерв).',
 'T826': 'Telegram 27 июля «не успеваю решения принимать на стриме за 12-ти столами, поэтому офнул стрим» (резерв).',
 'T835': 'Telegram 11 сентября «Начнем пробежку по лимитам)».',
 'T837': 'Telegram 12 сентября «ДЕП + ДОДЕП»: «хз как люди со 100БИ на лимит живут… БРМ = 40БИ… особенно после 10БИ на лимит😉… Главное во время переобуться😅».',
 'T842': 'Telegram 4 октября: «Провел провод себе с интернетом к компьютеру».',
 'T843': 'Telegram 4 октября, название стрима «С $400 ДО НОРМАЛЬНОГО БАНКРОЛЛА».',
}
used = {}
for l in lines:
    for tok in re.split(r',\s*', l['meta'].get('src', '')):
        tok = tok.strip()
        if tok: used.setdefault(tok, []).append(l['id'])
sl = ['# Что собрать для монтажа: скриншоты, кадры, фото\n',
      'Скриншоты постов снимать в браузере шириной 1600–1920 px, масштаб 125–150%, чтобы нужная строка в кадре была крупной. Ссылки ниже — на конкретный пост. '
      'Дословные тексты всех постов уже лежат в `07-RESEARCH/forum-posts/` — по ним удобно найти нужную строку.\n',
      '## Видео со стримов (VK, публичные записи канала героя)\n',
      '| Что | Запись | Таймкод | Как использовать |', '|---|---|---|---|',
      '| Лицо с бровями, 18 июля | https://live.vkvideo.ru/logovoshegla/record/9c0ac620-c003-4870-9834-fe35e282d175 | ~4:58 | Только вебка (кроп лица), 2–4 с. Хук и раскрытие в главе 08 |',
      '| Лицо без бровей, 27 июля | https://live.vkvideo.ru/logovoshegla/record/c57cd1f1-55ba-4d39-9801-ff8c1fb3a848 | ~15:01 | Только вебка; резервный блок 08 и финал |',
      '| Подсчёт итогов, 14 июля | https://live.vkvideo.ru/logovoshegla/record/0810b043-860d-49f5-8e7b-5060a5344491 | ~7:06:40 и ~7:08:20 | Кроп на таблицу и калькулятор, без окна трекера |',
      '| Четыре стола, 22 июня (необязательно) | https://live.vkvideo.ru/logovoshegla/record/b1a0c21f-7ebd-45fa-b53a-b0db9bba5e70 | ~1:00:00 | Только если сознательно рискуете 18+. По умолчанию — графика G14a |',
      '\nСкачивать и использовать — в объёме, нужном для истории (секунды, не минуты), с комментарием поверх. Звук стримов не использовать.\n',
      '## Посты и картинки\n']
for tok, ids in used.items():
    u = src_url(tok)
    chs_used = sorted({i.split('-')[0] for i in ids})
    sl.append(f"### {tok}\n{u}\n\nГлава: {', '.join(chs_used)} · первая реплика с этим источником: {ids[0]}\n\n{NOTES.get(tok, '')}\n")
sl.append('## Своё видео\n\n- ASMR-нарезка мыла, 2 с (реплика с советом про мыло): снять самим на телефон крупным планом (кусок мыла + нож, хруст) или взять сток с лицензией на коммерческое использование.\n')
w('03-EDIT/SOURCES-SHOTLIST.md', '\n'.join(sl))

# ---------- задания на полноэкранную графику и индекс ----------
from urllib.parse import urlencode
SJ = []
def find(prefix):
    for l in lines:
        if l['text'].startswith(prefix): return l['id']
    raise SystemExit('нет реплики: ' + prefix)
def job(scene, out, q=None, dur=None, frame=None, at=None):
    j = dict(scene=scene, out='render/' + out)
    if at: j['lines'] = at
    if q: j['q'] = urlencode(q)
    if dur: j['dur'] = dur
    if frame is not None: j['frame'] = frame
    SJ.append(j)
job('G00-bg', 'G00-bg.mp4', at=['под все скриншоты постов'])
job('G01-bet', 'G01-bet.mp4', at=[find('Он поспорил со зрителем')]); job('G01-bet', 'G01-bet-two-sides.mp4', {'v': '2'}, at=[find('Будет десять тысяч на балансе')])
job('G02-balance', 'G02-balance.mp4')
job('G03-title', 'G03-title.mp4', at=[find('Там будут пятьдесят')] )
for c in chs:
    if c['num'] == '00':
        continue
    job('G04-chapter', f"G04-chapter-{c['num']}.mp4", {'n': c['num'], 't': c['title'].replace('«', '').replace('»', ''), 'd': c['when']}, at=[c['lines'][0]['id']])
job('G05-timeline-full', 'G05-timeline-full.mp4')
job('G06-dota', 'G06-dota.mp4'); job('G07-sleep', 'G07-sleep.mp4'); job('G08-pages', 'G08-pages.mp4', at=[find('Доту он потом удалил')]); job('G08-pages', 'G08-pages-dip.mp4', {'dip': '1'}, dur=4, at=[find('Причём одиннадцатым')])
job('G09-limits', 'G09-limits.mp4', at=[find('Лимиты называются')]); job('G09-limits', 'G09-limits-nl100.mp4', {'hi': 'NL100'}, at=[find('Июнь. Марафон')]); job('G09-limits', 'G09-limits-arrow.mp4', {'hi': 'NL100', 'arrow': '1'}, at=[find('На тех же ставках')])
job('G10-ladder', 'G10-ladder-10.mp4', {'hi': '10'}, at=[find('Банкролл — это деньги')]); job('G10-ladder', 'G10-ladder-40.mp4', {'hi': '40'}, at=[find('Читатель предлагает')]); job('G10-ladder', 'G10-ladder-100.mp4', {'hi': '100'}, at=[find('Бровей на кону больше нет')])
for g in ['G11-variance', 'G12-x10', 'G13-rakeback', 'G14b-18tables', 'G15-live', 'G16-marathon', 'G18-hour', 'G19-pace', 'G21-split', 'G22-map', 'G23-thesis', 'G32-cta', 'G35-race', 'G38-ludka']:
    job(g, g + '.mp4')
job('G24-end', 'G24-end.mp4', at=[find('И расскажите про самый')]); job('G34-strike', 'G34-strike.mp4', at=[find('То есть мало заработать')]); job('G36-timer', 'G36-timer.mp4', at=[find('С первой раздачи')]); job('G39-earned', 'G39-earned.mp4', at=[find('До конца лета шесть недель')])
job('G14a-4tables', 'G14a-4tables.mp4', dur=16, at=[find('Он открывает восемнадцать')])
job('G17-profit', 'G17-profit.mp4', at=[find('Прибыль — около')]); job('G17-profit', 'G17-profit-ev.mp4', {'stage': '2'}, at=[find('Он и сам не скрывает')])
job('G20-bi', 'G20-bi-100.mp4', at=[find('С его двумястами долларами')]); job('G20-bi', 'G20-bi-40.mp4', {'v': '40'}, at=[find('Поэтому я принял решение')])
job('G33-calendar', 'G33-calendar-6w.mp4', {'v': '6w'}, at=[find('До конца лета шесть недель')]); job('G33-calendar', 'G33-calendar-flip.mp4', {'v': 'flip'}, at=[find('Проходит шесть дней')])
job('G29-hand', 'G29-hand-placeholder-a.mp4', {'reveal': '0', 'bettor': 'hero'}, at=[find('У Максима шестёрка')]); job('G29-hand', 'G29-hand-placeholder-b.mp4', {'reveal': '1', 'bettor': 'hero'}, at=[find('Соперник отвечает')])
job('G26-lower', 'G26-lower.mov')
job('G28-no', 'no-counter/G28-no-01-center.mov', {'n': '1', 'center': '1'}, at=[find('Правило, конечно, было для анкеты')])
job('G28-no', 'no-counter/G28-no-10-final.mov', {'n': '10', 'final': '1'}, at=[find('Анкета просила писать')])
for term, d, line in [('шот', 'попытка сыграть лимитом выше', 'В чём прикол'), ('ГСЧ', 'генератор случайных чисел', 'Возможно, когда-то ГСЧ'), ('рыба', 'слабый игрок', 'Возможно, когда-то ГСЧ'),
                ('сотка', 'NL100', 'Хотел идти на сотку'), ('колл', 'уравнять ставку', '«Нажать колл»'), ('тильт', 'когда эмоции играют вместо тебя', 'Каждая раздача на стек')]:
    job('G30-chip', f'chips/G30-{term}.png', {'term': term, 'def': d}, at=[find(line)])
for t in ['T01', 'T02', 'T03']:
    job(f'{t}-thumb', f'../../06-YOUTUBE/thumbnails/{t}-1920.png')
w('05-GFX/jobs-scenes.json', json.dumps(SJ, ensure_ascii=False, indent=1))

idx = ['# Графика: файл → где используется\n', 'Все файлы — в `05-GFX/render/`. Полноэкранные `.mp4` (1920×1080, 30 fps), прозрачные `.mov` (QuickTime PNG) и `.png`. Если реплика длиннее ролика графики — заморозить последний кадр.\n',
       '| Файл | Что | Реплики |', '|---|---|---|']
uses = {}
for l in lines:
    g = l['meta'].get('gfx', '')
    for gid in re.findall(r'\bG(\d\d[ab]?)\b', g):
        uses.setdefault('G' + gid, []).append(l['id'])
DESC = {'G00': 'фон «сукно» (подложка под скриншоты)', 'G01': 'условия пари', 'G02': 'баланс против цели', 'G03': 'титр «Брови на кону»', 'G04': 'карточки глав',
        'G05': 'таймлайн истории целиком', 'G06': '89 матчей в Доте', 'G07': 'режим сна (резерв)', 'G08': '48 страниц', 'G09': 'лестница лимитов', 'G10': 'банкролл в бай-инах',
        'G11': 'дисперсия (условный пример)', 'G12': 'одна ошибка ×10', 'G13': 'рейкбек', 'G14a': 'четыре стола (челлендж)', 'G14b': '18 столов', 'G15': '20 000 раздач vs живая игра',
        'G16': '11 недель / 336 часов / 316 000 раздач', 'G17': 'состав прибыли $3 500', 'G18': '$10 в час', 'G19': 'темп ×4,4', 'G20': '100 → 40 бай-инов', 'G21': '$100 → $80 + $20',
        'G22': 'Астана → Дананг', 'G23': 'тезис «Брови отрастают. Банкролл — нет.»', 'G24': 'конечная заставка', 'G25': 'карточки цитат (render/quotes)', 'G26': 'подпись героя',
        'G28': 'счётчик «НО» (render/no-counter)', 'G29': 'перерисовка раздачи 6-3', 'G30': 'сноски-термины (render/chips)', 'G32': 'вопрос зрителю', 'G33': 'календарь',
        'G34': '«заработать» → «на счёте»', 'G35': 'таблица гонки', 'G36': 'таймер 27 ч 42 мин', 'G38': 'лудка', 'G39': 'заработано ≠ на счёте'}
for j in SJ:
    gid = re.match(r'(G\d\d[ab]?|T\d\d)', j['scene']).group(1)
    if gid.startswith('T'):
        continue  # обложки — в 06-YOUTUBE/thumbnails
    where = j.get('lines') or uses.get(gid, [])
    idx.append(f"| `{j['out'].replace('render/', '')}` | {DESC.get(gid, 'обложка 16:9' if gid.startswith('T') else '')} | {', '.join(where) or '—'} |")
w('05-GFX/GFX-INDEX.md', '\n'.join(idx) + '\n')

# ---------- 04-SOUND/SFX-CUES.csv ----------
SFX_MAP = [
 (r'машинк', 'sfx01-trimmer-buzz.wav'), (r'свуш', 'sfx02-whoosh-soft.wav'), (r'щелч\w* фишки', 'sfx04-chip-click.wav'),
 (r'фишки уезжают|фишки вразнобой|щелчки фишек', 'sfx05-chips-push.wav'), (r'вжик', 'sfx06-metal-swish.wav'),
 (r'тикань|метроном', 'sfx07-clock-tick.wav'), (r'низкий удар|глухой удар', 'sfx08-low-thump.wav'), (r'перемотк', 'sfx09-tape-rewind.wav'),
 (r'скрип маркера', 'sfx10-marker-squeak.wav'), (r'тик» счётчика|«тик»|тики|«тиков»|писк', 'sfx11-ui-tick.wav'), (r'динь', 'sfx12-counter-ding.wav'),
 (r'вибрац', 'sfx13-phone-vibrate.wav'), (r'шелест', 'sfx14-paper-shuffle.wav'), (r'«тук»', 'sfx15-paper-thud.wav'),
 (r'дробь кликов|кликов мышью', 'sfx16-mouse-clicks.wav'), (r'коротких кликов|закрыть окно', 'sfx17-single-click.wav'), (r'затвор', 'sfx18-camera-shutter.wav'),
 (r'райзер', 'sfx19-riser-short.wav'), (r'самолёт', 'sfx21-airplane-pass.wav'), (r'моря', 'sfx22-sea-waves.wav'), (r'перелистыв', 'sfx23-calendar-flip.wav'),
 (r'калькулятор', 'sfx24-calculator.wav'), (r'мягкий звон', 'sfx25-soft-bell.wav'), (r'рулетки', 'sfx26-roulette-ball.wav'), (r'зачёркиван', 'sfx27-pen-strike.wav'),
 (r'рум-тон', 'sfx28-room-tone.wav'), (r'sfx29|бас-акцент', 'sfx29-bass-drop.wav'), (r'эхо|пэд', 'эффект эха в монтажке (хвост 0,3 с) + sfx25-soft-bell.wav с длинным ревербом'),
 (r'хруст', 'звук из своей съёмки мыла'), (r'глитч', 'sfx20-glitch.wav'),
]
buf = io.StringIO(); cw = csv.writer(buf, delimiter=';', lineterminator='\n')
cw.writerow(['id', 'тег', 'старт_оценка', 'реплика', 'что в сценарии', 'файлы'])
for l in lines:
    txt = ' '.join(x for x in [l['meta'].get('sfx', ''), l['meta'].get('mus', '') if 'рум-тон' in l['meta'].get('mus', '') else '', l['meta'].get('win', '') if 'жужжание' in l['meta'].get('win', '') else ''] if x)
    if not txt: continue
    files = []
    for pat, f in SFX_MAP:
        if re.search(pat, txt, re.I) and f not in files: files.append(f)
    cw.writerow([l['id'], l['tag'], mmss(l['est_start']) if l['est_start'] is not None else '', l['text'][:70], txt, ' + '.join(files) or 'без звука'])
w('04-SOUND/SFX-CUES.csv', '\ufeff' + buf.getvalue())

print(f'OK: {len(lines)} реплик, базовая версия {core_words} {plural(core_words, "слово", "слова", "слов")}, ≈ {mmss(TOTAL)}; карточек цитат {sum(1 for j in jobs if j["scene"]=="G25-quote")}')
