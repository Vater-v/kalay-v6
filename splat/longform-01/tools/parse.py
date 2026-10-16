"""Парсер script.src.txt → структура глав и реплик. Общий модуль для build.py."""
import re, sys, json

DIRS = {
    'D1': ('[Живо объясняешь другу, короткими фразами, без дикторской растяжки]', 'проверена в Shorts'),
    'D2': ('[С удивлением и азартом, разговорно, без вздохов]', 'проверена в Shorts'),
    'D3': ('[Удивлённо и увлечённо, без крика и вздохов]', 'проверена в Shorts'),
    'D4': ('[Уверенно и разговорно, с живым интересом в вопросе]', 'проверена в Shorts'),
    'D5': ('[Бодро, увлечённо, без вздохов]', 'базовая из voice-config'),
    'D6': ('[С лёгкой иронией, разговорно, без вздохов]', 'новая — проверить'),
    'D7': ('[Сухо и коротко, с едва заметной усмешкой]', 'новая — проверить'),
    'D8': ('[Спокойно и тепло, разговорно, без вздохов]', 'новая — проверить'),
    'M1': ('[Непринуждённо, как будто пишешь другу, без вздохов]', 'новая — роль Максима'),
    'M2': ('[Весело, с самоиронией, без вздохов]', 'новая — роль Максима'),
    'M3': ('[Спокойно и честно, разговорно, без вздохов]', 'новая — роль Максима'),
    'F1': ('[Коротко и насмешливо, разговорно]', 'новая — роль форума'),
    'F2': ('[Удивлённо, разговорно, без вздохов]', 'новая — роль форума'),
    'F3': ('[Дружелюбно, с улыбкой в голосе]', 'новая — роль форума'),
}
KEYS = {'tts', 'vis', 'mot', 'gfx', 'txt', 'sfx', 'mus', 'pause', 'win', 'src', 'note', 'no', 'drop'}
DEFAULT_PAUSE = 0.25


def parse(path):
    chapters = []
    cur = None
    line_obj = None
    for raw in open(path, encoding='utf-8-sig'):  # -sig: файл, сохранённый с BOM (Блокнот), читается так же
        line = raw.rstrip('\n')
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        if line.startswith('==='):
            parts = [p.strip() for p in line[3:].split('|')]
            cur = dict(num=parts[0], title=parts[1], when=parts[2], music=parts[3], lines=[])
            chapters.append(cur)
            line_obj = None
            continue
        if line.startswith('  ') and line_obj is not None:
            m = re.match(r'\s+(\w+):\s?(.*)$', line)
            if not m or m.group(1) not in KEYS:
                raise SystemExit(f'bad key line: {line}')
            k, v = m.group(1), m.group(2).strip()
            if k in line_obj['meta']:
                line_obj['meta'][k] += ' ' + v
            else:
                line_obj['meta'][k] = v
            continue
        parts = [p.strip() for p in line.split('|', 3)]
        if len(parts) != 4:
            raise SystemExit(f'bad line: {line}')
        role, d, tag, text = parts
        if d not in DIRS:
            raise SystemExit(f'unknown dir {d}: {line}')
        if tag not in ('core', 'cut', 'res'):
            raise SystemExit(f'bad tag {tag}')
        line_obj = dict(role=role, dir=d, tag=tag, text=text, meta={})
        cur['lines'].append(line_obj)
    # ids + счётчик «но»
    ORD = {1:'первое',2:'второе',3:'третье',4:'четвёртое',5:'пятое',6:'шестое',7:'седьмое',8:'восьмое',9:'девятое',10:'десятое',11:'одиннадцатое',12:'двенадцатое',13:'тринадцатое',14:'четырнадцатое',15:'пятнадцатое'}
    no_count = 0
    for ch in chapters:
        for l in ch['lines']:
            if l['meta'].get('no') == '+':
                if l['tag'] == 'res':
                    raise SystemExit('no:+ not allowed on reserve lines')
                no_count += 1
                l['no_n'] = no_count
            else:
                l['no_n'] = None
            l['no_running'] = no_count
            if '{NO_ORD}' in l['text'] or '{NO_ORD}' in l['meta'].get('tts', ''):
                l['text'] = l['text'].replace('{NO_ORD}', ORD[no_count])
                if 'tts' in l['meta']:
                    l['meta']['tts'] = l['meta']['tts'].replace('{NO_ORD}', ORD[no_count])
    for ch in chapters:
        for i, l in enumerate(ch['lines'], 1):
            l['id'] = f"{ch['num']}-{i:02d}"
            m = l['meta']
            l['tts'] = m.get('tts', l['text'])
            l['pause'] = float(m.get('pause', DEFAULT_PAUSE))
            w = m.get('win')
            if w:
                mm = re.match(r'([\d.]+)\s*(?:—\s*(.*))?$', w)
                l['win'] = float(mm.group(1)); l['win_desc'] = (mm.group(2) or '').strip()
            else:
                l['win'] = 0.0; l['win_desc'] = ''
            if l['role'] == 'N':
                l['role_name'] = 'Рассказчик'; l['voice'] = 'narrator'
            elif l['role'] == 'M':
                l['role_name'] = 'Максим (чтение поста)'; l['voice'] = 'maxim'
            else:
                l['role_name'] = 'Форум · ' + l['role'].split(':', 1)[1]; l['voice'] = 'forum'
            # sentence count sanity: max two sentences per TTS request
            sents = [s for s in re.split(r'(?<=[.!?…])\s+', l['tts']) if s.strip()]
            l['sentences'] = len(sents)
    return chapters


WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9$]+(?:[-–][A-Za-zА-Яа-яЁё0-9]+)*")


def words(s):
    return len(WORD_RE.findall(s))


def est_seconds(l, cps):
    # длительность клипа: символы TTS / скорость + поля клипа 0.2 с
    return len(l['tts']) / cps + 0.2


if __name__ == '__main__':
    chs = parse(sys.argv[1])
    tot_w = 0
    for cps in (11.0, 12.5, 14.0):
        t = 0.0
        for ch in chs:
            for l in ch['lines']:
                t += est_seconds(l, cps) + l['pause'] + l['win']
        print(f'cps {cps}: {t/60:.2f} min')
    for ch in chs:
        w = sum(words(l['text']) for l in ch['lines'])
        tot_w += w
        t = sum(est_seconds(l, 12.5) + l['pause'] + l['win'] for l in ch['lines'])
        print(ch['num'], ch['title'], len(ch['lines']), 'lines', w, 'words', f'{t:.0f}s')
    print('total words', tot_w, 'lines', sum(len(c['lines']) for c in chs))
    bad = [l['id'] for c in chs for l in c['lines'] if l['sentences'] > 2]
    print('lines with >2 sentences:', bad)
