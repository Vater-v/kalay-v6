"""Auditable chip/BB reporting and poker-statistic opportunity denominators."""
from collections import Counter
import math
import statistics

from poker.arena.slumbot.protocol import BIG_BLIND, STACK_SIZE, parse_history


def rates(winnings):
    n = len(winnings)
    values = [float(x) / BIG_BLIND for x in winnings]
    result = dict(hands=n, profit_chips=sum(winnings), profit_bb=sum(values),
                  bb_per_100=100 * statistics.mean(values) if n else None,
                  ci95_bb_per_100=None, fixed_n_hoeffding95_bb_per_100=None)
    if n:
        mean = statistics.mean(values)
        radius = 2 * STACK_SIZE / BIG_BLIND * math.sqrt(math.log(40) / (2 * n))
        result['fixed_n_hoeffding95_bb_per_100'] = [100 * (mean-radius), 100 * (mean+radius)]
    if n >= 2:
        from scipy.stats import t
        stderr = statistics.stdev(values) / math.sqrt(n)
        radius = float(t.ppf(.975, n-1)) * stderr
        result['standard_error_bb_per_100'] = 100 * stderr
        result['ci95_bb_per_100'] = [100 * (statistics.mean(values)-radius),
                                    100 * (statistics.mean(values)+radius)]
    return result


def equal_position_rates(completed):
    """Equal SB/BB weighting, with Welch uncertainty under independent hands."""
    values=[[h['winnings_chips']/BIG_BLIND for h in completed if h['client_pos']==seat] for seat in (0,1)]
    result=dict(bb_per_100=None,ci95_bb_per_100=None)
    if not all(values):return result
    mean=(statistics.mean(values[0])+statistics.mean(values[1]))/2
    result['bb_per_100']=100*mean
    if min(map(len,values))<2:return result
    from scipy.stats import t
    terms=[statistics.variance(x)/len(x) for x in values]
    variance=sum(terms)/4
    denominator=sum(term*term/(len(x)-1) for term,x in zip(terms,values))
    df=sum(terms)**2/denominator if denominator else float('inf')
    radius=float(t.ppf(.975,df))*math.sqrt(variance)
    result['ci95_bb_per_100']=[100*(mean-radius),100*(mean+radius)]
    return result


def _can_aggress(event):
    """A legal bet/raise exists at this validated heads-up decision.

    Need chips beyond the call and a non-all-in opponent. A short raise can
    always use the actor's remaining stack. In this HU protocol the only
    non-reopening raise is a short all-in, whose raiser has zero chips, so the
    opponent-stack guard also excludes that case. Multiway reopening must not
    be inferred with this heads-up helper.
    """
    return (event.stacks_before[event.actor] > event.to_call_before and
            event.stacks_before[1-event.actor] > 0)


def hand_statistics(hand):
    history = parse_history(hand['action'])
    hero = hand['client_pos']
    result = {}
    for seat, label in [(hero, 'hero'), (1-hero, 'opponent')]:
        events = [e for e in history.events if e.actor == seat]
        pre = [e for e in events if e.street == 0]
        post = [e for e in events if e.street > 0]
        won = hand['winnings_chips'] * (1 if label == 'hero' else -1)
        saw_flop = len(hand['board']) >= 3
        showdown = history.terminal_reason == 'showdown'
        counts = Counter(hands=1, vpip=int(any(e.contribution > 0 for e in pre)),
            pfr=int(any(e.kind == 'raise' for e in pre)), saw_flop=int(saw_flop),
            won_when_saw_flop=int(saw_flop and won > 0), showdown=int(showdown),
            won_showdown=int(showdown and won > 0), postflop_raises=sum(e.kind == 'raise' for e in post),
            postflop_calls=sum(e.kind == 'call' for e in post))
        for e in events:
            counts[f'action_{e.kind}'] += 1
            counts[f'street{e.street}_{e.kind}'] += 1
        pre_raises = 0
        last_pre_raiser = None
        for e in history.events:
            if e.street != 0:
                break
            if e.actor == seat and e.to_call_before > 0 and pre_raises == 1 and _can_aggress(e):
                counts['three_bet_opportunities'] += 1
                counts['three_bets'] += e.kind == 'raise'
            if e.actor == seat and e.to_call_before > 0 and pre_raises == 2:
                counts['fold_to_three_bet_opportunities'] += 1
                counts['fold_to_three_bet'] += e.kind == 'fold'
            if e.kind == 'raise':
                pre_raises += 1
                last_pre_raiser = e.actor
        flop_raise = False
        cbet_actor = None
        for e in history.events:
            if e.street != 1:
                continue
            if e.actor == last_pre_raiser and not flop_raise and _can_aggress(e):
                if e.actor == seat:
                    counts['cbet_opportunities'] += 1
                    counts['cbets'] += e.kind == 'raise'
                # Each player acts at most once before the first flop raise.
                if e.kind == 'raise':
                    cbet_actor = e.actor
            if e.actor == seat and cbet_actor is not None and cbet_actor != seat and e.to_call_before:
                counts['fold_to_cbet_opportunities'] += 1
                counts['fold_to_cbet'] += e.kind == 'fold'
                cbet_actor = None
            flop_raise |= e.kind == 'raise'
        result[label] = counts
    return result


def summary(records, requested_hands):
    completed = [r['result']['completed'][0] for r in records if r['status'] == 'completed']
    failed = [r['number'] for r in records if r['status'] == 'failed']
    unresolved = [r['number'] for r in records if r['status'] not in ('completed', 'failed')]
    combined = {label: Counter() for label in ['hero', 'opponent']}
    for record in records:
        if record['status'] != 'completed':
            continue
        per_hand = record.get('player_stats') or hand_statistics(record['result']['completed'][0])
        for label, counts in per_hand.items():
            combined[label].update(counts)
    player_stats = {}
    for label, counts in combined.items():
        stats = dict(counts)
        for name, numerator, denominator in [('vpip','vpip','hands'),('pfr','pfr','hands'),
            ('three_bet','three_bets','three_bet_opportunities'),
            ('fold_to_three_bet','fold_to_three_bet','fold_to_three_bet_opportunities'),
            ('cbet','cbets','cbet_opportunities'),('fold_to_cbet','fold_to_cbet','fold_to_cbet_opportunities'),
            ('wwsf','won_when_saw_flop','saw_flop'),('wtsd','showdown','saw_flop'),('wsd','won_showdown','showdown')]:
            stats[name+'_percent'] = 100*counts[numerator]/counts[denominator] if counts[denominator] else None
        stats['aggression_factor'] = counts['postflop_raises']/counts['postflop_calls'] if counts['postflop_calls'] else None
        player_stats[label] = stats
    complete = len(completed) + len(failed) == requested_hands and not unresolved
    scored = rates([h['winnings_chips'] for h in completed])
    played = len(completed) + len(failed)
    # the worst case scores every failed hand as a lost stack (the opponent could have taken it all): a bound, not an estimate
    worst = (100 * (scored['profit_chips'] - STACK_SIZE * len(failed)) / BIG_BLIND / played) if played else None
    return dict(status='completed' if complete else 'unresolved' if unresolved else 'paused',
        requested_hands=requested_hands, completed_hands=len(completed), unresolved_hands=unresolved,
        failed_hands=failed, failed_reasons={str(r['number']): r.get('failure_reason') for r in records if r['status'] == 'failed'},
        worst_case_bb_per_100_with_failed_hands_lost=worst,
        **scored,
        positions={name:rates([h['winnings_chips'] for h in completed if h['client_pos']==seat])
                   for seat,name in [(0,'bb'),(1,'sb')]}, player_stats=player_stats,
        equal_position_weighted=equal_position_rates(completed),
        wins=sum(h['winnings_chips']>0 for h in completed), ties=sum(h['winnings_chips']==0 for h in completed),
        losses=sum(h['winnings_chips']<0 for h in completed), quality_verdict=None,
        rates_scope='all_requested_hands' if complete else 'completed_hands_only; no complete-match conclusion',
        confidence_method='Student-t 95% approximate fixed-sample interval; independent identically distributed hand profits assumed; not anytime-valid',
        bounded_confidence_method='Hoeffding 95% fixed-n bound with profit in [-200,200] BB; independent hands assumed; not anytime-valid',
        retries=0)


def hud(report):
    rate = report['bb_per_100']
    ci = report['ci95_bb_per_100']
    values = 'n/a' if rate is None else f'{rate:+.2f} BB/100'
    interval = 'n/a' if ci is None else f'[{ci[0]:+.2f},{ci[1]:+.2f}]'
    hero = report['player_stats']['hero']
    def pct(value):return 'n/a' if value is None else f'{value:.1f}%'
    opponent = report['player_stats']['opponent']
    vpip,pfr=pct(hero.get('vpip_percent')),pct(hero.get('pfr_percent'))
    return (f"Slumbot {report['completed_hands']}/{report['requested_hands']} | {values} | "
            f"95% CI {interval} | VPIP/PFR {vpip}/{pfr} | "
            f"unresolved={len(report['unresolved_hands'])} | status={report['status']}\n"
            f"Seats SB/BB {report['positions']['sb']['hands']}/{report['positions']['bb']['hands']} | "
            f"Opponent VPIP/PFR {pct(opponent.get('vpip_percent'))}/{pct(opponent.get('pfr_percent'))} | "
            f"Hero 3bet/Cbet {pct(hero.get('three_bet_percent'))}/{pct(hero.get('cbet_percent'))} | retries=0")
