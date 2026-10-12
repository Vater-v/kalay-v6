"""Bounded HTTP concurrency, durable per-hand evidence, and explicit safe resume.

No mutating request is retried: Slumbot exposes no idempotency key or hand-status
recovery endpoint in its public sample API. Resume can extend a clean hand
boundary; an in-flight/ambiguous hand blocks continuation of that match.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from threading import Event, Lock
import time

import numpy as np

from poker.arena.slumbot.util import atomic_json, json_hash, file_sha256
from poker.arena.slumbot.client import PooledHTTPTransport, MatchInterrupted, run_match
from poker.arena.slumbot.metrics import summary, hand_statistics


class IncompleteMatchError(RuntimeError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class _SessionTransport:
    """Session token remains in memory; each HTTP worker has a separate session."""
    def __init__(self, transport):
        self.transport = transport
        self.token = None
        self.requests = 0
        self.http_seconds = 0.

    def post(self, endpoint, body):
        started = time.perf_counter()
        self.requests += 1
        try:
            payload = self.transport.post(endpoint, body)
        finally:
            self.http_seconds += time.perf_counter()-started
        if isinstance(payload, dict) and isinstance(payload.get('token'), str) and payload['token']:
            self.token = payload['token']
        return payload


def _exclusive_json(path,value):
    # Exclusive creation prevents two processes from overwriting the same
    # request intent. This is a durable record, not a renewable resource lease.
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())


def _write_record(path, record, *, initial=False):
    record = dict(record)
    record['record_sha256'] = json_hash(record)
    (_exclusive_json if initial else atomic_json)(path, record)


def _compact_record(record):
    """Keep summaries in RAM; full redacted request evidence remains on disk."""
    return {key:value for key,value in record.items() if key not in ('decisions','result')} | {
        'result':{'completed':record.get('result',{}).get('completed',[])}}


def _load_records(directory, policy_id, hands):
    records = []
    for path in sorted(directory.glob('*.json')):
        record = json.loads(path.read_text(encoding='utf-8'))
        digest = record.pop('record_sha256', None)
        if digest != json_hash(record):
            raise ValueError(f'Hand evidence checksum mismatch: {path.name}')
        number = record.get('number')
        if type(number) is not int or not 1 <= number <= hands or path.name != f'{number:08d}.json':
            raise ValueError('Hand evidence number is invalid')
        if record.get('policy_id') != policy_id:
            raise ValueError('Hand evidence belongs to another policy')
        if record.get('status') == 'completed':
            result = record.get('result', {})
            if result.get('status') != 'completed' or len(result.get('completed', [])) != 1:
                raise ValueError('Invalid completed-hand evidence')
        if record.get('status') == 'failed' and record.get('mutation_outcome_unknown', True):
            raise ValueError('A failed hand must have a known mutation outcome')
        records.append(_compact_record(record))
    return records


def run_parallel(policy, *, output, hands, workers=16, seed=42, hand_cap=10000,
                 action_cap_per_hand=512, transport_factory=None, resume=False,
                 stop_event=None, stop_file=None, max_new_hands=None, log_every=100,
                 on_progress=None, timeout=15.):
    """Evaluate a fixed artifact; all network work happens only inside this call.

    `max_new_hands` pauses on a clean boundary for bounded execution/resume. It
    does not change the preregistered requested hand count or replace errors.
    A crash leaves a pending record written before new_hand; resume rejects it.
    """
    for name, value in [('hands',hands),('workers',workers),('hand_cap',hand_cap),
                        ('action_cap_per_hand',action_cap_per_hand),('log_every',log_every)]:
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if workers > 64 or hands > hand_cap:
        raise ValueError('Workers must be <=64 and hands must not exceed hand_cap')
    if type(seed) is not int or seed < 0 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Invalid seed or timeout')
    if max_new_hands is not None and (type(max_new_hands) is not int or max_new_hands < 1):
        raise ValueError('max_new_hands must be a positive integer')
    if not isinstance(policy.policy_id, str) or not policy.policy_id:
        raise ValueError('A stable live configuration hash is required')
    output = Path(output)
    directory = output/'hands'
    directory.mkdir(parents=True, exist_ok=True)
    path = output/'manifest.json'
    contract = dict(schema_version=1, policy_id=policy.policy_id, hands=hands, workers=workers,
        seed=seed, hand_cap=hand_cap, action_cap_per_hand=action_cap_per_hand,
        timeout=timeout, api_origin='https://slumbot.com',
        algorithm='critic_backed_live_pcfr_stochastic_sampling',
        rng='numpy.default_rng(SeedSequence([seed, global_hand_number]))',
        game={'variant':'nlh','players':2,'blinds':[50,100],'stacks':[20000,20000]},
        policy_metadata=getattr(policy,'metadata',None), retries=0,
        inference_config=getattr(policy,'inference_config',None),
        implementation_sha256={p.name:file_sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
    if path.exists():
        if not resume:
            raise FileExistsError('Evaluation exists; use explicit resume with the identical contract')
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved['contract_hash'] != json_hash(contract) or saved['contract'] != contract:
            raise ValueError('Resume contract or live configuration changed')
    else:
        if resume or any(directory.iterdir()):
            raise FileNotFoundError('Resume requires a valid existing manifest')
        _exclusive_json(path,dict(created_at=utc_now(),contract=contract,contract_hash=json_hash(contract)))
    records = _load_records(directory, policy.policy_id, hands)
    if any(r['status'] not in ('completed', 'failed') for r in records):
        raise IncompleteMatchError('An unresolved/in-flight hand prevents safe resume; retain this partial match, never replace its missing outcome')
    existing = {r['number'] for r in records}
    numbers = [n for n in range(1,hands+1) if n not in existing]
    if max_new_hands is not None:
        numbers = numbers[:max_new_hands]
    stop = stop_event or Event()
    stop_file = Path(stop_file) if stop_file else None
    lock = Lock()
    started = time.monotonic()
    next_index = 0
    request_count = 0
    http_seconds = 0.
    transport_factory = transport_factory or (lambda _: PooledHTTPTransport(timeout=timeout))

    def worker(worker_id):
        transport = None
        try:
            transport = _SessionTransport(transport_factory(worker_id))
            return worker_loop(worker_id,transport)
        except BaseException:
            stop.set()
            raise
        finally:
            if transport is not None and hasattr(transport.transport,'close'):
                transport.transport.close()

    def worker_loop(worker_id,transport):
        nonlocal next_index,request_count,http_seconds
        while True:
            with lock:
                if stop.is_set() or (stop_file and stop_file.exists()) or next_index == len(numbers):
                    return
                number = numbers[next_index]
                next_index += 1
                record = dict(number=number,worker=worker_id,policy_id=policy.policy_id,
                    status='in_flight',started_at=utc_now(),mutation_outcome_unknown=True)
                record_path = directory/f'{number:08d}.json'
                _write_record(record_path,record,initial=True)  # Before the first possibly mutating request.
            rng = np.random.default_rng(np.random.SeedSequence([seed,number]))
            decisions = []
            before_requests,before_http = transport.requests,transport.http_seconds
            def choose(response):
                decision = policy.choose(response,rng)
                decisions.append(asdict(decision))
                return decision.increment
            try:
                result = run_match(choose,transport,hands=1,hand_cap=1,
                    policy_id=policy.policy_id,action_cap_per_hand=action_cap_per_hand,
                    initial_token=transport.token)
                record.update(status='completed',result=result.to_dict(),mutation_outcome_unknown=False)
                record['result']['completed'][0]['number'] = number
                record['player_stats'] = hand_statistics(record['result']['completed'][0])
            except MatchInterrupted as error:
                ambiguous = error.result.unresolved.mutation_outcome_unknown
                # A decision failure (BudgetExceeded, NonFinite, ...) is a FAILED hand: recorded with its reason, scored at the
                # worst case in the summary, never replayed by a default action; the match goes on and resume passes it
                # (live audit 2026-10-12, F1: one failure used to stop the whole evaluation and block its resume). An ambiguous
                # transport outcome (the server may have applied an action) still stops everything.
                record.update(status='unresolved' if ambiguous else 'failed', result=error.result.to_dict(),
                    mutation_outcome_unknown=ambiguous, failure_reason=error.result.unresolved.reason)
                if ambiguous:
                    stop.set()
            except BaseException:
                # The persisted in-flight record is deliberately retained on a
                # disk/process failure instead of asserting no request occurred.
                stop.set()
                raise
            record.update(completed_at=utc_now(),decisions=decisions,
                requests=transport.requests-before_requests,http_seconds=transport.http_seconds-before_http)
            _write_record(record_path,record)
            with lock:
                records.append(_compact_record(record))
                request_count += record['requests']
                http_seconds += record['http_seconds']
                completed_count = sum(r['status']=='completed' for r in records)
                # Event histories are parsed once per hand; progress aggregates
                # cached counters and profits, with no extra inference.
                if on_progress and (completed_count % log_every == 0 or record['status'] != 'completed'):
                    progress = summary(records,hands)
                    progress['elapsed_seconds'] = time.monotonic()-started
                    if progress['status']=='paused':progress['status']='running'
                    atomic_json(output/'summary.json',progress)
                    on_progress(progress)
    with ThreadPoolExecutor(max_workers=workers,thread_name_prefix='slumbot-http') as pool:
        futures = [pool.submit(worker,i) for i in range(workers)]
        for future in futures:
            future.result()
    records.sort(key=lambda record:record['number'])
    report = summary(records,hands)
    report.update(policy_id=policy.policy_id,contract_hash=json_hash(contract),
        elapsed_seconds=time.monotonic()-started,requests_this_run=request_count,
        http_seconds_this_run=http_seconds,created_at=utc_now(),
        inference_batches=getattr(policy,'batches',None),
        inference_decisions=getattr(policy,'decisions',None),
        max_inference_batch=getattr(policy,'max_observed_batch',None))
    atomic_json(output/'summary.json',report)
    return report
