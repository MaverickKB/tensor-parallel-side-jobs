"""Paired sealed job start for a two-rank tensor-parallel server.

Extracted from the resident trainer. The engine-specific parts are hooks at module
level (native_constants, prepare, run, synchronize, write_receipt); bind them for
your engine. The decoder object is expected to expose the attributes used below
(see tests/test_protocol.py for a complete fake).
"""

import json
import os
import time
from pathlib import Path

THIN_START = 29768        # 0x7448: trainer start op, carried in the sealed serving sequence
THIN_RESUME = 29769       # head finished a nested serving iteration; worker returns to the job
JOIN_TIMEOUT = 10.0       # seconds the head waits for the worker to join; then cancel the job only
STEP_TIMEOUT = 600.0      # seconds a trainer-channel exchange may wait inside a running job
CHANNEL_TAG = 'paired-jobs-v1'


def encode_id(identifier):
    """A 32-hex job identity as eight 16-bit int32-safe words."""
    return [int(identifier[i:i + 4], 16) for i in range(0, 32, 4)]


def decode_id(words):
    return ''.join('%04x' % int(word) for word in words)


def strip_start(message):
    """Split a message body into (ops without THIN_START, start payloads)."""
    body = list(message)
    kept, starts, index = [], [], 0
    while index < len(body):
        op, size = int(body[index]), int(body[index + 1])
        chunk = body[index:index + 2 + size]
        if len(chunk) != 2 + size:
            raise RuntimeError('Malformed serving message while scanning for trainer start')
        if op == THIN_START:
            starts.append(chunk[2:])
        elif op == THIN_RESUME:
            starts.append(None)
        else:
            kept += chunk
        index += 2 + size
    return kept, starts


class TrainerChannel:
    """Trainer-only exchange over the rendezvous store, never the chat all-gather.

    Every value is a tagged JSON record naming the job, the paired sequence,
    the exchange name and the sending rank; anything else is rejected.
    """

    def __init__(self, store, identifier, sequence, rank, world=2):
        self.store = store
        self.identifier = identifier
        self.sequence = int(sequence)
        self.rank = rank
        self.world = world

    def key(self, name, rank):
        return '/'.join(['paired-jobs', self.identifier, str(self.sequence), name, 'rank' + str(rank)])

    def put(self, name, value):
        record = {'tag': CHANNEL_TAG, 'job': self.identifier, 'seq': self.sequence,
                  'name': name, 'rank': self.rank, 'value': value}
        self.store.set(self.key(name, self.rank), json.dumps(record).encode())

    def take(self, name, rank, timeout):
        from datetime import timedelta
        key = self.key(name, rank)
        try:
            self.store.wait([key], timedelta(seconds=timeout))
        except Exception as error:
            raise TimeoutError('Trainer channel ' + name + ' from rank ' + str(rank)
                               + ' not received in ' + str(timeout) + 's') from error
        record = json.loads(self.store.get(key))
        expected = (CHANNEL_TAG, self.identifier, self.sequence, name, rank)
        actual = (record.get('tag'), record.get('job'), record.get('seq'), record.get('name'),
                  record.get('rank'))
        if actual != expected:
            raise RuntimeError('Trainer channel record mismatch: ' + str(actual))
        return record['value']

    def exchange(self, name, value, timeout=STEP_TIMEOUT):
        self.put(name, value)
        return [value if rank == self.rank else self.take(name, rank, timeout)
                for rank in range(self.world)]


def store_of(decoder):
    store = getattr(decoder.g.comm, 'store', None)
    if store is None:
        raise RuntimeError('Paired trainer requires the rendezvous store')
    return store


def attach(decoder, record):
    """Install the paired start protocol once on this rank (no received patching)."""
    if record['sid'] not in decoder.lanes:
        raise ValueError('Resident job admission lacks its waking request')
    # Other chats may be live: the job itself starts only once the pair is idle
    # (checked at every finish) and is announced in the sealed sequence.
    if getattr(decoder, '_thin_finish_installed', False):
        return
    decoder._thin_pending = {}
    decoder._thin_running = False
    decoder._thin_start = None
    if decoder.w.rank == 0:
        original_finish = decoder.finish

        def finish(done):
            original_finish(done)
            if (decoder._thin_pending and not decoder.lanes and decoder.broken is None
                    and not decoder._thin_running):
                sid = next(iter(decoder._thin_pending))
                lead(decoder, decoder._thin_pending.pop(sid))

        decoder.finish = finish
    else:
        original_apply = decoder.apply
        original_bell = decoder.g._await_bell

        def apply(message):
            kept, starts = strip_start(message)
            original_apply(kept)
            for payload in starts:
                if payload is None:
                    decoder._thin_resumed = True
                    continue
                # received still names this message; follow increments it after apply.
                decoder._thin_start = (int(payload[0]), decode_id(payload[1:]), decoder.received)

        def await_bell():
            start = decoder._thin_start
            if start is not None and not decoder._thin_running:
                decoder._thin_start = None
                join(decoder, *start)
                if not decoder.idle:
                    # The job's last THIN_RESUME left chats live: rank 0 is not
                    # idle and rings no doorbell; follow reads its next message.
                    return None
            return original_bell()

        decoder.apply = apply
        decoder.g._await_bell = await_bell
    decoder._thin_finish_installed = True


def pending_by_id(decoder, identifier):
    for sid, pending in list(decoder._thin_pending.items()):
        if pending['policy']['thin_job']['id'] == identifier:
            return decoder._thin_pending.pop(sid)
    return None


def write_receipt(rank, identifier, report):
    """Hook: persist a per-rank job receipt. Replace for your system."""
    path = Path(os.environ.get('PAIRED_JOBS_RECEIPTS', 'receipts'))
    path.mkdir(parents=True, exist_ok=True)
    (path / ('job-' + identifier + '-rank' + str(rank) + '.json')).write_text(json.dumps(report, indent=2) + '\n')


def cancel_receipt(decoder, identifier, reason, phases):
    phases.append(['cancelled', time.time()])
    report = {'id': identifier, 'rank': decoder.w.rank, 'status': 'cancelled', 'error': reason,
              'serving_restart': False, 'serving_affected': False, 'phases': phases}
    write_receipt(decoder.w.rank, identifier, report)
    return report


def native_constants():
    """Hook: return (IDLE op code, sealed-sequence modulus) from your serving engine."""
    raise NotImplementedError('Bind native_constants to your engine')


def lead(decoder, record, join_timeout=JOIN_TIMEOUT, runner=None):
    """Head: announce the job in the sealed sequence, then wait for the worker to join.

    Never starts alone: without the worker's matching confirmation only this
    trainer job is cancelled; serving continues.
    """
    runner = runner or run
    IDLE, SEAL_MOD = native_constants()
    identifier = record['policy']['thin_job']['id']
    sequence = getattr(decoder, '_thin_sequence', 0) + 1
    decoder._thin_sequence = sequence
    phases = [['announce', time.time()]]
    decoder._emit(THIN_START, [sequence, *encode_id(identifier)])
    decoder._emit(IDLE, [])
    decoder._flush()                      # rings the idle worker, seals as message `sent`
    decoder.idle = True
    message = (decoder.sent - 1) % SEAL_MOD
    channel = TrainerChannel(store_of(decoder), identifier, sequence, 0)
    ready = prepare(decoder, record)
    channel.put('join', {'message': message, 'ready': ready is None, 'error': ready})
    decoder._thin_running = True        # new chats wait in the scheduler queue meanwhile
    try:
        try:
            peer = channel.take('join', 1, join_timeout)
        except TimeoutError as error:
            channel.put('decision', 'cancel')
            return cancel_receipt(decoder, identifier, str(error), phases)
        phases.append(['worker_joined', time.time()])
        if peer['message'] != message or not peer['ready'] or ready is not None:
            channel.put('decision', 'cancel')
            return cancel_receipt(decoder, identifier, 'Paired admission failed: head message '
                                  + str(message) + ' ready ' + str(ready) + '; worker ' + str(peer),
                                  phases)
        channel.put('decision', 'go')
        phases.append(['both_in_job', time.time()])
        return runner(decoder, record, channel, phases)
    finally:
        decoder._thin_running = False


def join(decoder, sequence, identifier, message, decision_timeout=JOIN_TIMEOUT * 3, runner=None):
    """Worker: confirm the sealed message number, then follow the head's decision."""
    runner = runner or run
    phases = [['start_received', time.time()]]
    channel = TrainerChannel(store_of(decoder), identifier, sequence, 1)
    record = pending_by_id(decoder, identifier)
    ready = 'Worker lacks the admitted job record' if record is None else prepare(decoder, record)
    channel.put('join', {'message': message, 'ready': ready is None, 'error': ready})
    decoder._thin_running = True
    try:
        try:
            decision = channel.take('decision', 0, decision_timeout)
        except TimeoutError as error:
            return cancel_receipt(decoder, identifier, str(error), phases)
        if decision != 'go':
            return cancel_receipt(decoder, identifier, 'Head cancelled the job', phases)
        phases.append(['both_in_job', time.time()])
        return runner(decoder, record, channel, phases)
    finally:
        decoder._thin_running = False


def prepare(decoder, record):
    """Hook: validate the sealed job locally before joining; return an error string or None."""
    return None


def run(decoder, record, channel, phases):
    """Hook: the job body. Both ranks call it with the same record and channel."""
    raise NotImplementedError('Bind run to your job body')


def synchronize(decoder):
    """Hook: wait for this rank's device work (e.g. torch.cuda.synchronize) before serving."""


def serve_waiting(decoder, channel, counter):
    """Both ranks pause training at the same boundary for ordinary work.

    The decision travels on the trainer channel; the nested iteration then uses
    the ordinary sealed serving sequence exactly as native serving does.
    """
    weights = decoder.w
    if weights.rank == 0:
        scheduler = decoder.g.scheduler
        wanted = bool(decoder.live() or not scheduler.waiting.empty()
                      or scheduler.held is not None or scheduler.loading or scheduler.ready)
        channel.put('serve-' + str(counter), int(wanted))
    else:
        wanted = bool(channel.take('serve-' + str(counter), 0, STEP_TIMEOUT))
    if not wanted:
        return False
    synchronize(decoder)
    if weights.rank == 0:
        # One scheduler iteration may flush several sealed messages (rounds,
        # FINISH, IDLE). Close it with an explicit RESUME so the worker knows
        # exactly when to return to the job.
        decoder.g.scheduler._iteration()
        resume(decoder)
    else:
        decoder._thin_resumed = False
        while not decoder._thin_resumed:
            decoder.follow(once=True)
    return True


def resume(decoder):
    IDLE, _ = native_constants()
    decoder._emit(THIN_RESUME, [])
    idle = not decoder.lanes
    if idle:
        decoder._emit(IDLE, [])
    decoder._flush()
    if idle:
        decoder.idle = True

