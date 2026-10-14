"""CPU proofs for the paired start, trainer-only channel, resume marker and doorbell skip."""

import threading
import time
from types import SimpleNamespace

import pytest
from tp_side_jobs import protocol as thin_job

IDLE, FINISH, SEAL_MOD = 8, 7, 2_147_483_647


class Store:
    """In-memory rendezvous store with the TCPStore wait/get/set surface."""

    def __init__(self):
        self.values = {}
        self.condition = threading.Condition()

    def set(self, key, value):
        with self.condition:
            self.values[key] = value
            self.condition.notify_all()

    def get(self, key):
        with self.condition:
            return self.values[key]

    def wait(self, keys, timeout):
        with self.condition:
            if not self.condition.wait_for(lambda: all(k in self.values for k in keys),
                                           timeout=timeout.total_seconds()):
                raise RuntimeError('Socket Timeout')


class Wire:
    """The sealed serving message sequence: rank 0 seals, rank 1 unseals in order."""

    def __init__(self, base=0):
        self.base = base
        self.messages = []
        self.condition = threading.Condition()

    def send(self, message):
        with self.condition:
            self.messages.append(message)
            self.condition.notify_all()

    def receive(self, number):
        with self.condition:
            index = number - self.base
            assert self.condition.wait_for(lambda: len(self.messages) > index, timeout=5)
            return self.messages[index]


def record(identifier='ab' * 16):
    return {'sid': 3, 'policy': {'thin_job': {'id': identifier, 'sha256': 'x'}}}


def decoder(rank, store, wire):
    head = SimpleNamespace(outbox=[], sent=0, received=0, idle=True, lanes={}, broken=None,
                           w=SimpleNamespace(rank=rank),
                           g=SimpleNamespace(comm=SimpleNamespace(store=store)))
    head.bells = 0

    def emit(op, payload):
        head.outbox += [op, len(payload), *payload]

    def flush():
        if head.idle:
            head.bells += 1
            head.idle = False
        wire.send((list(head.outbox), head.sent))
        head.outbox = []
        head.sent = (head.sent + 1) % SEAL_MOD

    def apply(message):
        for index in range(0, len(message), 2):
            if message[index] == IDLE:
                head.idle = True
            elif message[index] == THIN_GUARD:
                raise RuntimeError('rank 1: unknown op')

    def finish(done):
        head._emit(FINISH, [3, 0])
        head._flush()

    head._emit, head._flush, head.apply, head.finish = emit, flush, apply, finish
    head.g._await_bell = lambda: None
    return head


THIN_GUARD = thin_job.THIN_START


@pytest.fixture(autouse=True)
def native(monkeypatch):
    monkeypatch.setattr(thin_job, 'native_constants', lambda: (IDLE, SEAL_MOD))
    monkeypatch.setattr(thin_job, 'prepare', lambda decoder, record: None)
    receipts = []
    monkeypatch.setattr(thin_job, 'write_receipt', lambda rank, ident, report: receipts.append(report))
    return receipts


def follow_once(worker, wire):
    """Native follow: unseal message `received`, apply, increment, then await the bell."""
    body, number = wire.receive(worker.received)
    assert number == worker.received, 'ranks out of step'
    worker.apply(body)
    worker.received = (worker.received + 1) % SEAL_MOD
    if worker.idle:
        worker.g._await_bell()


def test_start_rides_the_sealed_sequence_and_both_ranks_enter_at_the_same_message():
    store, wire = Store(), Wire(32394)
    head, worker = decoder(0, store, wire), decoder(1, store, wire)
    head.sent = worker.received = 32394
    entered = {}

    def runner(dec, rec, channel, phases):
        entered[dec.w.rank] = (channel.sequence, dec.received if dec.w.rank else dec.sent)
        return {'status': 'completed', 'phases': phases}

    for rank, dec in ((0, head), (1, worker)):
        dec.lanes = {3: object()}
        thin_job.attach(dec, record())
        dec.lanes = {}
    worker_ready = threading.Event()
    original_join = thin_job.join

    def follower():
        follow_once(worker, wire)                # the job's own FINISH message
        worker._thin_pending[3] = record()
        follow_once(worker, wire)                # THIN_START + IDLE, then joins
        worker_ready.set()

    thin_job.join = lambda *a, **k: original_join(*a, runner=runner, **k)
    try:
        thread = threading.Thread(target=follower)
        thread.start()
        head._thin_pending[3] = record()
        original_lead = thin_job.lead
        thin_job.lead = lambda d, r: original_lead(d, r, runner=runner)
        try:
            head.finish([])
        finally:
            thin_job.lead = original_lead
        thread.join(5)
    finally:
        thin_job.join = original_join
    assert worker_ready.is_set()
    assert entered[0] == entered[1] == (1, 32396)
    # The worker counter was advanced only by native follow, never hand-patched.
    assert worker.received == head.sent
    start, _ = wire.messages[1]
    assert start[0] == thin_job.THIN_START and start[-2:] == [IDLE, 0]


def test_head_never_starts_alone_and_cancels_only_the_job(native):
    store, wire = Store(), Wire()
    head = decoder(0, store, wire)
    ran = []
    began = time.monotonic()
    report = thin_job.lead(head, record(), join_timeout=.2,
                           runner=lambda *a: ran.append(1))
    assert time.monotonic() - began < 2
    assert not ran
    assert report['status'] == 'cancelled' and report['serving_affected'] is False
    assert head._thin_running is False and head.broken is None
    assert b'cancel' in store.values['paired-jobs/' + 'ab' * 16 + '/1/decision/rank0']


def test_worker_without_the_record_makes_head_cancel():
    store, wire = Store(), Wire()
    head, worker = decoder(0, store, wire), decoder(1, store, wire)
    worker._thin_pending = {}
    result = {}
    thread = threading.Thread(target=lambda: result.setdefault(
        'w', thin_job.join(worker, 1, 'ab' * 16, 0, runner=lambda *a: pytest.fail('ran'))))
    thread.start()
    report = thin_job.lead(head, record(), join_timeout=2, runner=lambda *a: pytest.fail('ran'))
    thread.join(5)
    assert report['status'] == 'cancelled' and 'lacks' in report['error']
    assert result['w']['status'] == 'cancelled'


def test_head_defers_the_job_while_chats_are_live():
    store, wire = Store(), Wire()
    head = decoder(0, store, wire)
    head.lanes = {3: object()}
    thin_job.attach(head, record())
    head._thin_pending[3] = record()
    head.lanes = {9: object()}
    head.finish([])
    assert 3 in head._thin_pending and len(wire.messages) == 1


def test_strip_start_keeps_serving_ops_and_rejects_truncation():
    words = thin_job.encode_id('0123456789abcdef' * 2)
    kept, starts = thin_job.strip_start([FINISH, 2, 3, 0, thin_job.THIN_START, 9, 1, *words, IDLE, 0])
    assert kept == [FINISH, 2, 3, 0, IDLE, 0]
    assert thin_job.decode_id(starts[0][1:]) == '0123456789abcdef' * 2
    with pytest.raises(RuntimeError, match='Malformed'):
        thin_job.strip_start([FINISH, 5, 1])


def test_trainer_channel_rejects_foreign_records():
    store = Store()
    head = thin_job.TrainerChannel(store, 'ab' * 16, 1, 0)
    worker = thin_job.TrainerChannel(store, 'ab' * 16, 1, 1)
    store.set(head.key('check-1', 1), b'{"tag": "chat", "value": 45309}')
    with pytest.raises(RuntimeError, match='mismatch'):
        head.take('check-1', 1, .1)
    with pytest.raises(TimeoutError):
        worker.take('missing', 0, .05)


def test_nested_serving_iteration_ends_with_resume_and_worker_consumes_every_message(monkeypatch):
    store, wire = Store(), Wire()
    head, worker = decoder(0, store, wire), decoder(1, store, wire)
    for dec in (head, worker):
        dec.lanes = {3: object()}
        thin_job.attach(dec, record())
        dec.lanes = {}
    head.idle = worker.idle = True

    def iteration():                      # one iteration, two sealed messages (round, then FINISH+IDLE)
        head._emit(FINISH + 100, [])
        head._flush()
        head._emit(FINISH, [3, 0])
        head._emit(IDLE, [])
        head._flush()
        head.idle = True

    head.live = lambda: 0
    head.g.scheduler = SimpleNamespace(waiting=SimpleNamespace(empty=lambda: False), held=None,
                                       loading=[], ready=[], _iteration=iteration)
    worker.follow = lambda once=False: follow_once(worker, wire)
    head_channel = thin_job.TrainerChannel(store, 'ab' * 16, 1, 0)
    worker_channel = thin_job.TrainerChannel(store, 'ab' * 16, 1, 1)
    head.w.device = worker.w.device = 'cpu'
    done = {}
    thread = threading.Thread(target=lambda: done.setdefault('w', thin_job.serve_waiting(worker, worker_channel, 1)))
    thread.start()
    assert thin_job.serve_waiting(head, head_channel, 1)
    thread.join(5)
    assert done['w'] is True
    assert worker.received == head.sent == 3 and worker.idle and head.idle


def test_worker_skips_the_doorbell_when_the_job_ends_with_live_chats(monkeypatch):
    store, wire = Store(), Wire()
    worker = decoder(1, store, wire)
    rang = []
    worker.g._await_bell = lambda: rang.append(1)
    worker.lanes = {3: object()}
    thin_job.attach(worker, record())

    def fake_join(dec, *args):
        dec.idle = False                   # last RESUME carried no IDLE

    monkeypatch.setattr(thin_job, 'join', fake_join)
    worker._thin_start = (1, 'ab' * 16, 0)
    worker.g._await_bell()
    assert rang == []
    worker.idle = True
    worker.g._await_bell()
    assert rang == [1]
