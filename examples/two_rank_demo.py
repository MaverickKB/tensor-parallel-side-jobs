"""Two-process CPU demo: a side job inside a live "server", with torch.distributed (gloo).

Two OS processes play the two tensor-parallel ranks of a tiny server:

- Rank 0 (head) owns the request queue. Every serving step it sends one *sealed message*
  (an ordered list of ops) to rank 1 with `broadcast_object_list`. A `CHAT` op makes both
  ranks run the serving collective (an `all_gather`), as a real TP forward would.
- Part-way through, the head puts a `THIN_START` op into the sealed stream. Both ranks see it
  at the same message number, confirm that number on a trainer-only channel (the rendezvous
  `TCPStore`, never the serving collective), and enter the side job together.
- The side job runs a few steps. Before each step both ranks agree over the trainer channel
  whether to serve waiting chats; if so the head sends one nested serving message that ends
  with an explicit `THIN_RESUME` op, and the worker keeps reading until it sees it.
- The last step hot-loads a staged module on both ranks (hash, agree, exec exact bytes).
- With `--mismatch`, the worker reports the wrong message number: only the job is cancelled
  and serving continues.

Run:  python examples/two_rank_demo.py [--mismatch]
"""

import argparse
import hashlib
import json
import os
import socket
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tp_side_jobs import protocol
from tp_side_jobs.hotload import hotload

CHAT = 1
IDLE = 8
JOB_ID = '0123456789abcdef' * 2
STAGED = 'VERSION = 2\n'


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def serve_chat(rank, step):
    """The serving collective: both ranks must call it in the same order."""
    local = torch.tensor([float(rank * 100 + step)])
    out = [torch.zeros(1) for _ in range(2)]
    dist.all_gather(out, local)
    return [float(t) for t in out]


MAX_WORDS = 64


def send(message):
    """Seal one message: a fixed-size int64 frame [length, ops...] broadcast from rank 0."""
    frame = torch.zeros(MAX_WORDS + 1, dtype=torch.int64)
    frame[0] = len(message)
    frame[1:1 + len(message)] = torch.tensor(message, dtype=torch.int64)
    dist.broadcast(frame, src=0)
    return message


def receive():
    frame = torch.zeros(MAX_WORDS + 1, dtype=torch.int64)
    dist.broadcast(frame, src=0)
    return [int(x) for x in frame[1:1 + int(frame[0])]]


def apply(rank, body, step, log):
    """Run the serving ops of one sealed message; return the start payloads it carried."""
    kept, starts = protocol.strip_start(body)
    index = 0
    while index < len(kept):
        op, size = kept[index], kept[index + 1]
        if op == CHAT:
            log.append(('chat', step, serve_chat(rank, step)))
        index += 2 + size
    return starts


def staged_package(rank):
    root = Path(tempfile.mkdtemp(prefix='side-jobs-rank%d-' % rank)) / 'demo_stage'
    root.mkdir()
    (root / '__init__.py').write_text('')
    (root / 'stage.py').write_text('VERSION = 1\n')
    sys.path.insert(0, str(root.parent))
    import demo_stage.stage  # noqa: F401
    (root / 'stage.py').write_text(STAGED)
    return root


def side_job(rank, channel, step_box, log, steps=3):
    """Both ranks run this in lockstep; between steps they may serve chats."""
    total = 0
    for i in range(steps):
        wanted = channel.exchange('serve-%d' % i, int(rank == 0 and i == 1))[0]
        if wanted:
            if rank == 0:
                step_box[0] += 1
                send([CHAT, 0, protocol.THIN_RESUME, 0])
                apply(0, [CHAT, 0], step_box[0], log)
            else:
                while True:
                    step_box[0] += 1
                    body = receive()
                    starts = apply(1, body, step_box[0], log)
                    if None in starts:          # THIN_RESUME: return to the job
                        break
            log.append(('served_during_job', i))
        values = channel.exchange('step-%d' % i, rank + i)
        total += sum(values)
        log.append(('job_step', i, values))
    root = staged_package(rank)
    expected = {'stage.py': hashlib.sha256(STAGED.encode()).hexdigest()}
    report = hotload('demo_stage', root, expected, channel.exchange, order=('stage',))
    import demo_stage.stage as stage
    log.append(('hotload', report['hotload'], stage.VERSION))
    return total


def worker_main(rank, port, mismatch, results):
    store = dist.TCPStore('127.0.0.1', port, 2, rank == 0, timeout=timedelta(seconds=30))
    dist.init_process_group('gloo', store=dist.PrefixStore('serving', store), rank=rank, world_size=2)
    log, step = [], [0]
    try:
        plan = [[CHAT, 0], [CHAT, 0], 'start', [CHAT, 0], [IDLE, 0]]
        for item in plan if rank == 0 else range(len(plan)):
            step[0] += 1
            if rank == 0:
                if item == 'start':
                    body = [protocol.THIN_START, 9, 1, *protocol.encode_id(JOB_ID), IDLE, 0]
                else:
                    body = item
                send(body)
            else:
                body = receive()
            starts = apply(rank, body, step[0], log)
            if starts and starts[0] is not None:
                sequence, identifier = int(starts[0][0]), protocol.decode_id(starts[0][1:])
                message = step[0] + (7 if (mismatch and rank == 1) else 0)
                channel = protocol.TrainerChannel(store, identifier, sequence, rank)
                joined = channel.exchange('join', {'message': message})
                if joined[0]['message'] != joined[1]['message']:
                    log.append(('cancelled', 'message numbers differ: ' + json.dumps(joined)))
                else:
                    log.append(('job_total', side_job(rank, channel, step, log)))
        dist.barrier()
    finally:
        dist.destroy_process_group()
    results.put((rank, log))


def run(mismatch=False):
    port = free_port()
    context = mp.get_context('spawn')
    results = context.Queue()
    procs = [context.Process(target=worker_main, args=(r, port, mismatch, results)) for r in range(2)]
    for p in procs:
        p.start()
    try:
        logs = dict(results.get(timeout=90) for _ in procs)
    finally:
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.kill()
    return logs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mismatch', action='store_true')
    args = parser.parse_args()
    began = time.time()
    for rank, log in sorted(run(args.mismatch).items()):
        print('rank', rank)
        for row in log:
            print('  ', row)
    print('seconds %.1f' % (time.time() - began), file=sys.stderr)
    sys.stdout.flush()
    os._exit(0)
