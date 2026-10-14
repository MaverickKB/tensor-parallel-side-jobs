# Porting guide

The protocol module talks to the engine through a small surface. Map each item to your engine.

## Hooks in `protocol.py` (module-level functions you rebind)

| Hook | Provide |
|---|---|
| `native_constants()` | `(IDLE op code, sealed-sequence modulus)` from your engine |
| `prepare(decoder, record)` | Local validation before joining; return an error string or `None` |
| `run(decoder, record, channel, phases)` | The job body; both ranks call it with the same arguments |
| `synchronize(decoder)` | Wait for device work before nested serving (e.g. `torch.cuda.synchronize`) |
| `write_receipt(rank, id, report)` | Persist a per-rank receipt (default: `PAIRED_JOBS_RECEIPTS` dir) |

## Decoder surface `attach()` expects

| Attribute | Meaning |
|---|---|
| `w.rank` | 0 for the head, 1 for the worker |
| `g.comm.store` | A key-value store with `set`, `get`, `wait(keys, timedelta)` (PyTorch `TCPStore` works) |
| `_emit(op, payload)`, `_flush()` | Head: append an op to the outgoing sealed message; seal and send it |
| `sent`, `received` | Message counters on head and worker |
| `apply(message)` | Worker: apply one unsealed message |
| `finish(done)` | Head: called when requests finish (the start hook goes here) |
| `g._await_bell()` | Worker: block until the head rings the idle doorbell |
| `lanes`, `idle`, `broken` | Live requests, idle flag, failure state |
| `live()`, `g.scheduler` | Head: whether chats are waiting; one scheduler iteration |
| `follow(once=True)` | Worker: read and apply the next message |

`tests/test_protocol.py` contains a complete fake of this surface, and
`examples/two_rank_demo.py` shows the same ideas with torch.distributed gloo.

## Steps

1. Find where your engine seals or orders head-to-worker messages. If it has no ordered stream,
   build one first; the whole pattern depends on a shared message number.
2. Choose op codes that cannot collide with serving ops.
3. Pick a side channel both ranks reach that is not a serving collective.
4. Wire `attach` at the head's request-finish point and the worker's doorbell wait.
5. Set `JOIN_TIMEOUT` short (a failed join should cancel fast) and `STEP_TIMEOUT` above your
   longest job step.
6. Keep a restart-only list for hot-load: any file whose objects are captured by hooks or
   closures.
7. Test in this order: the CPU tests, the gloo demo, then your engine with a no-op job, then a
   real job while sending chats.

## More than two ranks (untested)

`TrainerChannel.exchange` already loops over `world`. `attach`, `lead` and `join` assume one
head and one worker; generalise `join` to every non-head rank and have the head wait for all.
