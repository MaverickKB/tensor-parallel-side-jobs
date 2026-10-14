# tensor-parallel-side-jobs

Run a side job (training step, activation capture, evaluation, code update) **inside a live
two-rank tensor-parallel server** without desynchronising its collectives, without blocking
chats for long, and without restarting it.

> **Teaching example, not a product.** It captures a pattern and the hard lessons behind it.
> Adapt the ideas to your engine; do not copy it 1:1.

> **About this work.** This is an honest, ongoing investigation into what is possible, not a
> claim of perfection. We make mistakes and backfill transparency as we go, in the hope that
> others benefit and can skip the hard lessons we hit.

## The problem

In tensor-parallel serving, every rank must call the same collectives in the same order. A
side job that starts on one rank a message earlier than on the other pairs a job all-gather
with a serving all-gather, and serving hangs for everyone. Restarting to change job code
takes the model down.

## The pattern (four parts)

1. **Paired sealed start.** The head puts a `THIN_START` op into the ordered (sealed) serving
   message stream, so the worker sees it at the same message number. Both ranks confirm that
   number and their readiness on a **trainer-only channel** (a key-value store, never the
   serving collectives). Any mismatch or timeout cancels only the job.
2. **Nested serving with a resume marker.** Between job steps both ranks agree whether chats
   are waiting. If so, the head runs one serving iteration and closes it with an explicit
   `THIN_RESUME` op; the worker keeps reading messages until it sees it.
3. **Doorbell skip.** After the job, if chats are still live, the worker reads its next
   message instead of waiting for an idle doorbell that will not ring.
4. **Hot-load.** Both ranks hash the staged files, agree, then execute exactly those bytes into
   the existing module objects. Files whose objects are held by live hooks are restart-only.

See `docs/SEQUENCES.md` for diagrams and `docs/PORTING.md` to adapt it.

## What is here

| Path | What |
|---|---|
| `src/tp_side_jobs/protocol.py` | Start, join, cancel, trainer channel, nested serving, resume, doorbell skip. Engine-specific parts are hooks. |
| `src/tp_side_jobs/hotload.py` | Paired no-restart code loading. |
| `examples/two_rank_demo.py` | Two OS processes, torch.distributed **gloo** on CPU: a tiny server with a sealed message stream and a serving all-gather, a side job that serves a chat mid-job and hot-loads a module, and a mismatch mode that cancels only the job. |
| `tests/` | Protocol tests with fake decoders, hot-load tests, and the two-process demo end to end. |

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[test]'
pytest                                   # 14 tests, CPU only
python examples/two_rank_demo.py         # add --mismatch to see a cancelled job
```

## Evidence and limits

- **Ran live** on one setup: GLM-5.3-Flash (EXL3) tensor-parallel across two machines, inside the
  resident trainer of the origin project. It fixed three observed failures (a serving freeze from
  collective desync, a backward-pass stall, a post-job wait) and the stale-bytecode reload problem
  (see `LESSONS.md`).
- The gloo demo is a **simulation**: its "server" is a few lines, not a real engine. It shows the
  ordering logic, not performance.
- **Untested:** more than two ranks, pipeline parallelism, engines that do not seal messages in
  order, and NCCL-based serving collectives (the origin used its own engine's transport).
- This is an engineering pattern, not a research result.

## License

MIT. See `LICENSE` and `NOTICE`.

## Why, how and origin

- [docs/WHY_HOW.md](docs/WHY_HOW.md): why this exists and how it works.
- [docs/ORIGIN_STORY.md](docs/ORIGIN_STORY.md): how we got here (links to the master story and lessons in moe-live-learning-tools).
