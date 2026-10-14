# AGENTS.md

Use this repository when you must run a side job or change code inside a live multi-rank
tensor-parallel server. **Adapt the ideas; do not copy the code 1:1.** It was built for one
engine and two ranks.

## Map
- `src/tp_side_jobs/protocol.py`: start, join, cancel, trainer channel, nested serving, resume,
  doorbell skip. Hooks at module level are the engine boundary.
- `src/tp_side_jobs/hotload.py`: paired hot-load.
- `examples/two_rank_demo.py`: runnable two-process gloo version of the same logic.
- `tests/`: fakes for the decoder and store; read them to learn the interfaces.
- `docs/SEQUENCES.md`, `docs/PORTING.md`, `LESSONS.md`.

## Invariants
1. Never split the ranks: both run the job or neither does.
2. Both ranks enter at the same sealed message number.
3. Job traffic never uses the serving collectives.
4. A failed join cancels the job, never serving.
5. Nested serving always ends with an explicit resume marker.
6. Hot-load executes only bytes that every rank hashed and agreed on.

## Pitfalls
- One scheduler iteration can flush several messages. Never assume one.
- `importlib.reload` can run stale bytecode.
- Installed closures keep their old bodies after hot-load; only call-time lookups change.
- Long `JOIN_TIMEOUT` holds chats; keep it short.

## Before claiming it works on your stack
Run `pytest`, run the gloo demo with and without `--mismatch`, then test on your engine with
chats arriving during a job.
