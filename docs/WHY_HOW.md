# Why this exists and how it works

**Why.** Side jobs (training, capture, scoring) had to run inside a live two-rank
tensor-parallel server without restarts and without disturbing neighbours. The first naive
start froze serving; a normal reload served stale code. See `../LESSONS.md`.

**How.**
- `protocol.py`: the paired sealed start. The head puts the job start in the sealed serving
  stream; the worker confirms the message number on a separate channel; nested serving ends
  with an explicit resume op; the doorbell is skipped when chats are live.
- `hotload.py`: every rank hashes the staged files, ranks agree, and exactly those bytes are
  compiled and executed into the existing module objects; restart-only files are refused.
- Sequences: `SEQUENCES.md`. Porting: `PORTING.md`. Demo: `examples/two_rank_demo.py`.

Rules we follow around it: a free-memory floor that aborts only the side job, a single owner
for pair recovery that other jobs only read, and announced windows for anything disruptive.
No paths or hosts are hard-coded; pass them as settings.
