# Lessons

Each entry: symptom, cause, fix. Entries marked *inferred* are reconstructed from code and notes, not from a written incident record.

## 1. Collective desync froze serving
- **Symptom:** Serving hung for everyone when a job started.
- **Cause:** The ranks entered the job's collectives at different points of the serving stream, so a job all-gather paired with a serving all-gather.
- **Fix:** The paired sealed start: carry the start in the sealed stream, confirm the message number on a trainer-only channel, cancel only the job on mismatch.

## 2. Backward-pass stall from a half-read iteration
- **Symptom:** The backward pass stalled while chats waited.
- **Cause:** One head scheduler iteration can flush several sealed messages; the worker read one of two and returned to the job early.
- **Fix:** End every nested serving iteration with an explicit `THIN_RESUME`; the worker follows messages until it sees it.

## 3. Post-job wait while a chat was live
- **Symptom:** After a job the worker waited for a doorbell while a chat was live.
- **Cause:** The head rings the doorbell only when idle; the job's last resume left it busy.
- **Fix:** The doorbell skip: if not idle after the job, read the next message instead of waiting.
Each entry: symptom, cause, fix. Entries marked *inferred* are reconstructed from code and notes, not from a written incident record.

## 4. Stale bytecode on reload
- **Symptom:** After a code change a rank kept running old code.
- **Cause:** Python's normal reload path can serve cached bytecode that looks fresh (same size and mtime).
- **Fix:** Hash on every rank, agree, then compile and exec exactly those bytes. `test_executes_new_bytes_even_when_size_and_mtime_match` reproduces the trap.
