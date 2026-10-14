# Origin story (short)

The full chronological account lives in one master document:
[moe-live-learning-tools/docs/ORIGIN_STORY.md](https://github.com/MaverickKB/moe-live-learning-tools/blob/main/docs/ORIGIN_STORY.md). Project-wide research
lessons: [moe-live-learning-tools/docs/LESSONS.md](https://github.com/MaverickKB/moe-live-learning-tools/blob/main/docs/LESSONS.md).

## How this repository fits in the story

1. **Belief:** a trainer could simply start on both tensor-parallel ranks inside live serving.
   **Result:** serving froze for everyone; the ranks entered the job's collectives at
   different points of the serving stream, so a job all-gather paired with a serving one.
2. **Decision:** carry the start inside the sealed serving stream and confirm the message
   number on a trainer-only channel; on mismatch or timeout cancel only the job.
3. **Further evidence:** a backward pass stalled because one scheduler iteration flushed two
   sealed messages and the worker read one (fixed with an explicit resume op); a post-job
   wait delayed a live chat (fixed with the doorbell skip).
4. **Belief:** code changes could be picked up with a normal module reload. **Result:** a rank
   kept running stale bytecode. **Decision:** hash on every rank, agree, then execute exactly
   those bytes; refuse files that genuinely need a restart.
5. **Result:** both mechanisms ran live inside the resident trainer on the origin setup; this
   extracted version is CPU-tested only.

One setup, no independent replication; treat this as a starting point.
