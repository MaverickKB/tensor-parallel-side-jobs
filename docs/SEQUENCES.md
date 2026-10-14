# Sequence diagrams

## 1. Paired sealed start

```mermaid
sequenceDiagram
    participant H as Head (rank 0)
    participant S as Sealed serving stream
    participant W as Worker (rank 1)
    participant C as Trainer channel (KV store)
    Note over H: request finished, pair idle, job pending
    H->>S: message N = [THIN_START seq id..., IDLE]
    S->>W: message N (in order)
    W->>W: strip THIN_START, remember (seq, id, N)
    H->>C: join {message: N, ready}
    W->>C: join {message: N, ready}
    C-->>H: worker join
    alt same N and both ready
        H->>C: decision = go
        C-->>W: go
        Note over H,W: both run the job body together
    else mismatch or timeout
        H->>C: decision = cancel
        Note over H,W: job cancelled, serving untouched
    end
```

## 2. Nested serving during a job (resume marker)

```mermaid
sequenceDiagram
    participant H as Head
    participant C as Trainer channel
    participant S as Sealed stream
    participant W as Worker
    H->>C: serve-k = 1 (chats waiting)
    C-->>W: serve-k = 1
    H->>S: message: round ops
    H->>S: message: FINISH, IDLE
    H->>S: message: THIN_RESUME [, IDLE]
    S->>W: worker follows every message
    W->>W: sees THIN_RESUME, returns to the job
    Note over H,W: next job step on both ranks
```

Before the fix the worker returned after the first message and stalled the backward pass.

## 3. Doorbell skip after the job

```mermaid
sequenceDiagram
    participant H as Head
    participant W as Worker
    Note over H,W: job ends; last THIN_RESUME carried no IDLE (chats live)
    alt worker idle
        W->>W: wait for doorbell
    else chats live
        W->>W: skip doorbell, read next sealed message
    end
```

## 4. Hot-load

```mermaid
sequenceDiagram
    participant H as Head
    participant C as Trainer channel
    participant W as Worker
    H->>H: hash staged files, refuse restart-only
    W->>W: hash staged files, refuse restart-only
    H->>C: verdict + hashes
    W->>C: verdict + hashes
    alt both ok and hashes equal
        H->>H: exec exact hashed bytes into existing modules
        W->>W: exec exact hashed bytes into existing modules
        H->>C: reloaded list
        W->>C: reloaded list (must match)
    else
        Note over H,W: refused on both ranks, nothing changes
    end
```
