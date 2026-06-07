# micrograd-journey

Learning Andrej Karpathy's [micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)
by building it from scratch — and documenting the real, messy journey as I go.

This repo is both the code and the blog.

## Boot up a session

```bash
cd ~/micrograd-journey && claude
```

Claude reads `STATUS.md`, picks up where we left off, and keeps the journal updated
as we work. Progress auto-saves to git after every turn — no manual saving.

## Layout

| Path        | What                                                            |
|-------------|-----------------------------------------------------------------|
| `STATUS.md` | Where we are right now. The boot-up snapshot.                   |
| `journal/`  | Dated session logs — the raw thoughts, detours, and aha moments.|
| `code/`     | The micrograd I write, step by step.                            |
| `posts/`    | Clean blog series, assembled from the journal at the end.       |
| `CLAUDE.md` | How Claude coaches + scribes in this repo.                      |
