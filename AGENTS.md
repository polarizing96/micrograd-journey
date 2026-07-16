# AGENTS.md

Coaching + scribe workflow, repo layout, and session-start behavior live in `CLAUDE.md`
and `STATUS.md` — read those first. Key rule to remember: **Kevin writes the micrograd
code in `code/` himself**; coach with hints, don't hand him finished solutions.

## Cursor Cloud specific instructions

This is a learning-journey repo, not a deployable app. There is no test suite, lint
config, or build step yet, and `code/` is intentionally empty (Kevin fills it in as the
video progresses). The two runnable things are the Python learning stack and the static
viz demos.

### Python environment
- A virtualenv lives at `.venv` (gitignored). Use `.venv/bin/python` / `.venv/bin/pip`,
  or `source .venv/bin/activate`. The update script creates it and installs the stack:
  `numpy`, `matplotlib`, `graphviz`, `jupyter`, `ipython`, and CPU `torch` (from the
  pytorch CPU wheel index). These mirror what Karpathy uses in the video (numpy/matplotlib
  for plots, graphviz for the DAG, torch for the M8 parity check).
- System deps (baked into the VM, NOT in the update script since they're apt packages):
  the graphviz `dot` binary and `python3.12-venv`. If a fresh pod is missing them:
  `sudo apt-get update && sudo apt-get install -y graphviz python3.12-venv`. The `dot`
  binary is what `graphviz` (Python) shells out to when rendering the `Value` graph to
  SVG (milestone M2) — without it, `.render(...)` fails even though the import succeeds.

### Viz demos (`viz/*.html`)
- Fully self-contained HTML/CSS/JS (no external CDNs, no build). Serve the folder with
  `python3 -m http.server 8000` from `viz/` and open e.g.
  `http://localhost:8000/01-gradient-is-a-slope.html`, or just open the file directly.

### Auto-commit hook
- `.claude/settings.json` defines a `Stop` hook that auto-commits after each turn. That's
  a Claude Code hook and does not run under Cursor cloud agents — commit/push manually.
