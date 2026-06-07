# STATUS — where we are right now

> This is the **boot-up file**. Read this first every session. It is the single
> source of truth for "where did we leave off." Keep it short and current.

**Last updated:** 2026-06-07
**Video:** Karpathy — "The spelled-out intro to neural networks and backpropagation: building micrograd"
**Current position in video:** `00:01:27` (intro / overview)
**Current milestone:** M0 → heading into M1

---

## The map (milestones)

Rough arc of the video. We tick these off as we go; order/exact boundaries can shift.

- [x] **M0 — Setup.** Repo, journal, auto-save working.
- [ ] **M1 — The `Value` object.** Wrap a scalar; track `data`. Add `+` and `*`. Build an expression graph (children + the op that made each node).
- [ ] **M2 — Visualizing the graph.** Draw the DAG so we can *see* the forward pass.
- [ ] **M3 — Manual backprop.** Compute gradients by hand for a tiny graph. Understand the chain rule node-by-node.
- [ ] **M4 — `tanh` + the neuron.** Add an activation; build a single neuron's forward pass as a graph.
- [ ] **M5 — Autograd.** `_backward` per op + topological sort → `.backward()` does the whole graph automatically.
- [ ] **M6 — Bug hunt: `+=` for grads.** Why a node used twice needs accumulation, not assignment.
- [ ] **M7 — Breaking down `tanh`.** Reimplement it from `exp`/`pow`/`div` to prove the graph composes.
- [ ] **M8 — PyTorch parity.** Same tiny net in PyTorch; confirm our grads match.
- [ ] **M9 — Neuron / Layer / MLP.** Build the network abstractions.
- [ ] **M10 — Training loop.** Loss, `zero_grad`, gradient descent. Watch loss go down. 🎉

---

## Right now

**Next step:** Keep watching from ~1:27. When the `Value` class first appears, pause
and create `code/value.py`. Talk through it with me as you type.

**Currently stuck on:** nothing — gradient intuition clicked via the "warmer/colder"
analogy + the rolling-ball / step-size demos (see journal & viz/). Understands:
slope = downhill direction for the error; step opposite it = training; step size =
learning rate (too big → overshoot/explode). Built viz 01–03.

**Concepts solid now:** gradient = slope (direction + steepness); **nudge = the test
that measures the slope; step = the real move that uses it (learning rate = how big)**;
step too big → overshoot/explode. Kevin derived the nudge-vs-step distinction himself —
it clicked. Built viz 01–04.

**Not yet covered:** the chain rule / how the slope is computed through a *chain* of
operations (the actual backprop) — natural next intuition, then back to the video and
the `Value` object.

**Open questions / things to revisit:**
- Re-read the autograd/backprop journal note *after* finishing M5 (`.backward()`) and
  check whether it actually clicked the way the intro promised.
