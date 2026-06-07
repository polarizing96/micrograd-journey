# viz/ — interactive learning demos

Self-contained HTML pages built while learning. Open any in a browser (`open <file>`).
They capture the intuition-building detour before writing the micrograd code — and are
candidates to embed in the final blog posts.

| File | What it teaches |
|------|-----------------|
| `01-gradient-is-a-slope.html` | A gradient is a slope. Drag x and the nudge size h; watch the secant settle onto the tangent as h→0. |
| `02-why-slopes-matter.html` | The "so what": a ball rolls downhill on the error curve. Slope = downhill direction; step opposite it = training. |
| `03-step-size-learning-rate.html` | Step size = learning rate. Presets from tiny → explode show overshoot and divergence. |
| `04-the-nudge.html` | The nudge = how the slope is *measured*. Poke the knob, watch the output react, divide. (Speedometer idea.) |
| `05-calculus-crash-course.html` | The only 5 calculus pieces micrograd needs (slope, power, sum, product, chain rule) — lessons + instant-feedback quiz, each tied to the code. |

**Through-line:** gradient = slope (01) → why we care, rolling downhill (02) → how far to
step = learning rate (03) → how the slope is measured = the nudge (04) → how calculus
computes it instantly + the chain rule = backprop (05).
