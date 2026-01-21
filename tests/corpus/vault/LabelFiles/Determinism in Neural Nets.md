# Determinism in Neural Nets

### Floating Point math

The assertion (unproven) is that floating point - especially quantized fp, is inherently non deterministic. This is especially observable in aggregate model flows, like MoE. It is also asserted that GPU hardware does have mitigations for this.

<https://news.ycombinator.com/item?id=42957436>

> If temperature is zero, and weights are weights, where is the non-deterministic behavior coming from?

> This isn't really true unfortunately -- mixture of experts routing seems to suffer from batch non-determinism. No one has stated publicly exactly why this is, but you can easily replicate the behavior yourself or find bug reports / discussion with a bit of searching. The outcome and observed behavior of the major closed-weight LLM APIs is that a temperature of zero no longer corresponds to deterministic greedy sampling.

### MoE Non determinism 

<https://152334h.github.io/blog/non-determinism-in-gpt-4/>

> Under capacity constraints, all Sparse MoE approaches route tokens in groups of a fixed size and enforce (or encourage) balance within the group. When groups contain tokens from different sequences or inputs, these tokens often compete against each other for available spots in expert buffers. **As a consequence, the model is no longer deterministic at the sequence-level, but only at the batch-level**, as some input sequences may affect the final prediction for other inputs


