# Limitations and Interpretation Boundaries

## What the evidence supports

Under the frozen synthetic tasks, prompts, model, and execution settings, the market-framed condition produced final probabilities closer to the exact Bayesian posterior than the tested centralized baselines. V1 also found lower market error than two ensemble aggregators.

## What the evidence does not support

The experiments do not establish that prediction markets generally outperform single LLMs, ensembles, or other multi-agent architectures.

## Main limitations

### One model and one local deployment

Both experiments use the same locally served `qwen3:8b` model. V2 records an 8.2B-parameter Q4_K_M model digest. Results may not transfer to other model sizes, families, quantizations, providers, or inference implementations.

### Synthetic, explicitly structured evidence

Signals are binary, conditionally independent, and paired with known reliabilities supplied directly to the model. Real forecasting involves ambiguous language, correlated evidence, source selection, conflicting reports, and uncertain reliability.

### Framing is confounded with organization

In V2, `central_compact` and `market` receive the same numerical state but different descriptions. One is instructed to act as the same analyst; the other as a new trader under a logarithmic market scoring rule. Because the underlying Ollama calls are stateless, the measured advantage may be caused by prompt framing or role assignment rather than decentralization in a stronger economic sense.

### Agents are not persistent economic actors

Traders do not learn across tasks, maintain constrained wealth, choose whether to acquire information, or face real consequences from their payoffs. The market-scoring rule is part of the decision frame, but payoff and bankruptcy constraints are deliberately excluded.

### Limited prompt robustness

Each mechanism uses one frozen prompt family. The design prevents post-result prompt tuning, but it does not show that results are robust to neutral paraphrases or alternative instructions.

### Exhaustive does not mean universal

V2 covers every prior-by-binary-signal-pattern cell in the defined finite task. It does not enumerate every possible reliability permutation, reveal-order design, model response, prompt, or real-world information structure.

### No repeated stochastic response surface

Each V2 cell is executed once per condition with a matched generation seed and temperature zero. The weighted result is exact for the saved response surface, but it does not measure inference-level variability across repeated runs or seeds.

### Compute is not lower

V2 uses approximately 1,952 tokens per market cell, compared with 1,927 for `central_compact` and 1,771 for `central_full`. The result is an accuracy finding, not an accuracy-per-token advantage.

### Ensembles are not included in V2

The ensemble comparison comes from the smaller V1 pilot. V2 focuses on distinguishing the market from centralized full-evidence and compressed-state updating.

## Appropriate wording

Recommended:

> In a controlled synthetic experiment with Qwen3 8B, market-framed sequential updates produced lower oracle aggregation error than centralized full-evidence and compressed-state baselines.

Avoid:

> Multi-agent prediction markets are proven to aggregate information better than single LLMs.

