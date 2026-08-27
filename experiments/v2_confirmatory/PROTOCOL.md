# Confirmatory protocol: market mechanism vs centralized aggregation

Protocol version: `aggregation-confirmatory-v2.0`

## Short protocol summary

This experiment asks whether a five-agent prediction-market mechanism aggregates the same noisy
information better than a single LLM. It compares three systems on 160 paired information cells:

1. `central_full`: one analyst repeatedly sees all signals revealed so far;
2. `central_compact`: one analyst updates a public probability using one new signal at a time;
3. `market`: five one-shot traders update a public market price in sequence.

The second baseline is decisive. If the market beats `central_full` but not `central_compact`, the
benefit comes from compressed sequential state, not from multi-agent organization.

## Frozen research question and interpretation

The primary mechanism contrast is:

`market oracle squared error - central_compact oracle squared error`.

The V1 replication contrast is:

`market oracle squared error - central_full oracle squared error`.

Negative values favor the market. A genuine multi-agent advantage requires the market to beat both
central baselines on the population-weighted primary metric. If it only beats `central_full`, the
result supports a computational-scaffolding explanation. If `central_compact` matches or beats the
market, a market is not necessary for this task.

## Exhaustive paired design

The experiment has five priors (`0.30` through `0.70`) and five binary private signals with fixed
reliabilities (`0.55`, `0.60`, `0.65`, `0.70`, `0.75`). All `5 x 2^5 = 160` prior-by-signal-pattern
cells are evaluated exactly once by every system. The execution order is shuffled deterministically.
Reliabilities occupy every sequential position equally often.

Each cell receives its exact probability under the data-generating process. Metrics are weighted by
these probabilities, so the final mean is the expected performance under the intended model rather
than the average of an artificial uniform distribution of signal patterns. No realized outcome is
sampled. Expected Brier score and expected log loss are computed analytically from the exact oracle
posterior, eliminating outcome luck.

## Information and inference controls

Every LLM condition uses exactly five calls per cell, the same `qwen3:8b` model, temperature zero,
the same probability bounds, and a matched generation seed at each position. Condition execution
order rotates across cells. The model digest is recorded and must remain unchanged on resume.

- `central_full` receives the prior, every signal visible so far, and its previous probability.
- `central_compact` receives the current public probability, its public path, and one new signal.
- `market` receives the same numerical state as `central_compact`, but is framed as a new one-shot
  trader under a logarithmic market scoring rule.

Neither compressed condition sees previous signal values or reasoning. Neither any prompt sees the
oracle posterior. Prompt and output tokens are recorded. Scheduled calls and maximum output budgets
are exactly matched; observed token differences caused by task content are reported explicitly.

## Metrics

Primary: population-weighted squared distance between the final forecast and the exact posterior.

Secondary: weighted absolute error, RMSE, expected Brier score, expected log loss, weighted win
probability, unweighted robustness results, tokens, retries, probability-path movement, local
Bayesian-update error, and directional response to each signal.

Because the complete finite signal state space is evaluated, the population-weighted result is exact
for this frozen model response surface. A conventional confidence interval over randomly sampled
synthetic worlds would be misleading and is not used. The experiment does not establish generality
across models, prompts, or real-world forecasting tasks.

## Validity rules

A complete run is mechanically valid only if all pre-specified checks pass, including exact world
regeneration, exhaustive coverage, probability weights, oracle recomputation, five calls per
condition and cell, matched seeds, counterbalanced execution order, prompt reconstruction,
information firewalls, probability bounds, state-chain accounting, raw-response parsing, token
accounting, unique observations, fingerprints, model digest, and complete sample.

There is no behavioral fallback. Persistent invalid output stops the run and preserves completed
cells for resume. Development runs can be mechanically valid but are never marked confirmatory.
