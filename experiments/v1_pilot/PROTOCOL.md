# Experimental Protocol: Central Analyst vs Ensemble vs Prediction Market

Protocol version: `aggregation-pilot-v1.0`

## Research question

When the same noisy private information is distributed across a fixed number of
LLM calls, does a prediction market aggregate it more accurately than a single
sequential LLM analyst or a conventional ensemble?

The pilot evaluates an architecture, not a claim that LLMs possess preferences
or beliefs. All reported behavior is behavior induced by the specified model,
prompt, inference settings, and mechanism.

## Primary estimand and hypotheses

For event `e` and system `s`, let `p*e` be the exact Bayesian posterior after all
five signals and `pse` the system forecast. The primary loss is:

`aggregation_error_se = (pse - p*e)^2`.

Primary paired contrasts are:

1. market minus log-odds ensemble aggregation error;
2. market minus central analyst aggregation error.

A negative contrast favors the market. Market minus arithmetic-mean ensemble is
reported as a secondary contrast. Brier score against the realized outcome and
log loss are secondary because, in a small sample, they contain outcome noise
that is absent from the known-posterior aggregation metric.

This is a pilot. Confidence intervals describe uncertainty across the sampled
synthetic worlds; they are not a claim of generality across models, prompts, or
real forecasting tasks.

## Experimental unit and paired worlds

The experimental unit is one synthetic binary event. The default run contains
30 events. Every event is evaluated by every system using exactly the same:

- prior probability;
- realized binary outcome;
- five conditionally independent private signals;
- signal reliabilities;
- mapping of signals to agents;
- signal reveal/trading order;
- matched generation seeds by position.

The paired design prevents differences in world difficulty or lucky outcomes
from being mistaken for mechanism effects.

## Data-generating process

Five prior probabilities (`0.30`, `0.40`, `0.50`, `0.60`, `0.70`) occur equally
often in the 30-event default run and are shuffled before use. For each event:

1. outcome `Y` is drawn from `Bernoulli(prior)`;
2. reliabilities `0.55`, `0.60`, `0.65`, `0.70`, `0.75` are randomly assigned to
   the five agent identities;
3. each signal independently equals `Y` with its assigned reliability;
4. the order of the five positions is randomized;
5. the exact full-information posterior is calculated from Bayes' rule.

The model is told the common prior, conditional independence assumption, and
relevant reliabilities. It is never shown the realized outcome or the oracle
posterior.

## Conditions and inference-budget control

Each condition schedules exactly five model decisions per event. Formatting or
network retries are recorded. Because prompts necessarily differ in length,
Ollama prompt and output token counts are also saved and reported rather than
pretending that equal call counts imply identical token cost.

### Central sequential analyst

One analyst receives the five signals sequentially in the market order. At step
`k`, it sees exactly the first `k` signals, its previous forecast and its private
reasoning note. Its fifth forecast is the condition's final forecast.

### Independent ensemble

Five isolated agents each see the prior and exactly one private signal. They do
not see other agents' signals, forecasts, or reasoning. Their five forecasts are
combined without further model calls in two pre-specified ways:

- arithmetic mean;
- prior-corrected logarithmic opinion pool (the primary ensemble comparator).

The log pool is included because a market should not be credited merely for
beating a weak averaging rule.

### Prediction market

Five traders act once each in randomized order under a logarithmic market
scoring rule. The initial market price equals the prior. A trader sees its
private signal, the current public price, the sequence of previous prices, and
the public reliabilities/identities of previous traders, but never their signals
or private reasoning. It chooses a target probability in `[0.01, 0.99]`; the
market price moves to that target. The fifth price is the market forecast.

The same probability bounds apply to central and ensemble forecasts. This
prevents the market from being mechanically disadvantaged by finite-price
bounds that do not apply to its comparators.

Trader realized payoff is the change in logarithmic score caused by its price
move. Payoffs are diagnostic and do not constrain trading in this aggregation
pilot. This avoids liquidity and bankruptcy effects becoming additional
treatments.

### Benchmarks

- `prior`: no aggregation;
- `oracle_bayes`: exact posterior given all five signals.

No LLM calls are used for the benchmarks.

## Prompt and model controls

- All conditions use the same Ollama model and decoding settings.
- Output examples containing literal probabilities or signal choices are
  prohibited to avoid anchoring/copying.
- Prompts use the same neutral forecasting objective and the same explanation
  of conditional signal reliability.
- Each response must contain a probability and a short private reason.
- Invalid outputs are retried; there is no behavioral fallback. Persistent
  failure stops the run, which can then be resumed.
- Events are independent and no cross-event performance history is supplied.

## Metrics

Primary:

- mean squared distance from the exact posterior;
- paired market-minus-comparator differences with deterministic bootstrap 95%
  confidence intervals.

Secondary:

- absolute distance from the posterior;
- Brier score;
- log loss;
- event-level win rates;
- individual private-forecast error;
- market price path, trading payoff, and total movement;
- central forecast path;
- ensemble disagreement.

## Pre-specified validity requirements

A run is marked complete and valid only if all checks pass:

- all configured events use the pre-generated paired worlds;
- every event has five central, five ensemble, and five market calls;
- all forecasts and prices are finite and within declared bounds;
- no condition receives hidden signals;
- all oracle posteriors recompute exactly from saved worlds;
- ensemble aggregations and market scoring payoffs recompute exactly;
- no silent fallback or skipped model decision occurs;
- configuration and protocol fingerprints match on resume;
- all expected event-system observations are present exactly once.

## Interpretation rule

The market is considered promising only if it has lower average posterior error
than both ensemble aggregators and the central analyst, or if it offers a clear
accuracy/compute advantage that survives paired analysis. If it only beats the
arithmetic mean but loses to the log pool, the conclusion is that standard
forecast aggregation dominates the added market mechanism in this environment.

Only after this aggregation pilot would a later version add endogenous
information purchase and budgets.

## Closest related work

This pilot does not claim to be the first AI-agent prediction market. Galanis
(`Information Aggregation with AI Agents`, arXiv:2604.20050) studies LLM traders,
including homogeneous `qwen3:8b` teams, across several logical information
structures. The distinguishing test here is the paired, scheduled-call-matched
comparison against a central sequential analyst and two ensemble aggregators on
noisy probabilistic signals. The market mechanism follows Hanson's logarithmic
market scoring rule.
