# Methodology

## Research question

The experiments ask whether a prediction-market-style sequence of LLM calls aggregates the same noisy private evidence more accurately than centralized or simpler multi-forecast baselines.

The study evaluates induced model behavior under fixed prompts and mechanisms. It does not assume that LLM agents possess beliefs, preferences, or economic agency.

## Shared data-generating process

Each task concerns a binary outcome. The common prior is one of `0.30`, `0.40`, `0.50`, `0.60`, or `0.70`. Five conditionally independent binary signals have fixed reliabilities of `0.55`, `0.60`, `0.65`, `0.70`, and `0.75`.

The exact posterior after all five signals is computed using Bayes' rule. This posterior is never shown to an LLM condition and serves as the oracle aggregation target.

## V1 pilot

V1 contains 30 sampled paired events. Every condition sees the same underlying prior, outcome, signals, reliability assignment, reveal order, and position-matched generation seeds.

The compared mechanisms are:

1. **Central analyst:** five sequential updates by one analyst who sees all signals revealed so far, its previous forecast, and its previous private reasoning note.
2. **Arithmetic-mean ensemble:** five isolated agents each see one signal; their probabilities are averaged.
3. **Prior-corrected log-odds pool:** the same isolated forecasts are combined in log-odds space while correcting repeated use of the common prior.
4. **Prediction market:** five one-shot traders sequentially move a public probability under a logarithmic market scoring rule.
5. **Benchmarks:** the prior and the exact Bayesian posterior.

The primary statistic is squared distance to the oracle posterior. Paired market-minus-comparator differences use a deterministic 10,000-repetition bootstrap interval across the 30 sampled events.

## V2 exhaustive design

V2 evaluates every combination of five priors and 32 binary signal patterns, for 160 cells. Each cell is evaluated once by each LLM condition. Reliability placement is balanced across sequential positions, and condition execution order is counterbalanced.

The compared mechanisms are:

1. **`central_full`:** the analyst sees the prior, all currently visible raw signals, and its previous probability.
2. **`central_compact`:** the analyst sees the current public probability, its public path, and one new signal. Earlier raw signals and reasoning are hidden.
3. **`market`:** a new one-shot trader sees the same numerical state as `central_compact`, framed as a trade under a logarithmic market scoring rule.
4. **Benchmarks:** the prior and exact Bayesian posterior.

Each cell receives its probability under the data-generating process. Reported weighted means therefore represent expected performance under that frozen process rather than an equal-weight average of artificially uniform signal patterns.

V2 does not sample a realized outcome. Expected Brier score and expected log loss are computed analytically from the oracle posterior, removing outcome luck.

## Inference controls

- Model: `qwen3:8b` through Ollama
- Calls: five per LLM condition and task
- V1 temperature: `0.2`
- V2 temperature: `0.0`
- Shared probability interval: `[0.01, 0.99]`
- Matched generation seeds by position and task
- No behavioral fallback for invalid responses
- Prompt, output, retry, and token records saved

Equal call counts do not imply equal compute. Prompt and output tokens are reported for this reason.

## Validation

Both implementations include checks for paired-world integrity, oracle recomputation, call counts, information firewalls, exact prompt reconstruction, probability bounds, output parsing, token accounting, unique observations, and complete samples. V2 adds model-digest checks, exhaustive state-space coverage, probability-weight accounting, counterbalanced execution order, raw-response audits, and probability-chain accounting.

The saved `validation.json` files are the authoritative mechanical validity records. Passing these checks does not remove the external-validity and framing limitations described separately.

