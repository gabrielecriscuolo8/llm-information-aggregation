# Proposed V3: Framing Ablation

**Status: proposed, not run.**

## Motivation

V2 gives `central_compact` and `market` the same numerical state, but their instructions differ. The market condition is described as a new one-shot trader acting under a logarithmic market scoring rule; the centralized condition is described as the same analyst updating compressed state.

The observed difference may therefore be driven by:

- a generic sequential-update scaffold;
- independent actor identities;
- explicit prediction-market language;
- the scoring-rule instruction;
- a weakness specific to the central-analyst wording.

## Proposed research question

Does prediction-market framing improve sequential evidence aggregation, and is any advantage driven by independent actor identity, market language, or the scoring-rule instruction?

## Candidate conditions

All conditions should receive the same prior, public probability, public probability path, new signal, probability bounds, call count, generation seed, and output contract.

1. **Neutral updater:** no analyst, agent, trader, market, or scoring-rule labels.
2. **Same analyst:** persistent-analyst wording similar to V2 `central_compact`.
3. **New forecaster:** a new independent forecaster at each step, without market language.
4. **Market trader:** a new trader at each step with the V2 market-scoring-rule frame.

The primary metric would remain population-weighted squared distance from the exact posterior over the same exhaustive 160-cell design.

## Prompt robustness

A single wording per condition would leave another prompt-specific confound. The preferred confirmatory version should pre-specify at least two semantically equivalent prompt templates per condition and aggregate across templates. Template choice must be frozen before inspecting outcomes.

## Interpretation map

| Pattern | Interpretation |
|---|---|
| Market beats all other conditions | Evidence for a specifically useful market scaffold |
| New forecaster matches market and both beat same analyst | Independent actor framing, not the market mechanism, drives the gain |
| Neutral updater matches market | V2's central-analyst prompt was the likely weakness |
| Results vary materially by paraphrase | The V2 effect is prompt-sensitive and not mechanism-robust |

## Boundary

Even a positive V3 would establish a useful LLM orchestration scaffold, not a full artificial economy with persistent strategic agents, endogenous information acquisition, or binding wealth constraints.

