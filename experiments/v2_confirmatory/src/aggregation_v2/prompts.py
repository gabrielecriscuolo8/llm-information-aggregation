from __future__ import annotations

import json
from typing import Any

from .world import PrivateSignal


SYSTEM_PROMPT = """You are a probabilistic forecasting component in a controlled experiment.
Your only objective is to report your best calibrated probability that a binary outcome is YES.
Signals are conditionally independent given the outcome. A signal with reliability r matches the
true outcome with probability r, symmetrically for YES and NO. Treat the supplied prior or current
public probability as probabilistic information, not as ground truth. Do not invent observations.

OUTPUT CONTRACT
Return only one JSON object with exactly two fields:
- probability: a decimal inside the permitted interval;
- reason: one concise sentence, no longer than the stated word limit.
Do not use markdown or add other text."""


def signal_payload(signal: PrivateSignal) -> dict[str, Any]:
    return {
        "agent_id": signal.agent_id,
        "reliability": signal.reliability,
        "signal": "YES" if signal.signal_yes else "NO",
    }


def _render(description: str, context: dict[str, Any]) -> str:
    encoded = json.dumps(context, sort_keys=True, separators=(",", ":"))
    return f"{description}\n\n<CONTEXT_JSON>\n{encoded}\n</CONTEXT_JSON>"


def central_full_prompt(
    event_id: int,
    position: int,
    total_positions: int,
    prior: float,
    visible_signals: list[PrivateSignal],
    previous_probability: float | None,
    minimum: float,
    maximum: float,
    reason_word_limit: int,
) -> tuple[str, dict[str, Any]]:
    context: dict[str, Any] = {
        "task": "central_full_update",
        "event_id": event_id,
        "position": position,
        "total_positions": total_positions,
        "actor_id": "central_full_analyst",
        "prior": prior,
        "visible_signals": [signal_payload(signal) for signal in visible_signals],
        "previous_probability": previous_probability,
        "permitted_probability_interval": [minimum, maximum],
        "reason_word_limit": reason_word_limit,
    }
    description = (
        "TASK: CENTRAL FULL-EVIDENCE UPDATE. You are the same analyst at every position. "
        "Recalculate the probability using the common prior and every signal currently visible; "
        "the previous probability is only your earlier estimate and may be corrected."
    )
    return _render(description, context), context


def central_compact_prompt(
    event_id: int,
    position: int,
    total_positions: int,
    prior: float,
    signal: PrivateSignal,
    current_probability: float,
    public_history: list[dict[str, Any]],
    minimum: float,
    maximum: float,
    reason_word_limit: int,
) -> tuple[str, dict[str, Any]]:
    context: dict[str, Any] = {
        "task": "central_compact_update",
        "event_id": event_id,
        "position": position,
        "total_positions": total_positions,
        "actor_id": "central_compact_analyst",
        "prior": prior,
        "current_public_probability": current_probability,
        "new_signal": signal_payload(signal),
        "public_probability_history": public_history,
        "permitted_probability_interval": [minimum, maximum],
        "reason_word_limit": reason_word_limit,
    }
    description = (
        "TASK: CENTRAL COMPRESSED-STATE UPDATE. You are the same analyst at every position. "
        "The current public probability compresses all earlier evidence. Update it using only the "
        "new signal; do not apply the original prior again or invent earlier signal values."
    )
    return _render(description, context), context


def market_prompt(
    event_id: int,
    position: int,
    total_positions: int,
    prior: float,
    signal: PrivateSignal,
    current_probability: float,
    public_history: list[dict[str, Any]],
    minimum: float,
    maximum: float,
    reason_word_limit: int,
) -> tuple[str, dict[str, Any]]:
    context: dict[str, Any] = {
        "task": "market_trade",
        "event_id": event_id,
        "position": position,
        "total_positions": total_positions,
        "actor_id": signal.agent_id,
        "prior": prior,
        "current_public_probability": current_probability,
        "new_signal": signal_payload(signal),
        "public_probability_history": public_history,
        "permitted_probability_interval": [minimum, maximum],
        "reason_word_limit": reason_word_limit,
    }
    description = (
        "TASK: PREDICTION-MARKET TRADE. You are a new trader acting once. The current public "
        "probability compresses earlier traders' evidence. Under a logarithmic market scoring "
        "rule, move it to your posterior after using only your private signal; do not apply the "
        "original prior again or invent earlier signal values."
    )
    return _render(description, context), context
