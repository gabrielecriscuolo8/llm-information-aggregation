from __future__ import annotations

import json
from typing import Any

from .world import PrivateSignal


SYSTEM_PROMPT = """You are a probabilistic forecasting agent in a controlled experiment.
Your sole objective is to report your best calibrated probability that the binary outcome is YES.
Events are independent. A signal with reliability r matches the true outcome with probability r,
symmetrically for YES and NO. Signals are conditionally independent given the outcome.
Do not invent observations. Do not treat a previous price or forecast as ground truth.

OUTPUT CONTRACT
Return only one JSON object with exactly two fields:
- probability: a decimal probability inside the interval stated in the task;
- reason: one concise sentence explaining the informational basis of the estimate.
Do not wrap the object in markdown and do not add other text."""


def _signal_payload(signal: PrivateSignal) -> dict[str, Any]:
    return {
        "agent_id": signal.agent_id,
        "reliability": signal.reliability,
        "signal": "YES" if signal.signal_yes else "NO",
    }


def _render(task_description: str, context: dict[str, Any]) -> str:
    encoded = json.dumps(context, sort_keys=True, separators=(",", ":"))
    return f"{task_description}\n\n<CONTEXT_JSON>\n{encoded}\n</CONTEXT_JSON>"


def central_prompt(
    event_id: int,
    step: int,
    total_steps: int,
    prior: float,
    visible_signals: list[PrivateSignal],
    previous_probability: float | None,
    previous_reason: str | None,
    minimum: float,
    maximum: float,
) -> tuple[str, dict[str, Any]]:
    context: dict[str, Any] = {
        "task": "central_update",
        "event_id": event_id,
        "step": step,
        "total_steps": total_steps,
        "prior": prior,
        "visible_signals": [_signal_payload(signal) for signal in visible_signals],
        "previous_private_memory": None
        if previous_probability is None
        else {"probability": previous_probability, "reason": previous_reason},
        "permitted_probability_interval": [minimum, maximum],
    }
    description = (
        "TASK: CENTRAL SEQUENTIAL UPDATE. You are one analyst receiving evidence sequentially. "
        "Use every signal currently listed, reconsider earlier work when warranted, and report the "
        "probability conditional on all evidence visible at this step. No other signals are visible."
    )
    return _render(description, context), context


def ensemble_prompt(
    event_id: int,
    prior: float,
    private_signal: PrivateSignal,
    minimum: float,
    maximum: float,
) -> tuple[str, dict[str, Any]]:
    context: dict[str, Any] = {
        "task": "private_forecast",
        "event_id": event_id,
        "prior": prior,
        "private_signal": _signal_payload(private_signal),
        "permitted_probability_interval": [minimum, maximum],
    }
    description = (
        "TASK: INDEPENDENT PRIVATE FORECAST. You are isolated from all other agents. Update the "
        "common prior using only your one private signal. Report your own posterior probability."
    )
    return _render(description, context), context


def market_prompt(
    event_id: int,
    position: int,
    total_positions: int,
    prior: float,
    private_signal: PrivateSignal,
    current_price: float,
    price_history: list[dict[str, Any]],
    minimum: float,
    maximum: float,
) -> tuple[str, dict[str, Any]]:
    context: dict[str, Any] = {
        "task": "market_trade",
        "event_id": event_id,
        "position": position,
        "total_positions": total_positions,
        "prior": prior,
        "trader_id": private_signal.agent_id,
        "private_signal": _signal_payload(private_signal),
        "current_price": current_price,
        "public_price_history": price_history,
        "forecast_min_probability": minimum,
        "forecast_max_probability": maximum,
        "permitted_probability_interval": [minimum, maximum],
    }
    description = (
        "TASK: PREDICTION-MARKET TRADE. The current price is the public forecast produced by earlier "
        "trades and may encode their private evidence imperfectly. Previous traders' signal values "
        "and reasoning are not visible. Under the logarithmic market scoring rule, a risk-neutral "
        "myopic trader maximizes expected payoff by moving the price to its own posterior belief. "
        "Use the public price history and your private signal without inventing hidden signals, then "
        "report the target price."
    )
    return _render(description, context), context
