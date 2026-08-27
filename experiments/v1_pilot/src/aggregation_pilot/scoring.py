from __future__ import annotations

import math
import random
from statistics import mean
from typing import Iterable


EPSILON = 1e-12


def clamp_probability(value: float, low: float = EPSILON, high: float = 1.0 - EPSILON) -> float:
    if not math.isfinite(value):
        raise ValueError("probability must be finite")
    return min(max(float(value), low), high)


def logit(probability: float) -> float:
    p = clamp_probability(probability)
    return math.log(p / (1.0 - p))


def logistic(log_odds: float) -> float:
    if log_odds >= 0:
        z = math.exp(-log_odds)
        return 1.0 / (1.0 + z)
    z = math.exp(log_odds)
    return z / (1.0 + z)


def bayesian_posterior(
    prior: float,
    observations: Iterable[tuple[bool, float]],
) -> float:
    current_log_odds = logit(prior)
    for signal_yes, reliability in observations:
        if not 0.5 < reliability < 1.0:
            raise ValueError("signal reliability must be strictly between 0.5 and 1")
        evidence = math.log(reliability / (1.0 - reliability))
        current_log_odds += evidence if signal_yes else -evidence
    return logistic(current_log_odds)


def arithmetic_pool(probabilities: Iterable[float]) -> float:
    values = [clamp_probability(value) for value in probabilities]
    if not values:
        raise ValueError("cannot pool an empty sequence")
    return mean(values)


def prior_corrected_log_pool(probabilities: Iterable[float], prior: float) -> float:
    values = [clamp_probability(value) for value in probabilities]
    if not values:
        raise ValueError("cannot pool an empty sequence")
    pooled_log_odds = sum(logit(value) for value in values) - (len(values) - 1) * logit(prior)
    return logistic(pooled_log_odds)


def brier_score(probability: float, outcome: bool) -> float:
    return (clamp_probability(probability) - float(outcome)) ** 2


def log_loss(probability: float, outcome: bool) -> float:
    p = clamp_probability(probability)
    return -(math.log(p) if outcome else math.log(1.0 - p))


def log_score(probability: float, outcome: bool) -> float:
    return -log_loss(probability, outcome)


def market_scoring_profit(old_price: float, new_price: float, outcome: bool) -> float:
    return log_score(new_price, outcome) - log_score(old_price, outcome)


def quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability must lie in [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_bootstrap_mean_ci(
    differences: list[float],
    repetitions: int,
    seed: int,
) -> tuple[float, float, float]:
    if not differences:
        raise ValueError("paired bootstrap requires observations")
    rng = random.Random(seed)
    n = len(differences)
    estimates = [
        mean(differences[rng.randrange(n)] for _ in range(n))
        for _ in range(repetitions)
    ]
    return mean(differences), quantile(estimates, 0.025), quantile(estimates, 0.975)

