from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import random
from typing import Any

from .config import ExperimentConfig
from .scoring import bayesian_posterior


@dataclass(frozen=True)
class PrivateSignal:
    agent_id: str
    reliability: float
    signal_yes: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PrivateSignal":
        return cls(
            agent_id=str(raw["agent_id"]),
            reliability=float(raw["reliability"]),
            signal_yes=bool(raw["signal_yes"]),
        )


@dataclass(frozen=True)
class World:
    event_id: int
    prior: float
    outcome: bool
    signals: tuple[PrivateSignal, ...]
    order: tuple[str, ...]
    oracle_posterior: float

    def signal_for(self, agent_id: str) -> PrivateSignal:
        matches = [signal for signal in self.signals if signal.agent_id == agent_id]
        if len(matches) != 1:
            raise ValueError(f"world has {len(matches)} signals for {agent_id}")
        return matches[0]

    def ordered_signals(self) -> list[PrivateSignal]:
        return [self.signal_for(agent_id) for agent_id in self.order]

    def recompute_oracle(self) -> float:
        observations = [(signal.signal_yes, signal.reliability) for signal in self.signals]
        return bayesian_posterior(self.prior, observations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "prior": self.prior,
            "outcome": self.outcome,
            "signals": [signal.to_dict() for signal in self.signals],
            "order": list(self.order),
            "oracle_posterior": self.oracle_posterior,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "World":
        return cls(
            event_id=int(raw["event_id"]),
            prior=float(raw["prior"]),
            outcome=bool(raw["outcome"]),
            signals=tuple(PrivateSignal.from_dict(item) for item in raw["signals"]),
            order=tuple(str(item) for item in raw["order"]),
            oracle_posterior=float(raw["oracle_posterior"]),
        )


def generate_worlds(config: ExperimentConfig) -> list[World]:
    rng = random.Random(config.seed)
    prior_schedule = [config.priors[index % len(config.priors)] for index in range(config.events)]
    rng.shuffle(prior_schedule)
    agent_ids = [f"agent_{index + 1}" for index in range(config.n_agents)]
    worlds: list[World] = []

    for zero_index, prior in enumerate(prior_schedule):
        outcome = rng.random() < prior
        assigned_reliabilities = list(config.reliabilities)
        rng.shuffle(assigned_reliabilities)
        signals: list[PrivateSignal] = []
        for agent_id, reliability in zip(agent_ids, assigned_reliabilities, strict=True):
            matches_outcome = rng.random() < reliability
            signal_yes = outcome if matches_outcome else not outcome
            signals.append(
                PrivateSignal(
                    agent_id=agent_id,
                    reliability=reliability,
                    signal_yes=signal_yes,
                )
            )
        order = list(agent_ids)
        rng.shuffle(order)
        oracle = bayesian_posterior(
            prior,
            [(signal.signal_yes, signal.reliability) for signal in signals],
        )
        worlds.append(
            World(
                event_id=zero_index + 1,
                prior=prior,
                outcome=outcome,
                signals=tuple(signals),
                order=tuple(order),
                oracle_posterior=oracle,
            )
        )
    return worlds


def stable_call_seed(base_seed: int, event_id: int, position: int) -> int:
    payload = f"{base_seed}|{event_id}|{position}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % 2_147_483_647

