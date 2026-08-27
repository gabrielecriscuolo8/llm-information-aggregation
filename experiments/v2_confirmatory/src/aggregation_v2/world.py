from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
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
    cell_id: str
    prior: float
    signals: tuple[PrivateSignal, ...]
    order: tuple[str, ...]
    oracle_posterior: float
    population_weight: float
    evaluation_weight: float

    def signal_for(self, agent_id: str) -> PrivateSignal:
        matches = [signal for signal in self.signals if signal.agent_id == agent_id]
        if len(matches) != 1:
            raise ValueError(f"world has {len(matches)} signals for {agent_id}")
        return matches[0]

    def ordered_signals(self) -> list[PrivateSignal]:
        return [self.signal_for(agent_id) for agent_id in self.order]

    def recompute_oracle(self) -> float:
        return bayesian_posterior(
            self.prior,
            [(signal.signal_yes, signal.reliability) for signal in self.signals],
        )

    def recompute_population_weight(self, number_of_priors: int) -> float:
        likelihood_yes = 1.0
        likelihood_no = 1.0
        for signal in self.signals:
            likelihood_yes *= signal.reliability if signal.signal_yes else 1.0 - signal.reliability
            likelihood_no *= 1.0 - signal.reliability if signal.signal_yes else signal.reliability
        marginal_pattern_probability = (
            self.prior * likelihood_yes + (1.0 - self.prior) * likelihood_no
        )
        return marginal_pattern_probability / number_of_priors

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "cell_id": self.cell_id,
            "prior": self.prior,
            "signals": [signal.to_dict() for signal in self.signals],
            "order": list(self.order),
            "oracle_posterior": self.oracle_posterior,
            "population_weight": self.population_weight,
            "evaluation_weight": self.evaluation_weight,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "World":
        return cls(
            event_id=int(raw["event_id"]),
            cell_id=str(raw["cell_id"]),
            prior=float(raw["prior"]),
            signals=tuple(PrivateSignal.from_dict(item) for item in raw["signals"]),
            order=tuple(str(item) for item in raw["order"]),
            oracle_posterior=float(raw["oracle_posterior"]),
            population_weight=float(raw["population_weight"]),
            evaluation_weight=float(raw["evaluation_weight"]),
        )


def _pattern_probability(prior: float, signals: list[PrivateSignal]) -> float:
    likelihood_yes = 1.0
    likelihood_no = 1.0
    for signal in signals:
        likelihood_yes *= signal.reliability if signal.signal_yes else 1.0 - signal.reliability
        likelihood_no *= 1.0 - signal.reliability if signal.signal_yes else signal.reliability
    return prior * likelihood_yes + (1.0 - prior) * likelihood_no


def generate_worlds(config: ExperimentConfig) -> list[World]:
    """Enumerate the prior x signal-pattern state space, then shuffle execution order."""
    agent_ids = [f"agent_{index + 1}" for index in range(config.n_agents)]
    cells: list[dict[str, Any]] = []
    cyclic_orders = [
        tuple(agent_ids[offset:] + agent_ids[:offset]) for offset in range(config.n_agents)
    ]

    for prior_index, prior in enumerate(config.priors):
        for pattern_index, bits in enumerate(
            itertools.product((False, True), repeat=config.n_agents)
        ):
            signals = [
                PrivateSignal(agent_id, reliability, signal_yes)
                for agent_id, reliability, signal_yes in zip(
                    agent_ids, config.reliabilities, bits, strict=True
                )
            ]
            pattern = "".join("1" if bit else "0" for bit in bits)
            cells.append(
                {
                    "cell_id": f"prior_{prior:.1f}_signals_{pattern}",
                    "prior": prior,
                    "signals": tuple(signals),
                    "order": cyclic_orders[(prior_index + pattern_index) % config.n_agents],
                    "oracle_posterior": bayesian_posterior(
                        prior,
                        [(signal.signal_yes, signal.reliability) for signal in signals],
                    ),
                    "population_weight": _pattern_probability(prior, signals)
                    / len(config.priors),
                }
            )

    rng = random.Random(config.seed)
    rng.shuffle(cells)
    selected = cells[: config.events]
    selected_mass = sum(float(cell["population_weight"]) for cell in selected)
    if selected_mass <= 0.0:
        raise ValueError("selected exhaustive cells have zero probability mass")

    worlds: list[World] = []
    for event_id, cell in enumerate(selected, start=1):
        worlds.append(
            World(
                event_id=event_id,
                cell_id=str(cell["cell_id"]),
                prior=float(cell["prior"]),
                signals=cell["signals"],
                order=cell["order"],
                oracle_posterior=float(cell["oracle_posterior"]),
                population_weight=float(cell["population_weight"]),
                evaluation_weight=float(cell["population_weight"]) / selected_mass,
            )
        )
    return worlds


def stable_call_seed(base_seed: int, cell_id: str, position: int) -> int:
    payload = f"{base_seed}|{cell_id}|{position}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % 2_147_483_647


def condition_execution_order(event_id: int) -> tuple[str, str, str]:
    conditions = ("central_full", "central_compact", "market")
    offset = (event_id - 1) % len(conditions)
    return conditions[offset:] + conditions[:offset]
