from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from . import PROTOCOL_VERSION


@dataclass
class ExperimentConfig:
    events: int
    seed: int
    n_agents: int
    priors: list[float]
    reliabilities: list[float]
    forecast_min_probability: float
    forecast_max_probability: float
    reason_word_limit: int
    token_imbalance_warning_fraction: float

    @property
    def exhaustive_event_count(self) -> int:
        return len(self.priors) * (2 ** self.n_agents)

    def validate(self) -> None:
        if self.events < 1 or self.events > self.exhaustive_event_count:
            raise ValueError(
                f"experiment.events must lie in [1, {self.exhaustive_event_count}]"
            )
        if self.n_agents != 5:
            raise ValueError("V2 is frozen for exactly five evidence positions")
        if len(self.reliabilities) != self.n_agents:
            raise ValueError("one reliability is required per agent")
        if not self.priors:
            raise ValueError("experiment.priors cannot be empty")
        if any(not 0.0 < value < 1.0 for value in self.priors):
            raise ValueError("all priors must be strictly between 0 and 1")
        if any(not 0.5 < value < 1.0 for value in self.reliabilities):
            raise ValueError("all reliabilities must be strictly between 0.5 and 1")
        if len(set(self.reliabilities)) != len(self.reliabilities):
            raise ValueError("reliabilities must be distinct")
        if not 0.0 < self.forecast_min_probability < self.forecast_max_probability < 1.0:
            raise ValueError("forecast probability bounds must lie strictly inside (0, 1)")
        if self.reason_word_limit < 5:
            raise ValueError("reason_word_limit must be at least 5")
        if not 0.0 <= self.token_imbalance_warning_fraction <= 1.0:
            raise ValueError("token imbalance warning fraction must lie in [0, 1]")


@dataclass
class ModelConfig:
    provider: str
    base_url: str
    name: str
    temperature: float
    timeout_seconds: int
    max_retries: int
    max_output_tokens: int

    def validate(self) -> None:
        if self.provider not in {"ollama", "mock"}:
            raise ValueError("model.provider must be 'ollama' or 'mock'")
        if not self.base_url:
            raise ValueError("model.base_url cannot be empty")
        if not self.name:
            raise ValueError("model.name cannot be empty")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("model.temperature must be between 0 and 2")
        if self.timeout_seconds < 1:
            raise ValueError("model.timeout_seconds must be positive")
        if self.max_retries < 1:
            raise ValueError("model.max_retries must be positive")
        if self.max_output_tokens < 50:
            raise ValueError("model.max_output_tokens must be at least 50")


@dataclass
class AppConfig:
    protocol_version: str
    experiment: ExperimentConfig
    model: ModelConfig

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AppConfig":
        config = cls(
            protocol_version=str(raw["protocol_version"]),
            experiment=ExperimentConfig(**raw["experiment"]),
            model=ModelConfig(**raw["model"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported protocol version {self.protocol_version!r}; "
                f"expected {PROTOCOL_VERSION!r}"
            )
        self.experiment.validate()
        self.model.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def is_confirmatory_design(self) -> bool:
        return (
            self.experiment.events == self.experiment.exhaustive_event_count
            and self.experiment.n_agents == 5
            and self.experiment.priors == [0.3, 0.4, 0.5, 0.6, 0.7]
            and self.experiment.reliabilities == [0.55, 0.6, 0.65, 0.7, 0.75]
            and self.model.provider == "ollama"
            and self.model.name == "qwen3:8b"
            and self.model.temperature == 0.0
        )
