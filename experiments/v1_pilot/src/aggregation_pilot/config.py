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
    bootstrap_repetitions: int

    def validate(self) -> None:
        if self.events < 1:
            raise ValueError("experiment.events must be at least 1")
        if self.n_agents < 2:
            raise ValueError("experiment.n_agents must be at least 2")
        if len(self.reliabilities) != self.n_agents:
            raise ValueError("one reliability is required per agent")
        if not self.priors:
            raise ValueError("experiment.priors cannot be empty")
        if any(not 0.0 < value < 1.0 for value in self.priors):
            raise ValueError("all priors must be strictly between 0 and 1")
        if any(not 0.5 < value < 1.0 for value in self.reliabilities):
            raise ValueError("all reliabilities must be strictly between 0.5 and 1")
        if len(set(self.reliabilities)) != len(self.reliabilities):
            raise ValueError("reliabilities must be distinct so public price moves are identifiable")
        if not 0.0 < self.forecast_min_probability < self.forecast_max_probability < 1.0:
            raise ValueError("forecast probability bounds must lie strictly inside (0, 1)")
        if self.bootstrap_repetitions < 100:
            raise ValueError("bootstrap_repetitions must be at least 100")


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
        config = cls(
            protocol_version=str(raw["protocol_version"]),
            experiment=ExperimentConfig(**raw["experiment"]),
            model=ModelConfig(**raw["model"]),
        )
        config.validate()
        return config

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
                f"unsupported protocol version {self.protocol_version!r}; expected {PROTOCOL_VERSION!r}"
            )
        self.experiment.validate()
        self.model.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
