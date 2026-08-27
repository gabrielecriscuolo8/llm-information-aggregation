from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from . import PROTOCOL_VERSION
from .config import AppConfig
from .llm import Completion, ForecastModel, ModelError
from .prompts import SYSTEM_PROMPT, central_prompt, ensemble_prompt, market_prompt
from .reporting import write_run_artifacts
from .scoring import (
    arithmetic_pool,
    brier_score,
    clamp_probability,
    log_loss,
    market_scoring_profit,
    prior_corrected_log_pool,
)
from .world import PrivateSignal, World, generate_worlds, stable_call_seed


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_fingerprint(project_dir: Path) -> str:
    relevant = [project_dir / "PROTOCOL.md", project_dir / "config.json", project_dir / "run_pilot.py"]
    relevant.extend(sorted((project_dir / "src" / "aggregation_pilot").glob("*.py")))
    digest = hashlib.sha256()
    for path in relevant:
        if not path.exists():
            continue
        digest.update(path.relative_to(project_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_state(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "state.json"
    if not path.exists():
        raise ValueError(f"resume directory has no state.json: {output_dir}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("state.json must contain a JSON object")
    return value


def _signal_dict(signal: PrivateSignal) -> dict[str, Any]:
    return {
        "agent_id": signal.agent_id,
        "reliability": signal.reliability,
        "signal": "YES" if signal.signal_yes else "NO",
    }


def _completion_record(
    condition: str,
    event_id: int,
    position: int,
    actor_id: str,
    generation_seed: int,
    prompt: str,
    context: dict[str, Any],
    completion: Completion,
) -> dict[str, Any]:
    return {
        "condition": condition,
        "event_id": event_id,
        "position": position,
        "actor_id": actor_id,
        "generation_seed": generation_seed,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": prompt,
        "context": context,
        "probability": completion.probability,
        "reason": completion.reason,
        "raw_responses": list(completion.raw_responses),
        "attempts": completion.attempts,
        "elapsed_seconds": completion.elapsed_seconds,
        "prompt_tokens": completion.prompt_tokens,
        "output_tokens": completion.output_tokens,
    }


def _observation(
    system: str,
    world: World,
    forecast: float,
    scheduled_calls: int,
    inference_attempts: int,
    prompt_tokens: int = 0,
    output_tokens: int = 0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "event_id": world.event_id,
        "system": system,
        "forecast": forecast,
        "prior": world.prior,
        "oracle_posterior": world.oracle_posterior,
        "outcome": int(world.outcome),
        "oracle_squared_error": (forecast - world.oracle_posterior) ** 2,
        "oracle_absolute_error": abs(forecast - world.oracle_posterior),
        "brier": brier_score(forecast, world.outcome),
        "log_loss": log_loss(forecast, world.outcome),
        "scheduled_calls": scheduled_calls,
        "inference_attempts": inference_attempts,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
    }
    if extra:
        record.update(extra)
    return record


def run_event(config: AppConfig, model: ForecastModel, world: World) -> dict[str, Any]:
    n_agents = config.experiment.n_agents
    ordered = world.ordered_signals()
    calls: list[dict[str, Any]] = []

    central_probabilities: list[float] = []
    visible: list[PrivateSignal] = []
    previous_probability: float | None = None
    previous_reason: str | None = None
    for position, signal in enumerate(ordered, start=1):
        visible.append(signal)
        prompt, context = central_prompt(
            world.event_id,
            position,
            n_agents,
            world.prior,
            list(visible),
            previous_probability,
            previous_reason,
            config.experiment.forecast_min_probability,
            config.experiment.forecast_max_probability,
        )
        generation_seed = stable_call_seed(config.experiment.seed, world.event_id, position)
        completion = model.complete_forecast(SYSTEM_PROMPT, prompt, generation_seed)
        calls.append(
            _completion_record(
                "central",
                world.event_id,
                position,
                "central_analyst",
                generation_seed,
                prompt,
                context,
                completion,
            )
        )
        central_probabilities.append(completion.probability)
        previous_probability = completion.probability
        previous_reason = completion.reason

    ensemble_probabilities: list[float] = []
    for position, signal in enumerate(ordered, start=1):
        prompt, context = ensemble_prompt(
            world.event_id,
            world.prior,
            signal,
            config.experiment.forecast_min_probability,
            config.experiment.forecast_max_probability,
        )
        generation_seed = stable_call_seed(config.experiment.seed, world.event_id, position)
        completion = model.complete_forecast(SYSTEM_PROMPT, prompt, generation_seed)
        calls.append(
            _completion_record(
                "ensemble",
                world.event_id,
                position,
                signal.agent_id,
                generation_seed,
                prompt,
                context,
                completion,
            )
        )
        ensemble_probabilities.append(completion.probability)

    market_price = world.prior
    market_history: list[dict[str, Any]] = []
    market_profits: list[float] = []
    for position, signal in enumerate(ordered, start=1):
        public_history = [
            {
                "position": item["position"],
                "trader_id": item["trader_id"],
                "reliability": item["reliability"],
                "price_before": item["price_before"],
                "price_after": item["price_after"],
            }
            for item in market_history
        ]
        prompt, context = market_prompt(
            world.event_id,
            position,
            n_agents,
            world.prior,
            signal,
            market_price,
            public_history,
            config.experiment.forecast_min_probability,
            config.experiment.forecast_max_probability,
        )
        generation_seed = stable_call_seed(config.experiment.seed, world.event_id, position)
        completion = model.complete_forecast(SYSTEM_PROMPT, prompt, generation_seed)
        new_price = completion.probability
        profit = market_scoring_profit(market_price, new_price, world.outcome)
        calls.append(
            _completion_record(
                "market",
                world.event_id,
                position,
                signal.agent_id,
                generation_seed,
                prompt,
                context,
                completion,
            )
        )
        market_history.append(
            {
                "position": position,
                "trader_id": signal.agent_id,
                "reliability": signal.reliability,
                "price_before": market_price,
                "price_after": new_price,
                "realized_profit": profit,
            }
        )
        market_profits.append(profit)
        market_price = new_price

    ensemble_mean = clamp_probability(
        arithmetic_pool(ensemble_probabilities),
        config.experiment.forecast_min_probability,
        config.experiment.forecast_max_probability,
    )
    ensemble_log_pool = clamp_probability(
        prior_corrected_log_pool(ensemble_probabilities, world.prior),
        config.experiment.forecast_min_probability,
        config.experiment.forecast_max_probability,
    )
    attempts_by_condition = {
        condition: sum(call["attempts"] for call in calls if call["condition"] == condition)
        for condition in ("central", "ensemble", "market")
    }
    prompt_tokens_by_condition = {
        condition: sum(int(call["prompt_tokens"]) for call in calls if call["condition"] == condition)
        for condition in ("central", "ensemble", "market")
    }
    output_tokens_by_condition = {
        condition: sum(int(call["output_tokens"]) for call in calls if call["condition"] == condition)
        for condition in ("central", "ensemble", "market")
    }
    ensemble_mean_value = sum(ensemble_probabilities) / len(ensemble_probabilities)
    ensemble_variance = sum(
        (probability - ensemble_mean_value) ** 2 for probability in ensemble_probabilities
    ) / len(ensemble_probabilities)

    observations = [
        _observation("prior", world, world.prior, 0, 0),
        _observation("oracle_bayes", world, world.oracle_posterior, 0, 0),
        _observation(
            "central",
            world,
            central_probabilities[-1],
            n_agents,
            attempts_by_condition["central"],
            prompt_tokens_by_condition["central"],
            output_tokens_by_condition["central"],
            {"path_total_movement": sum(abs(b - a) for a, b in zip([world.prior] + central_probabilities[:-1], central_probabilities))},
        ),
        _observation(
            "ensemble_mean",
            world,
            ensemble_mean,
            n_agents,
            attempts_by_condition["ensemble"],
            prompt_tokens_by_condition["ensemble"],
            output_tokens_by_condition["ensemble"],
            {"ensemble_standard_deviation": math.sqrt(ensemble_variance)},
        ),
        _observation(
            "ensemble_log_pool",
            world,
            ensemble_log_pool,
            n_agents,
            attempts_by_condition["ensemble"],
            prompt_tokens_by_condition["ensemble"],
            output_tokens_by_condition["ensemble"],
            {"ensemble_standard_deviation": math.sqrt(ensemble_variance)},
        ),
        _observation(
            "market",
            world,
            market_price,
            n_agents,
            attempts_by_condition["market"],
            prompt_tokens_by_condition["market"],
            output_tokens_by_condition["market"],
            {
                "path_total_movement": sum(abs(item["price_after"] - item["price_before"]) for item in market_history),
                "market_total_realized_profit": sum(market_profits),
            },
        ),
    ]
    return {
        "event_id": world.event_id,
        "calls": calls,
        "central_path": central_probabilities,
        "ensemble_private_forecasts": ensemble_probabilities,
        "market_history": market_history,
        "observations": observations,
    }


def initialize_state(config: AppConfig, project_dir: Path) -> dict[str, Any]:
    worlds = generate_worlds(config.experiment)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "config_fingerprint": config.fingerprint(),
        "implementation_fingerprint": implementation_fingerprint(project_dir),
        "protocol_sha256": file_sha256(project_dir / "PROTOCOL.md"),
        "config": config.to_dict(),
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "status": "running",
        "next_event_index": 0,
        "worlds": [world.to_dict() for world in worlds],
        "event_results": [],
        "last_error": None,
    }


def validate_resume_state(state: dict[str, Any], project_dir: Path) -> AppConfig:
    config = AppConfig.from_dict(state["config"])
    if state.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("run uses an incompatible protocol version")
    if state.get("config_fingerprint") != config.fingerprint():
        raise ValueError("saved configuration fingerprint does not match state contents")
    if state.get("implementation_fingerprint") != implementation_fingerprint(project_dir):
        raise ValueError("code changed since this run began; resume with the original package")
    if state.get("protocol_sha256") != file_sha256(project_dir / "PROTOCOL.md"):
        raise ValueError("protocol changed since this run began")
    if int(state.get("next_event_index", -1)) != len(state.get("event_results", [])):
        raise ValueError("checkpoint index is inconsistent with saved event results")
    return config


def execute_run(
    config: AppConfig,
    model: ForecastModel,
    output_dir: Path,
    project_dir: Path,
    resume: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if resume:
        state = load_state(output_dir)
        saved_config = validate_resume_state(state, project_dir)
        if saved_config.fingerprint() != config.fingerprint():
            raise ValueError("resume configuration differs from the saved run")
        if state.get("status") == "complete":
            validation = write_run_artifacts(state, output_dir)
            if not validation["valid"]:
                failed = [item["name"] for item in validation["checks"] if not item["passed"]]
                raise ValueError("completed checkpoint failed validation: " + ", ".join(failed))
            return state
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError(f"output directory is not empty: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        state = initialize_state(config, project_dir)
        atomic_write_json(output_dir / "state.json", state)
        write_run_artifacts(state, output_dir)

    worlds = [World.from_dict(item) for item in state["worlds"]]
    start_index = int(state["next_event_index"])
    for index in range(start_index, len(worlds)):
        world = worlds[index]
        try:
            event_result = run_event(config, model, world)
        except (ModelError, ValueError, OSError) as exc:
            state["status"] = "interrupted"
            state["last_error"] = {
                "event_id": world.event_id,
                "message": str(exc),
                "at": utc_now(),
            }
            state["updated_at"] = utc_now()
            atomic_write_json(output_dir / "state.json", state)
            write_run_artifacts(state, output_dir)
            raise

        state["event_results"].append(event_result)
        state["next_event_index"] = index + 1
        state["last_error"] = None
        state["status"] = "running"
        state["updated_at"] = utc_now()
        atomic_write_json(output_dir / "state.json", state)
        write_run_artifacts(state, output_dir)
        observations = {item["system"]: item for item in event_result["observations"]}
        print(
            f"Completed event {index + 1}/{len(worlds)} | "
            f"oracle={world.oracle_posterior:.3f}, central={observations['central']['forecast']:.3f}, "
            f"ensemble={observations['ensemble_log_pool']['forecast']:.3f}, "
            f"market={observations['market']['forecast']:.3f}",
            flush=True,
        )

    state["status"] = "complete"
    state["updated_at"] = utc_now()
    atomic_write_json(output_dir / "state.json", state)
    validation = write_run_artifacts(state, output_dir)
    if not validation["valid"]:
        state["status"] = "invalid"
        state["last_error"] = {
            "event_id": None,
            "message": "post-run validation failed",
            "at": utc_now(),
        }
        atomic_write_json(output_dir / "state.json", state)
        write_run_artifacts(state, output_dir)
        failed = [item["name"] for item in validation["checks"] if not item["passed"]]
        raise ValueError("post-run validation failed: " + ", ".join(failed))
    return state
