from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Callable

from . import PROTOCOL_VERSION
from .config import AppConfig
from .llm import Completion, ForecastModel, ModelError
from .prompts import (
    SYSTEM_PROMPT,
    central_compact_prompt,
    central_full_prompt,
    market_prompt,
)
from .reporting import write_run_artifacts
from .scoring import expected_brier_score, expected_log_loss
from .world import (
    PrivateSignal,
    World,
    condition_execution_order,
    generate_worlds,
    stable_call_seed,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_fingerprint(project_dir: Path) -> str:
    relevant = [project_dir / "PROTOCOL.md", project_dir / "config.json", project_dir / "run_v2.py"]
    relevant.extend(sorted((project_dir / "src" / "aggregation_v2").glob("*.py")))
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


def _completion_record(
    condition: str,
    world: World,
    position: int,
    actor_id: str,
    generation_seed: int,
    prompt: str,
    context: dict[str, Any],
    completion: Completion,
) -> dict[str, Any]:
    return {
        "condition": condition,
        "event_id": world.event_id,
        "cell_id": world.cell_id,
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
        "cell_id": world.cell_id,
        "evaluation_weight": world.evaluation_weight,
        "population_weight": world.population_weight,
        "system": system,
        "forecast": forecast,
        "prior": world.prior,
        "oracle_posterior": world.oracle_posterior,
        "oracle_squared_error": (forecast - world.oracle_posterior) ** 2,
        "oracle_absolute_error": abs(forecast - world.oracle_posterior),
        "expected_brier": expected_brier_score(forecast, world.oracle_posterior),
        "expected_log_loss": expected_log_loss(forecast, world.oracle_posterior),
        "scheduled_calls": scheduled_calls,
        "inference_attempts": inference_attempts,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
    }
    if extra:
        record.update(extra)
    return record


def _call_totals(calls: list[dict[str, Any]], condition: str) -> tuple[int, int, int]:
    selected = [call for call in calls if call["condition"] == condition]
    return (
        sum(int(call["attempts"]) for call in selected),
        sum(int(call["prompt_tokens"]) for call in selected),
        sum(int(call["output_tokens"]) for call in selected),
    )


def _run_central_full(
    config: AppConfig,
    model: ForecastModel,
    world: World,
) -> tuple[list[dict[str, Any]], list[float]]:
    calls: list[dict[str, Any]] = []
    path: list[float] = []
    visible: list[PrivateSignal] = []
    previous: float | None = None
    for position, signal in enumerate(world.ordered_signals(), start=1):
        visible.append(signal)
        prompt, context = central_full_prompt(
            world.event_id,
            position,
            config.experiment.n_agents,
            world.prior,
            list(visible),
            previous,
            config.experiment.forecast_min_probability,
            config.experiment.forecast_max_probability,
            config.experiment.reason_word_limit,
        )
        generation_seed = stable_call_seed(config.experiment.seed, world.cell_id, position)
        completion = model.complete_forecast(SYSTEM_PROMPT, prompt, generation_seed)
        calls.append(
            _completion_record(
                "central_full",
                world,
                position,
                "central_full_analyst",
                generation_seed,
                prompt,
                context,
                completion,
            )
        )
        path.append(completion.probability)
        previous = completion.probability
    return calls, path


def _public_history_row(
    position: int,
    reliability: float,
    probability_before: float,
    probability_after: float,
) -> dict[str, Any]:
    return {
        "position": position,
        "reliability": reliability,
        "probability_before": probability_before,
        "probability_after": probability_after,
    }


def _run_probability_chain(
    condition: str,
    prompt_builder: Callable[..., tuple[str, dict[str, Any]]],
    config: AppConfig,
    model: ForecastModel,
    world: World,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    current = world.prior
    for position, signal in enumerate(world.ordered_signals(), start=1):
        public_history = [dict(item) for item in history]
        prompt, context = prompt_builder(
            world.event_id,
            position,
            config.experiment.n_agents,
            world.prior,
            signal,
            current,
            public_history,
            config.experiment.forecast_min_probability,
            config.experiment.forecast_max_probability,
            config.experiment.reason_word_limit,
        )
        generation_seed = stable_call_seed(config.experiment.seed, world.cell_id, position)
        completion = model.complete_forecast(SYSTEM_PROMPT, prompt, generation_seed)
        actor_id = "central_compact_analyst" if condition == "central_compact" else signal.agent_id
        calls.append(
            _completion_record(
                condition,
                world,
                position,
                actor_id,
                generation_seed,
                prompt,
                context,
                completion,
            )
        )
        history.append(
            _public_history_row(position, signal.reliability, current, completion.probability)
        )
        current = completion.probability
    return calls, history


def run_event(config: AppConfig, model: ForecastModel, world: World) -> dict[str, Any]:
    condition_order = condition_execution_order(world.event_id)
    calls_by_condition: dict[str, list[dict[str, Any]]] = {}
    paths: dict[str, list[float]] = {}
    histories: dict[str, list[dict[str, Any]]] = {}

    for condition in condition_order:
        if condition == "central_full":
            condition_calls, path = _run_central_full(config, model, world)
            calls_by_condition[condition] = condition_calls
            paths[condition] = path
        elif condition == "central_compact":
            condition_calls, history = _run_probability_chain(
                condition, central_compact_prompt, config, model, world
            )
            calls_by_condition[condition] = condition_calls
            histories[condition] = history
            paths[condition] = [float(item["probability_after"]) for item in history]
        elif condition == "market":
            condition_calls, history = _run_probability_chain(
                condition, market_prompt, config, model, world
            )
            calls_by_condition[condition] = condition_calls
            histories[condition] = history
            paths[condition] = [float(item["probability_after"]) for item in history]
        else:
            raise ValueError(f"unknown condition: {condition}")

    calls = [call for condition in condition_order for call in calls_by_condition[condition]]
    observations = [
        _observation("prior", world, world.prior, 0, 0),
        _observation("oracle_bayes", world, world.oracle_posterior, 0, 0),
    ]
    for condition in ("central_full", "central_compact", "market"):
        attempts, prompt_tokens, output_tokens = _call_totals(calls, condition)
        path = paths[condition]
        observations.append(
            _observation(
                condition,
                world,
                path[-1],
                config.experiment.n_agents,
                attempts,
                prompt_tokens,
                output_tokens,
                {
                    "path_total_movement": sum(
                        abs(after - before)
                        for before, after in zip([world.prior] + path[:-1], path, strict=True)
                    )
                },
            )
        )

    return {
        "event_id": world.event_id,
        "cell_id": world.cell_id,
        "condition_execution_order": list(condition_order),
        "calls": calls,
        "central_full_path": paths["central_full"],
        "central_compact_history": histories["central_compact"],
        "market_history": histories["market"],
        "observations": observations,
    }


def initialize_state(
    config: AppConfig,
    project_dir: Path,
    model_identity: dict[str, Any],
) -> dict[str, Any]:
    worlds = generate_worlds(config.experiment)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "config_fingerprint": config.fingerprint(),
        "implementation_fingerprint": implementation_fingerprint(project_dir),
        "protocol_sha256": file_sha256(project_dir / "PROTOCOL.md"),
        "config": config.to_dict(),
        "model_identity": model_identity,
        "runtime_identity": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "status": "running",
        "next_event_index": 0,
        "worlds": [world.to_dict() for world in worlds],
        "event_results": [],
        "last_error": None,
    }


def _identity_key(identity: dict[str, Any]) -> tuple[Any, Any, Any]:
    return identity.get("provider"), identity.get("name"), identity.get("digest")


def validate_resume_state(
    state: dict[str, Any],
    project_dir: Path,
    current_model_identity: dict[str, Any],
) -> AppConfig:
    config = AppConfig.from_dict(state["config"])
    if state.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("run uses an incompatible protocol version")
    if state.get("config_fingerprint") != config.fingerprint():
        raise ValueError("saved configuration fingerprint does not match state contents")
    if state.get("implementation_fingerprint") != implementation_fingerprint(project_dir):
        raise ValueError("code changed since this run began; resume with the original package")
    if state.get("protocol_sha256") != file_sha256(project_dir / "PROTOCOL.md"):
        raise ValueError("protocol changed since this run began")
    if _identity_key(state.get("model_identity", {})) != _identity_key(current_model_identity):
        raise ValueError("Ollama model identity changed since this run began")
    if int(state.get("next_event_index", -1)) != len(state.get("event_results", [])):
        raise ValueError("checkpoint index is inconsistent with saved event results")
    return config


def execute_run(
    config: AppConfig,
    model: ForecastModel,
    model_identity: dict[str, Any],
    output_dir: Path,
    project_dir: Path,
    resume: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if resume:
        state = load_state(output_dir)
        saved_config = validate_resume_state(state, project_dir, model_identity)
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
        state = initialize_state(config, project_dir, model_identity)
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
            f"Completed cell {index + 1}/{len(worlds)} | "
            f"oracle={world.oracle_posterior:.3f}, "
            f"full={observations['central_full']['forecast']:.3f}, "
            f"compact={observations['central_compact']['forecast']:.3f}, "
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
