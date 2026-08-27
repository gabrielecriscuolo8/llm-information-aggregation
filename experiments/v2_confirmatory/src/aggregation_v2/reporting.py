from __future__ import annotations

import csv
from html import escape
import itertools
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .config import AppConfig
from .llm import parse_forecast
from .prompts import (
    SYSTEM_PROMPT,
    central_compact_prompt,
    central_full_prompt,
    market_prompt,
)
from .scoring import bayesian_posterior, expected_brier_score, expected_log_loss, weighted_mean
from .world import World, condition_execution_order, generate_worlds, stable_call_seed


SYSTEM_ORDER = ["prior", "central_full", "central_compact", "market", "oracle_bayes"]
MODEL_CONDITIONS = ["central_full", "central_compact", "market"]


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_write_text(
        path,
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _flatten_calls(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [call for result in state.get("event_results", []) for call in result["calls"]]


def _flatten_observations(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for result in state.get("event_results", []) for row in result["observations"]]


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def _expected_history(history: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    return [dict(item) for item in history[:count]]


def validate_run(state: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []
    config = AppConfig.from_dict(state["config"])
    worlds = [World.from_dict(item) for item in state.get("worlds", [])]
    results = state.get("event_results", [])
    expected_events = config.experiment.events
    n_agents = config.experiment.n_agents

    checks.append(
        _check(
            "frozen_fingerprints_present",
            bool(
                state.get("protocol_sha256")
                and state.get("config_fingerprint")
                and state.get("implementation_fingerprint")
            ),
            "Protocol, configuration, and implementation fingerprints are recorded.",
        )
    )
    model_identity = state.get("model_identity", {})
    checks.append(
        _check(
            "model_identity_recorded",
            bool(model_identity.get("provider") and model_identity.get("name") and model_identity.get("digest")),
            "The model provider, exact name, and digest are recorded.",
        )
    )
    checks.append(
        _check(
            "world_count",
            len(worlds) == expected_events,
            f"Found {len(worlds)} of {expected_events} pre-generated cells.",
        )
    )
    world_ids = [world.event_id for world in worlds]
    cell_ids = [world.cell_id for world in worlds]
    checks.append(
        _check(
            "unique_world_and_cell_ids",
            world_ids == list(range(1, len(worlds) + 1)) and len(cell_ids) == len(set(cell_ids)),
            "Event identifiers are sequential and cell identifiers are unique.",
        )
    )
    expected_worlds = [world.to_dict() for world in generate_worlds(config.experiment)]
    checks.append(
        _check(
            "exact_frozen_world_schedule",
            [world.to_dict() for world in worlds] == expected_worlds,
            "Worlds, weights, signal assignments, and reveal orders match deterministic generation.",
        )
    )
    weights_ok = _close(sum(world.evaluation_weight for world in worlds), 1.0, 1e-11)
    weights_ok = weights_ok and all(
        _close(
            world.population_weight,
            world.recompute_population_weight(len(config.experiment.priors)),
            1e-12,
        )
        for world in worlds
    )
    checks.append(
        _check(
            "probability_weight_accounting",
            weights_ok,
            "Evaluation weights sum to one and population weights recompute from the data-generating process.",
        )
    )
    oracle_ok = all(_close(world.oracle_posterior, world.recompute_oracle(), 1e-12) for world in worlds)
    checks.append(
        _check("oracle_recomputation", oracle_ok, "Every oracle posterior recomputes exactly.")
    )
    full_design = config.experiment.events == config.experiment.exhaustive_event_count
    expected_cells = {
        f"prior_{prior:.1f}_signals_{''.join(bits)}"
        for prior in config.experiment.priors
        for bits in itertools.product("01", repeat=n_agents)
    }
    checks.append(
        _check(
            "exhaustive_state_space_when_configured",
            (not full_design) or set(cell_ids) == expected_cells,
            "The confirmatory design contains every prior-by-signal-pattern cell exactly once.",
        )
    )
    if full_design:
        position_counts = {
            reliability: [
                sum(
                    _close(world.ordered_signals()[position].reliability, reliability)
                    for world in worlds
                )
                for position in range(n_agents)
            ]
            for reliability in config.experiment.reliabilities
        }
        position_balance_ok = all(len(set(counts)) == 1 for counts in position_counts.values())
    else:
        position_counts = {}
        position_balance_ok = True
    checks.append(
        _check(
            "reliability_position_balance",
            position_balance_ok,
            "Each reliability occupies every evidence position equally often in the full design.",
        )
    )

    result_ids = [int(result["event_id"]) for result in results]
    checks.append(
        _check(
            "checkpoint_consistency",
            int(state.get("next_event_index", -1)) == len(results)
            and result_ids == list(range(1, len(results) + 1)),
            f"Checkpoint contains {len(results)} completed cells without gaps or duplicates.",
        )
    )

    call_counts_ok = True
    matched_seeds_ok = True
    execution_order_ok = True
    privacy_ok = True
    prompt_reconstruction_ok = True
    forecast_bounds_ok = True
    observation_math_ok = True
    chain_accounting_ok = True
    no_fallback_ok = True
    token_accounting_ok = True
    observation_uniqueness_ok = True
    path_accounting_ok = True
    raw_response_ok = True
    reason_lengths: list[int] = []

    for result in results:
        world = worlds[int(result["event_id"]) - 1]
        if result.get("cell_id") != world.cell_id:
            execution_order_ok = False
        expected_condition_order = list(condition_execution_order(world.event_id))
        if result.get("condition_execution_order") != expected_condition_order:
            execution_order_ok = False
        calls = result["calls"]
        by_condition = {
            condition: sorted(
                [call for call in calls if call["condition"] == condition],
                key=lambda item: int(item["position"]),
            )
            for condition in MODEL_CONDITIONS
        }
        if any(len(items) != n_agents for items in by_condition.values()):
            call_counts_ok = False
            continue
        flattened_order = []
        for call in calls:
            if not flattened_order or flattened_order[-1] != call["condition"]:
                flattened_order.append(call["condition"])
        if flattened_order != expected_condition_order:
            execution_order_ok = False

        ordered_signals = world.ordered_signals()
        for position in range(1, n_agents + 1):
            expected_seed = stable_call_seed(config.experiment.seed, world.cell_id, position)
            seeds = [
                int(by_condition[condition][position - 1]["generation_seed"])
                for condition in MODEL_CONDITIONS
            ]
            if seeds != [expected_seed] * len(MODEL_CONDITIONS):
                matched_seeds_ok = False

        previous_probability: float | None = None
        for position, call in enumerate(by_condition["central_full"], start=1):
            expected_prompt, expected_context = central_full_prompt(
                world.event_id,
                position,
                n_agents,
                world.prior,
                ordered_signals[:position],
                previous_probability,
                config.experiment.forecast_min_probability,
                config.experiment.forecast_max_probability,
                config.experiment.reason_word_limit,
            )
            if (
                call.get("system_prompt") != SYSTEM_PROMPT
                or call.get("user_prompt") != expected_prompt
                or call.get("context") != expected_context
            ):
                prompt_reconstruction_ok = False
            if any(key in call["context"] for key in ("oracle_posterior", "outcome", "new_signal")):
                privacy_ok = False
            previous_probability = float(call["probability"])

        for condition, builder, history_key in (
            ("central_compact", central_compact_prompt, "central_compact_history"),
            ("market", market_prompt, "market_history"),
        ):
            history = result[history_key]
            if len(history) != n_agents:
                chain_accounting_ok = False
                continue
            current = world.prior
            reconstructed: list[dict[str, Any]] = []
            for position, (call, signal, saved_row) in enumerate(
                zip(by_condition[condition], ordered_signals, history, strict=True), start=1
            ):
                expected_prompt, expected_context = builder(
                    world.event_id,
                    position,
                    n_agents,
                    world.prior,
                    signal,
                    current,
                    _expected_history(reconstructed, len(reconstructed)),
                    config.experiment.forecast_min_probability,
                    config.experiment.forecast_max_probability,
                    config.experiment.reason_word_limit,
                )
                if (
                    call.get("system_prompt") != SYSTEM_PROMPT
                    or call.get("user_prompt") != expected_prompt
                    or call.get("context") != expected_context
                ):
                    prompt_reconstruction_ok = False
                context = call["context"]
                if any(key in context for key in ("oracle_posterior", "outcome", "visible_signals")):
                    privacy_ok = False
                expected_row = {
                    "position": position,
                    "reliability": signal.reliability,
                    "probability_before": current,
                    "probability_after": float(call["probability"]),
                }
                if saved_row != expected_row:
                    chain_accounting_ok = False
                reconstructed.append(expected_row)
                current = float(call["probability"])

        for call in calls:
            probability = float(call["probability"])
            interval = call["context"]["permitted_probability_interval"]
            if not math.isfinite(probability) or not float(interval[0]) <= probability <= float(interval[1]):
                forecast_bounds_ok = False
            if int(call.get("attempts", 0)) < 1 or not call.get("raw_responses"):
                no_fallback_ok = False
            if int(call.get("prompt_tokens", -1)) < 0 or int(call.get("output_tokens", -1)) < 0:
                token_accounting_ok = False
            reason_lengths.append(len(str(call.get("reason", "")).split()))
            try:
                parsed_probability, parsed_reason = parse_forecast(str(call["raw_responses"][-1]))
                if not _close(parsed_probability, probability) or parsed_reason != call["reason"]:
                    raw_response_ok = False
            except Exception:
                raw_response_ok = False

        observations = result["observations"]
        if len(observations) != len(SYSTEM_ORDER) or {row["system"] for row in observations} != set(SYSTEM_ORDER):
            observation_uniqueness_ok = False
            continue
        observation_map = {row["system"]: row for row in observations}
        expected_forecasts = {
            "prior": world.prior,
            "oracle_bayes": world.oracle_posterior,
            "central_full": float(by_condition["central_full"][-1]["probability"]),
            "central_compact": float(by_condition["central_compact"][-1]["probability"]),
            "market": float(by_condition["market"][-1]["probability"]),
        }
        for system, forecast in expected_forecasts.items():
            row = observation_map[system]
            expected_values = {
                "forecast": forecast,
                "oracle_squared_error": (forecast - world.oracle_posterior) ** 2,
                "oracle_absolute_error": abs(forecast - world.oracle_posterior),
                "expected_brier": expected_brier_score(forecast, world.oracle_posterior),
                "expected_log_loss": expected_log_loss(forecast, world.oracle_posterior),
                "evaluation_weight": world.evaluation_weight,
                "population_weight": world.population_weight,
            }
            if any(not _close(row[key], value) for key, value in expected_values.items()):
                observation_math_ok = False

        for condition in MODEL_CONDITIONS:
            condition_calls = by_condition[condition]
            row = observation_map[condition]
            if int(row["scheduled_calls"]) != n_agents:
                call_counts_ok = False
            if int(row["inference_attempts"]) != sum(int(call["attempts"]) for call in condition_calls):
                no_fallback_ok = False
            if int(row["prompt_tokens"]) != sum(int(call["prompt_tokens"]) for call in condition_calls):
                token_accounting_ok = False
            if int(row["output_tokens"]) != sum(int(call["output_tokens"]) for call in condition_calls):
                token_accounting_ok = False
            path = (
                result["central_full_path"]
                if condition == "central_full"
                else [item["probability_after"] for item in result[f"{condition}_history"]]
            )
            expected_movement = sum(
                abs(after - before)
                for before, after in zip([world.prior] + list(path[:-1]), path, strict=True)
            )
            if not _close(row["path_total_movement"], expected_movement):
                path_accounting_ok = False

    checks.extend(
        [
            _check("scheduled_call_counts", call_counts_ok, "Each condition has exactly five decisions per completed cell."),
            _check("counterbalanced_execution_order", execution_order_ok, "Condition execution order follows the frozen rotation."),
            _check("matched_generation_seeds", matched_seeds_ok, "Corresponding positions use identical generation seeds."),
            _check("information_firewalls", privacy_ok, "No prompt contains an oracle, outcome, or unavailable raw signals."),
            _check("exact_prompt_reconstruction", prompt_reconstruction_ok, "Every prompt reconstructs exactly from legally visible state."),
            _check("forecast_bounds", forecast_bounds_ok, "All probabilities are finite and inside the common bounds."),
            _check("probability_chain_accounting", chain_accounting_ok, "Compressed and market states reproduce their saved probability chains."),
            _check("observation_recomputation", observation_math_ok, "Forecast errors and expected proper scores recompute exactly."),
            _check("path_movement_accounting", path_accounting_ok, "All path movement diagnostics recompute exactly."),
            _check("no_behavioral_fallback", no_fallback_ok, "Every scheduled decision has a parsed response; no fallback exists."),
            _check("raw_response_audit", raw_response_ok, "Saved parsed forecasts match the final raw model responses."),
            _check("token_accounting", token_accounting_ok, "Prompt and output token totals match all saved calls."),
            _check("unique_event_system_observations", observation_uniqueness_ok, "Each completed cell has one observation per system."),
        ]
    )

    complete = state.get("status") == "complete" and len(results) == expected_events
    checks.append(
        _check(
            "complete_sample",
            complete,
            f"Status is {state.get('status')!r}; {len(results)} of {expected_events} cells are complete.",
        )
    )

    observations = _flatten_observations(state)
    if complete and observations:
        summaries = summarize_observations(observations)
        token_values = {
            row["system"]: float(row["weighted_mean_total_tokens_per_cell"])
            for row in summaries
            if row["system"] in MODEL_CONDITIONS
        }
        if all(value > 0 for value in token_values.values()):
            smallest = min(token_values.values())
            largest = max(token_values.values())
            imbalance = (largest - smallest) / ((largest + smallest) / 2.0)
            if imbalance > config.experiment.token_imbalance_warning_fraction:
                warnings.append(
                    f"Observed token imbalance is {imbalance:.1%}, above the descriptive "
                    f"threshold of {config.experiment.token_imbalance_warning_fraction:.1%}."
                )
    if reason_lengths and max(reason_lengths) > config.experiment.reason_word_limit:
        warnings.append(
            f"At least one reason exceeded the requested {config.experiment.reason_word_limit}-word limit; "
            "reasons are never fed into another condition and do not affect forecasts."
        )

    valid = all(item["passed"] for item in checks)
    confirmatory_design = config.is_confirmatory_design()
    return {
        "protocol_version": config.protocol_version,
        "status": state.get("status"),
        "completed_events": len(results),
        "expected_events": expected_events,
        "valid": valid,
        "confirmatory_design": confirmatory_design,
        "confirmatory_valid": bool(valid and confirmatory_design),
        "warnings": warnings,
        "checks": checks,
    }


def summarize_observations(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for system in SYSTEM_ORDER:
        selected = [row for row in observations if row["system"] == system]
        if not selected:
            continue
        weights = [float(row["evaluation_weight"]) for row in selected]
        mse = weighted_mean([float(row["oracle_squared_error"]) for row in selected], weights)
        total_tokens = [float(row["prompt_tokens"]) + float(row["output_tokens"]) for row in selected]
        rows.append(
            {
                "system": system,
                "cells": len(selected),
                "evaluation_weight": sum(weights),
                "weighted_mean_oracle_squared_error": mse,
                "weighted_rmse_to_oracle": math.sqrt(mse),
                "weighted_mean_oracle_absolute_error": weighted_mean(
                    [float(row["oracle_absolute_error"]) for row in selected], weights
                ),
                "weighted_expected_brier": weighted_mean(
                    [float(row["expected_brier"]) for row in selected], weights
                ),
                "weighted_expected_log_loss": weighted_mean(
                    [float(row["expected_log_loss"]) for row in selected], weights
                ),
                "unweighted_mean_oracle_squared_error": mean(
                    float(row["oracle_squared_error"]) for row in selected
                ),
                "weighted_mean_forecast": weighted_mean(
                    [float(row["forecast"]) for row in selected], weights
                ),
                "scheduled_calls_per_cell": mean(float(row["scheduled_calls"]) for row in selected),
                "weighted_mean_inference_attempts_per_cell": weighted_mean(
                    [float(row["inference_attempts"]) for row in selected], weights
                ),
                "weighted_mean_prompt_tokens_per_cell": weighted_mean(
                    [float(row["prompt_tokens"]) for row in selected], weights
                ),
                "weighted_mean_output_tokens_per_cell": weighted_mean(
                    [float(row["output_tokens"]) for row in selected], weights
                ),
                "weighted_mean_total_tokens_per_cell": weighted_mean(total_tokens, weights),
            }
        )
    return rows


def paired_comparisons(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_system: dict[str, dict[int, dict[str, Any]]] = {}
    for system in SYSTEM_ORDER:
        by_system[system] = {
            int(row["event_id"]): row for row in observations if row["system"] == system
        }
    pairs = [
        ("market", "central_compact", "primary_mechanism_test"),
        ("market", "central_full", "v1_replication"),
        ("central_compact", "central_full", "compression_scaffold_test"),
        ("market", "prior", "market_vs_no_aggregation"),
    ]
    rows: list[dict[str, Any]] = []
    for first, second, role in pairs:
        common = sorted(set(by_system[first]) & set(by_system[second]))
        if not common:
            continue
        differences = [
            float(by_system[first][event_id]["oracle_squared_error"])
            - float(by_system[second][event_id]["oracle_squared_error"])
            for event_id in common
        ]
        weights = [float(by_system[first][event_id]["evaluation_weight"]) for event_id in common]
        normalized_total = sum(weights)
        normalized = [weight / normalized_total for weight in weights]
        rows.append(
            {
                "comparison": f"{first}_minus_{second}",
                "role": role,
                "cells": len(common),
                "weighted_mean_difference": sum(
                    difference * weight for difference, weight in zip(differences, normalized, strict=True)
                ),
                "weighted_first_strict_win_probability": sum(
                    weight
                    for difference, weight in zip(differences, normalized, strict=True)
                    if difference < -1e-15
                ),
                "weighted_tie_probability": sum(
                    weight
                    for difference, weight in zip(differences, normalized, strict=True)
                    if _close(difference, 0.0, 1e-15)
                ),
                "unweighted_mean_difference": mean(differences),
                "unweighted_first_strict_win_rate": sum(value < -1e-15 for value in differences)
                / len(differences),
                "interpretation": "negative_favors_first_system",
            }
        )
    return rows


def call_diagnostics(state: dict[str, Any]) -> list[dict[str, Any]]:
    worlds = {world.event_id: world for world in (World.from_dict(item) for item in state.get("worlds", []))}
    buckets: dict[tuple[str, int], list[dict[str, float]]] = {}
    for result in state.get("event_results", []):
        world = worlds[int(result["event_id"])]
        ordered = world.ordered_signals()
        for condition in MODEL_CONDITIONS:
            calls = sorted(
                [call for call in result["calls"] if call["condition"] == condition],
                key=lambda item: int(item["position"]),
            )
            for position, (call, signal) in enumerate(zip(calls, ordered, strict=True), start=1):
                actual = float(call["probability"])
                if condition == "central_full":
                    target = bayesian_posterior(
                        world.prior,
                        [(item.signal_yes, item.reliability) for item in ordered[:position]],
                    )
                    before = world.prior if position == 1 else float(calls[position - 2]["probability"])
                else:
                    before = float(call["context"]["current_public_probability"])
                    target = bayesian_posterior(before, [(signal.signal_yes, signal.reliability)])
                move = actual - before
                correct_direction = move > 0 if signal.signal_yes else move < 0
                flat = _close(move, 0.0, 1e-15)
                buckets.setdefault((condition, position), []).append(
                    {
                        "weight": world.evaluation_weight,
                        "error": actual - target,
                        "absolute_error": abs(actual - target),
                        "squared_error": (actual - target) ** 2,
                        "correct_direction": float(correct_direction),
                        "flat": float(flat),
                    }
                )
    rows: list[dict[str, Any]] = []
    for (condition, position), values in sorted(buckets.items()):
        weights = [item["weight"] for item in values]
        rows.append(
            {
                "condition": condition,
                "position": position,
                "calls": len(values),
                "weighted_local_target_mae": weighted_mean(
                    [item["absolute_error"] for item in values], weights
                ),
                "weighted_local_target_rmse": math.sqrt(
                    weighted_mean([item["squared_error"] for item in values], weights)
                ),
                "weighted_local_target_bias": weighted_mean(
                    [item["error"] for item in values], weights
                ),
                "weighted_correct_direction_rate": weighted_mean(
                    [item["correct_direction"] for item in values], weights
                ),
                "weighted_flat_rate": weighted_mean([item["flat"] for item in values], weights),
            }
        )
    return rows


def _html_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "".join(f"<th>{escape(label)}</th>" for _, label in columns)
    body_parts: list[str] = []
    for row in rows:
        cells: list[str] = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                shown = f"{value:.6f}"
            else:
                shown = str(value)
            cells.append(f"<td>{escape(shown)}</td>")
        body_parts.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_parts)}</tbody></table>"


def render_report(
    state: dict[str, Any],
    validation: dict[str, Any],
    summaries: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> str:
    status_class = "ok" if validation["confirmatory_valid"] else ("warn" if validation["valid"] else "bad")
    checks_html = "".join(
        f"<li class={'ok' if item['passed'] else 'bad'}>{escape(item['name'])}: "
        f"{escape(item['detail'])}</li>"
        for item in validation["checks"]
    )
    warnings_html = "".join(f"<li>{escape(item)}</li>" for item in validation["warnings"])
    summary_table = _html_table(
        summaries,
        [
            ("system", "System"),
            ("weighted_mean_oracle_squared_error", "Weighted oracle MSE"),
            ("weighted_expected_brier", "Expected Brier"),
            ("weighted_expected_log_loss", "Expected log loss"),
            ("weighted_mean_total_tokens_per_cell", "Tokens/cell"),
        ],
    )
    comparison_table = _html_table(
        comparisons,
        [
            ("comparison", "Comparison"),
            ("role", "Role"),
            ("weighted_mean_difference", "Weighted difference"),
            ("weighted_first_strict_win_probability", "First-system win probability"),
        ],
    )
    diagnostic_table = _html_table(
        diagnostics,
        [
            ("condition", "Condition"),
            ("position", "Position"),
            ("weighted_local_target_mae", "Local-target MAE"),
            ("weighted_correct_direction_rate", "Correct direction"),
        ],
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Aggregation mechanisms V2</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1180px;margin:32px auto;padding:0 18px;color:#172033}}
h1,h2{{color:#0f2447}} table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}
th,td{{border:1px solid #d8deea;padding:7px 9px;text-align:left}} th{{background:#eef3fa}}
.ok{{color:#137333}} .warn{{color:#9a6700}} .bad{{color:#b42318}} code{{background:#f2f4f7;padding:2px 4px}}
</style></head><body>
<h1>Aggregation mechanisms V2</h1>
<p class="{status_class}"><strong>Status:</strong> {escape(str(validation['status']).upper())} &mdash;
mechanically valid: {str(validation['valid']).lower()} &mdash;
confirmatory valid: {str(validation['confirmatory_valid']).lower()}</p>
<p>Protocol <code>{escape(str(validation['protocol_version']))}</code>; completed cells:
{validation['completed_events']} / {validation['expected_events']}.</p>
<h2>Weighted system results</h2>{summary_table}
<h2>Pre-specified paired contrasts</h2>{comparison_table}
<p>Negative weighted differences favor the first named system.</p>
<h2>Sequential diagnostics</h2>{diagnostic_table}
<h2>Warnings</h2><ul>{warnings_html or '<li>None.</li>'}</ul>
<h2>Validity audit</h2><ul>{checks_html}</ul>
</body></html>"""


def write_run_artifacts(state: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = validate_run(state)
    calls = _flatten_calls(state)
    observations = _flatten_observations(state)
    summaries = summarize_observations(observations)
    comparisons = paired_comparisons(observations)
    diagnostics = call_diagnostics(state)

    _write_jsonl(output_dir / "worlds.jsonl", state.get("worlds", []))
    _write_jsonl(output_dir / "model_calls.jsonl", calls)
    _write_csv(
        output_dir / "decisions.csv",
        calls,
        [
            "event_id", "cell_id", "condition", "position", "actor_id", "generation_seed",
            "probability", "reason", "attempts", "elapsed_seconds", "prompt_tokens", "output_tokens",
        ],
    )
    _write_csv(
        output_dir / "observations.csv",
        observations,
        [
            "event_id", "cell_id", "evaluation_weight", "population_weight", "system", "forecast",
            "prior", "oracle_posterior", "oracle_squared_error", "oracle_absolute_error",
            "expected_brier", "expected_log_loss", "scheduled_calls", "inference_attempts",
            "prompt_tokens", "output_tokens", "path_total_movement",
        ],
    )
    _write_csv(
        output_dir / "summary.csv",
        summaries,
        [
            "system", "cells", "evaluation_weight", "weighted_mean_oracle_squared_error",
            "weighted_rmse_to_oracle", "weighted_mean_oracle_absolute_error",
            "weighted_expected_brier", "weighted_expected_log_loss",
            "unweighted_mean_oracle_squared_error", "weighted_mean_forecast",
            "scheduled_calls_per_cell", "weighted_mean_inference_attempts_per_cell",
            "weighted_mean_prompt_tokens_per_cell", "weighted_mean_output_tokens_per_cell",
            "weighted_mean_total_tokens_per_cell",
        ],
    )
    _write_csv(
        output_dir / "comparisons.csv",
        comparisons,
        [
            "comparison", "role", "cells", "weighted_mean_difference",
            "weighted_first_strict_win_probability", "weighted_tie_probability",
            "unweighted_mean_difference", "unweighted_first_strict_win_rate", "interpretation",
        ],
    )
    _write_csv(
        output_dir / "diagnostics.csv",
        diagnostics,
        [
            "condition", "position", "calls", "weighted_local_target_mae",
            "weighted_local_target_rmse", "weighted_local_target_bias",
            "weighted_correct_direction_rate", "weighted_flat_rate",
        ],
    )
    _atomic_write_text(output_dir / "validation.json", json.dumps(validation, indent=2, sort_keys=True))
    _atomic_write_text(
        output_dir / "report.html",
        render_report(state, validation, summaries, comparisons, diagnostics),
    )
    return validation
