from __future__ import annotations

import csv
from html import escape
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .config import AppConfig
from .prompts import SYSTEM_PROMPT, central_prompt, ensemble_prompt, market_prompt
from .scoring import (
    arithmetic_pool,
    brier_score,
    clamp_probability,
    log_loss,
    market_scoring_profit,
    paired_bootstrap_mean_ci,
    prior_corrected_log_pool,
)
from .world import World, stable_call_seed


SYSTEM_ORDER = [
    "prior",
    "central",
    "ensemble_mean",
    "ensemble_log_pool",
    "market",
    "oracle_bayes",
]


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
    _atomic_write_text(path, text)


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


def validate_run(state: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    config = AppConfig.from_dict(state["config"])
    worlds = [World.from_dict(item) for item in state.get("worlds", [])]
    results = state.get("event_results", [])
    expected_events = config.experiment.events
    n_agents = config.experiment.n_agents

    checks.append(
        _check(
            "protocol_and_config_fingerprints_present",
            bool(state.get("protocol_sha256") and state.get("config_fingerprint") and state.get("implementation_fingerprint")),
            "Protocol, configuration, and implementation fingerprints are recorded.",
        )
    )
    checks.append(
        _check(
            "world_count",
            len(worlds) == expected_events,
            f"Found {len(worlds)} of {expected_events} pre-generated worlds.",
        )
    )
    world_ids = [world.event_id for world in worlds]
    checks.append(
        _check(
            "unique_ordered_world_ids",
            world_ids == list(range(1, len(worlds) + 1)),
            "World identifiers are unique and sequential.",
        )
    )
    prior_counts = {prior: sum(_close(world.prior, prior) for world in worlds) for prior in config.experiment.priors}
    balanced = max(prior_counts.values(), default=0) - min(prior_counts.values(), default=0) <= 1
    checks.append(
        _check(
            "balanced_prior_schedule",
            balanced,
            "Prior counts differ by no more than one: " + json.dumps(prior_counts, sort_keys=True),
        )
    )
    oracle_ok = all(_close(world.oracle_posterior, world.recompute_oracle(), 1e-11) for world in worlds)
    checks.append(_check("oracle_recomputation", oracle_ok, "Every saved oracle posterior recomputes from its world."))

    result_ids = [int(result["event_id"]) for result in results]
    checks.append(
        _check(
            "checkpoint_consistency",
            int(state.get("next_event_index", -1)) == len(results) and result_ids == list(range(1, len(results) + 1)),
            f"Checkpoint contains {len(results)} completed events without gaps or duplicates.",
        )
    )

    call_counts_ok = True
    matched_seeds_ok = True
    context_privacy_ok = True
    forecast_bounds_ok = True
    observation_math_ok = True
    market_accounting_ok = True
    no_fallback_ok = True
    attempts_consistent = True
    observation_uniqueness_ok = True
    prompt_reconstruction_ok = True
    token_accounting_ok = True

    for result in results:
        world = worlds[int(result["event_id"]) - 1]
        calls = result["calls"]
        by_condition = {
            condition: sorted(
                [call for call in calls if call["condition"] == condition],
                key=lambda item: int(item["position"]),
            )
            for condition in ("central", "ensemble", "market")
        }
        if any(len(items) != n_agents for items in by_condition.values()):
            call_counts_ok = False
            continue

        for position in range(1, n_agents + 1):
            expected_seed = stable_call_seed(config.experiment.seed, world.event_id, position)
            position_seeds = [
                int(by_condition[condition][position - 1]["generation_seed"])
                for condition in ("central", "ensemble", "market")
            ]
            if position_seeds != [expected_seed, expected_seed, expected_seed]:
                matched_seeds_ok = False

        ordered_signals = world.ordered_signals()
        previous_probability: float | None = None
        previous_reason: str | None = None
        for position, call in enumerate(by_condition["central"], start=1):
            context = call["context"]
            expected_visible = [
                {
                    "agent_id": signal.agent_id,
                    "reliability": signal.reliability,
                    "signal": "YES" if signal.signal_yes else "NO",
                }
                for signal in ordered_signals[:position]
            ]
            if context.get("visible_signals") != expected_visible or "private_signal" in context:
                context_privacy_ok = False
            expected_prompt, expected_context = central_prompt(
                world.event_id,
                position,
                n_agents,
                world.prior,
                ordered_signals[:position],
                previous_probability,
                previous_reason,
                config.experiment.forecast_min_probability,
                config.experiment.forecast_max_probability,
            )
            if (
                call.get("system_prompt") != SYSTEM_PROMPT
                or call.get("user_prompt") != expected_prompt
                or context != expected_context
            ):
                prompt_reconstruction_ok = False
            previous_probability = float(call["probability"])
            previous_reason = str(call["reason"])

        for position, call in enumerate(by_condition["ensemble"], start=1):
            context = call["context"]
            signal = ordered_signals[position - 1]
            expected_private = {
                "agent_id": signal.agent_id,
                "reliability": signal.reliability,
                "signal": "YES" if signal.signal_yes else "NO",
            }
            if context.get("private_signal") != expected_private:
                context_privacy_ok = False
            if "visible_signals" in context or "public_price_history" in context:
                context_privacy_ok = False
            expected_prompt, expected_context = ensemble_prompt(
                world.event_id,
                world.prior,
                signal,
                config.experiment.forecast_min_probability,
                config.experiment.forecast_max_probability,
            )
            if (
                call.get("system_prompt") != SYSTEM_PROMPT
                or call.get("user_prompt") != expected_prompt
                or context != expected_context
            ):
                prompt_reconstruction_ok = False

        reconstructed_history: list[dict[str, Any]] = []
        for position, call in enumerate(by_condition["market"], start=1):
            context = call["context"]
            signal = ordered_signals[position - 1]
            expected_current_price = (
                world.prior if not reconstructed_history else float(reconstructed_history[-1]["price_after"])
            )
            expected_private = {
                "agent_id": signal.agent_id,
                "reliability": signal.reliability,
                "signal": "YES" if signal.signal_yes else "NO",
            }
            if context.get("private_signal") != expected_private:
                context_privacy_ok = False
            if context.get("public_price_history") != reconstructed_history:
                context_privacy_ok = False
            if not _close(float(context.get("current_price", math.nan)), expected_current_price):
                prompt_reconstruction_ok = False
            if any(
                any(key in item for key in ("signal", "reason", "outcome", "realized_profit"))
                for item in context.get("public_price_history", [])
            ):
                context_privacy_ok = False
            expected_prompt, expected_context = market_prompt(
                world.event_id,
                position,
                n_agents,
                world.prior,
                signal,
                float(context["current_price"]),
                list(reconstructed_history),
                config.experiment.forecast_min_probability,
                config.experiment.forecast_max_probability,
            )
            if (
                call.get("system_prompt") != SYSTEM_PROMPT
                or call.get("user_prompt") != expected_prompt
                or context != expected_context
            ):
                prompt_reconstruction_ok = False
            history_row = result["market_history"][position - 1]
            reconstructed_history.append(
                {
                    "position": history_row["position"],
                    "trader_id": history_row["trader_id"],
                    "reliability": history_row["reliability"],
                    "price_before": history_row["price_before"],
                    "price_after": history_row["price_after"],
                }
            )

        for call in calls:
            probability = float(call["probability"])
            interval = call["context"]["permitted_probability_interval"]
            if not math.isfinite(probability) or not float(interval[0]) <= probability <= float(interval[1]):
                forecast_bounds_ok = False
            if int(call.get("attempts", 0)) < 1 or not call.get("raw_responses"):
                no_fallback_ok = False
            if int(call.get("prompt_tokens", -1)) < 0 or int(call.get("output_tokens", -1)) < 0:
                token_accounting_ok = False

        observations = result["observations"]
        if len(observations) != len(SYSTEM_ORDER) or {row["system"] for row in observations} != set(SYSTEM_ORDER):
            observation_uniqueness_ok = False
            continue
        observation_map = {row["system"]: row for row in observations}
        central_final = float(by_condition["central"][-1]["probability"])
        ensemble_values = [float(call["probability"]) for call in by_condition["ensemble"]]
        market_final = float(by_condition["market"][-1]["probability"])
        expected_forecasts = {
            "prior": world.prior,
            "oracle_bayes": world.oracle_posterior,
            "central": central_final,
            "ensemble_mean": clamp_probability(
                arithmetic_pool(ensemble_values),
                config.experiment.forecast_min_probability,
                config.experiment.forecast_max_probability,
            ),
            "ensemble_log_pool": clamp_probability(
                prior_corrected_log_pool(ensemble_values, world.prior),
                config.experiment.forecast_min_probability,
                config.experiment.forecast_max_probability,
            ),
            "market": market_final,
        }
        for system, expected_forecast in expected_forecasts.items():
            row = observation_map[system]
            if not _close(row["forecast"], expected_forecast):
                observation_math_ok = False
            if not _close(row["oracle_squared_error"], (expected_forecast - world.oracle_posterior) ** 2):
                observation_math_ok = False
            if not _close(row["brier"], brier_score(expected_forecast, world.outcome)):
                observation_math_ok = False
            if not _close(row["log_loss"], log_loss(expected_forecast, world.outcome)):
                observation_math_ok = False

        for condition in ("central", "ensemble", "market"):
            expected_attempts = sum(int(call["attempts"]) for call in by_condition[condition])
            observation_system = "ensemble_mean" if condition == "ensemble" else condition
            if int(observation_map[observation_system]["inference_attempts"]) != expected_attempts:
                attempts_consistent = False
            expected_prompt_tokens = sum(int(call["prompt_tokens"]) for call in by_condition[condition])
            expected_output_tokens = sum(int(call["output_tokens"]) for call in by_condition[condition])
            if (
                int(observation_map[observation_system]["prompt_tokens"]) != expected_prompt_tokens
                or int(observation_map[observation_system]["output_tokens"]) != expected_output_tokens
            ):
                token_accounting_ok = False

        market_history = result["market_history"]
        if len(market_history) != n_agents:
            market_accounting_ok = False
        else:
            previous = world.prior
            profits = 0.0
            for item in market_history:
                if not _close(item["price_before"], previous):
                    market_accounting_ok = False
                expected_profit = market_scoring_profit(previous, item["price_after"], world.outcome)
                if not _close(item["realized_profit"], expected_profit):
                    market_accounting_ok = False
                profits += float(item["realized_profit"])
                previous = float(item["price_after"])
            telescoping = market_scoring_profit(world.prior, market_final, world.outcome)
            if not _close(profits, telescoping):
                market_accounting_ok = False

    checks.extend(
        [
            _check("scheduled_call_counts", call_counts_ok, "Each completed event has five central, ensemble, and market decisions."),
            _check("matched_generation_seeds", matched_seeds_ok, "Corresponding positions use common random generation seeds."),
            _check("information_firewalls", context_privacy_ok, "No ensemble or market prompt context contains hidden signal values."),
            _check("exact_prompt_reconstruction", prompt_reconstruction_ok, "Every saved prompt reconstructs exactly from its legally visible state."),
            _check("forecast_and_price_bounds", forecast_bounds_ok, "All decisions respect their pre-specified probability interval."),
            _check("observation_recomputation", observation_math_ok, "Forecast aggregations and scoring fields recompute exactly."),
            _check("market_scoring_accounting", market_accounting_ok, "Market scoring profits telescope from initial to final price."),
            _check("no_behavioral_fallback", no_fallback_ok, "Every scheduled decision has a parsed model response; no fallback exists."),
            _check("inference_attempt_accounting", attempts_consistent, "Formatting/network retry attempts are recorded consistently."),
            _check("token_accounting", token_accounting_ok, "Available Ollama prompt and output token counts are aggregated consistently."),
            _check("unique_event_system_observations", observation_uniqueness_ok, "Every completed event has one observation per system."),
        ]
    )
    complete = state.get("status") == "complete" and len(results) == expected_events
    checks.append(
        _check(
            "complete_sample",
            complete,
            f"Status is {state.get('status')!r}; {len(results)} of {expected_events} events are complete.",
        )
    )
    return {
        "valid": all(item["passed"] for item in checks),
        "status": state.get("status"),
        "completed_events": len(results),
        "expected_events": expected_events,
        "checks": checks,
    }


def build_summary(state: dict[str, Any]) -> list[dict[str, Any]]:
    observations = _flatten_observations(state)
    rows: list[dict[str, Any]] = []
    for system in SYSTEM_ORDER:
        selected = [row for row in observations if row["system"] == system]
        if not selected:
            continue
        rows.append(
            {
                "system": system,
                "events": len(selected),
                "mean_oracle_squared_error": mean(float(row["oracle_squared_error"]) for row in selected),
                "rmse_to_oracle": math.sqrt(mean(float(row["oracle_squared_error"]) for row in selected)),
                "mean_oracle_absolute_error": mean(float(row["oracle_absolute_error"]) for row in selected),
                "mean_brier": mean(float(row["brier"]) for row in selected),
                "mean_log_loss": mean(float(row["log_loss"]) for row in selected),
                "mean_forecast": mean(float(row["forecast"]) for row in selected),
                "scheduled_calls_per_event": mean(float(row["scheduled_calls"]) for row in selected),
                "mean_inference_attempts_per_event": mean(float(row["inference_attempts"]) for row in selected),
                "mean_prompt_tokens_per_event": mean(float(row["prompt_tokens"]) for row in selected),
                "mean_output_tokens_per_event": mean(float(row["output_tokens"]) for row in selected),
                "mean_total_tokens_per_event": mean(
                    float(row["prompt_tokens"]) + float(row["output_tokens"]) for row in selected
                ),
            }
        )
    return rows


def build_comparisons(state: dict[str, Any]) -> list[dict[str, Any]]:
    config = AppConfig.from_dict(state["config"])
    observations = _flatten_observations(state)
    by_event: dict[int, dict[str, dict[str, Any]]] = {}
    for row in observations:
        by_event.setdefault(int(row["event_id"]), {})[str(row["system"])] = row
    comparisons = ["ensemble_log_pool", "central", "ensemble_mean", "prior"]
    rows: list[dict[str, Any]] = []
    for offset, comparator in enumerate(comparisons, start=1):
        differences: list[float] = []
        wins = 0
        ties = 0
        for event_id in sorted(by_event):
            systems = by_event[event_id]
            if "market" not in systems or comparator not in systems:
                continue
            market_error = float(systems["market"]["oracle_squared_error"])
            comparator_error = float(systems[comparator]["oracle_squared_error"])
            differences.append(market_error - comparator_error)
            if _close(market_error, comparator_error, 1e-12):
                ties += 1
            elif market_error < comparator_error:
                wins += 1
        if not differences:
            continue
        estimate, low, high = paired_bootstrap_mean_ci(
            differences,
            config.experiment.bootstrap_repetitions,
            config.experiment.seed + 100_000 + offset,
        )
        rows.append(
            {
                "comparison": f"market_minus_{comparator}",
                "events": len(differences),
                "mean_paired_difference": estimate,
                "bootstrap_ci_2_5": low,
                "bootstrap_ci_97_5": high,
                "market_strict_win_rate": wins / len(differences),
                "tie_rate": ties / len(differences),
                "interpretation": "negative_favors_market",
            }
        )
    return rows


def _format(value: Any, digits: int = 5) -> str:
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return f"{float(value):.{digits}f}"
    return escape(str(value))


def render_report(
    state: dict[str, Any],
    summary: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    validation: dict[str, Any],
) -> str:
    status_class = "valid" if validation["valid"] else "incomplete"
    summary_rows = "".join(
        "<tr>"
        f"<td>{escape(row['system'])}</td>"
        f"<td>{int(row['events'])}</td>"
        f"<td>{_format(row['mean_oracle_squared_error'])}</td>"
        f"<td>{_format(row['rmse_to_oracle'])}</td>"
        f"<td>{_format(row['mean_brier'])}</td>"
        f"<td>{_format(row['mean_log_loss'])}</td>"
        f"<td>{_format(row['mean_inference_attempts_per_event'], 2)}</td>"
        f"<td>{_format(row['mean_total_tokens_per_event'], 1)}</td>"
        "</tr>"
        for row in summary
    )
    comparison_rows = "".join(
        "<tr>"
        f"<td>{escape(row['comparison'])}</td>"
        f"<td>{_format(row['mean_paired_difference'])}</td>"
        f"<td>[{_format(row['bootstrap_ci_2_5'])}, {_format(row['bootstrap_ci_97_5'])}]</td>"
        f"<td>{_format(row['market_strict_win_rate'], 3)}</td>"
        "</tr>"
        for row in comparisons
    ) or '<tr><td colspan="4">No completed paired events yet.</td></tr>'
    check_rows = "".join(
        "<tr>"
        f"<td>{'PASS' if item['passed'] else 'FAIL'}</td>"
        f"<td>{escape(item['name'])}</td>"
        f"<td>{escape(item['detail'])}</td>"
        "</tr>"
        for item in validation["checks"]
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM Aggregation Pilot Report</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;max-width:1120px;margin:32px auto;padding:0 18px;color:#1c2430;background:#f6f8fb}}
h1,h2{{color:#12213a}} .card{{background:white;border:1px solid #dbe2ea;border-radius:12px;padding:18px;margin:16px 0;box-shadow:0 2px 8px #12213a0d}}
.status{{font-weight:700;padding:8px 12px;border-radius:8px;display:inline-block}} .valid{{background:#dff6e8;color:#126534}} .incomplete{{background:#fff0d5;color:#7b4c00}}
table{{width:100%;border-collapse:collapse;font-size:14px}} th,td{{text-align:left;padding:9px;border-bottom:1px solid #e6ebf1}} th{{background:#eef3f8}}
code{{background:#eef3f8;padding:2px 5px;border-radius:4px}} .small{{color:#526173;font-size:13px}}
</style></head><body>
<h1>Central Analyst vs Ensemble vs Prediction Market</h1>
<div class="card"><span class="status {status_class}">{escape(str(validation['status']).upper())}</span>
<p>{validation['completed_events']} of {validation['expected_events']} paired events completed. Protocol <code>{escape(state['protocol_version'])}</code>.</p>
<p class="small">Primary loss: squared distance from the exact Bayesian full-information posterior. Lower is better.</p></div>
<div class="card"><h2>System performance</h2><table><thead><tr><th>System</th><th>N</th><th>Posterior MSE</th><th>Posterior RMSE</th><th>Brier</th><th>Log loss</th><th>Attempts/event</th><th>Tokens/event</th></tr></thead><tbody>{summary_rows}</tbody></table></div>
<div class="card"><h2>Paired market contrasts</h2><p class="small">Negative differences favor the market. Intervals are deterministic event-level bootstrap intervals.</p>
<table><thead><tr><th>Contrast</th><th>Mean difference</th><th>95% interval</th><th>Market win rate</th></tr></thead><tbody>{comparison_rows}</tbody></table></div>
<div class="card"><h2>Validity checks</h2><table><thead><tr><th>Result</th><th>Check</th><th>Detail</th></tr></thead><tbody>{check_rows}</tbody></table></div>
<div class="card"><h2>Interpretation boundary</h2><p>This pilot concerns one model, synthetic conditionally independent signals, and fixed prompts. It does not establish performance on real news or across model families. The market is promising only if it beats both ensemble aggregation rules and the central analyst, or establishes a defensible accuracy/cost advantage.</p></div>
</body></html>"""


def write_run_artifacts(state: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    worlds = state.get("worlds", [])
    calls = _flatten_calls(state)
    observations = _flatten_observations(state)
    summary = build_summary(state)
    comparisons = build_comparisons(state)
    validation = validate_run(state)

    _write_jsonl(output_dir / "worlds.jsonl", worlds)
    _write_jsonl(output_dir / "model_calls.jsonl", calls)
    _write_csv(
        output_dir / "decisions.csv",
        calls,
        [
            "event_id",
            "condition",
            "position",
            "actor_id",
            "generation_seed",
            "probability",
            "reason",
            "attempts",
            "elapsed_seconds",
            "prompt_tokens",
            "output_tokens",
        ],
    )
    _write_csv(
        output_dir / "observations.csv",
        observations,
        [
            "event_id",
            "system",
            "forecast",
            "prior",
            "oracle_posterior",
            "outcome",
            "oracle_squared_error",
            "oracle_absolute_error",
            "brier",
            "log_loss",
            "scheduled_calls",
            "inference_attempts",
            "prompt_tokens",
            "output_tokens",
            "path_total_movement",
            "ensemble_standard_deviation",
            "market_total_realized_profit",
        ],
    )
    _write_csv(
        output_dir / "summary.csv",
        summary,
        [
            "system",
            "events",
            "mean_oracle_squared_error",
            "rmse_to_oracle",
            "mean_oracle_absolute_error",
            "mean_brier",
            "mean_log_loss",
            "mean_forecast",
            "scheduled_calls_per_event",
            "mean_inference_attempts_per_event",
            "mean_prompt_tokens_per_event",
            "mean_output_tokens_per_event",
            "mean_total_tokens_per_event",
        ],
    )
    _write_csv(
        output_dir / "comparisons.csv",
        comparisons,
        [
            "comparison",
            "events",
            "mean_paired_difference",
            "bootstrap_ci_2_5",
            "bootstrap_ci_97_5",
            "market_strict_win_rate",
            "tie_rate",
            "interpretation",
        ],
    )
    _atomic_write_text(
        output_dir / "validation.json",
        json.dumps(validation, indent=2, sort_keys=True, ensure_ascii=False),
    )
    _atomic_write_text(output_dir / "report.html", render_report(state, summary, comparisons, validation))
    return validation
