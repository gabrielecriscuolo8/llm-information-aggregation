from __future__ import annotations

import copy
import itertools
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from aggregation_v2.config import AppConfig  # noqa: E402
from aggregation_v2.llm import Completion, MockClient, ModelError, parse_forecast  # noqa: E402
from aggregation_v2.prompts import SYSTEM_PROMPT  # noqa: E402
from aggregation_v2.reporting import paired_comparisons, summarize_observations, validate_run  # noqa: E402
from aggregation_v2.scoring import (  # noqa: E402
    bayesian_posterior,
    expected_brier_score,
    expected_log_loss,
)
from aggregation_v2.simulation import (  # noqa: E402
    execute_run,
    initialize_state,
    load_state,
    run_event,
)
from aggregation_v2.world import (  # noqa: E402
    condition_execution_order,
    generate_worlds,
    stable_call_seed,
)


def development_config(events: int = 3) -> AppConfig:
    config = AppConfig.load(PROJECT_DIR / "config.json")
    config.experiment.events = events
    config.model.provider = "mock"
    config.validate()
    return config


class FailingAfterClient:
    def __init__(self, allowed_calls: int) -> None:
        self.allowed_calls = allowed_calls
        self.calls = 0
        self.delegate = MockClient()

    def identity(self):
        return self.delegate.identity()

    def complete_forecast(self, system: str, user: str, seed: int) -> Completion:
        self.calls += 1
        if self.calls > self.allowed_calls:
            raise ModelError("intentional test interruption")
        return self.delegate.complete_forecast(system, user, seed)


class AlwaysFailClient:
    def identity(self):
        return {"provider": "mock", "name": "always-fail", "digest": "test"}

    def complete_forecast(self, system: str, user: str, seed: int) -> Completion:
        raise ModelError("intentional failure without fallback")


class V2Tests(unittest.TestCase):
    def test_default_config_is_frozen_confirmatory_design(self) -> None:
        config = AppConfig.load(PROJECT_DIR / "config.json")
        self.assertEqual(config.protocol_version, "aggregation-confirmatory-v2.0")
        self.assertEqual(config.experiment.events, 160)
        self.assertEqual(config.experiment.exhaustive_event_count, 160)
        self.assertEqual(config.model.name, "qwen3:8b")
        self.assertEqual(config.model.temperature, 0.0)
        self.assertTrue(config.is_confirmatory_design())

    def test_exhaustive_worlds_are_reproducible_unique_and_weighted(self) -> None:
        config = AppConfig.load(PROJECT_DIR / "config.json")
        first = generate_worlds(config.experiment)
        second = generate_worlds(config.experiment)
        self.assertEqual([world.to_dict() for world in first], [world.to_dict() for world in second])
        self.assertEqual(len(first), 160)
        self.assertEqual(len({world.cell_id for world in first}), 160)
        self.assertAlmostEqual(sum(world.population_weight for world in first), 1.0, places=12)
        self.assertAlmostEqual(sum(world.evaluation_weight for world in first), 1.0, places=12)
        for prior in config.experiment.priors:
            selected = [world for world in first if math.isclose(world.prior, prior)]
            self.assertEqual(len(selected), 32)
            self.assertEqual(
                {tuple(signal.signal_yes for signal in world.signals) for world in selected},
                set(itertools.product((False, True), repeat=5)),
            )

    def test_reliabilities_are_exactly_balanced_across_positions(self) -> None:
        config = AppConfig.load(PROJECT_DIR / "config.json")
        worlds = generate_worlds(config.experiment)
        for reliability in config.experiment.reliabilities:
            counts = [
                sum(
                    math.isclose(world.ordered_signals()[position].reliability, reliability)
                    for world in worlds
                )
                for position in range(5)
            ]
            self.assertEqual(counts, [32] * 5)

    def test_population_weight_and_oracle_laws(self) -> None:
        config = AppConfig.load(PROJECT_DIR / "config.json")
        worlds = generate_worlds(config.experiment)
        for world in worlds:
            self.assertAlmostEqual(world.oracle_posterior, world.recompute_oracle(), places=12)
            self.assertAlmostEqual(
                world.population_weight,
                world.recompute_population_weight(len(config.experiment.priors)),
                places=12,
            )
        weighted_oracle_mean = sum(
            world.population_weight * world.oracle_posterior for world in worlds
        )
        self.assertAlmostEqual(weighted_oracle_mean, sum(config.experiment.priors) / 5, places=12)

    def test_entire_state_space_respects_probability_bounds(self) -> None:
        config = AppConfig.load(PROJECT_DIR / "config.json")
        worlds = generate_worlds(config.experiment)
        self.assertGreaterEqual(
            min(world.oracle_posterior for world in worlds),
            config.experiment.forecast_min_probability,
        )
        self.assertLessEqual(
            max(world.oracle_posterior for world in worlds),
            config.experiment.forecast_max_probability,
        )

    def test_expected_scores_are_proper_and_outcome_free(self) -> None:
        probability = 0.37
        truth = 0.62
        expected = truth * (1 - truth) + (probability - truth) ** 2
        self.assertAlmostEqual(expected_brier_score(probability, truth), expected, places=12)
        self.assertLess(expected_brier_score(truth, truth), expected_brier_score(probability, truth))
        self.assertLess(expected_log_loss(truth, truth), expected_log_loss(probability, truth))

    def test_probability_parser_accepts_json_and_rejects_invalid(self) -> None:
        probability, reason = parse_forecast('{"probability":"42%","reason":"private evidence"}')
        self.assertAlmostEqual(probability, 0.42)
        self.assertEqual(reason, "private evidence")
        with self.assertRaises(ModelError):
            parse_forecast('{"probability":1.2,"reason":"outside"}')
        with self.assertRaises(ModelError):
            parse_forecast('{"probability":0.4}')

    def test_prompts_have_no_literal_output_example(self) -> None:
        self.assertNotIn('{"probability"', SYSTEM_PROMPT)

    def test_rational_mock_makes_all_three_conditions_equal_oracle(self) -> None:
        config = development_config(1)
        world = generate_worlds(config.experiment)[0]
        result = run_event(config, MockClient(), world)
        counts = {
            condition: sum(call["condition"] == condition for call in result["calls"])
            for condition in ("central_full", "central_compact", "market")
        }
        self.assertEqual(counts, {"central_full": 5, "central_compact": 5, "market": 5})
        observations = {row["system"]: row for row in result["observations"]}
        for condition in ("central_full", "central_compact", "market"):
            self.assertAlmostEqual(
                observations[condition]["forecast"], world.oracle_posterior, places=11
            )

    def test_compact_and_market_receive_same_numerical_state(self) -> None:
        config = development_config(1)
        result = run_event(config, MockClient(), generate_worlds(config.experiment)[0])
        compact = sorted(
            [call for call in result["calls"] if call["condition"] == "central_compact"],
            key=lambda item: item["position"],
        )
        market = sorted(
            [call for call in result["calls"] if call["condition"] == "market"],
            key=lambda item: item["position"],
        )
        for left, right in zip(compact, market, strict=True):
            left_context = copy.deepcopy(left["context"])
            right_context = copy.deepcopy(right["context"])
            left_context.pop("task")
            right_context.pop("task")
            left_context.pop("actor_id")
            right_context.pop("actor_id")
            self.assertEqual(left_context, right_context)

    def test_generation_seeds_match_across_all_conditions(self) -> None:
        config = development_config(1)
        world = generate_worlds(config.experiment)[0]
        result = run_event(config, MockClient(), world)
        for position in range(1, 6):
            calls = [call for call in result["calls"] if call["position"] == position]
            self.assertEqual(
                {call["generation_seed"] for call in calls},
                {stable_call_seed(config.experiment.seed, world.cell_id, position)},
            )

    def test_condition_execution_order_is_counterbalanced(self) -> None:
        orders = [condition_execution_order(event_id) for event_id in range(1, 161)]
        first_counts = {
            condition: sum(order[0] == condition for order in orders)
            for condition in ("central_full", "central_compact", "market")
        }
        self.assertLessEqual(max(first_counts.values()) - min(first_counts.values()), 1)

    def test_end_to_end_mock_run_writes_valid_artifacts(self) -> None:
        config = development_config(3)
        model = MockClient()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            state = execute_run(
                config, model, model.identity(), output, PROJECT_DIR, resume=False
            )
            validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "complete")
            self.assertTrue(validation["valid"])
            self.assertFalse(validation["confirmatory_valid"])
            self.assertEqual(sum(len(item["calls"]) for item in state["event_results"]), 45)
            for name in (
                "report.html", "worlds.jsonl", "model_calls.jsonl", "decisions.csv",
                "observations.csv", "summary.csv", "comparisons.csv", "diagnostics.csv",
                "validation.json", "state.json",
            ):
                self.assertTrue((output / name).is_file(), name)

    def test_weighted_summaries_and_comparisons_recompute(self) -> None:
        config = development_config(4)
        model = MockClient()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            state = execute_run(
                config, model, model.identity(), output, PROJECT_DIR, resume=False
            )
            observations = [row for result in state["event_results"] for row in result["observations"]]
            summaries = {row["system"]: row for row in summarize_observations(observations)}
            for condition in ("central_full", "central_compact", "market", "oracle_bayes"):
                self.assertAlmostEqual(
                    summaries[condition]["weighted_mean_oracle_squared_error"], 0.0, places=12
                )
            comparisons = paired_comparisons(observations)
            self.assertTrue(all(abs(row["weighted_mean_difference"]) < 1e-12 for row in comparisons[:3]))

    def test_interrupted_run_resumes_without_duplicates(self) -> None:
        config = development_config(2)
        identity = MockClient().identity()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            with self.assertRaises(ModelError):
                execute_run(
                    config, FailingAfterClient(15), identity, output, PROJECT_DIR, resume=False
                )
            interrupted = load_state(output)
            self.assertEqual(interrupted["status"], "interrupted")
            self.assertEqual(interrupted["next_event_index"], 1)
            completed = execute_run(
                config, MockClient(), identity, output, PROJECT_DIR, resume=True
            )
            self.assertEqual(completed["status"], "complete")
            self.assertEqual([item["event_id"] for item in completed["event_results"]], [1, 2])
            self.assertTrue(validate_run(completed)["valid"])

    def test_resume_rejects_changed_model_digest(self) -> None:
        config = development_config(1)
        model = MockClient()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            execute_run(config, model, model.identity(), output, PROJECT_DIR, resume=False)
            changed = dict(model.identity())
            changed["digest"] = "different"
            with self.assertRaisesRegex(ValueError, "model identity changed"):
                execute_run(config, model, changed, output, PROJECT_DIR, resume=True)

    def test_failure_stops_without_silent_fallback(self) -> None:
        config = development_config(1)
        model = AlwaysFailClient()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            with self.assertRaises(ModelError):
                execute_run(
                    config, model, model.identity(), output, PROJECT_DIR, resume=False
                )
            state = load_state(output)
            self.assertEqual(state["status"], "interrupted")
            self.assertEqual(state["event_results"], [])
            self.assertIn("intentional failure", state["last_error"]["message"])

    def _complete_one_cell_state(self):
        config = development_config(1)
        model = MockClient()
        state = initialize_state(config, PROJECT_DIR, model.identity())
        world = generate_worlds(config.experiment)[0]
        state["event_results"] = [run_event(config, model, world)]
        state["next_event_index"] = 1
        state["status"] = "complete"
        return state

    def test_validator_detects_information_leak_and_prompt_tamper(self) -> None:
        state = self._complete_one_cell_state()
        tampered = copy.deepcopy(state)
        market = [
            call for call in tampered["event_results"][0]["calls"] if call["condition"] == "market"
        ][-1]
        market["context"]["oracle_posterior"] = 0.9
        validation = validate_run(tampered)
        checks = {item["name"]: item["passed"] for item in validation["checks"]}
        self.assertFalse(checks["information_firewalls"])
        self.assertFalse(checks["exact_prompt_reconstruction"])
        self.assertFalse(validation["valid"])

    def test_validator_detects_weight_and_chain_tampering(self) -> None:
        state = self._complete_one_cell_state()
        weighted = copy.deepcopy(state)
        weighted["worlds"][0]["evaluation_weight"] = 0.5
        validation = validate_run(weighted)
        checks = {item["name"]: item["passed"] for item in validation["checks"]}
        self.assertFalse(checks["exact_frozen_world_schedule"])
        self.assertFalse(checks["probability_weight_accounting"])

        chained = copy.deepcopy(state)
        chained["event_results"][0]["market_history"][0]["probability_after"] += 0.01
        validation = validate_run(chained)
        checks = {item["name"]: item["passed"] for item in validation["checks"]}
        self.assertFalse(checks["probability_chain_accounting"])

    def test_validator_detects_observation_and_raw_response_tampering(self) -> None:
        state = self._complete_one_cell_state()
        observation = copy.deepcopy(state)
        target = next(
            row for row in observation["event_results"][0]["observations"] if row["system"] == "market"
        )
        target["oracle_squared_error"] += 0.01
        checks = {item["name"]: item["passed"] for item in validate_run(observation)["checks"]}
        self.assertFalse(checks["observation_recomputation"])

        raw = copy.deepcopy(state)
        call = raw["event_results"][0]["calls"][0]
        call["raw_responses"][-1] = '{"probability":0.5,"reason":"tampered"}'
        checks = {item["name"]: item["passed"] for item in validate_run(raw)["checks"]}
        self.assertFalse(checks["raw_response_audit"])


if __name__ == "__main__":
    unittest.main()
