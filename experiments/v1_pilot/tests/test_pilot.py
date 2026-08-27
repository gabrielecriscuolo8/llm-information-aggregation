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

from aggregation_pilot.config import AppConfig  # noqa: E402
from aggregation_pilot.llm import Completion, MockClient, ModelError, parse_forecast  # noqa: E402
from aggregation_pilot.prompts import SYSTEM_PROMPT, ensemble_prompt, market_prompt  # noqa: E402
from aggregation_pilot.reporting import validate_run  # noqa: E402
from aggregation_pilot.scoring import (  # noqa: E402
    bayesian_posterior,
    market_scoring_profit,
    prior_corrected_log_pool,
)
from aggregation_pilot.simulation import execute_run, load_state, run_event  # noqa: E402
from aggregation_pilot.world import generate_worlds, stable_call_seed  # noqa: E402


def small_config(events: int = 3) -> AppConfig:
    config = AppConfig.load(PROJECT_DIR / "config.json")
    config.experiment.events = events
    config.experiment.bootstrap_repetitions = 200
    config.model.provider = "mock"
    config.validate()
    return config


class FailingAfterClient:
    def __init__(self, allowed_calls: int) -> None:
        self.allowed_calls = allowed_calls
        self.calls = 0
        self.delegate = MockClient()

    def complete_forecast(self, system: str, user: str, seed: int) -> Completion:
        self.calls += 1
        if self.calls > self.allowed_calls:
            raise ModelError("intentional test interruption")
        return self.delegate.complete_forecast(system, user, seed)


class AlwaysFailClient:
    def complete_forecast(self, system: str, user: str, seed: int) -> Completion:
        raise ModelError("intentional failure without fallback")


class PilotTests(unittest.TestCase):
    def test_default_config_is_frozen_and_valid(self) -> None:
        config = AppConfig.load(PROJECT_DIR / "config.json")
        self.assertEqual(config.protocol_version, "aggregation-pilot-v1.0")
        self.assertEqual(config.experiment.events, 30)
        self.assertEqual(config.experiment.n_agents, 5)
        self.assertEqual(config.model.name, "qwen3:8b")

    def test_world_generation_is_reproducible_and_balanced(self) -> None:
        config = AppConfig.load(PROJECT_DIR / "config.json")
        first = [world.to_dict() for world in generate_worlds(config.experiment)]
        second = [world.to_dict() for world in generate_worlds(config.experiment)]
        self.assertEqual(first, second)
        counts = {
            prior: sum(math.isclose(world["prior"], prior) for world in first)
            for prior in config.experiment.priors
        }
        self.assertEqual(set(counts.values()), {6})
        for world in first:
            self.assertEqual(len(world["signals"]), 5)
            self.assertEqual(len(set(world["order"])), 5)

    def test_bayesian_posterior_known_case(self) -> None:
        posterior = bayesian_posterior(0.5, [(True, 0.8), (False, 0.6)])
        self.assertAlmostEqual(posterior, 8.0 / 11.0, places=12)

    def test_log_pool_recovers_oracle_from_private_bayes_forecasts(self) -> None:
        prior = 0.35
        observations = [(True, 0.6), (False, 0.68), (True, 0.75), (True, 0.82), (False, 0.9)]
        private = [bayesian_posterior(prior, [observation]) for observation in observations]
        pooled = prior_corrected_log_pool(private, prior)
        oracle = bayesian_posterior(prior, observations)
        self.assertAlmostEqual(pooled, oracle, places=12)

    def test_market_scoring_payoffs_telescope(self) -> None:
        prices = [0.35, 0.52, 0.41, 0.77, 0.66]
        for outcome in (False, True):
            individual = sum(
                market_scoring_profit(before, after, outcome)
                for before, after in zip(prices, prices[1:])
            )
            total = market_scoring_profit(prices[0], prices[-1], outcome)
            self.assertAlmostEqual(individual, total, places=12)

    def test_entire_signal_state_space_has_no_bound_saturation_and_rational_equivalence(self) -> None:
        config = AppConfig.load(PROJECT_DIR / "config.json")
        reliabilities = config.experiment.reliabilities
        for prior in config.experiment.priors:
            for bits in itertools.product((False, True), repeat=config.experiment.n_agents):
                observations = list(zip(bits, reliabilities, strict=True))
                oracle = bayesian_posterior(prior, observations)
                self.assertGreaterEqual(oracle, config.experiment.forecast_min_probability)
                self.assertLessEqual(oracle, config.experiment.forecast_max_probability)
                private = [
                    bayesian_posterior(prior, [observation]) for observation in observations
                ]
                self.assertAlmostEqual(
                    prior_corrected_log_pool(private, prior), oracle, places=12
                )
                market_price = prior
                for observation in observations:
                    market_price = bayesian_posterior(market_price, [observation])
                self.assertAlmostEqual(market_price, oracle, places=12)

    def test_probability_parser_accepts_json_and_rejects_invalid(self) -> None:
        probability, reason = parse_forecast('{"probability":"42%","reason":"private evidence"}')
        self.assertAlmostEqual(probability, 0.42)
        self.assertEqual(reason, "private evidence")
        with self.assertRaises(ModelError):
            parse_forecast('{"probability":1.2,"reason":"outside"}')
        with self.assertRaises(ModelError):
            parse_forecast('{"probability":0.4}')

    def test_prompts_do_not_use_literal_output_examples_or_hidden_data(self) -> None:
        self.assertNotIn('{"probability"', SYSTEM_PROMPT)
        config = small_config(1)
        world = generate_worlds(config.experiment)[0]
        signal = world.ordered_signals()[0]
        ensemble_text, ensemble_context = ensemble_prompt(
            world.event_id,
            world.prior,
            signal,
            config.experiment.forecast_min_probability,
            config.experiment.forecast_max_probability,
        )
        self.assertNotIn("oracle_posterior", ensemble_text)
        self.assertNotIn("outcome", ensemble_context)
        market_text, market_context = market_prompt(
            world.event_id,
            1,
            5,
            world.prior,
            signal,
            world.prior,
            [],
            config.experiment.forecast_min_probability,
            config.experiment.forecast_max_probability,
        )
        self.assertNotIn("oracle_posterior", market_text)
        self.assertNotIn("outcome", market_context)

    def test_rational_mock_event_obeys_budget_and_aggregates(self) -> None:
        config = small_config(1)
        world = generate_worlds(config.experiment)[0]
        result = run_event(config, MockClient(), world)
        counts = {
            condition: sum(call["condition"] == condition for call in result["calls"])
            for condition in ("central", "ensemble", "market")
        }
        self.assertEqual(counts, {"central": 5, "ensemble": 5, "market": 5})
        observations = {row["system"]: row for row in result["observations"]}
        expected_bounded = min(
            max(world.oracle_posterior, config.experiment.forecast_min_probability),
            config.experiment.forecast_max_probability,
        )
        self.assertAlmostEqual(observations["central"]["forecast"], expected_bounded, places=11)
        self.assertAlmostEqual(observations["ensemble_log_pool"]["forecast"], expected_bounded, places=11)
        expected_market = expected_bounded
        self.assertAlmostEqual(observations["market"]["forecast"], expected_market, places=11)

    def test_common_random_seeds_are_matched_across_conditions(self) -> None:
        config = small_config(1)
        world = generate_worlds(config.experiment)[0]
        result = run_event(config, MockClient(), world)
        for position in range(1, 6):
            values = {
                call["condition"]: call["generation_seed"]
                for call in result["calls"]
                if call["position"] == position
            }
            expected = stable_call_seed(config.experiment.seed, world.event_id, position)
            self.assertEqual(values, {"central": expected, "ensemble": expected, "market": expected})

    def test_market_public_history_never_leaks_signal_outcome_or_profit(self) -> None:
        config = small_config(1)
        world = generate_worlds(config.experiment)[0]
        result = run_event(config, MockClient(), world)
        market_calls = [call for call in result["calls"] if call["condition"] == "market"]
        for call in market_calls:
            for history_row in call["context"]["public_price_history"]:
                self.assertTrue(
                    {"signal", "outcome", "reason", "realized_profit"}.isdisjoint(history_row)
                )

    def test_end_to_end_run_writes_valid_complete_artifacts(self) -> None:
        config = small_config(3)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            state = execute_run(config, MockClient(), output, PROJECT_DIR, resume=False)
            self.assertEqual(state["status"], "complete")
            validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
            self.assertTrue(validation["valid"])
            self.assertEqual(len(state["event_results"]), 3)
            self.assertEqual(sum(len(item["calls"]) for item in state["event_results"]), 45)
            self.assertEqual(sum(len(item["observations"]) for item in state["event_results"]), 18)
            for name in (
                "report.html",
                "worlds.jsonl",
                "model_calls.jsonl",
                "decisions.csv",
                "observations.csv",
                "summary.csv",
                "comparisons.csv",
                "state.json",
            ):
                self.assertTrue((output / name).is_file(), name)

    def test_interrupted_run_resumes_without_duplicates(self) -> None:
        config = small_config(2)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            with self.assertRaises(ModelError):
                execute_run(config, FailingAfterClient(15), output, PROJECT_DIR, resume=False)
            interrupted = load_state(output)
            self.assertEqual(interrupted["status"], "interrupted")
            self.assertEqual(interrupted["next_event_index"], 1)
            self.assertEqual(len(interrupted["event_results"]), 1)
            completed = execute_run(config, MockClient(), output, PROJECT_DIR, resume=True)
            self.assertEqual(completed["status"], "complete")
            self.assertEqual([item["event_id"] for item in completed["event_results"]], [1, 2])
            self.assertTrue(validate_run(completed)["valid"])

    def test_failure_stops_without_silent_fallback(self) -> None:
        config = small_config(1)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            with self.assertRaises(ModelError):
                execute_run(config, AlwaysFailClient(), output, PROJECT_DIR, resume=False)
            state = load_state(output)
            self.assertEqual(state["status"], "interrupted")
            self.assertEqual(state["event_results"], [])
            self.assertIn("intentional failure", state["last_error"]["message"])

    def test_validator_detects_market_information_leak(self) -> None:
        config = small_config(1)
        world = generate_worlds(config.experiment)[0]
        result = run_event(config, MockClient(), world)
        fake_state = {
            "protocol_sha256": "recorded",
            "config_fingerprint": "recorded",
            "implementation_fingerprint": "recorded",
            "config": config.to_dict(),
            "status": "complete",
            "next_event_index": 1,
            "worlds": [world.to_dict()],
            "event_results": [copy.deepcopy(result)],
        }
        market_calls = [
            call for call in fake_state["event_results"][0]["calls"] if call["condition"] == "market"
        ]
        market_calls[-1]["context"]["public_price_history"][0]["signal"] = "YES"
        validation = validate_run(fake_state)
        checks = {item["name"]: item["passed"] for item in validation["checks"]}
        self.assertFalse(checks["information_firewalls"])
        self.assertFalse(validation["valid"])


if __name__ == "__main__":
    unittest.main()
