from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from aggregation_pilot.config import AppConfig  # noqa: E402
from aggregation_pilot.llm import MockClient, ModelError, OllamaClient  # noqa: E402
from aggregation_pilot.simulation import execute_run, load_state  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the controlled central-vs-ensemble-vs-market aggregation pilot."
    )
    parser.add_argument("--config", default=str(PROJECT_DIR / "config.json"), help="JSON config path")
    parser.add_argument("--events", type=int, help="Override event count for development only")
    parser.add_argument("--backend", choices=("ollama", "mock"), help="Override model backend")
    parser.add_argument("--output", help="New run output directory")
    parser.add_argument("--resume", help="Resume an interrupted run directory")
    parser.add_argument("--check", action="store_true", help="Only verify Ollama and the configured model")
    return parser.parse_args()


def make_model(config: AppConfig):
    if config.model.provider == "mock":
        return MockClient()
    return OllamaClient(
        base_url=config.model.base_url,
        model=config.model.name,
        temperature=config.model.temperature,
        timeout_seconds=config.model.timeout_seconds,
        max_retries=config.model.max_retries,
        max_output_tokens=config.model.max_output_tokens,
    )


def main() -> int:
    args = parse_args()
    if args.output and args.resume:
        raise ValueError("use either --output or --resume, not both")

    if args.resume:
        if args.events is not None or args.backend is not None:
            raise ValueError("do not override events or backend when resuming")
        output_dir = Path(args.resume).resolve()
        state = load_state(output_dir)
        config = AppConfig.from_dict(state["config"])
        resume = True
    else:
        config = AppConfig.load(args.config)
        if args.events is not None:
            config.experiment.events = args.events
        if args.backend is not None:
            config.model.provider = args.backend
        config.validate()
        output_dir = (
            Path(args.output).resolve()
            if args.output
            else PROJECT_DIR / "runs" / ("run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        )
        resume = False

    model = make_model(config)
    if isinstance(model, OllamaClient):
        model.check()
        print(f"Ollama is ready and model {config.model.name!r} is available.")
    else:
        print("Using deterministic rational mock backend (validation only).")
    if args.check:
        return 0

    print(f"Protocol: {config.protocol_version}")
    print(f"Paired events: {config.experiment.events}")
    print(f"Scheduled model decisions: {config.experiment.events * config.experiment.n_agents * 3}")
    print(f"Run output: {output_dir}", flush=True)
    print(f'Resume if interrupted: python run_pilot.py --resume "{output_dir}"', flush=True)

    execute_run(
        config=config,
        model=model,
        output_dir=output_dir,
        project_dir=PROJECT_DIR,
        resume=resume,
    )
    print(f"\nExperiment complete and validation passed. Open: {output_dir / 'report.html'}")
    print(f"Validation details: {output_dir / 'validation.json'}")
    print(f'Create a ZIP of this entire folder and send it to me: "{output_dir}"')
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted safely. Use the resume command printed above.", file=sys.stderr)
        raise SystemExit(130)
    except (ModelError, ValueError, OSError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

