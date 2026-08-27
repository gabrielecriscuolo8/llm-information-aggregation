# LLM Information Aggregation Pilot

This package runs one controlled, paired experiment comparing:

1. a single sequential LLM analyst;
2. five independent LLM forecasts aggregated by arithmetic mean and a
   prior-corrected log-odds pool;
3. five LLM traders in a logarithmic market scoring rule;
4. prior-only and exact Bayesian benchmarks.

All systems receive the same synthetic worlds and collectively receive the same
five noisy signals. Central, ensemble, and market each use five scheduled model
decisions per event; actual prompt and output tokens are recorded. The primary metric is squared distance from the exact
full-information Bayesian posterior, not just realized-outcome accuracy.

The complete design is frozen in `PROTOCOL.md`. Do not modify `config.json`,
prompts, or source code before the main run.

## Requirements

- Windows, macOS, or Linux;
- Python 3.10 or newer;
- Ollama running locally;
- Ollama model `qwen3:8b` installed.

No third-party Python packages are required.

## Run on Windows PowerShell

Open PowerShell inside this `llm-aggregation-pilot` folder.

First verify Ollama without starting an experiment:

```powershell
python run_pilot.py --check
```

Optional local code tests (no Ollama calls):

```powershell
python -m unittest discover -s tests -v
```

Start the pre-specified 30-event experiment once:

```powershell
python run_pilot.py
```

The full run schedules 450 model decisions and may take a long time locally.
Closing the terminal or stopping the computer does not invalidate completed
events. Use the exact resume command printed at startup, for example:

```powershell
python run_pilot.py --resume "C:\path\to\runs\run_..."
```

Do not use `--events` for the real experiment. That option exists only for
automated development checks.

## Valid completion

The terminal prints `Experiment complete and validation passed` only after all
pre-specified mechanical and information-firewall checks succeed. Open
`report.html` and confirm that its status is `COMPLETE`. The authoritative
machine-readable audit is `validation.json`.

The program never substitutes a fallback forecast. If Ollama repeatedly returns
an unusable response, the run stops and preserves every earlier completed event;
resume it after correcting the operational problem.

## Outputs

Each `runs/run_...` folder contains:

- `report.html`: human-readable results and validity checks;
- `validation.json`: authoritative validity audit;
- `summary.csv`: system-level performance;
- `comparisons.csv`: paired market contrasts and bootstrap intervals;
- `observations.csv`: event-by-system results;
- `decisions.csv`: compact model decision log;
- `model_calls.jsonl`: complete prompts, parsed decisions, and retry audit;
- `worlds.jsonl`: frozen paired synthetic worlds;
- `state.json`: atomic checkpoint used for resume.

After completion, compress the entire run folder and send the ZIP for analysis.

## Interpretation boundary

This pilot evaluates one local model, one frozen prompt family, and synthetic
conditionally independent signals. It does not by itself establish performance
on real forecasting tasks or across model families. Information purchase is
deliberately excluded; it belongs to a later experiment only if the aggregation
mechanism is promising.
