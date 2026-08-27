# Aggregation Mechanisms V2

## What this experiment tests

The frozen V2 compares a five-agent prediction market with two single-LLM baselines on every one of
the 160 possible prior-by-signal configurations. Its main purpose is to distinguish a true
multi-agent advantage from the simpler benefit of compressing earlier evidence into a probability.

Read `PROTOCOL.md` for the complete pre-specified design. Do not edit the configuration, prompts, or
source before running.

## Requirements

- Python 3.10 or newer;
- Ollama running locally;
- the exact Ollama model `qwen3:8b` installed.

No third-party Python packages are required.

## Run on Windows PowerShell

Open PowerShell inside this `aggregation-mechanisms-v2` folder and run:

```powershell
python run_v2.py --check
python run_v2.py
```

The full experiment schedules 2,400 model decisions. On the PC used for V1 it should take roughly
5-7 hours. It checkpoints after every completed cell.

If interrupted, use the exact command printed at startup:

```powershell
python run_v2.py --resume "C:\path\to\runs\run_..."
```

Do not start a second run and do not modify `config.json`. A short Ollama test is deliberately absent:
`--check` verifies connectivity without consuming experimental cells.

## Completion and files to send

Only trust a terminal message saying `Experiment complete and validation passed`. The output folder
contains:

- `report.html`: readable results;
- `validation.json`: authoritative audit;
- `summary.csv` and `comparisons.csv`: final weighted metrics;
- `diagnostics.csv`: sequential-update diagnostics;
- `observations.csv` and `decisions.csv`: analysis tables;
- `model_calls.jsonl`: complete prompt and response audit;
- `worlds.jsonl`: exhaustive weighted cells;
- `state.json`: resumable checkpoint.

Compress the entire completed `runs\run_...` folder and send the ZIP for analysis.

## Optional local code tests

These use a deterministic mock and never call Ollama:

```powershell
python -m unittest discover -s tests -v
```
