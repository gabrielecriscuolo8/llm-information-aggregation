from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
import time
from typing import Any, Protocol
from urllib import error, request

from .scoring import bayesian_posterior


class ModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class Completion:
    probability: float
    reason: str
    raw_responses: tuple[str, ...]
    attempts: int
    elapsed_seconds: float
    prompt_tokens: int
    output_tokens: int


class ForecastModel(Protocol):
    def complete_forecast(self, system: str, user: str, seed: int) -> Completion: ...

    def identity(self) -> dict[str, Any]: ...


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for position, character in enumerate(cleaned):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(cleaned[position:])
                break
            except json.JSONDecodeError:
                continue
        else:
            raise ModelError("model response did not contain a JSON object")
    if not isinstance(value, dict):
        raise ModelError("model JSON response must be an object")
    return value


def _parse_probability(value: Any) -> float:
    if isinstance(value, bool):
        raise ModelError("probability cannot be Boolean")
    if isinstance(value, (int, float)):
        probability = float(value)
    elif isinstance(value, str):
        cleaned = value.strip()
        try:
            if cleaned.endswith("%"):
                probability = float(cleaned[:-1].strip()) / 100.0
            else:
                probability = float(cleaned)
        except ValueError as exc:
            raise ModelError("probability string is not numeric") from exc
    else:
        raise ModelError("probability must be a number or numeric string")
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ModelError("probability must be finite and lie inside [0, 1]")
    return probability


def parse_forecast(text: str) -> tuple[float, str]:
    payload = extract_json_object(text)
    if "probability" not in payload:
        raise ModelError("model JSON is missing the probability field")
    probability = _parse_probability(payload["probability"])
    reason_value = payload.get("reason", payload.get("reasoning", ""))
    if not isinstance(reason_value, str) or not reason_value.strip():
        raise ModelError("model JSON is missing a non-empty reason field")
    return probability, reason_value.strip()


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float,
        timeout_seconds: int,
        max_retries: int,
        max_output_tokens: int,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.max_output_tokens = max_output_tokens

    def check(self) -> dict[str, Any]:
        try:
            with request.urlopen(f"{self.base_url}/api/tags", timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ModelError(
                f"Cannot reach Ollama at {self.base_url}. Start Ollama and try again. ({exc})"
            ) from exc
        models = [item for item in payload.get("models", []) if isinstance(item, dict)]
        matches = [item for item in models if item.get("name") == self.model]
        if not matches:
            names = {item.get("name") for item in models}
            shown = ", ".join(sorted(name for name in names if name)) or "none"
            raise ModelError(
                f"Model {self.model!r} is not installed in Ollama. Available models: {shown}"
            )
        selected = matches[0]
        return {
            "provider": "ollama",
            "name": self.model,
            "digest": selected.get("digest"),
            "size": selected.get("size"),
            "modified_at": selected.get("modified_at"),
            "details": selected.get("details"),
        }

    def identity(self) -> dict[str, Any]:
        return self.check()

    def complete_forecast(self, system: str, user: str, seed: int) -> Completion:
        raw_responses: list[str] = []
        last_error: Exception | None = None
        started = time.monotonic()
        corrective_suffix = ""
        prompt_tokens = 0
        output_tokens = 0
        context = _context_from_prompt(user)
        interval = context.get("permitted_probability_interval", [0.0, 1.0])
        minimum, maximum = float(interval[0]), float(interval[1])

        for attempt in range(1, self.max_retries + 1):
            payload = {
                "model": self.model,
                "stream": False,
                "format": "json",
                "think": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user + corrective_suffix},
                ],
                "options": {
                    "temperature": self.temperature,
                    "seed": (seed + attempt - 1) % 2_147_483_647,
                    "num_predict": self.max_output_tokens,
                },
            }
            body = json.dumps(payload).encode("utf-8")
            try:
                req = request.Request(
                    f"{self.base_url}/api/chat",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with request.urlopen(req, timeout=self.timeout_seconds) as response:
                    raw_payload = json.loads(response.read().decode("utf-8"))
                prompt_tokens += int(raw_payload.get("prompt_eval_count", 0) or 0)
                output_tokens += int(raw_payload.get("eval_count", 0) or 0)
                content = str(raw_payload.get("message", {}).get("content", ""))
                raw_responses.append(content)
                probability, reason = parse_forecast(content)
                if not minimum <= probability <= maximum:
                    raise ModelError(
                        f"probability {probability} lies outside permitted interval "
                        f"[{minimum}, {maximum}]"
                    )
                return Completion(
                    probability=probability,
                    reason=reason,
                    raw_responses=tuple(raw_responses),
                    attempts=attempt,
                    elapsed_seconds=time.monotonic() - started,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                )
            except (error.URLError, TimeoutError, json.JSONDecodeError, ModelError, ValueError) as exc:
                last_error = exc
                corrective_suffix = (
                    "\n\nYour preceding response could not be parsed. Return only the two required "
                    "fields with a decimal probability inside the permitted interval and a non-empty reason."
                )
                if attempt < self.max_retries:
                    time.sleep(0.5 * attempt)

        raise ModelError(
            f"Ollama failed after {self.max_retries} attempts; no fallback was used. Last error: {last_error}"
        )


def _context_from_prompt(user: str) -> dict[str, Any]:
    match = re.search(r"<CONTEXT_JSON>\s*(.*?)\s*</CONTEXT_JSON>", user, re.DOTALL)
    if not match:
        raise ModelError("mock backend could not find structured context")
    return json.loads(match.group(1))


class MockClient:
    """Deterministic rational backend used only for tests and mechanical validation."""

    def identity(self) -> dict[str, Any]:
        return {"provider": "mock", "name": "deterministic-rational-mock", "digest": "built-in"}

    def complete_forecast(self, system: str, user: str, seed: int) -> Completion:
        started = time.monotonic()
        context = _context_from_prompt(user)
        task = str(context["task"])
        prior = float(context["prior"])

        if task == "central_full_update":
            observations = [
                (item["signal"] == "YES", float(item["reliability"]))
                for item in context["visible_signals"]
            ]
            probability = bayesian_posterior(prior, observations)
            probability = min(
                max(probability, float(context["permitted_probability_interval"][0])),
                float(context["permitted_probability_interval"][1]),
            )
            reason = "Rational mock posterior from every signal visible to the central analyst."
        elif task in {"central_compact_update", "market_trade"}:
            signal = context["new_signal"]
            probability = bayesian_posterior(
                float(context["current_public_probability"]),
                [(signal["signal"] == "YES", float(signal["reliability"]))],
            )
            probability = min(
                max(probability, float(context["permitted_probability_interval"][0])),
                float(context["permitted_probability_interval"][1]),
            )
            reason = "Rational mock update of current public odds with the new signal."
        else:
            raise ModelError(f"unknown mock task: {task}")

        raw = json.dumps({"probability": probability, "reason": reason})
        return Completion(
            probability=probability,
            reason=reason,
            raw_responses=(raw,),
            attempts=1,
            elapsed_seconds=time.monotonic() - started,
            prompt_tokens=0,
            output_tokens=0,
        )
