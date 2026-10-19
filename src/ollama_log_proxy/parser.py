"""Parse Ollama API responses for token usage and model information."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

INFERENCE_ENDPOINTS = {
    "/api/generate",
    "/api/chat",
    "/api/embeddings",
    "/api/embed",
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/responses",
}
POLL_ENDPOINTS = {"/api/tags", "/api/ps", "/api/version", "/api/show"}


@dataclass
class LogEntry:
    """Single inference log record with timing and token counts."""

    timestamp: str
    caller_ip: str
    model: str
    endpoint: str
    request_type: str
    status_code: int
    duration_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entry to a plain dict for JSON encoding."""
        return {
            "timestamp": self.timestamp,
            "caller_ip": self.caller_ip,
            "model": self.model,
            "endpoint": self.endpoint,
            "request_type": self.request_type,
            "status_code": self.status_code,
            "duration_ms": self.duration_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "metadata": self.metadata,
        }


def _normalize_endpoint(endpoint: str) -> str:
    """Normalize an endpoint path so trailing slashes still match."""
    normalized = endpoint.rstrip("/")
    return normalized or "/"


def classify_request(endpoint: str) -> str:
    """Classify an Ollama API endpoint as 'inference', 'poll', or 'other'."""
    endpoint = _normalize_endpoint(endpoint)
    if endpoint in INFERENCE_ENDPOINTS:
        return "inference"
    if endpoint in POLL_ENDPOINTS:
        return "poll"
    return "other"


def extract_model_from_body(body: bytes) -> str:
    """Return the 'model' field from a JSON request body, or empty string on failure."""
    if not body:
        return ""
    try:
        data = json.loads(body)
        return data.get("model", "")
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ""


def _extract_from_obj(
    obj: dict,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
) -> tuple[int, int, int]:
    prompt_count = obj.get("prompt_eval_count")
    if prompt_count:
        prompt_tokens = prompt_count

    eval_count = obj.get("eval_count")
    if eval_count:
        completion_tokens = eval_count

    usage = obj.get("usage")
    if isinstance(usage, dict):
        usage_prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
        if usage_prompt:
            prompt_tokens = usage_prompt

        usage_completion = usage.get("completion_tokens", usage.get("output_tokens"))
        if usage_completion:
            completion_tokens = usage_completion

        usage_total = usage.get("total_tokens")
        if usage_total:
            total_tokens = usage_total

    return prompt_tokens, completion_tokens, total_tokens


def _iter_response_objects(text: str):
    """Yield JSON objects from a plain JSON response, NDJSON, or SSE stream."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(obj, dict):
            yield obj
        return

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("data:"):
            line = line[len("data:") :].strip()
            if not line or line == "[DONE]":
                continue
        elif line.startswith(("event:", "id:", "retry:", ":")):
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        if isinstance(obj, dict):
            yield obj


def extract_usage_from_response(response_bytes: bytes, endpoint: str) -> tuple[int, int, int]:
    """Extract (prompt, completion, total) token counts from an Ollama response.

    Handles single JSON objects, streaming NDJSON, and OpenAI-style SSE streams.
    """
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    text = response_bytes.decode("utf-8", errors="replace").strip()
    if not text:
        return 0, 0, 0

    for obj in _iter_response_objects(text):
        prompt_tokens, completion_tokens, total_tokens = _extract_from_obj(
            obj, prompt_tokens, completion_tokens, total_tokens
        )

    if not total_tokens:
        total_tokens = prompt_tokens + completion_tokens

    return prompt_tokens, completion_tokens, total_tokens


def now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()
