from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import dataclass
from typing import List, Protocol


class ActionProposer(Protocol):
    def propose(self, model_code: str, action_names: List[str], budget: int) -> List[str]:
        ...


@dataclass
class OpenAICompatibleProposer:
    """Provider-agnostic proposer for OpenAI-compatible chat completion APIs."""

    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = 45.0
    retry_count: int = 2
    max_feedback_items: int = 8
    feedback_log: List[str] | None = None

    def propose(self, model_code: str, action_names: List[str], budget: int) -> List[str]:
        recent_feedback = [] if self.feedback_log is None else self.feedback_log[-self.max_feedback_items :]
        prompt = {
            "role": "user",
            "content": (
                "You are evolving a signaling model. "
                "Return ONLY a JSON object with key 'actions' containing a list of action names. "
                f"Allowed actions: {action_names}. "
                f"Budget: {budget}. "
                f"Recent fitness feedback: {recent_feedback}. "
                f"Model: {model_code}"
            ),
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Return strict JSON only."},
                prompt,
            ],
            "temperature": 0.2,
        }
        raw = None
        last_error: Exception | None = None
        for attempt in range(self.retry_count + 1):
            try:
                req = urllib.request.Request(
                    url=f"{self.base_url.rstrip('/')}/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                break
            except Exception as exc:
                last_error = exc
                if attempt < self.retry_count:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                break
        if raw is None:
            raise RuntimeError(f"OpenAI-compatible proposer request failed: {last_error}") from last_error

        content = raw["choices"][0]["message"]["content"]
        data = self._parse_json_from_text(content)
        actions = data.get("actions", []) if isinstance(data, dict) else []
        return [str(x) for x in actions]

    def _parse_json_from_text(self, text: str) -> dict:
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            start = match.start()
            try:
                obj, _ = decoder.raw_decode(text[start:])
                if isinstance(obj, dict) and "actions" in obj:
                    return obj
            except json.JSONDecodeError:
                continue

        fenced = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
        if fenced:
            try:
                obj = json.loads(fenced.group(1))
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass

        if not text:
            return {"actions": []}
        return {"actions": []}

    def record_feedback(self, score: float, notes: str) -> None:
        if self.feedback_log is None:
            self.feedback_log = []
        self.feedback_log.append(f"score={score:.4f};{notes}")


@dataclass
class SafeActionFilterProposer:
    """Wraps another proposer and enforces budget + allowlist constraints."""

    inner: ActionProposer

    def propose(self, model_code: str, action_names: List[str], budget: int) -> List[str]:
        allowed = set(action_names)
        proposed = self.inner.propose(model_code, action_names, budget)
        filtered = [a for a in proposed if a in allowed]
        if not filtered:
            return action_names[: max(1, min(budget, len(action_names)))]
        return filtered[: max(1, budget)]

    def record_feedback(self, score: float, notes: str) -> None:
        if hasattr(self.inner, "record_feedback"):
            self.inner.record_feedback(score, notes)
