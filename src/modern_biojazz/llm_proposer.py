from __future__ import annotations

import ipaddress
import json
import re
import socket
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPSConnection
from typing import List, Protocol


class _ValidatedHTTPSConnection(HTTPSConnection):
    def __init__(self, host, port=None, safe_ip=None, **kwargs):
        self._safe_ip = safe_ip
        super().__init__(host, port, **kwargs)

    def connect(self):
        self.sock = socket.create_connection((self._safe_ip, self.port), self.timeout, self.source_address)
        if self._tunnel_host:
            self._tunnel()
        if self._context:
            server_hostname = self.host
            self.sock = self._context.wrap_socket(self.sock, server_hostname=server_hostname)


class _ValidatedHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, safe_ip, **kwargs):
        self._safe_ip = safe_ip
        super().__init__(**kwargs)

    def https_open(self, req):
        def build(host, port=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, **kwargs):
            return _ValidatedHTTPSConnection(host, port=port, timeout=timeout, safe_ip=self._safe_ip, **kwargs)

        return self.do_open(build, req)


class _ValidatedHTTPConnection(HTTPConnection):
    def __init__(self, host, port=None, safe_ip=None, **kwargs):
        self._safe_ip = safe_ip
        super().__init__(host, port, **kwargs)

    def connect(self):
        self.sock = socket.create_connection((self._safe_ip, self.port), self.timeout, self.source_address)
        if self._tunnel_host:
            self._tunnel()


class _ValidatedHTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, safe_ip, **kwargs):
        self._safe_ip = safe_ip
        super().__init__(**kwargs)

    def http_open(self, req):
        def build(host, port=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, **kwargs):
            return _ValidatedHTTPConnection(host, port=port, timeout=timeout, safe_ip=self._safe_ip, **kwargs)

        return self.do_open(build, req)


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

    def _validate_url(self, url: str) -> str:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https":
            raise ValueError(f"Only HTTPS URLs are allowed, got: {parsed.scheme}")
        hostname = parsed.hostname
        if not hostname:
            raise ValueError("Invalid hostname in URL")
        try:
            addr_info = socket.getaddrinfo(hostname, None)
        except socket.gaierror as e:
            raise ValueError(f"Could not resolve hostname {hostname}: {e}")

        safe_ip = None
        for _, _, _, _, sockaddr in addr_info:
            ip = sockaddr[0]
            if safe_ip is None:
                safe_ip = ip
            try:
                ip_obj = ipaddress.ip_address(ip)
            except ValueError as e:
                raise ValueError(f"Invalid IP address resolved {ip}: {e}")
            if (
                ip_obj.is_loopback
                or ip_obj.is_private
                or ip_obj.is_link_local
                or ip_obj.is_multicast
                or ip_obj.is_reserved
            ):
                raise ValueError(f"URL resolves to an internal or reserved IP address: {ip}")
        if safe_ip is None:
            raise ValueError(f"Could not resolve any IP for hostname {hostname}")
        return safe_ip

    def propose(self, model_code: str, action_names: List[str], budget: int) -> List[str]:
        safe_ip = self._validate_url(self.base_url)

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
        opener = urllib.request.build_opener(
            _ValidatedHTTPSHandler(safe_ip=safe_ip),
            _ValidatedHTTPHandler(safe_ip=safe_ip),
        )
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
                with opener.open(req, timeout=self.timeout_seconds) as response:
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
class LLMDenoisingProposer:
    """Implements discrete diffusion denoising using an LLM to restore/improve a mutated network."""

    inner: OpenAICompatibleProposer

    def propose(self, model_code: str, action_names: List[str], budget: int) -> List[str]:
        self.inner._validate_url(self.inner.base_url)
        prompt = {
            "role": "user",
            "content": (
                "You are an expert biological network builder. We have applied random noise "
                "(mutations) to a signaling network. Your task is to act as a denoising diffusion step: "
                "propose structural actions to repair, connect, or improve this network back to a biologically "
                "valid and functional state. "
                "Return ONLY a JSON object with key 'actions' containing a list of action names. "
                f"Allowed actions: {action_names}. "
                f"Budget: {budget}. "
                f"Noisy Model: {model_code}"
            ),
        }

        payload = {
            "model": self.inner.model,
            "messages": [
                {"role": "system", "content": "Return strict JSON only."},
                prompt,
            ],
            "temperature": 0.4,
        }

        raw = None
        last_error: Exception | None = None
        for attempt in range(self.inner.retry_count + 1):
            try:
                req = urllib.request.Request(
                    url=f"{self.inner.base_url.rstrip('/')}/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.inner.api_key}",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.inner.timeout_seconds) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                break
            except Exception as exc:
                last_error = exc
                if attempt < self.inner.retry_count:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                break

        if raw is None:
            raise RuntimeError(f"Denoising request failed: {last_error}") from last_error

        content = raw["choices"][0]["message"]["content"]
        data = self.inner._parse_json_from_text(content)
        actions = data.get("actions", []) if isinstance(data, dict) else []
        return [str(x) for x in actions]

    def record_feedback(self, score: float, notes: str) -> None:
        self.inner.record_feedback(score, notes)

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
