from __future__ import annotations

from modern_biojazz.llm_proposer import OpenAICompatibleProposer, SafeActionFilterProposer


def test_parse_json_handles_markdown_fences():
    proposer = OpenAICompatibleProposer(base_url="http://example", api_key="k", model="m")
    text = "Response:\n```json\n{\"actions\":[\"add_site\"]}\n```"
    payload = proposer._parse_json_from_text(text)
    assert payload["actions"] == ["add_site"]


def test_parse_json_returns_empty_actions_on_malformed_response():
    proposer = OpenAICompatibleProposer(base_url="http://example", api_key="k", model="m")
    payload = proposer._parse_json_from_text("{not valid json")
    assert payload == {"actions": []}


class _Inner:
    def __init__(self):
        self.feedback = []

    def propose(self, model_code: str, action_names: list[str], budget: int) -> list[str]:
        del model_code
        del budget
        return action_names[:1]

    def record_feedback(self, score: float, notes: str) -> None:
        self.feedback.append((score, notes))


def test_safe_filter_proposer_propagates_feedback():
    inner = _Inner()
    wrapper = SafeActionFilterProposer(inner)
    wrapper.record_feedback(0.9, "ok")
    assert inner.feedback == [(0.9, "ok")]
