from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from modern_biojazz.llm_proposer import OpenAICompatibleProposer, SafeActionFilterProposer


class _LLMHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        body = {
            "choices": [
                {
                    "message": {
                        "content": "{\"actions\":[\"add_binding\",\"DROP_TABLE\",\"add_site\"]}"
                    }
                }
            ]
        }
        raw = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format, *args):
        return


def test_openai_compatible_proposer_with_safety_filter_e2e():
    server = HTTPServer(("127.0.0.1", 0), _LLMHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        raw = OpenAICompatibleProposer(base_url=base, api_key="test", model="x")
        proposer = SafeActionFilterProposer(raw)
        actions = proposer.propose(
            model_code="{}",
            action_names=["add_binding", "add_site", "add_phosphorylation"],
            budget=2,
        )
        assert actions == ["add_binding", "add_site"]
    finally:
        server.shutdown()
        thread.join()
