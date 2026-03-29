from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from modern_biojazz.grounding_sources import (
    OmniPathClient,
    INDRAClient,
    load_grounding_snapshot,
    build_grounding_payload_from_sources,
)


class _SourcesHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path.startswith("/interactions/"):
            payload = [
                {
                    "source_genesymbol": "STAT3_HUMAN",
                    "target_genesymbol": "SOCS3_HUMAN",
                }
            ]
        else:
            payload = []
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):  # noqa: N802
        if self.path.startswith("/statements/from_agents"):
            payload = {
                "statements": [
                    {
                        "type": "phosphorylation",
                        "agents": [{"name": "STAT3_HUMAN"}, {"name": "SOCS3_HUMAN"}],
                    }
                ]
            }
        else:
            payload = {"statements": []}
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format, *args):
        return


def test_grounding_sources_and_snapshot_e2e(fixtures_dir):
    snapshot = load_grounding_snapshot(fixtures_dir / "grounding_snapshot.json")
    assert "abstract_types" in snapshot

    server = HTTPServer(("127.0.0.1", 0), _SourcesHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        op = OmniPathClient(base_url=base)
        indra = INDRAClient(base_url=base)

        op_rows = op.fetch_interactions(["STAT3_HUMAN", "SOCS3_HUMAN"])
        stmts = indra.fetch_statements(["STAT3_HUMAN", "SOCS3_HUMAN"])
        payload = build_grounding_payload_from_sources(
            abstract_types=snapshot["abstract_types"],
            omnipath_rows=op_rows,
            indra_statements=stmts,
        )

        assert len(op_rows) == 1
        assert len(stmts) == 1
        assert "real_nodes" in payload
        assert "real_interactions" in payload
    finally:
        server.shutdown()
        thread.join()
