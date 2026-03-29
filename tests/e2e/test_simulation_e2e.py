from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from modern_biojazz.simulation import CatalystHTTPClient, FitnessEvaluator, LocalCatalystEngine


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(length)
        payload = json.loads(raw.decode("utf-8"))
        rules = payload["network"]["rules"]
        response = {
            "solver": payload["solver"],
            "trajectory": [{"t": 0.0, "output": 0.1}, {"t": payload["t_end"], "output": 1.0}],
            "stats": {"n_rules": len(rules), "n_species": 2, "stiff": True},
        }
        body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


def test_local_catalyst_engine_e2e(seed_network):
    engine = LocalCatalystEngine()
    result = engine.simulate(seed_network, t_end=10.0, dt=1.0, solver="FBDF")
    score = FitnessEvaluator(target_output=1.0).score(result)

    assert result["solver"] in {"BDF", "EulerFallback"}
    assert len(result["trajectory"]) == 11
    assert score > 0.0


def test_http_catalyst_client_e2e(seed_network):
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        client = CatalystHTTPClient(base_url)
        result = client.simulate(seed_network, t_end=8.0, dt=1.0, solver="Rodas5P")
        score = FitnessEvaluator(target_output=1.0).score(result)

        assert result["solver"] == "Rodas5P"
        assert result["stats"]["stiff"] is True
        assert score >= 1.0
    finally:
        server.shutdown()
        thread.join()
