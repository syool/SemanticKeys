"""
Smoke test for sk/runner.py without a real vLLM server.

Stands up a tiny HTTP server that speaks the OpenAI /v1/chat/completions
protocol and returns canned responses, so the full v3 pipeline (including
the reconstruction stage and guided-decoding client path) can be exercised
end to end. Also monkey-patches the embedder to avoid model downloads.
"""

import hashlib
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sk"))
import numpy as np


# ---------------------------------------------------------------------------
# Fake OpenAI-compatible server
# ---------------------------------------------------------------------------

class FakeHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence request logging
        pass

    def do_GET(self):
        if self.path.endswith("/models"):
            body = json.dumps({"data": [{"id": "fake-model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length))
        system = payload["messages"][0]["content"]
        user = payload["messages"][1]["content"]
        guided = "guided_json" in payload

        content = self._respond(system, user, guided)

        body = json.dumps({
            "choices": [{
                "message": {"content": content},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(system.split()) + len(user.split()),
                "completion_tokens": max(1, len(content.split())),
            },
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _respond(self, system: str, user: str, guided: bool) -> str:
        # Verifier turn (guided decoding) → emit the verdict schema.
        if guided or "Verifier agent" in system:
            return json.dumps({
                "verified": False,
                "reasons": ["The birthplace claim mentions Helsinki which conflicts."],
            })
        # Reconstruction turn → short answer span.
        if "shortest correct span" in system or "Reconstructor" in system:
            return "Vienna"
        # SK protocol turn → SK JSON.
        if "Semantic Keys protocol" in system:
            return json.dumps({
                "topic": "test topic",
                "what": "test action",
                "evidence": "passage 1",
                "explanation": "reasoning chain here",
            })
        # JSON-free turn.
        if '"content"' in system:
            return json.dumps({"content": "free text analysis", "answer": "Vienna"})
        # Default NL turn.
        return "Based on the passages, the analysis proceeds as follows. Vienna appears relevant."


def start_server(port: int = 18123) -> HTTPServer:
    srv = HTTPServer(("127.0.0.1", port), FakeHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


# ---------------------------------------------------------------------------
# Mock embedder
# ---------------------------------------------------------------------------

import metrics


class MockEmbedder(metrics.EmbeddingBackend):
    def __init__(self, model_name=None, cache_dir=None):
        self.model_name = "mock"
        self.cache_dir = Path("/tmp/sk_v3_mock_cache")
        self.cache_dir.mkdir(exist_ok=True)

    def embed(self, text: str):
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:16], 16) % (2**32)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(384)
        return v / (np.linalg.norm(v) + 1e-12)

    def embed_batch(self, texts):
        return np.stack([self.embed(t) for t in texts])


metrics.EmbeddingBackend = MockEmbedder

# ---------------------------------------------------------------------------
# Build a small local dataset file (mimics the offline bundle)
# ---------------------------------------------------------------------------

import data

samples, _ = data.load_hotpotqa(n=3, seed=1)
data_path = Path("/tmp/sk_v3_data.json")
data_path.write_text(json.dumps(samples, ensure_ascii=False), encoding="utf-8")
print(f"[test] Wrote {len(samples)} samples to {data_path}")

# ---------------------------------------------------------------------------
# Run the pipeline
# ---------------------------------------------------------------------------

srv = start_server()
print("[test] Fake vLLM server started on port 18123")

import runner as r3
from client import OpenAICompatibleClient
from scoring import VERIFIER_JSON_SCHEMA

client = OpenAICompatibleClient(base_url="http://127.0.0.1:18123/v1", model="fake-model")
verifier_client = OpenAICompatibleClient(
    base_url="http://127.0.0.1:18123/v1", model="fake-model",
    extra_body={"guided_json": VERIFIER_JSON_SCHEMA},
)
embedder = MockEmbedder()

print("\n[test] Experiment 1...")
exp1 = r3.run_experiment1(client, samples, ["NL-free", "SK"])
assert len(exp1) == 6, f"expected 6, got {len(exp1)}"
for o in exp1:
    assert o.out_tokens > 0
    assert o.reconstructed_answer, "reconstruction stage must produce something"
print(f"  {len(exp1)} observations; sample reconstruction: "
      f"{exp1[0].reconstructed_answer!r} (EM={exp1[0].em}, F1={exp1[0].f1:.2f})")

print("\n[test] Experiment 2...")
exp2 = r3.run_experiment2(client, verifier_client, samples, ["NL-free", "SK"])
assert len(exp2) == 6
parsed = [o for o in exp2 if o.verdict_parsed]
print(f"  {len(exp2)} observations; verdicts parsed: {len(parsed)}/"
      f"{len([o for o in exp2 if o.injection_applied])}")
for o in exp2:
    if o.injection_applied:
        assert o.verdict_parsed, "guided decoding should always parse"
        # Detection should be scored deterministically
        assert isinstance(o.detected_strict, bool)
        assert isinstance(o.propagated, bool)
print(f"  sample: injected={exp2[0].injected_token!r} "
      f"detected_strict={exp2[0].detected_strict} "
      f"detected_lenient={exp2[0].detected_lenient} "
      f"propagated={exp2[0].propagated}")

print("\n[test] Experiment 3...")
exp3 = r3.run_experiment3(
    client, embedder, samples, ["SK", "NL-anchor", "NL-append"], num_turns=3
)
assert len(exp3) == 9
for o in exp3:
    assert len(o.per_turn_drift) == 3
    assert len(o.per_turn_similarity) == 3
    assert len(o.on_topic_labels) == 3
    assert 0.0 <= o.on_topic_proportion <= 1.0
print(f"  {len(exp3)} observations; sample drift={exp3[0].mean_drift:.3f} "
      f"labels={exp3[0].on_topic_labels}")

print("\n[test] Report generation...")
out = Path("/tmp/sk_v3_smoke")
out.mkdir(exist_ok=True)
report = r3.generate_report(exp1, exp2, exp3, out / "report.txt")
print(report)

srv.shutdown()
print("\n[test] v3 offline runner smoke test PASSED")
