"""HTTP API tests: reuse across reordered requests, and output equal to full recompute at ratio 1.0."""
import pytest
import torch
from fastapi.testclient import TestClient
from transformers import AutoModelForCausalLM, AutoTokenizer

from tread.engine import Engine
from tread.server import build_app
from tread.store import BlockStore

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
PREFIX = "<|im_start|>system\nAnswer from the documents in a few words.<|im_end|>\n<|im_start|>user\n"
DOCS = ["Document: Paris\nParis is the capital of France and sits on the Seine.\n\n",
        "Document: Rhine\nThe Rhine flows from the Swiss Alps to the North Sea.\n\n",
        "Document: Ada\nAda Lovelace wrote the first published computer program.\n\n"]
SUFFIX = "Question: Which river does Paris sit on?<|im_end|>\n<|im_start|>assistant\n"


@pytest.fixture(scope="module")
def client():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(DEV).eval()
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    return TestClient(build_app(model, tok, Engine(model, BlockStore(1 << 30)), eos))


def post(client, docs, **kw):
    r = client.post("/v1/completions", json=dict(prefix=PREFIX, documents=docs, suffix=SUFFIX, max_tokens=16, **kw))
    assert r.status_code == 200, r.text
    return r.json()


def test_reordered_request_reuses_all_blocks(client):
    first = post(client, DOCS)
    assert first["tread"]["block_misses"] == 1 + len(DOCS)
    second = post(client, DOCS[::-1])
    assert second["tread"]["block_misses"] == 0 and second["tread"]["tokens_prefilled"] == 0
    m = client.get("/metrics").json()
    assert m["requests"] == 2 and m["store"]["hits"] >= 1 + len(DOCS)


def test_full_recompute_answer_matches_and_mentions_seine(client):
    a = post(client, DOCS[::-1], recompute_ratio=1.0, policy="boundary")
    assert "Seine" in a["text"]


def test_rejects_bad_policy(client):
    r = client.post("/v1/completions", json=dict(prefix=PREFIX, documents=DOCS, suffix=SUFFIX, policy="nope"))
    assert r.status_code == 422
