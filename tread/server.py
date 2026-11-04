"""HTTP serving layer: POST /v1/completions with explicit segments, GET /metrics, GET /health.

Requests carry their prompt as segments, [prefix][documents...][suffix], each tokenized on its
own. That is the API contract that makes chunk reuse well-defined (no tokenization-boundary
ambiguity). The engine is single-GPU and not thread-safe, so requests are serialized with a lock.

Trust model: `prefix` and `suffix` are operator-controlled and may contain chat control tokens.
`documents` are untrusted: control-token text in them stays plain text. There is no authentication,
so the server binds to localhost by default. Reuse stats and timing reveal whether a document is
already cached, so a shared deployment must sit behind an authenticating proxy that sets `tenant`
itself: blocks are never reused across tenants.

Run: .venv/Scripts/python -m tread.server --model Qwen/Qwen2.5-0.5B-Instruct --port 8000
"""
import argparse
import threading
import time
from typing import Annotated

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, StringConstraints

from tread.engine import Engine
from tread.store import BlockStore

MAX_SEGMENT_CHARS = 200_000
MAX_DOCUMENTS = 256


class CompletionRequest(BaseModel):
    prefix: str = Field(min_length=1, max_length=MAX_SEGMENT_CHARS,
                        description="shared system prompt / instructions, reused as an exact prefix")
    documents: list[Annotated[str, StringConstraints(min_length=1, max_length=MAX_SEGMENT_CHARS)]] = Field(
        default_factory=list, max_length=MAX_DOCUMENTS, description="chunks reused at any position")
    suffix: str = Field(min_length=1, max_length=MAX_SEGMENT_CHARS,
                        description="the per-request part: question + assistant header")
    tenant: str = Field("", max_length=128, description="cache namespace; blocks are never shared across tenants")
    max_tokens: int = Field(64, ge=1, le=2048)
    recompute_ratio: float = Field(0.2, ge=0.0, le=1.0)
    policy: str = Field("deviation", pattern="^(deviation|boundary|random)$")


def build_app(model, tok, engine: Engine, eos_ids, max_context=16384):
    app = FastAPI(title="tread")
    lock = threading.Lock()
    counters = dict(requests=0, ttft_ms_sum=0.0, tokens_reused=0, tokens_prefilled=0, generated_tokens=0)
    dev = next(model.parameters()).device

    def enc(s, untrusted=False):
        ids = tok(s, add_special_tokens=False, split_special_tokens=untrusted).input_ids
        return torch.tensor(ids, dtype=torch.long, device=dev)

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model.config._name_or_path}

    @app.post("/v1/completions")
    def complete(req: CompletionRequest):
        prefix, docs, suffix = enc(req.prefix), [enc(d, untrusted=True) for d in req.documents], enc(req.suffix)
        n_prompt = len(prefix) + sum(map(len, docs)) + len(suffix)
        if n_prompt + req.max_tokens > max_context:
            raise HTTPException(413, f"{n_prompt} prompt + {req.max_tokens} new tokens exceeds limit {max_context}")
        with lock, torch.inference_mode():
            t0 = time.perf_counter()
            logits, cache, st = engine.prefill(prefix, docs, suffix, ratio=req.recompute_ratio, policy=req.policy,
                                               namespace=req.tenant)
            out = []
            nxt = logits.argmax()
            while nxt.item() not in eos_ids and len(out) < req.max_tokens:
                out.append(nxt.item())
                o = model(nxt.view(1, 1), past_key_values=cache, use_cache=True, num_logits_to_keep=1)
                cache, nxt = o.past_key_values, o.logits[0, -1].argmax()
            total = time.perf_counter() - t0
        counters["requests"] += 1
        counters["ttft_ms_sum"] += st.t["ttft"] * 1000
        counters["tokens_reused"] += st.tokens_reused
        counters["tokens_prefilled"] += st.tokens_prefilled
        counters["generated_tokens"] += len(out)
        return {
            "text": tok.decode(out),
            "usage": {"prompt_tokens": n_prompt, "completion_tokens": len(out)},
            "tread": {"block_hits": st.hits, "block_misses": st.misses, "tokens_reused": st.tokens_reused,
                      "tokens_prefilled": st.tokens_prefilled, "recompute_eff": round(st.recompute_eff, 4),
                      "ttft_ms": round(st.t["ttft"] * 1000, 2), "total_ms": round(total * 1000, 2)},
        }

    @app.get("/metrics")
    def metrics():
        s = engine.store.stats
        n = max(1, counters["requests"])
        return {**counters, "ttft_ms_mean": counters["ttft_ms_sum"] / n,
                "store": {"blocks_gpu": len(engine.store.blocks), "blocks_cpu": len(engine.store.cpu_blocks),
                          "gpu_bytes": s.bytes_used, "cpu_bytes": s.cpu_bytes_used, "hits": s.hits,
                          "misses": s.misses, "cpu_hits": s.cpu_hits, "offloads": s.offloads,
                          "evictions": s.evictions, "cpu_evictions": s.cpu_evictions}}

    return app


def main():
    import uvicorn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--gpu-mb", type=int, default=256)
    ap.add_argument("--cpu-mb", type=int, default=2048)
    ap.add_argument("--graphs", action="store_true", help="CUDA-graph recompute for contexts <= 2k tokens")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype)).to("cuda").eval()
    engine = Engine(model, BlockStore(args.gpu_mb << 20, args.cpu_mb << 20), graphs=args.graphs)
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    uvicorn.run(build_app(model, tok, engine, eos), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
