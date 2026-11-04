"""HTTP serving layer: POST /v1/completions with explicit segments, GET /metrics, GET /health.

Requests carry their prompt as segments, [prefix][documents...][suffix], each tokenized on its
own. That is the API contract that makes chunk reuse well-defined (no tokenization-boundary
ambiguity). The engine is single-GPU and not thread-safe, so requests are serialized with a lock.

Decoding is plain greedy (no sampling, no repetition penalty), so output equals the argmax of a full prefill,
not `model.generate()` with a model's default generation_config (Qwen2.5 sets repetition_penalty=1.1).

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
from fastapi.responses import JSONResponse
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


class LockedTokenizer:
    """Serializes all tokenizer use.

    HF fast tokenizers write `encode_special_tokens` onto the shared Rust tokenizer on every call, so
    concurrent calls with different `split_special_tokens` race: in testing, concurrent use failed with
    tokenizer exceptions, and the same shared flag could let untrusted text be parsed as control tokens.
    Every encode/decode takes this lock and the flag is passed explicitly on every call.
    """

    def __init__(self, tok):
        self.tok, self._lock = tok, threading.Lock()

    def encode(self, text, untrusted=False):
        with self._lock:
            return self.tok(text, add_special_tokens=False, split_special_tokens=untrusted).input_ids

    def decode(self, ids):
        with self._lock:
            return self.tok.decode(ids)


# A token covers at least one character in any sane vocabulary and ~4 on English text; 32 is a loose upper
# bound on chars/token that still rejects multi-megabyte inputs before they are tokenized.
MAX_CHARS_PER_TOKEN = 32
# JSON escapes non-ASCII as \uXXXX (6 bytes/char), so a body can be ~6x its character count.
MAX_BODY_BYTES_PER_CHAR = 6


def build_app(model, tok, engine: Engine, eos_ids, max_context=16384):
    app = FastAPI(title="tread")
    lock = threading.Lock()
    tk = LockedTokenizer(tok)
    counters = dict(requests=0, ttft_ms_sum=0.0, tokens_reused=0, tokens_prefilled=0, generated_tokens=0)
    dev = next(model.parameters()).device
    max_chars = max_context * MAX_CHARS_PER_TOKEN
    max_body = max_chars * MAX_BODY_BYTES_PER_CHAR + (1 << 16)

    @app.middleware("http")
    async def cap_body(request, call_next):
        # Reject on the declared length before the body is read or parsed. Chunked uploads (no length) are refused.
        if request.method == "POST":
            n = request.headers.get("content-length")
            if n is None:
                return JSONResponse({"detail": "Content-Length required"}, status_code=411)
            if not n.isdigit() or int(n) > max_body:
                return JSONResponse({"detail": f"body exceeds {max_body} bytes"}, status_code=413)
        return await call_next(request)

    def tokenize(req):
        """All segments to token-id lists, with every size limit enforced before any tensor is allocated."""
        segments = [(req.prefix, False)] + [(d, True) for d in req.documents] + [(req.suffix, False)]
        if sum(len(s) for s, _ in segments) > max_chars:
            raise HTTPException(413, f"prompt exceeds {max_chars} characters")
        budget, out = max_context - req.max_tokens, []
        for text, untrusted in segments:
            ids = tk.encode(text, untrusted)
            budget -= len(ids)
            if budget < 0:  # stop as soon as the running total is over the limit
                raise HTTPException(413, f"prompt + {req.max_tokens} new tokens exceeds limit {max_context}")
            out.append(ids)
        return out

    app.state.tokenize, app.state.tk = tokenize, tk  # exposed for tests

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model.config._name_or_path}

    @app.post("/v1/completions")
    def complete(req: CompletionRequest):
        ids = tokenize(req)
        n_prompt = sum(map(len, ids))
        as_tensor = lambda x: torch.tensor(x, dtype=torch.long, device=dev)
        prefix, docs, suffix = as_tensor(ids[0]), [as_tensor(d) for d in ids[1:-1]], as_tensor(ids[-1])
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
            "text": tk.decode(out),
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
