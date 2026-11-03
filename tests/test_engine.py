"""Correctness tests for the reuse engine. Uses Qwen2.5-0.5B in fp32 so exactness checks are tight.

Run: .venv/Scripts/python -m pytest tests -q
"""
import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from tread.blend import assemble, prefill_kv, rotate_keys, to_cache
from tread.engine import Engine, prefill_full
from tread.store import BlockStore

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def mt():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(DEV).eval()
    return model, tok


def enc(tok, s):
    return torch.tensor(tok(s, add_special_tokens=False).input_ids, device=DEV)


@pytest.fixture(scope="module")
def req(mt):
    _, tok = mt
    prefix = enc(tok, "<|im_start|>system\nAnswer from the documents.<|im_end|>\n<|im_start|>user\n")
    docs = [enc(tok, f"Document: {t}\n{b}\n\n") for t, b in [
        ("Paris", "Paris is the capital of France and sits on the Seine."),
        ("Rhine", "The Rhine flows from the Swiss Alps to the North Sea."),
        ("Ada", "Ada Lovelace wrote the first published computer program."),
    ]]
    query = enc(tok, "Question: Which river does Paris sit on?<|im_end|>\n<|im_start|>assistant\n")
    return prefix, docs, query


def kl(a, b):
    return F.kl_div(F.log_softmax(a.float(), -1), F.log_softmax(b.float(), -1), log_target=True, reduction="sum").item()


@torch.inference_mode()
def test_rotation_composes(mt, req):
    """Keys computed at position 0 and shifted by d equal keys computed at position d."""
    model, _ = mt
    prefix, docs, _ = req
    K0, _ = prefill_kv(model, docs[0])
    d = 37
    Kd, _ = prefill_kv(model, torch.cat([torch.zeros(d, dtype=torch.long, device=DEV), docs[0]]))
    # Only layer 0 is context-free; compare the doc slice there.
    assert torch.allclose(rotate_keys(model, K0[0], d), Kd[0][:, :, d:], atol=1e-3)


@torch.inference_mode()
def test_full_recompute_matches_full_prefill(mt, req):
    model, _ = mt
    prefix, docs, query = req
    eng = Engine(model, BlockStore(1 << 30))
    ref, _, _ = prefill_full(model, prefix, docs, query)
    out, _, _ = eng.prefill(prefix, docs, query, ratio=1.0, policy="boundary")
    assert kl(ref, out) < 1e-5


@torch.inference_mode()
def test_prefix_only_matches_full_prefill(mt, req):
    model, _ = mt
    prefix, docs, query = req
    eng = Engine(model, BlockStore(1 << 30))
    ref, _, _ = prefill_full(model, prefix, docs, query)
    out, _, _ = eng.prefill_prefix_only(prefix, docs, query)
    assert kl(ref, out) < 1e-5


@torch.inference_mode()
def test_reordered_request_hits_cache_and_matches_cold_assembly(mt, req):
    """A reordered request reuses every block and gives the same result as assembling cold."""
    model, _ = mt
    prefix, docs, query = req
    eng = Engine(model, BlockStore(1 << 30))
    _, _, st1 = eng.prefill(prefix, docs, query)
    assert st1.hits == 0 and st1.misses == 1 + len(docs)

    reordered = docs[::-1]
    warm, _, st2 = eng.prefill(prefix, reordered, query)
    assert st2.hits == 1 + len(docs) and st2.misses == 0 and st2.tokens_prefilled == 0

    cold, _, _ = Engine(model, BlockStore(1 << 30)).prefill(prefix, reordered, query)
    assert kl(cold, warm) < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
@torch.inference_mode()
@pytest.mark.parametrize("policy,ratio", [("deviation", 0.2), ("boundary", 0.3), ("boundary", 1.0)])
def test_cuda_graph_blend_matches_eager(mt, req, policy, ratio):
    """Graphed recompute (padded into a size bucket) gives the same result as the eager loop."""
    model, _ = mt
    prefix, docs, query = req
    eager = Engine(model, BlockStore(1 << 30))
    graphed = Engine(model, BlockStore(1 << 30), graphs=True)
    eager.prefill(prefix, docs, query)
    graphed.prefill(prefix, docs, query)
    a, _, _ = eager.prefill(prefix, docs[::-1], query, ratio=ratio, policy=policy)
    for _ in range(2):  # second call replays the captured graph
        b, _, _ = graphed.prefill(prefix, docs[::-1], query, ratio=ratio, policy=policy)
    assert graphed.runner.captures == 1
    assert kl(a, b) < 1e-4
    if ratio == 1.0:
        ref, _, _ = prefill_full(model, prefix, docs[::-1], query)
        assert kl(ref, b) < 1e-4


@torch.inference_mode()
def test_batched_cold_path_matches_per_doc_prefill(mt, req):
    """All misses prefilled in one packed forward == each doc prefilled separately after the prefix."""
    model, _ = mt
    prefix, docs, query = req
    K_ref, V_ref = assemble(model, prefix, docs, "shared")
    ref = model(query[None], past_key_values=to_cache(K_ref, V_ref)).logits[0, -1]
    eng = Engine(model, BlockStore(1 << 30))
    out, cache, st = eng.prefill(prefix, docs, query)
    assert st.misses == 1 + len(docs)
    T = len(prefix) + sum(len(d) for d in docs)
    for l in (0, len(K_ref) // 2, len(K_ref) - 1):
        assert torch.allclose(cache.key_cache[l][:, :, :T], K_ref[l], atol=1e-3)
    assert kl(ref, out) < 1e-5


@torch.inference_mode()
def test_cpu_tier_offloads_and_serves_hits(mt, req):
    """With a tiny GPU budget, evicted blocks go to pinned CPU and still hit (and stay exact)."""
    model, _ = mt
    prefix, docs, query = req
    probe = BlockStore(1 << 30)
    Engine(model, probe).prefill(prefix, docs, query)
    one_block = max(b.nbytes for b in probe.blocks.values())
    store = BlockStore(2 * one_block, cpu_budget_bytes=1 << 30)
    eng = Engine(model, store)
    eng.prefill(prefix, docs, query)
    assert store.stats.offloads > 0 and store.stats.bytes_used <= store.budget
    assert store.stats.cpu_evictions == 0
    warm, _, st = eng.prefill(prefix, docs[::-1], query)
    assert st.misses == 0 and store.stats.cpu_hits > 0
    cold, _, _ = Engine(model, BlockStore(1 << 30)).prefill(prefix, docs[::-1], query)
    assert kl(cold, warm) < 1e-5


@torch.inference_mode()
def test_block_key_depends_on_prefix(mt, req):
    """The same doc after a different prefix must not hit: its KV depends on the prefix."""
    model, tok = mt
    prefix, docs, query = req
    eng = Engine(model, BlockStore(1 << 30))
    eng.prefill(prefix, docs, query)
    other = enc(tok, "<|im_start|>system\nBe terse.<|im_end|>\n<|im_start|>user\n")
    _, _, st = eng.prefill(other, docs, query)
    assert st.hits == 0


@torch.inference_mode()
def test_eviction_respects_budget(mt, req):
    model, _ = mt
    prefix, docs, query = req
    store = BlockStore(1 << 30)
    Engine(model, store).prefill(prefix, docs, query)
    one_block = max(b.nbytes for b in store.blocks.values())
    small = BlockStore(2 * one_block)
    eng = Engine(model, small)
    eng.prefill(prefix, docs, query)
    assert small.stats.bytes_used <= small.budget
    assert small.stats.evictions > 0
    # Still correct when blocks were evicted mid-request.
    ref, _, _ = prefill_full(model, prefix, docs, query)
    out, _, _ = eng.prefill(prefix, docs, query, ratio=1.0, policy="boundary")
    assert kl(ref, out) < 1e-5
